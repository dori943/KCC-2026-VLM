"""
robot_controller.py
────────────────────────────────────────────────────────────────────────────────
VL-Grasp (IROS 2023) 기반 로봇팔 제어 모듈

포인트클라우드 파이프라인 (VL-Grasp demo.py 원본 방식):
  1. CameraInfo + create_point_cloud_from_depth_image → 카메라 좌표계 포인트클라우드
  2. bbox mask로 물체 영역만 필터링
  3. FGC-GraspNet 추론 (카메라 좌표계 입력)
  4. GraspGroup (직접 구현: NMS + sort_by_score) → 최적 grasp 선택
  5. 카메라 좌표 → 월드 좌표 변환
  6. PyBullet IK → Franka Panda 제어
────────────────────────────────────────────────────────────────────────────────
"""

import sys
import os

# ── sys.path 먼저 설정 ────────────────────────────────────────────────────────
VL_GRASP_ROOT = "/root/KCC2026_VLM/VL-Grasp"
sys.path.insert(0, VL_GRASP_ROOT)
sys.path.insert(0, os.path.join(VL_GRASP_ROOT, "RoboRefIt"))
sys.path.insert(0, os.path.join(VL_GRASP_ROOT, "GraspNet"))

# ── 일반 라이브러리 ───────────────────────────────────────────────────────────
import pybullet as p
import numpy as np
import torch
import time
import torchvision.transforms as T
from PIL import Image

# ── VL-Grasp 내부 모듈 ───────────────────────────────────────────────────────
from model.decode import pred_decode
from model.FGC_graspnet import FGC_graspnet as GraspNetModel
from models import build_reftr_seg
from main_vg import get_args_parser
from util.misc import nested_tensor_from_tensor_list
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image

# ── 카메라 파라미터 ───────────────────────────────────────────────────────────
CAM_WIDTH  = 640
CAM_HEIGHT = 480
CAM_FOV    = 60
CAM_NEAR   = 0.1
CAM_FAR    = 5.0

# ── Panda 상수 ────────────────────────────────────────────────────────────────
END_EFFECTOR_INDEX = 11
NUM_JOINTS         = 7

# ── Franka Panda 홈 포지션 (팔을 위로 펴서 작업 준비 자세) ───────────────────
PANDA_HOME_JOINTS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

# ── IK 관절 범위 (Franka Panda 실제 범위) ────────────────────────────────────
PANDA_LOWER = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
PANDA_UPPER = [ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973]
PANDA_RANGE = [ 5.7946,  3.5256,  5.7946,  3.0020,  5.7946,  3.7700,  5.7946]

# ════════════════════════════════════════════════════════════════════════════════
#  GraspGroup 직접 구현 (graspnetAPI 대체)
#  원본 GraspGroup과 동일한 기능:
#    - nms()          : 위치가 가까운 grasp 중복 제거
#    - sort_by_score(): score 기준 내림차순 정렬
#    - __getitem__    : 인덱스로 단일 Grasp 접근
#
#  grasp array 컬럼 구조 (17개):
#  [score, width, height, depth, r00,r01,r02,r10,r11,r12,r20,r21,r22, x,y,z, obj_id]
#   0      1      2       3      4~12(rotation matrix)                 13~15  16
# ════════════════════════════════════════════════════════════════════════════════

class Grasp:
    """단일 grasp pose를 나타내는 클래스."""
    def __init__(self, array):
        self._array = array  # (17,)

    @property
    def score(self):
        return float(self._array[0])

    @property
    def width(self):
        return float(self._array[1])

    @property
    def height(self):
        return float(self._array[2])

    @property
    def depth(self):
        return float(self._array[3])

    @property
    def rotation_matrix(self):
        return self._array[4:13].reshape(3, 3)

    @property
    def translation(self):
        return self._array[13:16]

    @property
    def object_id(self):
        return int(self._array[16])


class GraspGroup:
    """
    graspnetAPI의 GraspGroup을 Python 3.12 호환으로 직접 구현.
    NMS + sort_by_score 기능 포함.
    """
    def __init__(self, array):
        """
        array: (N, 17) numpy array
          각 행 = [score, width, height, depth, R(9), translation(3), obj_id]
        """
        self._array = np.array(array, dtype=np.float32)  # (N, 17)

    def __len__(self):
        return len(self._array)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return Grasp(self._array[idx])
        return GraspGroup(self._array[idx])

    def sort_by_score(self):
        """score(col 0) 기준 내림차순 정렬 (in-place)."""
        order = np.argsort(self._array[:, 0])[::-1]
        self._array = self._array[order]
        return self

    def nms(self, translation_thresh=0.03, rotation_thresh=30.0):
        """
        Non-Maximum Suppression:
        위치가 translation_thresh(m) 이내이고
        회전 각도 차이가 rotation_thresh(도) 이내인 grasp 중 score 낮은 것 제거.

        원본 graspnetAPI NMS와 동일한 로직.
        """
        if len(self._array) == 0:
            return self

        # score 기준으로 먼저 정렬
        self.sort_by_score()
        arr      = self._array
        keep     = np.ones(len(arr), dtype=bool)

        translations = arr[:, 13:16]  # (N, 3)
        rotations    = arr[:, 4:13].reshape(-1, 9)  # (N, 9)

        for i in range(len(arr)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(arr)):
                if not keep[j]:
                    continue
                # 위치 거리
                dist = np.linalg.norm(translations[i] - translations[j])
                if dist > translation_thresh:
                    continue
                # 회전 각도 차이 (Frobenius norm 근사)
                R_i   = rotations[i].reshape(3, 3)
                R_j   = rotations[j].reshape(3, 3)
                R_rel = R_i.T @ R_j
                # 회전 각도 = arccos((trace(R)-1)/2)
                trace = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(trace))
                if angle_deg < rotation_thresh:
                    keep[j] = False  # score 낮은 j 제거

        self._array = arr[keep]
        return self

    def to_numpy(self):
        return self._array.copy()


# ════════════════════════════════════════════════════════════════════════════════
#  1. PyBullet 가상 카메라 렌더링
# ════════════════════════════════════════════════════════════════════════════════
def render_camera(cam_target=[0.55, -0.35, 0.8], cam_distance=1.0,
                  cam_yaw=45, cam_pitch=-45):
    """
    PyBullet 가상 카메라로 RGB + Depth 렌더링.
    Returns:
        rgb         : (H, W, 3) uint8
        depth       : (H, W)    float32  실제 미터 단위
        proj_matrix : (16,) float
        view_matrix : (16,) float
    """
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=cam_target,
        distance=cam_distance,
        yaw=cam_yaw, pitch=cam_pitch, roll=0,
        upAxisIndex=2
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=CAM_FOV, aspect=CAM_WIDTH / CAM_HEIGHT,
        nearVal=CAM_NEAR, farVal=CAM_FAR
    )
    _, _, rgb_raw, depth_raw, _ = p.getCameraImage(
        CAM_WIDTH, CAM_HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_TINY_RENDERER
    )
    rgb = np.array(rgb_raw, dtype=np.uint8).reshape(CAM_HEIGHT, CAM_WIDTH, 4)[:, :, :3]
    depth_buf = np.array(depth_raw, dtype=np.float32).reshape(CAM_HEIGHT, CAM_WIDTH)
    depth = CAM_FAR * CAM_NEAR / (CAM_FAR - (CAM_FAR - CAM_NEAR) * depth_buf)
    return rgb, depth, proj_matrix, view_matrix


# ════════════════════════════════════════════════════════════════════════════════
#  2. Depth → 포인트클라우드 (VL-Grasp demo.py 원본 방식 — 카메라 좌표계)
# ════════════════════════════════════════════════════════════════════════════════
def depth_to_pointcloud_camera(depth, proj_matrix, workspace_mask=None):
    """
    VL-Grasp demo.py 원본과 동일한 방식으로 포인트클라우드 생성.
    → 카메라 좌표계 기준 (월드 변환 없음)
    → GraspNet이 카메라 좌표계로 학습됐으므로 이 방식이 올바름

    Returns:
        cloud_masked : (N, 3) float32  카메라 좌표계
    """
    h, w = depth.shape

    # proj_matrix → fx, fy, cx, cy 추출
    proj = np.array(proj_matrix).reshape(4, 4).T  # column-major → row-major
    fx = proj[0, 0] * w / 2.0
    fy = proj[1, 1] * h / 2.0
    cx = (1.0 - proj[0, 2]) * w / 2.0
    cy = (1.0 + proj[1, 2]) * h / 2.0

    # ✅ VL-Grasp 원본 CameraInfo 사용
    # depth가 미터 단위이므로 scale=1.0
    camera = CameraInfo(float(w), float(h), fx, fy, cx, cy, scale=1.0)
    cloud  = create_point_cloud_from_depth_image(depth, camera, organized=True)
    # cloud: (H, W, 3) 카메라 좌표계

    # 유효 포인트 마스크
    depth_mask = (depth > CAM_NEAR + 0.01) & (depth < CAM_FAR - 0.01)
    if workspace_mask is not None:
        mask = depth_mask & workspace_mask
    else:
        mask = depth_mask

    cloud_masked = cloud[mask]
    return cloud_masked.astype(np.float32)


def camera_to_world(points_cam, view_matrix):
    """
    카메라 좌표계 → PyBullet 월드 좌표계 변환
    - view_matrix: reshape(4,4).T 로 파싱 (PyBullet column-major)
    - GraspNet 포인트클라우드는 OpenCV 규칙(z=앞, y=아래)
    - PyBullet은 OpenGL 규칙(z=뒤, y=위) → y,z 반전 후 변환
    """
    vm     = np.array(view_matrix).reshape(4, 4).T
    inv_vm = np.linalg.inv(vm)

    pts = np.array(points_cam, dtype=np.float64)

    if pts.ndim == 1:
        pts_conv = np.array([pts[0], -pts[1], -pts[2], 1.0])  # y,z 반전
        return (inv_vm @ pts_conv)[:3]
    else:
        pts_conv = pts.copy()
        pts_conv[:, 1] *= -1  # y 반전
        pts_conv[:, 2] *= -1  # z 반전
        homo = np.hstack([pts_conv, np.ones((len(pts_conv), 1))])
        return (inv_vm @ homo.T).T[:, :3]
    


def make_bbox_mask(h, w, bbox):
    """bbox [x1,y1,x2,y2] 영역의 boolean mask (H, W) 생성."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    mask = np.zeros((h, w), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask

# ════════════════════════════════════════════════════════════════════════════════
#  3. GraspNet rotation → PyBullet EEF quaternion 변환
# ════════════════════════════════════════════════════════════════════════════════
def graspnet_rot_to_pybullet_quat(rot_cam, view_matrix):
    """
    GraspNet rotation matrix (카메라 좌표계) → PyBullet EEF quaternion (월드 좌표계)
 
    GraspNet rotation 축 정의:
      R[:,0] = approach vector  (그리퍼 접근 방향 = PyBullet EEF z축)
      R[:,1] = binormal
      R[:,2] = major axis (그리퍼 열리는 방향 = PyBullet EEF x축)
 
    PyBullet EEF 축 정의:
      z축 = 그리퍼 접근 방향 (아래를 향해야 함)
      x축 = 그리퍼 열리는 방향
 
    변환 순서:
      1. GraspNet cam → 월드 좌표 rotation
      2. 축 재배열: GraspNet[col0,col1,col2] → PyBullet[z,y,x] 매핑
    """
    # Step 1: 카메라 rotation → 월드 rotation (y,z 반전 포함)
    vm = np.array(view_matrix).reshape(4, 4).T
    R_view = vm[:3, :3]  # world→camera rotation
 
    # OpenCV→OpenGL 축 변환 행렬
    T_cv2gl = np.diag([1.0, -1.0, -1.0])
 
    # 카메라 rotation을 OpenGL로 변환 후 월드로
    rot_gl    = T_cv2gl @ rot_cam   # OpenCV→OpenGL
    rot_world = R_view.T @ rot_gl   # camera→world
 
    # Step 2: GraspNet 축 → PyBullet EEF 축 재배열
    # GraspNet col0=approach → PyBullet z
    # GraspNet col1=binormal → PyBullet y (또는 -y)
    # GraspNet col2=major    → PyBullet x
    approach = rot_world[:, 0]   # 접근 방향
    binormal = rot_world[:, 1]
    major    = rot_world[:, 2]   # 그리퍼 열리는 방향
 
    # PyBullet EEF 좌표계 구성
    # z = approach (정규화)
    z_axis = approach / (np.linalg.norm(approach) + 1e-8)
    # x = major (정규화)
    x_axis = major / (np.linalg.norm(major) + 1e-8)
    # y = z × x (우수 좌표계 유지)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)
    # x 재계산 (직교 보정)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
 
    # 회전 행렬 → 쿼터니언
    R_eef = np.stack([x_axis, y_axis, z_axis], axis=1)  # (3,3)
    quat  = rotation_matrix_to_quaternion(R_eef)
    return quat

# ════════════════════════════════════════════════════════════════════════════════
#  3. VL-Grasp 모델 로드 (싱글턴)
# ════════════════════════════════════════════════════════════════════════════════
_vl_grasp_models = None

def load_vl_grasp_models(
    vg_checkpoint = "/root/KCC2026_VLM/VL-Grasp/logs/checkpoint_best_r50.pth",
    gn_checkpoint = "/root/KCC2026_VLM/VL-Grasp/logs/checkpoint_fgc.tar",
    device        = "cuda"
):
    """RoboRefIt (VG) + FGC-GraspNet 모델을 한 번만 로드."""
    global _vl_grasp_models
    if _vl_grasp_models is not None:
        return _vl_grasp_models

    # ── Visual Grounding (RoboRefIt) ──────────────────────────────
    vg_args = get_args_parser().parse_args([])
    vg_args.device   = device
    vg_args.masks    = True
    vg_args.img_type = 'RGB'
    vg_model, _, _ = build_reftr_seg(vg_args)
    vg_ckpt = torch.load(vg_checkpoint, map_location=device, weights_only=False)
    vg_model.load_state_dict(vg_ckpt["model"], strict=False)
    vg_model.to(device).eval()
    print("[VL-Grasp] VG 모델 로드 완료")

    # ── FGC-GraspNet ──────────────────────────────────────────────
    # is_training=False, is_demo=True → process_grasp_labels 스킵
    gn_model = GraspNetModel(is_training=False, is_demo=True)
    gn_ckpt  = torch.load(gn_checkpoint, map_location=device, weights_only=False)
    gn_model.load_state_dict(gn_ckpt["model_state_dict"])
    gn_model.to(device).eval()
    print("[VL-Grasp] GraspNet 모델 로드 완료")

    # ── BertTokenizer ─────────────────────────────────────────────
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    _vl_grasp_models = {
        "vg":        vg_model,
        "gn":        gn_model,
        "tokenizer": tokenizer,
        "device":    device,
    }
    print("[VL-Grasp] 전체 모델 로드 완료")
    return _vl_grasp_models


# ════════════════════════════════════════════════════════════════════════════════
#  4. VL-Grasp 추론: 언어 명령 → 6-DoF grasp pose
# ════════════════════════════════════════════════════════════════════════════════
def get_grasp_pose_from_vl_grasp(
    rgb:        np.ndarray,   # (H, W, 3) uint8
    depth:      np.ndarray,   # (H, W)    float32 미터
    proj_matrix,
    view_matrix,
    language:   str,
    models:     dict = None,
    num_points: int  = 20000,
) -> dict:
    """
    VL-Grasp 전체 파이프라인.

    Returns:
        {
            "position":    [x, y, z],        # PyBullet 월드 좌표 (미터)
            "orientation": [qx, qy, qz, qw], # 쿼터니언
            "bbox":        [x1, y1, x2, y2]  # VG 탐지 bbox (픽셀)
        }
    """
    if models is None:
        models = load_vl_grasp_models()

    device    = models["device"]
    tokenizer = models["tokenizer"]

    # ── Step 1: 이미지 전처리 ────────────────────────────────────
    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)

    # ── Step 2: 언어 토크나이징 ──────────────────────────────────
    encoded = tokenizer(
        language, padding="max_length", max_length=40,
        truncation=True, return_tensors="pt"
    )
    sentence      = encoded["input_ids"].to(device)
    sentence_mask = encoded["attention_mask"].to(device)

    # ── Step 3: Visual Grounding → bbox ─────────────────────────
    samples = {
        "img":           nested_tensor_from_tensor_list(img_tensor),
        "sentence":      sentence,
        "sentence_mask": sentence_mask,
    }
    with torch.no_grad():
        vg_out = models["vg"](samples)

    # pred_boxes: (batch, n_phrase, n_query, 4) [cx,cy,w,h] normalized
    pred_box = vg_out["pred_boxes"][0, 0, 0].cpu().numpy()
    h_img, w_img = rgb.shape[:2]
    cx_b, cy_b, bw, bh = pred_box
    x1 = int((cx_b - bw / 2) * w_img)
    y1 = int((cy_b - bh / 2) * h_img)
    x2 = int((cx_b + bw / 2) * w_img)
    y2 = int((cy_b + bh / 2) * h_img)
    bbox = [x1, y1, x2, y2]
    print(f"[VG] '{language}' → bbox: {bbox}")

    # ── Step 4: 포인트클라우드 생성 (카메라 좌표계) ──────────────
    # ✅ VL-Grasp 원본 방식: CameraInfo + create_point_cloud_from_depth_image
    # bbox 영역만 workspace_mask로 필터링
    bbox_mask  = make_bbox_mask(h_img, w_img, bbox)
    cloud_bbox = depth_to_pointcloud_camera(depth, proj_matrix, workspace_mask=bbox_mask)
    cloud_full = depth_to_pointcloud_camera(depth, proj_matrix, workspace_mask=None)

    print(f"[포인트클라우드] 전체: {len(cloud_full)}, bbox: {len(cloud_bbox)}")

    # bbox 포인트가 부족하면 bbox 중심 반경 0.15m 이내 포인트 추가
    if len(cloud_bbox) >= num_points:
        cloud = cloud_bbox
    elif len(cloud_bbox) >= 50:
        center     = cloud_bbox.mean(axis=0)
        dists      = np.linalg.norm(cloud_full - center, axis=1)
        cloud_near = cloud_full[dists < 0.15]
        cloud      = np.concatenate([cloud_bbox, cloud_near], axis=0)
        print(f"[포인트클라우드] bbox({len(cloud_bbox)}) + 주변({len(cloud_near)}) = {len(cloud)}")
    else:
        cloud = cloud_full
        print(f"[포인트클라우드] bbox 부족 → 전체 사용")

    if len(cloud) < 50:
        raise ValueError(f"[GraspNet] 포인트클라우드 너무 적음: {len(cloud)}개")

    # num_points개로 샘플링
    np.random.seed(42)
    idx = np.random.choice(len(cloud), num_points,
                           replace=(len(cloud) < num_points))
    cloud_sampled = cloud[idx]

    # ── Step 5: GraspNet 추론 (카메라 좌표계 입력) ───────────────
    pc_tensor = torch.tensor(
        cloud_sampled, dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        end_points = models["gn"]({"point_clouds": pc_tensor})

    grasp_preds = pred_decode(end_points)
    preds_tensor = grasp_preds[0]  # (N, 17) tensor

    if preds_tensor.shape[0] == 0:
        raise ValueError("[GraspNet] 유효한 grasp pose가 없습니다.")

    preds_np = preds_tensor.detach().cpu().numpy()  # (N, 17)
    print(f"[GraspNet] 후처리 전 후보 수: {len(preds_np)}")

    # ── Step 6: GraspGroup으로 NMS + sort ───────────────────────
    # ✅ graspnetAPI 대신 직접 구현한 GraspGroup 사용
    gg = GraspGroup(preds_np)
    gg.nms(translation_thresh=0.03, rotation_thresh=30.0)
    gg.sort_by_score()
    print(f"[GraspGroup] NMS 후: {len(gg)}개 | best score: {gg[0].score:.4f}")

    best = gg[0]  # score 최고 grasp
    print(f"[GraspNet] score={best.score:.4f} | pos(카메라)={best.translation.round(4)}")

    # Step 7: 카메라 좌표 → 월드 좌표
    grasp_pos_world = camera_to_world(best.translation, view_matrix)
    print(f"[GraspNet] pos(월드)={[round(v,4) for v in grasp_pos_world.tolist()]}")
 
    # Step 8: GraspNet rotation → PyBullet EEF quaternion
    # 좌표계 변환 + 축 재배열 포함
    grasp_quat = graspnet_rot_to_pybullet_quat(best.rotation_matrix, view_matrix)
 
    return {
        "position":    [float(v) for v in grasp_pos_world],
        "orientation": grasp_quat,
        "bbox":        bbox,
    }


# ════════════════════════════════════════════════════════════════════════════════
#  6. 포인트클라우드 정확도 검증
# ════════════════════════════════════════════════════════════════════════════════
def verify_pointcloud_accuracy(depth, proj_matrix, view_matrix, known_objects):
    cloud_cam   = depth_to_pointcloud_camera(depth, proj_matrix)
    cloud_world = camera_to_world(cloud_cam, view_matrix)
    for label, true_pos in known_objects.items():
        true_pos = np.array(true_pos)
        dists    = np.linalg.norm(cloud_world - true_pos, axis=1)
        nearest  = cloud_world[dists.argmin()]
        print(f"[검증] {label}: 실제={true_pos.round(3)}, "
              f"최근접={nearest.round(4)}, 오차={(nearest-true_pos).round(4)}")
        
# ════════════════════════════════════════════════════════════════════════════════
#  6. 회전행렬 → 쿼터니언
# ════════════════════════════════════════════════════════════════════════════════
def rotation_matrix_to_quaternion(R):
    """3x3 회전행렬 → [qx, qy, qz, qw] (PyBullet 순서)."""
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    return [x, y, z, w]

# ════════════════════════════════════════════════════════════════════════════════
#  8. 홈 포지션 복귀
# ════════════════════════════════════════════════════════════════════════════════
def reset_to_home(panda_id, steps=800):
    """로봇을 홈 포지션으로 이동."""
    for i in range(NUM_JOINTS):
        p.setJointMotorControl2(
            panda_id, i, p.POSITION_CONTROL,
            targetPosition=PANDA_HOME_JOINTS[i],
            force=500, maxVelocity=1.0
        )
    for _ in range(steps):
        p.stepSimulation()
        time.sleep(1. / 240)
    print("[홈] 홈 포지션 복귀 완료")


# ════════════════════════════════════════════════════════════════════════════════
#  7. IK + 관절 제어
# ════════════════════════════════════════════════════════════════════════════════
def move_end_effector_to(panda_id, position, orientation=None, steps=1000):
    """IK로 Panda EEF를 target position/orientation으로 이동."""
    if orientation is None:
        orientation = p.getQuaternionFromEuler([np.pi, 0, 0])

    joint_poses = p.calculateInverseKinematics(
        panda_id, END_EFFECTOR_INDEX, position,
        targetOrientation=orientation,
        lowerLimits=PANDA_LOWER,
        upperLimits=PANDA_UPPER,
        jointRanges=PANDA_RANGE,
        restPoses=PANDA_HOME_JOINTS,
        maxNumIterations=1000,
        residualThreshold=1e-6
    )
    for i in range(NUM_JOINTS):
        p.setJointMotorControl2(
            panda_id, i, p.POSITION_CONTROL,
            targetPosition=joint_poses[i], force=500,
            maxVelocity=1.0
        )
    for _ in range(steps):
        p.stepSimulation()
        time.sleep(1. / 240)
    
    actual_eef = p.getLinkState(panda_id, END_EFFECTOR_INDEX)
    actual_pos = actual_eef[4]   # 월드 좌표 위치
    actual_orn = actual_eef[5]   # 월드 좌표 orientation
    actual_euler = p.getEulerFromQuaternion(actual_orn)
    print(f"[IK] 목표 pos: {[round(v,4) for v in position]}")
    print(f"[IK] 실제 pos: {[round(v,4) for v in actual_pos]}")
    print(f"[IK] 실제 orn(euler): {[round(np.degrees(v),1) for v in actual_euler]}")
    print(f"[IK] 목표 orn(euler): {[round(np.degrees(v),1) for v in p.getEulerFromQuaternion(orientation)]}")


def open_gripper(panda_id: int, steps: int = 100):
    for fj in [9, 10]:
        p.setJointMotorControl2(panda_id, fj, p.POSITION_CONTROL,
                                targetPosition=0.04, force=100)
    for _ in range(steps):
        p.stepSimulation()
        time.sleep(1. / 240)


def close_gripper(panda_id: int, steps: int = 100):
    for fj in [9, 10]:
        p.setJointMotorControl2(panda_id, fj, p.POSITION_CONTROL,
                                targetPosition=0.0, force=100)
    for _ in range(steps):
        p.stepSimulation()
        time.sleep(1. / 240)