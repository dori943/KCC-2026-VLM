"""
robot_controller.py
────────────────────────────────────────────────────────────────────────────────
VL-Grasp (IROS 2023) 기반 로봇팔 제어 모듈

주요 기능:
  1. PyBullet 가상 카메라 렌더링
  2. Depth → 포인트클라우드 (VL-Grasp demo.py 원본 방식, 카메라 좌표계)
  3. PyBullet AABB → 픽셀 bbox 직접 계산
  4. VL-Grasp 모델 로드 (RoboRefIt VG + FGC-GraspNet)
  5. VL-Grasp 추론: 언어 명령 → 6-DoF grasp pose
  6. GraspGroup (NMS + sort_by_score) 직접 구현 (graspnetAPI 대체)
  7. IK seed(restPoses) + 관절 범위 제한으로 안정적 IK 수렴
────────────────────────────────────────────────────────────────────────────────
"""

import sys
import os

# ── sys.path 먼저 설정 ────────────────────────────────────────────────────────
VL_GRASP_ROOT = "/workspace/KCC-2026-VLM/VL-Grasp"
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

# ── Franka Panda 홈 포지션 ────────────────────────────────────────────────────
PANDA_HOME_JOINTS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

# ── IK 관절 범위 ──────────────────────────────────────────────────────────────
PANDA_LOWER = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
PANDA_UPPER = [ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973]
PANDA_RANGE = [ 5.7946,  3.5256,  5.7946,  3.0020,  5.7946,  3.7700,  5.7946]


# ════════════════════════════════════════════════════════════════════════════════
#  GraspGroup 직접 구현 (graspnetAPI Python 3.12 호환 대체)
#
#  grasp array 컬럼 구조 (17개):
#  [score, width, height, depth, r00~r22(9), x,y,z, obj_id]
#   0      1      2       3      4~12         13~15  16
# ════════════════════════════════════════════════════════════════════════════════

class Grasp:
    def __init__(self, array):
        self._array = array

    @property
    def score(self):           return float(self._array[0])
    @property
    def width(self):           return float(self._array[1])
    @property
    def depth(self):           return float(self._array[3])
    @property
    def rotation_matrix(self): return self._array[4:13].reshape(3, 3)
    @property
    def translation(self):     return self._array[13:16]


class GraspGroup:
    def __init__(self, array):
        self._array = np.array(array, dtype=np.float32)

    def __len__(self):
        return len(self._array)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return Grasp(self._array[idx])
        return GraspGroup(self._array[idx])

    def sort_by_score(self):
        order = np.argsort(self._array[:, 0])[::-1]
        self._array = self._array[order]
        return self

    def nms(self, translation_thresh=0.03, rotation_thresh=30.0):
        if len(self._array) == 0:
            return self
        self.sort_by_score()
        keep         = np.ones(len(self._array), dtype=bool)
        translations = self._array[:, 13:16]
        rotations    = self._array[:, 4:13].reshape(-1, 9)

        for i in range(len(self._array)):
            if not keep[i]: continue
            for j in range(i + 1, len(self._array)):
                if not keep[j]: continue
                if np.linalg.norm(translations[i] - translations[j]) > translation_thresh:
                    continue
                R_i   = rotations[i].reshape(3, 3)
                R_j   = rotations[j].reshape(3, 3)
                trace = np.clip((np.trace(R_i.T @ R_j) - 1.0) / 2.0, -1.0, 1.0)
                if np.degrees(np.arccos(trace)) < rotation_thresh:
                    keep[j] = False
        self._array = self._array[keep]
        return self


# ════════════════════════════════════════════════════════════════════════════════
#  1. PyBullet 가상 카메라 렌더링
# ════════════════════════════════════════════════════════════════════════════════
def render_camera(cam_target=[0.55, -0.35, 0.8], cam_distance=1.2,
                  cam_yaw=0, cam_pitch=-45):
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
    rgb       = np.array(rgb_raw, dtype=np.uint8).reshape(CAM_HEIGHT, CAM_WIDTH, 4)[:, :, :3]
    depth_buf = np.array(depth_raw, dtype=np.float32).reshape(CAM_HEIGHT, CAM_WIDTH)
    depth     = CAM_FAR * CAM_NEAR / (CAM_FAR - (CAM_FAR - CAM_NEAR) * depth_buf)
    return rgb, depth, proj_matrix, view_matrix


# ════════════════════════════════════════════════════════════════════════════════
#  2. Depth → 포인트클라우드 (카메라 좌표계 — VL-Grasp demo.py 원본 방식)
# ════════════════════════════════════════════════════════════════════════════════
def depth_to_pointcloud_camera(depth, proj_matrix, workspace_mask=None):
    h, w  = depth.shape
    proj  = np.array(proj_matrix).reshape(4, 4).T
    fx    = proj[0, 0] * w / 2.0
    fy    = proj[1, 1] * h / 2.0
    cx    = (1.0 - proj[0, 2]) * w / 2.0
    cy    = (1.0 + proj[1, 2]) * h / 2.0

    camera = CameraInfo(float(w), float(h), fx, fy, cx, cy, scale=1.0)
    cloud  = create_point_cloud_from_depth_image(depth, camera, organized=True)

    depth_mask = (depth > CAM_NEAR + 0.01) & (depth < CAM_FAR - 0.01)
    mask       = depth_mask & workspace_mask if workspace_mask is not None else depth_mask
    return cloud[mask].astype(np.float32)


def camera_to_world(points_cam, view_matrix):
    """
    카메라 좌표(OpenCV) → PyBullet 월드 좌표 변환.
    검증된 변환: reshape(4,4).T + y,z 반전
    """
    vm     = np.array(view_matrix).reshape(4, 4).T
    inv_vm = np.linalg.inv(vm)

    pts = np.array(points_cam, dtype=np.float64)
    if pts.ndim == 1:
        pts_gl = np.array([pts[0], -pts[1], -pts[2], 1.0])
        return (inv_vm @ pts_gl)[:3]
    else:
        pts_gl = pts.copy()
        pts_gl[:, 1] *= -1
        pts_gl[:, 2] *= -1
        homo = np.hstack([pts_gl, np.ones((len(pts_gl), 1))])
        return (inv_vm @ homo.T).T[:, :3]


def make_bbox_mask(h, w, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    mask = np.zeros((h, w), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


# ════════════════════════════════════════════════════════════════════════════════
#  3. PyBullet AABB → 픽셀 bbox 직접 계산
# ════════════════════════════════════════════════════════════════════════════════
def get_bbox_from_pybullet(body_id, proj_matrix, view_matrix,
                            cam_width=CAM_WIDTH, cam_height=CAM_HEIGHT,
                            padding=1.3):
    """
    PyBullet AABB 8개 꼭짓점을 카메라로 투영해서 픽셀 bbox 계산.
    VG 모델보다 정확하고 항상 올바른 물체를 잡음.
    """
    aabb_min, aabb_max = p.getAABB(body_id)
    corners_world = np.array([
        [aabb_min[0], aabb_min[1], aabb_min[2]],
        [aabb_max[0], aabb_min[1], aabb_min[2]],
        [aabb_min[0], aabb_max[1], aabb_min[2]],
        [aabb_max[0], aabb_max[1], aabb_min[2]],
        [aabb_min[0], aabb_min[1], aabb_max[2]],
        [aabb_max[0], aabb_min[1], aabb_max[2]],
        [aabb_min[0], aabb_max[1], aabb_max[2]],
        [aabb_max[0], aabb_max[1], aabb_max[2]],
    ])

    vm   = np.array(view_matrix).reshape(4, 4).T
    proj = np.array(proj_matrix).reshape(4, 4).T
    fx   = proj[0, 0] * cam_width  / 2.0
    fy   = proj[1, 1] * cam_height / 2.0
    cx   = (1.0 - proj[0, 2]) * cam_width  / 2.0
    cy   = (1.0 + proj[1, 2]) * cam_height / 2.0

    us, vs = [], []
    for pt in corners_world:
        pt_h   = np.append(pt, 1.0)
        pt_cam = (vm @ pt_h)[:3]

        if pt_cam[2] >= 0:
            continue

        u = int(fx * pt_cam[0] / (-pt_cam[2]) + cx)
        v = int(fy * (-pt_cam[1]) / (-pt_cam[2]) + cy)
        us.append(u)
        vs.append(v)

    if not us:
        return None

    # 중심 기준 padding 배 확장
    u_center = (min(us) + max(us)) // 2
    v_center = (min(vs) + max(vs)) // 2
    half_w   = int((max(us) - min(us)) / 2 * padding)
    half_h   = int((max(vs) - min(vs)) / 2 * padding)

    x1 = max(0, u_center - half_w)
    y1 = max(0, v_center - half_h)
    x2 = min(cam_width,  u_center + half_w)
    y2 = min(cam_height, v_center + half_h)
    return [x1, y1, x2, y2]


# ════════════════════════════════════════════════════════════════════════════════
#  4. VL-Grasp 모델 로드 (싱글턴)
# ════════════════════════════════════════════════════════════════════════════════
_vl_grasp_models = None

def load_vl_grasp_models(
    vg_checkpoint = "/workspace/KCC-2026-VLM/VL-Grasp/logs/checkpoint_best_r50.pth",
    gn_checkpoint = "/workspace/KCC-2026-VLM/VL-Grasp/logs/checkpoint_fgc.tar",
    device        = "cuda"
):
    global _vl_grasp_models
    if _vl_grasp_models is not None:
        return _vl_grasp_models

    # Visual Grounding (RoboRefIt)
    vg_args          = get_args_parser().parse_args([])
    vg_args.device   = device
    vg_args.masks    = True
    vg_args.img_type = 'RGB'
    vg_model, _, _   = build_reftr_seg(vg_args)
    vg_ckpt          = torch.load(vg_checkpoint, map_location=device, weights_only=False)
    vg_model.load_state_dict(vg_ckpt["model"], strict=False)
    vg_model.to(device).eval()
    print("[VL-Grasp] VG 모델 로드 완료")

    # FGC-GraspNet
    gn_model = GraspNetModel(is_training=False, is_demo=True)
    gn_ckpt  = torch.load(gn_checkpoint, map_location=device, weights_only=False)
    gn_model.load_state_dict(gn_ckpt["model_state_dict"])
    gn_model.to(device).eval()
    print("[VL-Grasp] GraspNet 모델 로드 완료")

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
#  5. VL-Grasp 추론: 언어 명령 → 6-DoF grasp pose
# ════════════════════════════════════════════════════════════════════════════════
def get_grasp_pose_from_vl_grasp(
    rgb, depth, proj_matrix, view_matrix,
    language, models=None, num_points=20000,
    scene_info=None, target_label=None,
    pybullet_bbox=None,
):
    """
    Returns:
        {
            "position":    [x, y, z],        # PyBullet 월드 좌표
            "orientation": [qx, qy, qz, qw], # 쿼터니언
            "bbox":        [x1, y1, x2, y2]  # 사용된 bbox
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

    # ── Step 3: bbox 결정 ─────────────────────────────────────────
    h_img, w_img = rgb.shape[:2]
    samples = {
        "img":           nested_tensor_from_tensor_list(img_tensor),
        "sentence":      sentence,
        "sentence_mask": sentence_mask,
    }
    with torch.no_grad():
        vg_out = models["vg"](samples)

    if pybullet_bbox is not None:
        # ✅ PyBullet 직접 계산 bbox 사용 (항상 정확한 물체 bbox)
        x1, y1, x2, y2 = pybullet_bbox
        bbox = pybullet_bbox
        print(f"[bbox] PyBullet 직접 계산: {bbox}")
    else:
        # VG 모델 bbox 사용 (fallback)
        pred_box       = vg_out["pred_boxes"][0, 0, 0].cpu().numpy()
        cx_b, cy_b, bw, bh = pred_box
        x1 = int((cx_b - bw / 2) * w_img)
        y1 = int((cy_b - bh / 2) * h_img)
        x2 = int((cx_b + bw / 2) * w_img)
        y2 = int((cy_b + bh / 2) * h_img)
        bbox = [x1, y1, x2, y2]
        print(f"[VG] bbox: {bbox}")

    # ── Step 4: segmentation mask (bbox 내부만) ───────────────────
    import torch.nn.functional as F
    pred_mask_raw     = vg_out["pred_masks"][0, 0]
    pred_mask_resized = F.interpolate(
        pred_mask_raw.unsqueeze(0).unsqueeze(0),
        size=(h_img, w_img), mode='bilinear', align_corners=False
    )[0, 0]
    seg_mask = (pred_mask_resized.sigmoid() > 0.5).cpu().numpy()
    bbox_mask_for_seg = make_bbox_mask(h_img, w_img, bbox)
    seg_mask_filtered = seg_mask & bbox_mask_for_seg
    seg_pixel_count   = seg_mask_filtered.sum()
    print(f"[VG] segmentation 픽셀 수: {seg_pixel_count}")

    # ── Step 5: 포인트클라우드 생성 (카메라 좌표계) ───────────────
    if seg_pixel_count >= 50:
        cloud_obj = depth_to_pointcloud_camera(depth, proj_matrix,
                                               workspace_mask=seg_mask_filtered)
        print(f"[포인트클라우드] segmentation 기반: {len(cloud_obj)}개")
    else:
        bbox_mask = make_bbox_mask(h_img, w_img, bbox)
        cloud_obj = depth_to_pointcloud_camera(depth, proj_matrix,
                                               workspace_mask=bbox_mask)
        print(f"[포인트클라우드] bbox 기반: {len(cloud_obj)}개")

    cloud_full = depth_to_pointcloud_camera(depth, proj_matrix)

    # 물체 중심 (카메라 좌표계)
    object_center_cam = cloud_obj.mean(axis=0) if len(cloud_obj) >= 10 else None
    if object_center_cam is not None:
        print(f"[물체 중심] 카메라: {object_center_cam.round(4)}")

    # GraspNet용 포인트클라우드 구성
    if len(cloud_obj) >= num_points:
        cloud = cloud_obj
    elif len(cloud_obj) >= 50:
        center     = cloud_obj.mean(axis=0)
        dists      = np.linalg.norm(cloud_full - center, axis=1)
        cloud_near = cloud_full[dists < 0.15]
        cloud      = np.concatenate([cloud_obj, cloud_near], axis=0)
        print(f"[포인트클라우드] obj({len(cloud_obj)}) + 주변({len(cloud_near)}) = {len(cloud)}")
    else:
        cloud = cloud_full

    if len(cloud) < 50:
        raise ValueError(f"[GraspNet] 포인트클라우드 너무 적음: {len(cloud)}개")

    np.random.seed(42)
    idx = np.random.choice(len(cloud), num_points,
                           replace=(len(cloud) < num_points))
    cloud_sampled = cloud[idx]

    # ── Step 6: GraspNet 추론 ────────────────────────────────────
    pc_tensor = torch.tensor(
        cloud_sampled, dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        end_points = models["gn"]({"point_clouds": pc_tensor})

    grasp_preds = pred_decode(end_points)
    preds_np    = grasp_preds[0].detach().cpu().numpy()

    if len(preds_np) == 0:
        raise ValueError("[GraspNet] 유효한 grasp pose가 없습니다.")

    # ── Step 7: GraspGroup NMS + 물체 중심 기반 선택 ─────────────
    gg = GraspGroup(preds_np)
    gg.nms(translation_thresh=0.03, rotation_thresh=30.0)
    gg.sort_by_score()
    print(f"[GraspGroup] NMS 후: {len(gg)}개 | best score: {gg[0].score:.4f}")

    if object_center_cam is not None:
        top_k    = min(30, len(gg))
        best     = None
        min_dist = float('inf')
        for i in range(top_k):
            g    = gg[i]
            dist = np.linalg.norm(g.translation - object_center_cam)
            if dist < min_dist:
                min_dist = dist
                best = g
        print(f"[GraspNet] 중심 기반 선택 | dist={min_dist:.4f} | score={best.score:.4f}")
    else:
        best = gg[0]

    print(f"[GraspNet] pos(카메라)={best.translation.round(4)}")

    # ── Step 8: 카메라 → 월드 좌표 변환 ─────────────────────────
    grasp_pos_world = camera_to_world(best.translation, view_matrix)
    print(f"[GraspNet] pos(월드)={[round(v,4) for v in grasp_pos_world.tolist()]}")

    # rotation → PyBullet EEF quaternion
    grasp_quat = graspnet_rot_to_pybullet_quat(best.rotation_matrix, view_matrix)

    return {
        "position":    [float(v) for v in grasp_pos_world],
        "orientation": grasp_quat,
        "bbox":        bbox,
    }


# ════════════════════════════════════════════════════════════════════════════════
#  6. GraspNet rotation → PyBullet EEF quaternion 변환
# ════════════════════════════════════════════════════════════════════════════════
def graspnet_rot_to_pybullet_quat(rot_cam, view_matrix):
    vm     = np.array(view_matrix).reshape(4, 4).T
    R_view = vm[:3, :3]
    T_cv2gl = np.diag([1.0, -1.0, -1.0])

    rot_gl    = T_cv2gl @ rot_cam
    rot_world = R_view.T @ rot_gl

    approach = rot_world[:, 0]
    major    = rot_world[:, 2]

    z_axis = approach / (np.linalg.norm(approach) + 1e-8)
    x_axis = major    / (np.linalg.norm(major)    + 1e-8)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis   / (np.linalg.norm(y_axis)   + 1e-8)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis   / (np.linalg.norm(x_axis)   + 1e-8)

    R_eef = np.stack([x_axis, y_axis, z_axis], axis=1)
    return rotation_matrix_to_quaternion(R_eef)


# ════════════════════════════════════════════════════════════════════════════════
#  7. 포인트클라우드 정확도 검증 (디버그)
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
#  8. 회전행렬 → 쿼터니언
# ════════════════════════════════════════════════════════════════════════════════
def rotation_matrix_to_quaternion(R):
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return [(R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s, 0.25/s]
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return [0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s, (R[2,1]-R[1,2])/s]
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return [(R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s, (R[0,2]-R[2,0])/s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return [(R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s, (R[1,0]-R[0,1])/s]


# ════════════════════════════════════════════════════════════════════════════════
#  9. 홈 포지션 복귀
# ════════════════════════════════════════════════════════════════════════════════
def reset_to_home(panda_id, steps=800):
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
#  10. IK + 관절 제어 (seed 적용)
# ════════════════════════════════════════════════════════════════════════════════
def move_end_effector_to(panda_id, position, orientation=None, steps=1000):
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
            targetPosition=joint_poses[i],
            force=500, maxVelocity=1.0
        )
    for _ in range(steps):
        p.stepSimulation()
        time.sleep(1. / 240)

    # 실제 도달 위치 확인
    eef         = p.getLinkState(panda_id, END_EFFECTOR_INDEX)
    actual_pos  = eef[4]
    actual_orn  = eef[5]
    actual_euler = p.getEulerFromQuaternion(actual_orn)
    target_euler = p.getEulerFromQuaternion(orientation)
    print(f"[IK] 목표 pos: {[round(v,4) for v in position]}")
    print(f"[IK] 실제 pos: {[round(v,4) for v in actual_pos]}")
    print(f"[IK] 실제 orn(euler): {[round(np.degrees(v),1) for v in actual_euler]}")
    print(f"[IK] 목표 orn(euler): {[round(np.degrees(v),1) for v in target_euler]}")


def open_gripper(panda_id, steps=100):
    for fj in [9, 10]:
        p.setJointMotorControl2(panda_id, fj, p.POSITION_CONTROL,
                                targetPosition=0.04, force=100)
    for _ in range(steps):
        p.stepSimulation()
        time.sleep(1. / 240)


def close_gripper(panda_id, steps=100):
    for fj in [9, 10]:
        p.setJointMotorControl2(panda_id, fj, p.POSITION_CONTROL,
                                targetPosition=0.0, force=100)
    for _ in range(steps):
        p.stepSimulation()
        time.sleep(1. / 240)