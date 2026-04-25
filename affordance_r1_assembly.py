"""
Affordance-R1 + PyBullet 양팔 로봇팔 Assembly 시뮬레이션
=========================================================

Pipeline:
  1. AffordanceR1Predictor  - Qwen2.5-VL 기반 fine-tuned 모델로 물체별
                              grasp part + bounding box 예측
  2. AssemblyPlanner        - VLM 결과를 받아 어떤 물체를 어느 팔이 잡을지 결정
  3. PyBulletAssemblyEnv    - 양팔 UR5 시뮬레이션 환경
  4. DualArmAssemblyPipeline - 전체 orchestration

Model: hqking/affordance-r1 (Qwen2.5-VL-7B fine-tuned w/ GRPO)
Output format: <think>...</think><answer>[x1,y1,x2,y2] part_name</answer>

Requirements:
  pip install torch==2.6.0 torchvision
  pip install transformers qwen-vl-utils
  pip install pybullet numpy Pillow opencv-python
  pip install sam2  # optional: segment mask 생성 시
"""

import re
import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import cv2
import torch
from PIL import Image, ImageDraw

# ─── Transformers / Qwen2.5-VL ────────────────────────────────────────────────
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ─── PyBullet ─────────────────────────────────────────────────────────────────
import pybullet as p
import pybullet_data

# ── IK 관절 범위 (Franka Panda 기준) ─────────────────────────────────────────
PANDA_LOWER = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
PANDA_UPPER = [ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973]
PANDA_RANGE = [ 5.7946,  3.5256,  5.7946,  3.0020,  5.7946,  3.7700,  5.7946]
PANDA_REST  = [ 0.0,    -0.7854,  0.0,    -2.3562,  0.0,     1.5708,  0.7854]



# =============================================================================
# 1. Data Structures
# =============================================================================

@dataclass
class AffordanceResult:
    """Affordance-R1 단일 물체 예측 결과"""
    object_name: str
    affordance_part: str          # 예: "handle", "shaft", "grip"
    bbox_norm: list[float]        # [x1,y1,x2,y2] 0~1000 normalized (Qwen 포맷)
    bbox_pixel: list[int]         # 실제 픽셀 좌표
    grasp_center_2d: tuple[float, float]  # bbox 중심 (픽셀)
    confidence: float = 1.0
    raw_think: str = ""
    raw_answer: str = ""


@dataclass
class AssemblyTask:
    """VLM이 결정한 assembly 태스크"""
    tool_name: str                         # 만들 도구 이름
    components: list[str]                  # 사용할 물체 목록
    assembly_instruction: str              # 조립 방법 설명
    grasp_assignments: dict[str, str] = field(default_factory=dict)
    # {"object_name": "left"/"right"}


@dataclass
class GraspPose:
    """3D grasp pose"""
    position: np.ndarray        # [x, y, z] world frame
    orientation: np.ndarray     # quaternion [x, y, z, w]
    arm: str                    # "left" or "right"
    object_id: int              # pybullet body id
    approach_direction: np.ndarray = field(default_factory=lambda: np.array([0,0,-1]))


# =============================================================================
# 2. Affordance-R1 Predictor
# =============================================================================

class AffordanceR1Predictor:
    """
    Affordance-R1 모델 래퍼.
    입력: RGB 이미지 + 물체 이름 (조립 context 포함)
    출력: AffordanceResult (grasp part, bounding box)
    """

    MODEL_PATH = "/workspace/KCC-2026-VLM/affordance-r1/huggingface"  # HF repo subfolder

    # Affordance-R1의 <think>/<rethink>/<answer> 출력 파싱용 패턴
    ANSWER_PATTERN = re.compile(
        r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE
    )
    THINK_PATTERN = re.compile(
        r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE
    )
    # bbox: [x1, y1, x2, y2] (0~1000 normalized)
    BBOX_PATTERN = re.compile(
        r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]"
    )

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        if model_path is None:
            self.model_path = "/workspace/KCC-2026-VLM/affordance-r1/huggingface"
        else:
            self.model_path = model_path
        self._load_model()

    def _load_model(self):
        print(f"[AffordanceR1] Loading model from: {self.model_path}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else "cpu",
            attn_implementation="eager" if self.device == "cuda" else "eager",
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )
        self.model.eval()
        print("[AffordanceR1] Model loaded successfully.")

    def _build_prompt(self, object_name: str, assembly_context: str) -> str:
        """
        Affordance-R1 학습 포맷에 맞는 프롬프트 생성.
        모델은 <think>추론</think><answer>[x1,y1,x2,y2] part_name</answer> 형태로 응답.
        """
        return (
            f"You are an expert robot manipulation assistant for object assembly.\n"
            f"Assembly context: {assembly_context}\n\n"
            f"Task: Identify the best grasp affordance region on the '{object_name}' "
            f"for robotic assembly grasping.\n"
            f"Provide:\n"
            f"1. The functional part name to grasp (e.g., handle, shaft, grip, body, connector)\n"
            f"2. Bounding box of that part in format [x1, y1, x2, y2] "
            f"(normalized 0-1000, top-left origin)\n\n"
            f"Think step by step about:\n"
            f"- What role does this object play in the assembly?\n"
            f"- Which part should the robot grasp for stable manipulation?\n"
            f"- What is the precise location of that graspable part?\n\n"
            f"Output format: <think>reasoning</think>"
            f"<answer>[x1, y1, x2, y2] part_name</answer>"
        )

    def predict(
        self,
        image: Image.Image,
        object_name: str,
        assembly_context: str = "",
        max_new_tokens: int = 512,
    ) -> AffordanceResult:
        """
        단일 물체의 grasp affordance 예측.

        Args:
            image: PIL Image (물체가 포함된 씬 또는 크롭된 물체 이미지)
            object_name: 예측할 물체 이름
            assembly_context: 조립 목표 설명
            max_new_tokens: 생성 토큰 수

        Returns:
            AffordanceResult
        """
        prompt_text = self._build_prompt(object_name, assembly_context)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        # Tokenize
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=False,
            )

        # Decode
        trimmed_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return self._parse_output(output_text, object_name, image.size)

    def predict_batch(
        self,
        image: Image.Image,
        objects: list[dict],  # [{"name": str, "crop_bbox": [x1,y1,x2,y2]}]
        assembly_context: str = "",
    ) -> list[AffordanceResult]:
        """
        씬 이미지에서 여러 물체에 대해 순차적으로 affordance 예측.
        objects: [{"name": "hammer_handle", "crop_bbox": [x1,y1,x2,y2]}]
        """
        results = []
        img_w, img_h = image.size

        for obj in objects:
            name = obj["name"]
            # crop_bbox가 있으면 해당 영역 크롭하여 집중 예측
            if "crop_bbox" in obj:
                x1, y1, x2, y2 = obj["crop_bbox"]
                crop = image.crop((x1, y1, x2, y2))
            else:
                crop = image

            result = self.predict(crop, name, assembly_context)

            # 크롭 이미지 기준 좌표 → 원본 이미지 좌표로 변환
            if "crop_bbox" in obj:
                cx1, cy1, cx2, cy2 = obj["crop_bbox"]
                cw = cx2 - cx1
                ch = cy2 - cy1
                # pixel bbox (크롭 기준) → 원본 기준
                bx1 = result.bbox_pixel[0] + cx1
                by1 = result.bbox_pixel[1] + cy1
                bx2 = result.bbox_pixel[2] + cx1
                by2 = result.bbox_pixel[3] + cy1
                result.bbox_pixel = [bx1, by1, bx2, by2]
                result.grasp_center_2d = (
                    (bx1 + bx2) / 2,
                    (by1 + by2) / 2
                )

            results.append(result)
            print(f"  [{name}] part={result.affordance_part}, "
                  f"bbox={result.bbox_pixel}, center={result.grasp_center_2d}")

        return results

    def _parse_output(
        self,
        output_text: str,
        object_name: str,
        image_size: tuple[int, int],
    ) -> AffordanceResult:
        """모델 출력 파싱 → AffordanceResult"""
        img_w, img_h = image_size

        # <think> 추출
        think_match = self.THINK_PATTERN.search(output_text)
        raw_think = think_match.group(1).strip() if think_match else ""

        # <answer> 추출
        answer_match = self.ANSWER_PATTERN.search(output_text)
        raw_answer = answer_match.group(1).strip() if answer_match else output_text

        # bbox 파싱 (0~1000 normalized)
        bbox_match = self.BBOX_PATTERN.search(raw_answer)
        if bbox_match:
            x1_n, y1_n, x2_n, y2_n = [int(v) for v in bbox_match.groups()]
            # → 실제 픽셀
            x1_p = int(x1_n / 1000 * img_w)
            y1_p = int(y1_n / 1000 * img_h)
            x2_p = int(x2_n / 1000 * img_w)
            y2_p = int(y2_n / 1000 * img_h)
            bbox_norm = [x1_n, y1_n, x2_n, y2_n]
            bbox_pixel = [x1_p, y1_p, x2_p, y2_p]
        else:
            # fallback: 이미지 중앙
            print(f"  [WARN] bbox not found for '{object_name}', using center.")
            bbox_norm = [250, 250, 750, 750]
            x1_p, y1_p = int(0.25 * img_w), int(0.25 * img_h)
            x2_p, y2_p = int(0.75 * img_w), int(0.75 * img_h)
            bbox_pixel = [x1_p, y1_p, x2_p, y2_p]

        # part name 파싱 (bbox 뒤 텍스트)
        part_name = "body"  # default
        if bbox_match:
            after_bbox = raw_answer[bbox_match.end():].strip()
            # 첫 번째 단어를 part name으로
            part_tokens = after_bbox.split()
            if part_tokens:
                part_name = part_tokens[0].lower().strip(".,")

        center_2d = (
            (bbox_pixel[0] + bbox_pixel[2]) / 2,
            (bbox_pixel[1] + bbox_pixel[3]) / 2,
        )

        return AffordanceResult(
            object_name=object_name,
            affordance_part=part_name,
            bbox_norm=bbox_norm,
            bbox_pixel=bbox_pixel,
            grasp_center_2d=center_2d,
            raw_think=raw_think,
            raw_answer=raw_answer,
        )


# =============================================================================
# 3. Assembly Planner (VLM 결과 → 조립 계획)
# =============================================================================

class AssemblyPlanner:
    """
    VLM(예: GPT-4V, Claude)에서 받은 조립 계획을 파싱하고,
    각 물체를 어느 팔이 잡을지 결정하는 규칙 기반 플래너.
    실제 연구에서는 LLM API 호출로 대체 가능.
    """

    def plan_arm_assignment(
        self,
        task: AssemblyTask,
        affordance_results: list[AffordanceResult],
    ) -> dict[str, str]:
        """
        조립 역할에 따라 arm 배정:
        - 메인 부품(첫 번째): right arm (dominant)
        - 보조 부품(나머지): left arm
        - 부품 2개 초과 시: 순서대로 right/left 교대
        """
        assignments = {}
        for i, obj_name in enumerate(task.components):
            arm = "right" if i % 2 == 0 else "left"
            assignments[obj_name] = arm
        return assignments

    def determine_assembly_sequence(
        self,
        task: AssemblyTask,
        affordance_results: list[AffordanceResult],
    ) -> list[dict]:
        """
        조립 순서 결정:
        1. 모든 부품 grasp
        2. 메인 부품 고정 위치로 이동
        3. 보조 부품을 메인 부품에 결합
        4. 조립 완료
        """
        sequence = []

        # Step 1: 모든 부품 grasp
        for result in affordance_results:
            arm = task.grasp_assignments.get(result.object_name, "right")
            sequence.append({
                "action": "grasp",
                "object": result.object_name,
                "arm": arm,
                "affordance_part": result.affordance_part,
                "grasp_center_2d": result.grasp_center_2d,
            })

        # Step 2: 메인 부품을 조립 위치로 이동
        main_obj = task.components[0]
        sequence.append({
            "action": "move_to_assembly_pose",
            "object": main_obj,
            "arm": task.grasp_assignments.get(main_obj, "right"),
            "target_pose": [0.5, -0.1, 0.4],  # world frame 조립 위치
        })

        # Step 3: 보조 부품 결합
        for obj in task.components[1:]:
            arm = task.grasp_assignments.get(obj, "left")
            sequence.append({
                "action": "assemble",
                "object": obj,
                "arm": arm,
                "attach_to": main_obj,
                "contact_part": affordance_results[
                    task.components.index(obj)
                ].affordance_part,
            })

        # Step 4: 결과물 배치
        sequence.append({
            "action": "place",
            "object": task.tool_name,
            "arm": "both",
            "target_pose": [0.5, 0.0, 0.3],
        })

        return sequence


# =============================================================================
# 4. Camera-to-World 3D 좌표 변환
# =============================================================================

class DepthProjector:
    """
    2D 픽셀 좌표 + depth → 3D world 좌표 변환.
    PyBullet 카메라 내/외부 파라미터 사용.
    """

    def __init__(self, width: int = 640, height: int = 480, fov: float = 60.0):
        self.width = width
        self.height = height
        self.fov = fov
        self.aspect = width / height
        self.near = 0.01
        self.far = 10.0

        # Intrinsic matrix
        f = (height / 2) / np.tan(np.radians(fov / 2))
        self.K = np.array([
            [f, 0, width / 2],
            [0, f, height / 2],
            [0, 0, 1],
        ])
        self.K_inv = np.linalg.inv(self.K)

    def pixel_to_world(
        self,
        px: float, py: float,
        depth_buffer: np.ndarray,
        view_matrix: np.ndarray,
        proj_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        PyBullet depth buffer 픽셀 → world 좌표.
        """
        # depth buffer 값 → 실제 depth
        depth_raw = depth_buffer[int(py), int(px)]
        depth = self.far * self.near / (self.far - (self.far - self.near) * depth_raw)

        # NDC 좌표
        ndc_x = (2.0 * px / self.width) - 1.0
        ndc_y = 1.0 - (2.0 * py / self.height)

        # Clip space → view space
        proj = np.array(proj_matrix).reshape(4, 4).T
        view = np.array(view_matrix).reshape(4, 4).T
        proj_inv = np.linalg.inv(proj)
        view_inv = np.linalg.inv(view)

        clip = np.array([ndc_x, ndc_y, 2 * depth_raw - 1, 1.0])
        view_space = proj_inv @ clip
        view_space /= view_space[3]
        world_space = view_inv @ view_space

        return world_space[:3]

    def grasp_center_to_3d(
        self,
        center_2d: tuple[float, float],
        depth_buffer: np.ndarray,
        view_matrix,
        proj_matrix,
        z_offset: float = 0.05,
    ) -> np.ndarray:
        """
        Grasp center 2D → 3D world pose (z_offset으로 물체 위 grasp 위치 보정)
        """
        pos_3d = self.pixel_to_world(
            center_2d[0], center_2d[1],
            depth_buffer, view_matrix, proj_matrix
        )
        pos_3d[2] += z_offset
        return pos_3d


# =============================================================================
# 5. PyBullet 양팔 로봇 환경
# =============================================================================

class DualArmRobot:
    """
    양팔 UR5 로봇팔 PyBullet 시뮬레이션 환경.
    두 UR5를 좌/우에 배치하고 IK 기반 end-effector 제어.
    """

    # UR5 joint 인덱스 (0-based, shoulder to wrist)
    UR5_ARM_JOINTS = [1, 2, 3, 4, 5, 6]

    # Panda URDF joint 구조:
    #   0: panda_joint1 ~ 6: panda_joint7 (arm)
    #   7: panda_joint8 (fixed, flange)
    #   8: panda_hand_joint (fixed, hand)
    #   9: panda_finger_joint1  ← 왼쪽 핑거
    #  10: panda_finger_joint2  ← 오른쪽 핑거
    PANDA_EE_LINK        = 11   # panda_hand (손바닥 중심)
    PANDA_FINGER_JOINTS  = [9, 10]
    FINGER_OPEN          = 0.04  # 각 핑거 최대 개방 (m)
    FINGER_CLOSED        = 0.00  # 완전히 닫힘

    # 손바닥 중심(link 11) → 손가락 끝까지의 z 오프셋 (Panda 기준 ~10 cm)
    FINGER_TIP_OFFSET    = 0.105

    # 홈 포즈 (라디안)
    HOME_JOINTS_RIGHT = [0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]
    HOME_JOINTS_LEFT  = [0, -np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0]

    def __init__(self, gui: bool = True):
        self.gui = gui
        self.physics_client = None
        self.right_arm_id = None
        self.left_arm_id  = None
        self.object_ids: dict[str, int] = {}
        self.constraints: dict[str, int] = {}
        self._setup_simulation()

    def _setup_simulation(self):
        mode = p.GUI if self.gui else p.DIRECT
        self.physics_client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # 바닥
        self.plane_id = p.loadURDF("plane.urdf")

        # 작업 테이블
        table_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.6, 0.02])
        table_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.4, 0.6, 0.02],
            rgbaColor=[0.7, 0.5, 0.3, 1]
        )
        self.table_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=table_col,
            baseVisualShapeIndex=table_vis,
            basePosition=[0.5, 0, 0.3],
        )

        # UR5 로봇팔 로드 (우측, 좌측)
        # ※ 실제 환경에서는 ur5.urdf 경로를 지정 (pybullet_data에 포함됨)
        try:
            self.right_arm_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0.0, -0.4, 0.3],
            baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]),
            useFixedBase=True,
            )
            self.left_arm_id = p.loadURDF(
                "franka_panda/panda.urdf",
                basePosition=[0.0, 0.4, 0.3],
                baseOrientation=p.getQuaternionFromEuler([0, 0, -np.pi/2]),
                useFixedBase=True,
            )

            # 홈 포즈 초기화
            for robot_id in [self.right_arm_id, self.left_arm_id]:
                for j, angle in enumerate(PANDA_REST):
                    p.resetJointState(robot_id, j, angle)
        except Exception as e:
            print(f"[WARN] Robot URDF load failed: {e}")
            print("       Using placeholder – set correct URDF path for real use.")
            self.right_arm_id = None
            self.left_arm_id  = None

        # 카메라 파라미터
        self.cam_width  = 640
        self.cam_height = 480
        self.cam_fov    = 60
        self.cam_target = [0.5, 0.0, 0.3]
        self.cam_pos    = [0.5, -1.2, 0.9]

        print("[DualArmRobot] Simulation initialized.")

    # ── 카메라 ──────────────────────────────────────────────────────────────

    def get_camera_image(self) -> tuple[np.ndarray, np.ndarray, tuple, tuple]:
        """
        씬 RGB + depth 이미지 취득.
        Returns: (rgb, depth_buffer, view_matrix, proj_matrix)
        """
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.cam_pos,
            cameraTargetPosition=self.cam_target,
            cameraUpVector=[0, 0, 1],
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.cam_fov,
            aspect=self.cam_width / self.cam_height,
            nearVal=0.01,
            farVal=10.0,
        )
        _, _, rgb, depth, _ = p.getCameraImage(
            width=self.cam_width,
            height=self.cam_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER,
        )
        rgb_array = np.array(rgb, dtype=np.uint8).reshape(
            self.cam_height, self.cam_width, 4
        )[:, :, :3]
        depth_array = np.array(depth).reshape(self.cam_height, self.cam_width)
        return rgb_array, depth_array, view_matrix, proj_matrix

    # ── 물체 생성 ───────────────────────────────────────────────────────────

    def spawn_assembly_objects(self, objects: list[dict]) -> dict[str, int]:
        """
        조립용 물체 생성.
        objects: [{"name": str, "shape": str, "size": [...], "position": [...], "color": [...]}]
        """
        for obj in objects:
            name     = obj["name"]
            shape    = obj.get("shape", "box")
            pos      = obj.get("position", [0.5, 0.0, 0.35])
            color    = obj.get("color", [0.8, 0.2, 0.2, 1.0])
            size     = obj.get("size", [0.03, 0.03, 0.1])

            if shape == "cylinder":
                col = p.createCollisionShape(
                    p.GEOM_CYLINDER, radius=size[0], height=size[1]
                )
                vis = p.createVisualShape(
                    p.GEOM_CYLINDER, radius=size[0], length=size[1],
                    rgbaColor=color
                )
            elif shape == "sphere":
                col = p.createCollisionShape(p.GEOM_SPHERE, radius=size[0])
                vis = p.createVisualShape(
                    p.GEOM_SPHERE, radius=size[0], rgbaColor=color
                )
            else:  # box
                col = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
                vis = p.createVisualShape(
                    p.GEOM_BOX, halfExtents=size, rgbaColor=color
                )

            body_id = p.createMultiBody(
                baseMass=0.2,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=pos,
            )
            self.object_ids[name] = body_id
            print(f"  [Spawn] '{name}' → body_id={body_id} at {pos}")

        return self.object_ids

    # ── Grasp / Release ─────────────────────────────────────────────────────

    def move_end_effector(
        self,
        arm: str,
        target_pos: np.ndarray,
        target_orn: Optional[np.ndarray] = None,
        num_steps: int = 120,
    ):
        robot_id = self.right_arm_id if arm == "right" else self.left_arm_id
        if robot_id is None:
            print(f"  [WARN] {arm} arm not loaded, skipping.")
            return

        ee_link = self.PANDA_EE_LINK

        if target_orn is None:
            target_orn = p.getQuaternionFromEuler([0, np.pi, 0])

        # IK 목표를 finger tip 기준으로 보정:
        # panda_hand(link 11)는 finger tip보다 FINGER_TIP_OFFSET만큼 위에 있으므로
        # EE link를 finger tip 위치로 보내려면 목표를 그만큼 위로 올려야 함.
        # orientation이 [0, pi, 0] (z축 아래 방향)일 때 world z+ 방향이 로봇 EE z-
        orn_mat = np.array(p.getMatrixFromQuaternion(target_orn)).reshape(3, 3)
        # EE local z축의 world 방향 (손가락이 향하는 방향)
        ee_z_world = orn_mat[:, 2]
        # link 11 목표 = finger tip 목표 - tip_offset * ee_z_world
        ik_target = target_pos - self.FINGER_TIP_OFFSET * ee_z_world

        # 관절 범위 포함한 IK
        joint_poses = p.calculateInverseKinematics(
            robot_id,
            ee_link,
            ik_target.tolist(),
            target_orn,
            lowerLimits=PANDA_LOWER,
            upperLimits=PANDA_UPPER,
            jointRanges=PANDA_RANGE,
            restPoses=PANDA_REST,
            maxNumIterations=200,
            residualThreshold=1e-5,
        )

        # 7개 arm 관절만 제어 (0~6)
        for step in range(num_steps):
            for j in range(7):
                p.setJointMotorControl2(
                    robot_id, j,
                    p.POSITION_CONTROL,
                    targetPosition=joint_poses[j],
                    force=200,
                    maxVelocity=1.0,
                )
            p.stepSimulation()
            if self.gui:
                time.sleep(1.0 / 240.0)

    def open_gripper(self, arm: str, steps: int = 60):
        """그리퍼를 엽니다."""
        robot_id = self.right_arm_id if arm == "right" else self.left_arm_id
        if robot_id is None:
            return
        for _ in range(steps):
            for j in self.PANDA_FINGER_JOINTS:
                p.setJointMotorControl2(
                    robot_id, j,
                    p.POSITION_CONTROL,
                    targetPosition=self.FINGER_OPEN,
                    force=20,
                )
            p.stepSimulation()
            if self.gui:
                time.sleep(1.0 / 240.0)

    def close_gripper(self, arm: str, steps: int = 60):
        """그리퍼를 닫습니다 (물체 파지)."""
        robot_id = self.right_arm_id if arm == "right" else self.left_arm_id
        if robot_id is None:
            return
        for _ in range(steps):
            for j in self.PANDA_FINGER_JOINTS:
                p.setJointMotorControl2(
                    robot_id, j,
                    p.POSITION_CONTROL,
                    targetPosition=self.FINGER_CLOSED,
                    force=20,
                )
            p.stepSimulation()
            if self.gui:
                time.sleep(1.0 / 240.0)

    def grasp_object(self, arm: str, object_name: str, grasp_pos: np.ndarray):
        """
        실제로 end-effector가 물체 위치까지 이동한 후에만 constraint 생성.
        open → approach → descend → check proximity → close gripper → attach
        """
        obj_id = self.object_ids.get(object_name)
        if obj_id is None:
            print(f"  [WARN] Object '{object_name}' not found.")
            return

        robot_id = self.right_arm_id if arm == "right" else self.left_arm_id
        if robot_id is None:
            print(f"  [WARN] {arm} arm not loaded.")
            return

        ee_link = self.PANDA_EE_LINK
        down_orn = np.array(p.getQuaternionFromEuler([0, np.pi, 0]))

        # 0) 그리퍼 열기
        self.open_gripper(arm, steps=60)

        # 1) Approach: grasp 위 15cm
        approach = grasp_pos + np.array([0, 0, 0.15])
        print(f"    → approach {approach.round(3)}")
        self.move_end_effector(arm, approach, down_orn, num_steps=300)
        self.run_steps(60)  # 수렴 대기

        # 2) Descend: 실제 grasp 위치
        print(f"    → descend  {grasp_pos.round(3)}")
        self.move_end_effector(arm, grasp_pos, down_orn, num_steps=240)
        self.run_steps(60)  # 수렴 대기

        # 3) Proximity check: EE finger tip이 실제로 물체 근처에 있는지 확인
        ee_state = p.getLinkState(robot_id, ee_link)
        ee_pos = np.array(ee_state[0])
        # finger tip 보정: ee_pos는 panda_hand 중심이므로 tip offset 적용
        orn_mat = np.array(p.getMatrixFromQuaternion(ee_state[1])).reshape(3, 3)
        ee_z_world = orn_mat[:, 2]
        finger_tip_pos = ee_pos + self.FINGER_TIP_OFFSET * ee_z_world

        obj_pos = np.array(p.getBasePositionAndOrientation(obj_id)[0])
        dist = np.linalg.norm(finger_tip_pos - obj_pos)
        print(f"    → finger tip-Object distance: {dist:.4f}m")

        GRASP_THRESHOLD = 0.15  # 15cm 이내일 때만 grasp
        if dist > GRASP_THRESHOLD:
            print(f"    [WARN] Too far ({dist:.3f}m > {GRASP_THRESHOLD}m), skipping grasp.")
            return

        # 4) 그리퍼 닫기 (실제 파지 동작)
        self.close_gripper(arm, steps=80)

        # 5) Constraint 생성 (물체와 EE 연결)
        constraint_id = p.createConstraint(
            parentBodyUniqueId=robot_id,
            parentLinkIndex=ee_link,
            childBodyUniqueId=obj_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, self.FINGER_TIP_OFFSET],  # finger tip 기준
            childFramePosition=[0, 0, 0],
        )
        self.constraints[f"{arm}_{object_name}"] = constraint_id
        print(f"    [Grasp] {arm} arm grasped '{object_name}' (dist={dist:.3f}m)")

    def release_object(self, arm: str, object_name: str):
        """물체 release."""
        key = f"{arm}_{object_name}"
        if key in self.constraints:
            p.removeConstraint(self.constraints[key])
            del self.constraints[key]
            print(f"  [Release] {arm} arm released '{object_name}'")
        else:
            print(f"  [WARN] release_object: '{key}' constraint 없음")

    def assemble_objects(
        self,
        main_object: str,
        aux_object: str,
        contact_offset: list = None,
    ):
        """두 물체를 constraint로 연결 (assembly 완료)."""
        main_id = self.object_ids.get(main_object)
        aux_id  = self.object_ids.get(aux_object)
        if main_id is None or aux_id is None:
            print(f"  [WARN] assemble_objects: 물체 없음 ({main_object}, {aux_object})")
            return

        if contact_offset is None:
            contact_offset = [0, 0, 0.05]

        constraint_id = p.createConstraint(
            parentBodyUniqueId=main_id,
            parentLinkIndex=-1,
            childBodyUniqueId=aux_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=contact_offset,
            childFramePosition=[0, 0, 0],
        )
        key = f"assembly_{main_object}_{aux_object}"
        self.constraints[key] = constraint_id
        print(f"  [Assembly] '{main_object}' + '{aux_object}' 결합 완료.")

    def run_steps(self, n: int = 240):
        """시뮬레이션 스텝 실행."""
        for _ in range(n):
            p.stepSimulation()
            if self.gui:
                time.sleep(1.0 / 240.0)

    def disconnect(self):
        p.disconnect(self.physics_client)


# =============================================================================
# 6. 전체 파이프라인 Orchestrator
# =============================================================================

class DualArmAssemblyPipeline:
    """
    VLM 조립 계획 → Affordance-R1 grasp 예측 → PyBullet 실행
    전체 파이프라인 통합.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        gui: bool = True,
        device: str = "cuda",
    ):
        print("=" * 60)
        print("Dual-Arm Assembly Pipeline Initializing...")
        print("=" * 60)
        self.affordance_model = AffordanceR1Predictor(
            model_path=model_path, device=device
        )
        self.planner = AssemblyPlanner()
        self.env = DualArmRobot(gui=gui)
        self.projector = DepthProjector(
            width=self.env.cam_width,
            height=self.env.cam_height,
            fov=self.env.cam_fov,
        )

    def run(
        self,
        assembly_task: AssemblyTask,
        scene_objects: list[dict],
    ):
        """
        전체 assembly 실행.

        Args:
            assembly_task: VLM이 생성한 조립 태스크
            scene_objects: PyBullet에 생성할 물체 스펙
                           [{"name": str, "shape": ..., "position": ..., ...}]
        """
        print(f"\n[Pipeline] Task: {assembly_task.tool_name}")
        print(f"           Components: {assembly_task.components}")
        print(f"           Instruction: {assembly_task.assembly_instruction}")

        # ── Step 1: 씬 구성 ──────────────────────────────────────────────────
        print("\n[Step 1] Spawning objects in simulation...")
        self.env.spawn_assembly_objects(scene_objects)
        self.env.run_steps(60)  # 물리 안정화

        # ── Step 2: 카메라 이미지 취득 ────────────────────────────────────────
        print("\n[Step 2] Capturing scene image...")
        rgb, depth, view_mat, proj_mat = self.env.get_camera_image()
        scene_image = Image.fromarray(rgb)
        scene_image.save("scene_capture.png")
        print("         Saved: scene_capture.png")

        # ── Step 3: Affordance-R1 예측 ────────────────────────────────────────
        print("\n[Step 3] Running Affordance-R1 inference...")
        objects_for_affordance = [
            {"name": obj["name"]} for obj in scene_objects
            if obj["name"] in assembly_task.components
        ]
        affordance_results = self.affordance_model.predict_batch(
            image=scene_image,
            objects=objects_for_affordance,
            assembly_context=assembly_task.assembly_instruction,
        )

        # ── Step 4: Arm 배정 ─────────────────────────────────────────────────
        print("\n[Step 4] Planning arm assignments...")
        assembly_task.grasp_assignments = self.planner.plan_arm_assignment(
            assembly_task, affordance_results
        )
        for obj, arm in assembly_task.grasp_assignments.items():
            print(f"         '{obj}' → {arm} arm")

        # ── Step 5: PyBullet에서 직접 물체 위치 가져오기 ──────────────────────────────
        print("\n[Step 5] PyBullet에서 물체 3D 위치 직접 조회...")
        grasp_poses: dict[str, GraspPose] = {}

        for result in affordance_results:
            obj_id = self.env.object_ids.get(result.object_name)
            if obj_id is None:
                print(f"  [WARN] '{result.object_name}' object id 없음")
                continue

            # PyBullet에서 직접 위치 가져오기
            obj_pos, _ = p.getBasePositionAndOrientation(obj_id)
            pos_3d = np.array(obj_pos)
            # NOTE: z_offset은 grasp_object() 내부 approach 로직에서 처리하므로 여기서 추가하지 않음

            arm = assembly_task.grasp_assignments.get(result.object_name, "right")
            orn = np.array(p.getQuaternionFromEuler([0, np.pi, 0]))

            grasp_poses[result.object_name] = GraspPose(
                position=pos_3d,
                orientation=orn,
                arm=arm,
                object_id=obj_id,
            )
            print(f"  '{result.object_name}': part={result.affordance_part}, "
                f"3D={pos_3d.round(3)}, arm={arm}")

        # ── Step 6: 조립 시퀀스 생성 ─────────────────────────────────────────
        print("\n[Step 6] Generating assembly sequence...")
        sequence = self.planner.determine_assembly_sequence(
            assembly_task, affordance_results
        )
        for i, step in enumerate(sequence):
            print(f"         [{i+1}] {step['action']} - {step.get('object','')}")

        # ── Step 7: 실행 ─────────────────────────────────────────────────────
        print("\n[Step 7] Executing assembly sequence...")
        self._execute_sequence(sequence, grasp_poses, assembly_task)

        # ── Step 8: 결과 시각화 ───────────────────────────────────────────────
        print("\n[Step 8] Visualizing affordance results...")
        self._visualize_affordances(scene_image, affordance_results)

        print("\n[Pipeline] Assembly complete!")
        return affordance_results

    def _execute_sequence(
        self,
        sequence: list[dict],
        grasp_poses: dict[str, GraspPose],
        task: AssemblyTask,
    ):
        """조립 시퀀스 실행."""
        grasp_success: dict[str, bool] = {}  # grasp 성공 여부 추적

        for step in sequence:
            action = step["action"]
            obj_name = step.get("object", "")

            if action == "grasp":
                pose = grasp_poses.get(obj_name)
                if pose:
                    self.env.grasp_object(pose.arm, obj_name, pose.position)
                    # constraint가 실제로 생성됐는지 확인
                    key = f"{pose.arm}_{obj_name}"
                    success = key in self.env.constraints
                    grasp_success[obj_name] = success
                    if not success:
                        print(f"    [ERROR] Grasp failed for '{obj_name}', "
                              f"subsequent assembly will be skipped.")
                    self.env.run_steps(60)
                else:
                    grasp_success[obj_name] = False

            elif action == "move_to_assembly_pose":
                target = np.array(step["target_pose"])
                arm = step["arm"]
                obj = step.get("object", "")
                # grasp 실패한 물체는 이동 의미 없음
                if grasp_success.get(obj, True):
                    self.env.move_end_effector(arm, target)
                    self.env.run_steps(60)

            elif action == "assemble":
                main_obj = step["attach_to"]
                aux_obj  = obj_name
                # 두 물체 모두 grasp 성공한 경우에만 assemble
                if not grasp_success.get(main_obj, False):
                    print(f"  [SKIP] assemble: '{main_obj}' grasp 실패로 건너뜀")
                    continue
                if not grasp_success.get(aux_obj, False):
                    print(f"  [SKIP] assemble: '{aux_obj}' grasp 실패로 건너뜀")
                    continue
                # 보조 물체를 메인 물체 위치로 이동 후 결합
                main_id = self.env.object_ids.get(main_obj, -1)
                if main_id >= 0:
                    main_pos, _ = p.getBasePositionAndOrientation(main_id)
                    target = np.array(main_pos) + np.array([0, 0.05, 0.05])
                    arm = step["arm"]
                    self.env.move_end_effector(arm, target)
                    self.env.run_steps(60)
                self.env.assemble_objects(main_obj, aux_obj)
                self.env.run_steps(60)

            elif action == "place":
                target = np.array(step["target_pose"])
                for obj in task.components:
                    arm = task.grasp_assignments.get(obj, "right")
                    self.env.release_object(arm, obj)
                self.env.run_steps(120)

    def _visualize_affordances(
        self,
        image: Image.Image,
        results: list[AffordanceResult],
    ):
        """Affordance 예측 결과를 이미지에 시각화."""
        vis = image.copy()
        draw = ImageDraw.Draw(vis)

        colors = [
            (255, 80, 80),   # red
            (80, 255, 80),   # green
            (80, 80, 255),   # blue
            (255, 255, 80),  # yellow
        ]

        for i, result in enumerate(results):
            color = colors[i % len(colors)]
            x1, y1, x2, y2 = result.bbox_pixel
            cx, cy = result.grasp_center_2d

            # Bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Grasp center
            r = 8
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=color)

            # Label
            label = f"{result.object_name} [{result.affordance_part}]"
            draw.text((x1, max(0, y1 - 20)), label, fill=color)

        vis.save("affordance_visualization.png")
        print("         Saved: affordance_visualization.png")

    def close(self):
        self.env.disconnect()


# =============================================================================
# 7. 예시 실행
# =============================================================================

def create_example_task() -> tuple[AssemblyTask, list[dict]]:
    """
    예시: 망치 조립 (손잡이 + 헤드)
    실제 연구에서는 VLM API(GPT-4V, Claude 등)에서 task를 받아옴.
    """

    # VLM이 제안한 조립 태스크 (실제에서는 VLM 출력 파싱)
    task = AssemblyTask(
        tool_name="hammer",
        components=["hammer_head", "hammer_handle"],
        assembly_instruction=(
            "Assemble a hammer by attaching the metal hammer head to the "
            "top of the wooden handle. The handle should be grasped at its "
            "lower end, and the head should be aligned with the top of the handle."
        ),
    )

    # PyBullet 씬 물체 스펙
    scene_objects = [
        {
            "name": "hammer_head",
            "shape": "box",
            "size": [0.04, 0.02, 0.03],
            "position": [0.45, -0.1, 0.35],
            "color": [0.6, 0.6, 0.6, 1.0],  # gray (metal)
        },
        {
            "name": "hammer_handle",
            "shape": "cylinder",
            "size": [0.015, 0.15],
            "position": [0.55, 0.1, 0.38],
            "color": [0.6, 0.4, 0.2, 1.0],  # brown (wood)
        },
    ]

    return task, scene_objects


def create_screwdriver_task() -> tuple[AssemblyTask, list[dict]]:
    """예시: 드라이버 조립 (손잡이 + 샤프트 + 비트)"""
    task = AssemblyTask(
        tool_name="screwdriver",
        components=["screwdriver_handle", "screwdriver_shaft"],
        assembly_instruction=(
            "Assemble a screwdriver by inserting the metal shaft into the "
            "plastic handle. Grasp the handle at its center for stable holding, "
            "and align the shaft connector with the handle socket."
        ),
    )
    scene_objects = [
        {
            "name": "screwdriver_handle",
            "shape": "cylinder",
            "size": [0.025, 0.12],
            "position": [0.40, -0.05, 0.37],
            "color": [1.0, 0.3, 0.0, 1.0],  # orange
        },
        {
            "name": "screwdriver_shaft",
            "shape": "cylinder",
            "size": [0.008, 0.18],
            "position": [0.58, 0.08, 0.37],
            "color": [0.7, 0.7, 0.7, 1.0],  # silver
        },
    ]
    return task, scene_objects


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Affordance-R1 Assembly Pipeline")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/workspace/KCC-2026-VLM/affordance-r1/huggingface",
        help="HuggingFace model ID or local path to affordance-r1",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="hammer",
        choices=["hammer", "screwdriver"],
        help="Assembly task to run",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        default=True,
        help="Enable PyBullet GUI",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: cuda / cpu",
    )
    parser.add_argument(
        "--inference_only",
        action="store_true",
        help="Run only affordance inference on a test image (no PyBullet)",
    )
    args = parser.parse_args()

    # ── Inference-only 모드 ───────────────────────────────────────────────────
    if args.inference_only:
        print("[Mode] Inference-only: testing AffordanceR1Predictor")
        predictor = AffordanceR1Predictor(
            model_path=args.model_path,
            device=args.device,
        )
        # 테스트 이미지 (실제 사용 시 카메라 이미지로 교체)
        test_img = Image.new("RGB", (640, 480), color=(200, 200, 200))
        # 임의로 물체 모양 그리기
        draw = ImageDraw.Draw(test_img)
        draw.rectangle([200, 150, 440, 330], fill=(160, 100, 50))  # 손잡이
        draw.rectangle([280, 100, 360, 170], fill=(100, 100, 100))  # 헤드

        result = predictor.predict(
            image=test_img,
            object_name="hammer_handle",
            assembly_context="Assemble a hammer by attaching head to handle.",
        )
        print(f"\n[Result]")
        print(f"  Object:    {result.object_name}")
        print(f"  Part:      {result.affordance_part}")
        print(f"  BBox(px):  {result.bbox_pixel}")
        print(f"  Center2D:  {result.grasp_center_2d}")
        print(f"  Think:     {result.raw_think[:200]}...")
        print(f"  Answer:    {result.raw_answer}")

    # ── 전체 파이프라인 모드 ─────────────────────────────────────────────────
    else:
        if args.task == "hammer":
            task, scene_objects = create_example_task()
        else:
            task, scene_objects = create_screwdriver_task()

        pipeline = DualArmAssemblyPipeline(
            model_path=args.model_path,
            gui=args.gui,
            device=args.device,
        )

        try:
            results = pipeline.run(task, scene_objects)

            # 결과 요약 출력
            print("\n" + "=" * 60)
            print("Assembly Affordance Results Summary")
            print("=" * 60)
            for r in results:
                arm = task.grasp_assignments.get(r.object_name, "?")
                print(f"  {r.object_name}")
                print(f"    Part to grasp : {r.affordance_part}")
                print(f"    BBox (pixel)  : {r.bbox_pixel}")
                print(f"    Grasp center  : {r.grasp_center_2d}")
                print(f"    Assigned arm  : {arm}")
                print()

            if pipeline.env.gui:
                input("Press Enter to exit simulation...")

        finally:
            pipeline.close()