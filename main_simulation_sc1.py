"""
main_simulation_3.py
Boot-only dual-arm Panda simulation orchestration.

Notes:
- R1 model assets should be obtained from Hugging Face:
  https://huggingface.co/hqking/affordance-r1
- This boot path intentionally avoids GPT / OpenAI / VL-Grasp / open3d imports.
"""

import os
import time

import numpy as np
import pybullet as p
import pybullet_data

from robot_controller_3 import PandaController, render_camera
from assembly_manager import AssemblyManager


SIM_TIMESTEP = 1.0 / 240.0
STABILIZE_STEPS = 480
DEMO_HOLD_STEPS = 240
R1_HF_REPO = "https://huggingface.co/hqking/affordance-r1"
ENABLE_AFFORDANCE_R1_DEFAULT = False
ENABLE_SAM2_REFINEMENT_DEFAULT = False
ENABLE_MODULE1_DYNAMICS_DEFAULT = True
ENABLE_MODULE1_PROFILE_INFERENCE_DEFAULT = True
AFFORDANCE_MODEL_ID = "hqking/affordance-r1"
SAM2_MODEL_ID = "facebook/sam2-hiera-large"
AFFORDANCE_CAPTURE_WIDTH = 640
AFFORDANCE_CAPTURE_HEIGHT = 480
MODULE1_MAP_PATH = "/workspace/KCC-2026-VLM/module1-2B/configs/module1_to_pybullet_map.yaml"

LEFT_BASE_POSITION = [0.0, -0.35, 0.65]
RIGHT_BASE_POSITION = [0.0, 0.35, 0.65]
TABLE_BASE_POSITION = [0.6, 0.0, 0.0]
YCB_DIR = "/workspace/KCC-2026-VLM/data/object2urdf/examples/ycb"

YCB_OBJECT_SPECS = [
    ("cracker_box", "003_cracker_box.urdf", [1.2, 0.3, 0.82]),
    ("sugar_box", "004_sugar_box.urdf", [1.2, 0.10, 0.82]),
    ("pudding_box", "008_pudding_box.urdf", [1.2, -0.10, 0.82]),
    ("gelatin_box", "009_gelatin_box.urdf", [1.2, -0.3, 0.82]),
    ("bowl", "024_bowl.urdf", [1.0, 0.2, 0.82]),
    ("mug", "025_mug.urdf", [1.0, 0.0, 0.82]),
    ("plate", "029_plate.urdf", [1.0, -0.2, 0.82]),
    ("fork", "030_fork.urdf", [0.7, 0.3, 0.82]),
    ("spoon", "031_spoon.urdf", [0.7, 0.10, 0.82]),
    ("knife", "032_knife.urdf", [0.7, -0.1, 0.82]),
    ("spatula", "033_spatula.urdf", [0.7, -0.3, 0.82]),
    ("adjustable_wrench", "042_adjustable_wrench.urdf", [0.45, 0.2, 0.82]),
    ("large_marker", "040_large_marker.urdf", [0.45, 0.0, 0.82]),
    ("phillips_screwdriver", "043_phillips_screwdriver.urdf", [0.45, -0.2, 0.82]),
    ("flat_screwdriver", "044_flat_screwdriver.urdf", [0.45, -0.35, 0.82]),
]

def _load_module3_object_labels(json_path: str) -> list[str]:
    """
    module3_output.json에서 사용된 물체 이름 목록을 파싱해 반환.
    base_object / attach_object 모두 수집하며 순서를 유지한다.
    """
    import json as _json
    try:
        raw = _json.loads(open(json_path, encoding="utf-8").read())
        seen, labels = set(), []
        for step in raw.get("assembly_steps", []):
            for key in ("base_object", "attach_object"):
                val = step.get(key)
                if val and val not in seen:
                    seen.add(val)
                    labels.append(val)
        print(f"[Boot] module3 object labels: {labels}")
        return labels
    except Exception as exc:
        print(f"[Boot][WARN] could not parse module3_output.json ({exc}); using fallback YCB spec.")
        return []

_YCB_SPEC_BY_LABEL = {label: (label, urdf, pos) for label, urdf, pos in YCB_OBJECT_SPECS}

def _build_ycb_object_specs(json_path: str) -> list[tuple]:
    """
    module3_output.json 물체 목록을 기반으로 YCB_OBJECT_SPECS를 동적 생성.
    JSON에 없는 물체나 매핑이 없는 경우 경고 후 건너뜀.
    """
    labels = _load_module3_object_labels(json_path)
    if not labels:
        # fallback: 기존 YCB 전체 로드
        return [
            ("cracker_box", "003_cracker_box.urdf", [1.2, 0.3, 0.82]),
            ("sugar_box", "004_sugar_box.urdf", [1.2, 0.10, 0.82]),
            ("pudding_box", "008_pudding_box.urdf", [1.2, -0.10, 0.82]),
            ("gelatin_box", "009_gelatin_box.urdf", [1.2, -0.3, 0.82]),
            ("bowl", "024_bowl.urdf", [1.0, 0.2, 0.82]),
            ("mug", "025_mug.urdf", [1.0, 0.0, 0.82]),
            ("plate", "029_plate.urdf", [1.0, -0.2, 0.82]),
            ("fork", "030_fork.urdf", [0.7, 0.3, 0.82]),
            ("spoon", "031_spoon.urdf", [0.7, 0.10, 0.82]),
            ("knife", "032_knife.urdf", [0.7, -0.1, 0.82]),
            ("spatula", "033_spatula.urdf", [0.7, -0.3, 0.82]),
            ("adjustable_wrench", "042_adjustable_wrench.urdf", [0.45, 0.2, 0.82]),
            ("large_marker", "040_large_marker.urdf", [0.45, 0.0, 0.82]),
            ("phillips_screwdriver", "043_phillips_screwdriver.urdf", [0.45, -0.2, 0.82]),
            ("flat_screwdriver", "044_flat_screwdriver.urdf", [0.45, -0.35, 0.82]),
        ]
    specs = []
    for raw_label in labels:
        entry = _YCB_SPEC_BY_LABEL.get(raw_label)
        if entry is None:
            print(f"[Boot][WARN] '{raw_label}' is not in YCB_OBJECT_SPECS — skipping load.")
            continue
        specs.append(entry)
    return list(YCB_OBJECT_SPECS)


# JSON 경로를 미리 결정 (main() 호출 전에도 사용)
_MODULE3_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "module3_task1_output.json")

# 동적으로 결정된 YCB_OBJECT_SPECS
YCB_OBJECT_SPECS: list[tuple] = _build_ycb_object_specs(_MODULE3_JSON_PATH)

MODULE1_FALLBACK_MAP = {
    "clamp_ranges": {
        "lateral_friction": [0.05, 2.0],
        "restitution": [0.0, 0.95],
        "mass_kg": [0.02, 5.0],
        "linear_damping": [0.0, 0.2],
        "angular_damping": [0.0, 0.2],
    },
    "friction_map": {"low": 0.2, "medium": 0.6, "high": 1.0},
    "slip_penalty": {"low": -0.05, "medium": -0.15, "high": -0.3},
    "restitution_map": {"low": 0.15, "medium": 0.45, "high": 0.75},
    "mass_base_kg_by_category": {
        "very_light": 0.08,
        "light": 0.25,
        "medium": 0.75,
        "heavy": 1.8,
    },
    "density_multiplier": {
        "very_low": 0.7,
        "low": 0.9,
        "medium": 1.1,
        "high": 1.35,
    },
    "size_multiplier": {
        "very_small": 0.6,
        "small": 0.8,
        "medium": 1.0,
        "large": 1.25,
        "very_large": 1.5,
        "unknown": 1.0,
    },
}

MODULE1_PROFILE_DEFAULT = {
    "surface_friction": "medium",
    "slip_tendency": "medium",
    "restitution": "low",
    "mass_category": "medium",
    "density_category": "medium",
    "size_relative": "medium",
    "deformability": "low",
}

YCB_MODULE1_PROFILE_BY_LABEL = {
    "cracker_box": {"surface_friction": "medium", "slip_tendency": "medium", "mass_category": "light", "size_relative": "medium"},
    "sugar_box": {"surface_friction": "medium", "slip_tendency": "medium", "mass_category": "light", "size_relative": "small"},
    "pudding_box": {"surface_friction": "medium", "slip_tendency": "low", "mass_category": "light", "size_relative": "small"},
    "gelatin_box": {"surface_friction": "medium", "slip_tendency": "low", "mass_category": "light", "size_relative": "small"},
    "bowl": {"surface_friction": "low", "slip_tendency": "medium", "mass_category": "medium", "size_relative": "medium"},
    "mug": {"surface_friction": "medium", "slip_tendency": "low", "mass_category": "medium", "size_relative": "small"},
    "plate": {"surface_friction": "low", "slip_tendency": "high", "mass_category": "medium", "size_relative": "large"},
    "fork": {"surface_friction": "high", "slip_tendency": "low", "mass_category": "light", "size_relative": "small"},
    "spoon": {"surface_friction": "high", "slip_tendency": "low", "mass_category": "light", "size_relative": "small"},
    "knife": {"surface_friction": "medium", "slip_tendency": "medium", "mass_category": "light", "size_relative": "small"},
    "spatula": {"surface_friction": "medium", "slip_tendency": "medium", "mass_category": "light", "size_relative": "medium"},
    "key": {"surface_friction": "high", "slip_tendency": "low", "mass_category": "light", "size_relative": "small"},
    "large_marker": {"surface_friction": "medium", "slip_tendency": "medium", "mass_category": "light", "size_relative": "small"},
    "phillips_screwdriver": {"surface_friction": "medium", "slip_tendency": "low", "mass_category": "light", "size_relative": "medium"},
    "flat_screwdriver": {"surface_friction": "medium", "slip_tendency": "low", "mass_category": "light", "size_relative": "medium"},
}
YCB_DYNAMICS_OVERRIDE_BY_LABEL = {
    
}

def env_flag(env_name: str, default: bool = False) -> bool:
    raw = os.getenv(env_name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def clamp_value(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def read_loaded_body_features(body_id: int) -> dict:
    aabb_min, aabb_max = p.getAABB(body_id)
    size_x = max(1e-4, float(aabb_max[0] - aabb_min[0]))
    size_y = max(1e-4, float(aabb_max[1] - aabb_min[1]))
    size_z = max(1e-4, float(aabb_max[2] - aabb_min[2]))
    span_xy = max(size_x, size_y)
    min_xy = min(size_x, size_y)
    volume = max(1e-6, size_x * size_y * size_z)
    info = p.getDynamicsInfo(body_id, -1)
    mass = max(0.02, float(info[0])) if len(info) > 0 else 0.2
    density = mass / volume
    height_ratio = size_z / max(span_xy, 1e-4)
    roundness_xy = min_xy / max(span_xy, 1e-4)
    return {
        "size_x": size_x,
        "size_y": size_y,
        "size_z": size_z,
        "span_xy": span_xy,
        "volume": volume,
        "mass": mass,
        "density": density,
        "height_ratio": height_ratio,
        "roundness_xy": roundness_xy,
        "round_like": roundness_xy > 0.82 and height_ratio > 0.75,
        "slender_like": height_ratio > 1.25,
        "flat_like": height_ratio < 0.55,
    }


def _categorize_mass(mass: float) -> str:
    if mass < 0.10:
        return "very_light"
    if mass < 0.30:
        return "light"
    if mass < 1.10:
        return "medium"
    return "heavy"


def _categorize_density(density: float) -> str:
    if density < 130.0:
        return "very_low"
    if density < 320.0:
        return "low"
    if density < 850.0:
        return "medium"
    return "high"


def _categorize_size(span_xy: float) -> str:
    if span_xy < 0.035:
        return "very_small"
    if span_xy < 0.055:
        return "small"
    if span_xy < 0.085:
        return "medium"
    if span_xy < 0.12:
        return "large"
    return "very_large"


def infer_module1_profile_from_loaded_body(label: str, body_id: int) -> tuple[dict, dict]:
    """
    Infer module1 semantic profile from current loaded body features.
    This lets dynamics generalize to new objects without per-label tuning.
    """
    features = read_loaded_body_features(body_id)
    label_lower = str(label).lower()

    surface = "medium"
    if "clamp" in label_lower:
        surface = "high"
    elif "can" in label_lower or "bottle" in label_lower:
        surface = "medium"

    slip = "medium"
    if features["round_like"]:
        slip = "high"
    if features["flat_like"]:
        slip = "low"
    if "clamp" in label_lower:
        slip = "low"
    if "apple" in label_lower:
        slip = "low"

    deformability = "low"
    if any(token in label_lower for token in ("apple", "orange", "banana", "fruit")):
        deformability = "medium"

    restitution = "medium" if deformability == "low" else "low"

    inferred_profile = {
        "surface_friction": surface,
        "slip_tendency": slip,
        "restitution": restitution,
        "mass_category": _categorize_mass(features["mass"]),
        "density_category": _categorize_density(features["density"]),
        "size_relative": _categorize_size(features["span_xy"]),
        "deformability": deformability,
    }
    return inferred_profile, features


def load_module1_map(path: str = MODULE1_MAP_PATH) -> dict:
    map_cfg = {k: v.copy() if isinstance(v, dict) else v for k, v in MODULE1_FALLBACK_MAP.items()}
    if not os.path.isfile(path):
        print(f"[Dynamics][WARN] module1 map file not found, using fallback: {path}")
        return map_cfg

    try:
        import yaml
    except Exception as exc:
        print(f"[Dynamics][WARN] PyYAML unavailable ({exc}), using fallback mapping.")
        return map_cfg

    try:
        with open(path, "r", encoding="utf-8") as fp:
            loaded = yaml.safe_load(fp) or {}
        for key, fallback_value in MODULE1_FALLBACK_MAP.items():
            loaded_value = loaded.get(key)
            if isinstance(fallback_value, dict) and isinstance(loaded_value, dict):
                merged = fallback_value.copy()
                merged.update(loaded_value)
                map_cfg[key] = merged
            else:
                map_cfg[key] = loaded_value if loaded_value is not None else fallback_value
        return map_cfg
    except Exception as exc:
        print(f"[Dynamics][WARN] failed reading module1 map ({exc}), using fallback mapping.")
        return map_cfg


def compute_surrogate_dynamics(profile: dict, map_cfg: dict) -> dict:
    clamp_ranges = map_cfg["clamp_ranges"]
    friction_map = map_cfg["friction_map"]
    slip_penalty = map_cfg["slip_penalty"]
    restitution_map = map_cfg["restitution_map"]
    mass_base = map_cfg["mass_base_kg_by_category"]
    density_multiplier = map_cfg["density_multiplier"]
    size_multiplier = map_cfg["size_multiplier"]

    surface = profile.get("surface_friction", "medium")
    slip = profile.get("slip_tendency", "medium")
    restitution_label = profile.get("restitution", "low")
    mass_label = profile.get("mass_category", "medium")
    density_label = profile.get("density_category", "medium")
    size_label = profile.get("size_relative", "medium")
    deformability = profile.get("deformability", "medium")

    lateral_raw = float(friction_map.get(surface, friction_map["medium"])) + float(
        slip_penalty.get(slip, slip_penalty["medium"])
    )
    lateral_friction = clamp_value(
        lateral_raw,
        float(clamp_ranges["lateral_friction"][0]),
        float(clamp_ranges["lateral_friction"][1]),
    )

    restitution_raw = float(restitution_map.get(restitution_label, restitution_map["medium"]))
    restitution = clamp_value(
        restitution_raw,
        float(clamp_ranges["restitution"][0]),
        float(clamp_ranges["restitution"][1]),
    )

    mass_raw = (
        float(mass_base.get(mass_label, mass_base["medium"]))
        * float(density_multiplier.get(density_label, density_multiplier["medium"]))
        * float(size_multiplier.get(size_label, size_multiplier["unknown"]))
    )
    mass_kg = clamp_value(
        mass_raw,
        float(clamp_ranges["mass_kg"][0]),
        float(clamp_ranges["mass_kg"][1]),
    )

    linear_damping_raw = {"low": 0.02, "medium": 0.05, "high": 0.08}.get(deformability, 0.05)
    angular_damping_raw = {"low": 0.01, "medium": 0.03, "high": 0.06}.get(deformability, 0.03)
    linear_damping = clamp_value(
        linear_damping_raw,
        float(clamp_ranges["linear_damping"][0]),
        float(clamp_ranges["linear_damping"][1]),
    )
    angular_damping = clamp_value(
        angular_damping_raw,
        float(clamp_ranges["angular_damping"][0]),
        float(clamp_ranges["angular_damping"][1]),
    )

    return {
        "mass_kg": round(mass_kg, 6),
        "lateral_friction": round(lateral_friction, 6),
        "restitution": round(restitution, 6),
        "linear_damping": round(linear_damping, 6),
        "angular_damping": round(angular_damping, 6),
    }


def apply_module1_dynamics_to_loaded_objects(
    ycb_object_ids: dict,
    map_cfg: dict,
    infer_profile_from_body: bool = True,
) -> dict:
    applied = {}
    for label, body_id in ycb_object_ids.items():
        profile = MODULE1_PROFILE_DEFAULT.copy()
        inferred_profile = {}
        body_features = None
        if infer_profile_from_body:
            inferred_profile, body_features = infer_module1_profile_from_loaded_body(
                label=label,
                body_id=body_id,
            )
            profile.update(inferred_profile)
        profile.update(YCB_MODULE1_PROFILE_BY_LABEL.get(label, {}))
        surrogate = compute_surrogate_dynamics(profile=profile, map_cfg=map_cfg)
        override = YCB_DYNAMICS_OVERRIDE_BY_LABEL.get(label, {})
        if override:
            surrogate.update(override)
            print(f"[Dynamics][WARN] {label} dynamics override applied: {override}")
        p.changeDynamics(
            body_id,
            -1,
            mass=surrogate["mass_kg"],
            lateralFriction=surrogate["lateral_friction"],
            restitution=surrogate["restitution"],
            linearDamping=surrogate["linear_damping"],
            angularDamping=surrogate["angular_damping"],
            rollingFriction=max(0.001, surrogate["lateral_friction"] * 0.01),
            spinningFriction=max(0.001, surrogate["lateral_friction"] * 0.01),
        )
        info = p.getDynamicsInfo(body_id, -1)
        applied[label] = {
            "inferred_profile": inferred_profile,
            "profile": profile,
            "body_features": body_features,
            "requested": surrogate,
            "actual": {
                "mass_kg": round(float(info[0]), 6),
                "lateral_friction": round(float(info[1]), 6),
                "restitution": round(float(info[5]), 6),
            },
        }
    return applied


def summarize_applied_dynamics(applied: dict) -> dict:
    """
    Keep boot logs compact by printing only core requested/actual dynamics.
    """
    summary = {}
    for label, payload in applied.items():
        requested = payload.get("requested", {})
        actual = payload.get("actual", {})
        summary[label] = {
            "requested": {
                "mass_kg": requested.get("mass_kg"),
                "lateral_friction": requested.get("lateral_friction"),
                "restitution": requested.get("restitution"),
            },
            "actual": {
                "mass_kg": actual.get("mass_kg"),
                "lateral_friction": actual.get("lateral_friction"),
                "restitution": actual.get("restitution"),
            },
        }
    return summary


def configure_simulation() -> None:
    p.connect(p.DIRECT)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.0,
        cameraYaw=70,
        cameraPitch=-60,
        cameraTargetPosition=[1.3, 0.0, 0.8],
    )
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setTimeStep(SIM_TIMESTEP)
    p.setGravity(0, 0, -9.8)
    p.setPhysicsEngineParameter(
        fixedTimeStep=SIM_TIMESTEP,
        deterministicOverlappingPairs=1,
        numSolverIterations=150,
        numSubSteps=2,
    )


def load_static_scene() -> dict:
    plane_id = p.loadURDF("plane.urdf")
    table_id = p.loadURDF("table/table.urdf", basePosition=TABLE_BASE_POSITION)
    p.changeDynamics(plane_id, -1, lateralFriction=0.9, restitution=0.0)
    p.changeDynamics(table_id, -1, lateralFriction=0.9, restitution=0.0)
    return {
        "plane_id": plane_id,
        "table_id": table_id,
    }

def _get_table_surface_z(table_body_id=None) -> float:
    """테이블 AABB 상단 z를 반환. 모르면 기본값 사용."""
    if table_body_id is not None:
        try:
            _, aabb_max = p.getAABB(table_body_id)
            return float(aabb_max[2])
        except Exception:
            pass
    return 0.625 

def load_ycb_objects(ycb_dir: str = YCB_DIR, table_body_id=None) -> dict:
    if not os.path.isdir(ycb_dir):
        raise FileNotFoundError(
            f"YCB directory not found: {ycb_dir}. "
            "Extract data.zip first."
        )
 
    # globalScaling 제거: YCB URDF는 이미 미터 단위 실물 크기.
    flags = p.URDF_USE_INERTIA_FROM_FILE
    loaded = {}
 
    for label, urdf_name, base_position in YCB_OBJECT_SPECS:
        urdf_path = os.path.join(ycb_dir, urdf_name)
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"Missing URDF: {urdf_path}")
        spawn_pos = [base_position[0], base_position[1], base_position[2]]
        body_id = p.loadURDF(
            urdf_path,
            basePosition=spawn_pos,
            baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
            # YCB URDF는 실물 스케일(미터) 기준이므로 축소하지 않는다.
            globalScaling=0.1,
            flags=flags,
        )
        loaded[label] = body_id
        print(f"[Load] '{label}' body_id={body_id} at {[round(v,3) for v in spawn_pos]}")
    return loaded


def create_dual_arm_controllers() -> tuple[dict, dict]:
    left_controller = PandaController(name="left", base_position=LEFT_BASE_POSITION)
    right_controller = PandaController(name="right", base_position=RIGHT_BASE_POSITION)

    left_id = left_controller.load_panda()
    right_id = right_controller.load_panda()

    controllers = {
        "left": left_controller,
        "right": right_controller,
    }
    robot_ids = {
        "left": left_id,
        "right": right_id,
    }
    return controllers, robot_ids


def stabilize_scene(steps: int = STABILIZE_STEPS) -> None:
    for _ in range(steps):
        p.stepSimulation()
        time.sleep(SIM_TIMESTEP)


def capture_affordance_rgb(
    width: int = AFFORDANCE_CAPTURE_WIDTH,
    height: int = AFFORDANCE_CAPTURE_HEIGHT,
) -> np.ndarray:
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[1.05, -1.00, 1.35],
        cameraTargetPosition=[1.3, 0.0, 0.8],
        cameraUpVector=[0.0, 0.0, 1.0],
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=60.0,
        aspect=width / height,
        nearVal=0.01,
        farVal=5.0,
    )
    _, _, rgba, _, _ = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_TINY_RENDERER,
    )
    rgba_np = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)
    return rgba_np[:, :, :3]


# 수정 후
def _pixel_depth_to_world(
    px: float,
    py: float,
    depth_buf: np.ndarray,
    proj_matrix: tuple,
    view_matrix: tuple,
    img_w: int,
    img_h: int,
) -> np.ndarray | None:
    """
    2D 이미지 픽셀 (px, py) + PyBullet depth buffer → world 3D 좌표 변환.
    depth_buf: getCameraImage의 depthBuffer (float32 [H, W], 0~1 NDC)
    """
    ix = int(np.clip(px, 0, img_w - 1))
    iy = int(np.clip(py, 0, img_h - 1))
    depth_ndc = float(depth_buf[iy, ix])
    if depth_ndc >= 0.9999:   # 배경/무한대
        return None

    # NDC → clip → view → world
    # PyBullet projection은 OpenGL 컨벤션 (column-major, z [-1,1])
    proj = np.array(proj_matrix).reshape(4, 4).T
    view = np.array(view_matrix).reshape(4, 4).T

    # pixel → NDC
    ndc_x = (px / img_w) * 2.0 - 1.0
    ndc_y = 1.0 - (py / img_h) * 2.0   # y축 반전
    ndc_z = 2.0 * depth_ndc - 1.0

    clip = np.array([ndc_x, ndc_y, ndc_z, 1.0])
    view_inv = np.linalg.inv(proj)
    view_coord = view_inv @ clip
    view_coord /= view_coord[3]

    world_inv = np.linalg.inv(view)
    world = world_inv @ view_coord
    return world[:3]


def _crop_object_rgb(
    rgb: np.ndarray,
    depth: np.ndarray,
    bbox_pixel: list[int],
    pad: int = 8,
) -> np.ndarray:
    """bbox_pixel 영역을 잘라 R1 개별 추론용 crop 이미지 반환."""
    h, w = rgb.shape[:2]
    x1 = max(0, bbox_pixel[0] - pad)
    y1 = max(0, bbox_pixel[1] - pad)
    x2 = min(w, bbox_pixel[2] + pad)
    y2 = min(h, bbox_pixel[3] + pad)
    if x2 <= x1 or y2 <= y1:
        return rgb
    return rgb[y1:y2, x1:x2].copy()


def _project_body_to_pixel(
    body_id: int,
    proj_matrix: tuple,
    view_matrix: tuple,
    img_w: int,
    img_h: int,
) -> list[int] | None:
    """PyBullet body의 world pos를 카메라 픽셀 좌표로 투영."""
    try:
        pos, _ = p.getBasePositionAndOrientation(body_id)
        proj = np.array(proj_matrix).reshape(4, 4).T
        view = np.array(view_matrix).reshape(4, 4).T
        world_pt = np.array([pos[0], pos[1], pos[2], 1.0])
        clip = proj @ view @ world_pt
        if abs(clip[3]) < 1e-6:
            return None
        ndc = clip[:3] / clip[3]
        px = int((ndc[0] + 1.0) / 2.0 * img_w)
        py = int((1.0 - ndc[1]) / 2.0 * img_h)
        if 0 <= px < img_w and 0 <= py < img_h:
            return [px, py]
    except Exception:
        pass
    return None


def run_optional_affordance_probe(
    enable_affordance_r1: bool,
    enable_sam2_refinement: bool = False,
    controllers: dict | None = None,
    ycb_object_ids: dict | None = None,
    target_labels: list[str] | None = None,
) -> dict[str, dict]:
    """
    R1 추론을 물체별로 실행해 각 controller에 3D grasp hint를 주입.

    Returns:
        { label: { "world_pos": [x,y,z], "orientation": quaternion, "bbox_pixel": [...] } }
    """
    results: dict[str, dict] = {}

    if not enable_affordance_r1:
        print("[R1] optional affordance probe disabled (set ENABLE_AFFORDANCE_R1=1 to enable)")
        return results

    print("[R1] optional affordance probe enabled")

    try:
        from affordancegrasp_r1_adapter import AffordanceGraspR1Adapter
    except Exception as exc:
        print(f"[R1][WARN] adapter import failed: {exc}")
        return results

    try:
        # ── 1. 카메라 설정 (depth 포함) ──────────────────────────────────────
        IMG_W = AFFORDANCE_CAPTURE_WIDTH
        IMG_H = AFFORDANCE_CAPTURE_HEIGHT
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[1.05, -1.00, 1.35],
            cameraTargetPosition=[1.3, 0.0, 0.8],
            cameraUpVector=[0.0, 0.0, 1.0],
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=IMG_W / IMG_H,
            nearVal=0.01,
            farVal=5.0,
        )
        _, _, rgba_raw, depth_raw, _ = p.getCameraImage(
            width=IMG_W,
            height=IMG_H,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER,
        )
        rgb_full = np.asarray(rgba_raw, dtype=np.uint8).reshape(IMG_H, IMG_W, 4)[:, :, :3]
        depth_buf = np.asarray(depth_raw, dtype=np.float32).reshape(IMG_H, IMG_W)

        # ── 2. R1 adapter 로드 (CPU 강제) ────────────────────────────────────
        import os as _os
        _os.environ["CUDA_VISIBLE_DEVICES"] = ""   # GPU 비활성화 → device mismatch 방지

        adapter = AffordanceGraspR1Adapter(
            model_id=AFFORDANCE_MODEL_ID,
            device="cpu",
            local_model_dir=_os.getenv("AFFORDANCE_R1_LOCAL_DIR"),
            local_files_only=env_flag("AFFORDANCE_R1_LOCAL_ONLY", default=False),
        )
        adapter.load()
        if not adapter.is_available():
            print(f"[R1][WARN] model not available: {adapter._last_error}")
            return results

        # ── 3. 물체별 개별 추론 ─────────────────────────────────────────────
        labels_to_probe = target_labels or (list(ycb_object_ids.keys()) if ycb_object_ids else [])
        if not labels_to_probe:
            print("[R1][WARN] no target labels to probe.")
            return results

        for label in labels_to_probe:
            body_id = (ycb_object_ids or {}).get(label)
            if body_id is None:
                print(f"[R1][WARN] label '{label}' not in ycb_object_ids, skipping.")
                continue

            print(f"[R1] probing '{label}' (body_id={body_id})...")

            # 3-a. body를 픽셀로 투영해 crop 범위 결정
            center_px = _project_body_to_pixel(
                body_id, proj_matrix, view_matrix, IMG_W, IMG_H
            )
            if center_px is not None:
                # AABB 기반 bbox 크기 추정
                try:
                    aabb_min, aabb_max = p.getAABB(body_id)
                    # 물체 크기를 픽셀 크기로 대략 환산 (FOV 기반 rough estimate)
                    obj_span = max(
                        float(aabb_max[0] - aabb_min[0]),
                        float(aabb_max[1] - aabb_min[1]),
                        float(aabb_max[2] - aabb_min[2]),
                    )
                    half_px = max(20, int(obj_span * IMG_W * 0.5))
                    crop_bbox = [
                        max(0, center_px[0] - half_px),
                        max(0, center_px[1] - half_px),
                        min(IMG_W, center_px[0] + half_px),
                        min(IMG_H, center_px[1] + half_px),
                    ]
                except Exception:
                    crop_bbox = [
                        max(0, center_px[0] - 30),
                        max(0, center_px[1] - 30),
                        min(IMG_W, center_px[0] + 30),
                        min(IMG_H, center_px[1] + 30),
                    ]
                crop_img = _crop_object_rgb(rgb_full, depth_buf, crop_bbox)
            else:
                crop_img = rgb_full   # fallback: 전체 이미지
                crop_bbox = [0, 0, IMG_W, IMG_H]

            # 3-b. R1 추론 (물체 label을 prompt에 명시)
            r1_result = adapter.predict(
                image=crop_img,
                prompt=(
                    f"Find the best robot grasp region for '{label}'. "
                    "The robot gripper approaches from above. "
                    "Return [x1, y1, x2, y2] (0-1000 scale) and a part name."
                ),
                extra_context={
                    "target_object": label,
                    "scene": "dual-panda-tabletop",
                    "grasp_direction": "top-down",
                },
                max_new_tokens=128,
            )

            if not r1_result.get("success"):
                print(f"[R1][WARN] '{label}' inference failed: {r1_result.get('error')}")
                continue

            candidates = r1_result.get("grasp_candidates") or []
            if not candidates:
                print(f"[R1][INFO] '{label}' — no candidate parsed from output: {r1_result.get('raw_output')}")
                continue

            top = candidates[0]
            print(f"[R1] '{label}' top candidate: {top}")

            # 3-c. crop 내 픽셀 좌표 → 전체 이미지 픽셀 좌표로 역변환
            cx_crop, cy_crop = top["center_pixel"]
            crop_w = crop_bbox[2] - crop_bbox[0]
            crop_h = crop_bbox[3] - crop_bbox[1]
            if crop_w > 0 and crop_h > 0:
                cx_full = crop_bbox[0] + cx_crop * (crop_w / max(crop_img.shape[1], 1))
                cy_full = crop_bbox[1] + cy_crop * (crop_h / max(crop_img.shape[0], 1))
            else:
                cx_full, cy_full = cx_crop, cy_crop

            # 수정 후
            # 3-d. 2D pixel → 3D world 좌표 변환
            # center 1점이 배경에 걸릴 수 있으므로 bbox 내 9점을 샘플링해 유효한 값 사용
            _sample_pts = []
            _bpx = top["bbox_pixel"]  # crop 기준
            for _sy in [0.25, 0.5, 0.75]:
                for _sx in [0.25, 0.5, 0.75]:
                    _scx = _bpx[0] + (_bpx[2] - _bpx[0]) * _sx
                    _scy = _bpx[1] + (_bpx[3] - _bpx[1]) * _sy
                    # crop → full 역변환
                    if crop_w > 0 and crop_h > 0:
                        _fx = crop_bbox[0] + _scx * (crop_w / max(crop_img.shape[1], 1))
                        _fy = crop_bbox[1] + _scy * (crop_h / max(crop_img.shape[0], 1))
                    else:
                        _fx, _fy = _scx, _scy
                    _wp = _pixel_depth_to_world(
                        px=_fx, py=_fy,
                        depth_buf=depth_buf,
                        proj_matrix=proj_matrix,
                        view_matrix=view_matrix,
                        img_w=IMG_W,
                        img_h=IMG_H,
                    )
                    if _wp is not None:
                        _sample_pts.append(_wp)

            if _sample_pts:
                # 유효한 샘플들의 중앙값 사용 (이상치 제거)
                world_pos = np.median(np.array(_sample_pts), axis=0)
            else:
                # 모든 샘플이 배경 → body center pixel로 직접 재시도
                _cp = _project_body_to_pixel(body_id, proj_matrix, view_matrix, IMG_W, IMG_H)
                if _cp is not None:
                    world_pos = _pixel_depth_to_world(
                        px=float(_cp[0]), py=float(_cp[1]),
                        depth_buf=depth_buf,
                        proj_matrix=proj_matrix,
                        view_matrix=view_matrix,
                        img_w=IMG_W,
                        img_h=IMG_H,
                    )
                if world_pos is None:
                    # 최종 fallback: AABB top center
                    try:
                        aabb_min, aabb_max = p.getAABB(body_id)
                        world_pos = np.array([
                            (float(aabb_min[0]) + float(aabb_max[0])) / 2.0,
                            (float(aabb_min[1]) + float(aabb_max[1])) / 2.0,
                            float(aabb_max[2]),
                        ])
                        print(f"[R1] '{label}' depth miss (9-sample) → AABB fallback")
                    except Exception:
                        print(f"[R1][WARN] '{label}' could not compute world_pos, skipping.")
                        continue

            try:
                _aabb_min, _aabb_max = p.getAABB(body_id)
                _dx = float(_aabb_max[0] - _aabb_min[0])
                _dy = float(_aabb_max[1] - _aabb_min[1])
                _, _body_orn = p.getBasePositionAndOrientation(body_id)
                _pose_yaw = float(p.getEulerFromQuaternion(_body_orn)[2])
                if _dx > _dy * 1.15:
                    # 긴 축 X → gripper finger를 Y방향으로 벌려야 옆면 잡음 → yaw=90°
                    grasp_yaw = np.pi / 2.0
                elif _dy > _dx * 1.15:
                    # 긴 축 Y → gripper finger를 X방향으로 벌려야 옆면 잡음 → yaw=0°
                    grasp_yaw = 0.0
                else:
                    # 정방형: pose yaw 그대로
                    grasp_yaw = _pose_yaw
                print(f"[R1] '{label}' AABB yaw: dx={_dx:.3f} dy={_dy:.3f} → yaw={np.degrees(grasp_yaw):.1f}°")
            except Exception:
                grasp_yaw = 0.0
            grasp_orientation = p.getQuaternionFromEuler([np.pi, 0.0, grasp_yaw])

            print(
                f"[R1] '{label}' world_pos={[round(v, 4) for v in world_pos.tolist()]}, "
                f"yaw={np.degrees(grasp_yaw):.1f}°"
            )

            hint_payload = {
                "world_pos": world_pos.tolist(),
                "orientation": list(grasp_orientation),
                "bbox_pixel": top.get("bbox_pixel", []),
                "part": top.get("part", ""),
                "score": float(top.get("score", 1.0)),
            }
            results[label] = hint_payload

            # SAM2 refinement (선택)
            if enable_sam2_refinement:
                try:
                    from sam2_segmenter_adapter import SAM2SegmentationAdapter
                    sam2 = SAM2SegmentationAdapter(
                        model_id=_os.getenv("SAM2_MODEL_ID", SAM2_MODEL_ID),
                        local_model_dir=_os.getenv("SAM2_LOCAL_DIR"),
                        local_files_only=env_flag("SAM2_LOCAL_ONLY", default=False),
                    )
                    refined = sam2.refine_result(image=crop_img, affordance_result=r1_result, top_k=1)
                    if refined.get("success"):
                        print(f"[SAM2] '{label}' mask refined.")
                except Exception as sam_exc:
                    print(f"[SAM2][WARN] '{label}' refinement failed: {sam_exc}")

            # 3-f. controller에 hint 주입
            if isinstance(controllers, dict):
                for arm_name, controller in controllers.items():
                    # 이 label이 해당 arm의 타겟인지는 run_sequential_demo에서 결정
                    # 여기서는 전체 inference_result에 world_pos 포함해서 저장
                    r1_result_with_3d = dict(r1_result)
                    r1_result_with_3d["world_pos"] = world_pos.tolist()
                    r1_result_with_3d["orientation"] = list(grasp_orientation)
                    r1_result_with_3d["target_label"] = label
                    controller.set_affordance_hint(r1_result_with_3d)
                    print(f"[R1] hint injected into controller '{arm_name}' for '{label}'")

        return results

    except Exception as exc:
        print(f"[R1][WARN] affordance probe failed: {exc}")
        import traceback; traceback.print_exc()
        return results


def run_sequential_demo(
    controllers: dict,
    ycb_object_ids: dict,
    r1_hints: dict | None = None,   # ← run_optional_affordance_probe 결과
) -> None:
    left = controllers["left"]
    right = controllers["right"]
    down_orn = p.getQuaternionFromEuler([np.pi, 0.0, 0.0])
    r1_hints = r1_hints or {}
 
    CAM_CONFIG = {
    "cam_target":   [1.0, 0.0, 0.8],
    "cam_distance": 0.8,
    "cam_yaw":      70,
    "cam_pitch":    -60,
    }
 
    print("\n[3] 가상 카메라 렌더링 중...")
    rgb, depth, proj_matrix, view_matrix = render_camera(**CAM_CONFIG)
 
    try:
        from PIL import Image as PILImage
        PILImage.fromarray(rgb).save("scene_capture.png")
        print("[3] scene_capture.png 저장 완료")
    except Exception:
        pass
 
    print("[Boot] reset both robots to home")
    left.reset_to_home(steps=600)
    right.reset_to_home(steps=600)
 
    print("[Demo] open both grippers")
    left.open_gripper(steps=120)
    right.open_gripper(steps=120)
 
    if not ycb_object_ids:
        print("[Demo][WARN] no YCB objects were loaded, skipping grasp demo.")
        return
 
    # ── module3_output.json에서 grasp 대상 결정 ────────────────────────────
    # 물체 y 좌표 기준으로 팔 할당:
    #   left  base y=-0.35 → spawn y가 더 작은(음수에 가까운) 물체 담당
    #   right base y=+0.35 → spawn y가 더 큰(양수에 가까운) 물체 담당
    # 이렇게 하면 팔이 서로 교차하지 않음.
    _m3_labels = _load_module3_object_labels(_MODULE3_JSON_PATH)
    _loaded_labels = list(ycb_object_ids.keys())
 
    # YCB_OBJECT_SPECS에서 label→spawn_y 매핑
    _label_to_y = {lbl: pos[1] for lbl, _, pos in YCB_OBJECT_SPECS}
 
    # JSON 물체 중 실제 로드된 것만 후보
    _candidates = [lbl for lbl in _m3_labels if lbl in ycb_object_ids]
    if not _candidates:
        _candidates = _loaded_labels[:2]
 
    if len(_candidates) >= 2:
        # y 기준 정렬: 작은 y → left, 큰 y → right
        _candidates_sorted = sorted(_candidates, key=lambda l: _label_to_y.get(l, 0.0))
        left_target_label  = _candidates_sorted[0]
        right_target_label = _candidates_sorted[1]
    elif len(_candidates) == 1:
        left_target_label  = _candidates[0]
        right_target_label = _candidates[0]
    else:
        print("[Demo][WARN] could not determine grasp targets; skipping demo.")
        return
 
    if left_target_label is None:
        print("[Demo][WARN] could not determine left grasp target; skipping demo.")
        return
 
    left_body_id  = ycb_object_ids[left_target_label]
    right_body_id = ycb_object_ids[right_target_label]
 
    print(f"[Demo] grasp targets — left: '{left_target_label}', right: '{right_target_label}'")
 
    # 조립·배치 위치 (두 팔이 서로 다른 Y 방향에서 접근)
    ASSEMBLY_POS_LEFT  = [0.62, -0.10, 0.97]
    ASSEMBLY_POS_RIGHT = [0.62,  0.10, 1.13]
    PLACE_POS          = [0.62,  0.00, 0.85]
 
    # ══════════════════════════════════════════════════════
    # Step 1-L. Left arm: base_object 파지 → 제자리 대기
    # ══════════════════════════════════════════════════════
    # 수정 후
    print(f"[Demo] left-arm grasp target: {left_target_label}")
    _left_hint = r1_hints.get(left_target_label, {})
    _left_orn = _left_hint.get("orientation") or down_orn
    _left_r1_pos = _left_hint.get("world_pos")  # R1 world_pos (없으면 None → AABB fallback)
    if _left_hint.get("orientation"):
        print(f"[R1] using R1 orientation for '{left_target_label}': {[round(v,4) for v in _left_orn]}")
    if _left_r1_pos:
        print(f"[R1] using R1 world_pos for '{left_target_label}': {[round(v,4) for v in _left_r1_pos]}")
    left_ok = left.grasp_body(
        body_id=left_body_id,
        object_label=left_target_label,
        orientation=list(_left_orn),
        r1_world_pos=_left_r1_pos,
    )
    if not left_ok:
        print(f"[Demo][WARN] left-arm grasp failed for '{left_target_label}'")
    else:
        left.maintain_grasp_hold(steps=120)
        print("[Demo] left arm holding — waiting for right arm grasp.")
 
    # ══════════════════════════════════════════════════════
    # Step 1-R. Right arm: attach_object 파지 → 제자리 대기
    #           left가 들고 있는 동안 hold_companion으로 보호
    # ══════════════════════════════════════════════════════
    print(f"[Demo] right-arm grasp target: {right_target_label}")
    _right_hint = r1_hints.get(right_target_label, {})
    _right_orn = _right_hint.get("orientation") or down_orn
    _right_r1_pos = _right_hint.get("world_pos")
    if _right_hint.get("orientation"):
        print(f"[R1] using R1 orientation for '{right_target_label}': {[round(v,4) for v in _right_orn]}")
    if _right_r1_pos:
        print(f"[R1] using R1 world_pos for '{right_target_label}': {[round(v,4) for v in _right_r1_pos]}")
    right_ok = right.grasp_body(
        body_id=right_body_id,
        object_label=right_target_label,
        orientation=list(_right_orn),
        hold_companion=left if left_ok else None,
        r1_world_pos=_right_r1_pos,
    )
    if not right_ok:
        print(f"[Demo][WARN] right-arm grasp failed for '{right_target_label}'")
    else:
        right.maintain_grasp_hold(steps=120)
        print("[Demo] right arm holding — both arms ready.")
 
    # ══════════════════════════════════════════════════════
    # Step 2-L. Left arm: 조립 위치로 이동
    #           right가 들고 있는 동안 hold_companion으로 보호
    # ══════════════════════════════════════════════════════
    # move_end_effector_to 내부에서 매 스텝 _update_grasp_hold_feedback() +
    # _set_gripper_target()이 호출되므로 slip 감지 시 자동으로 force가 증가함.
    if left_ok:
        left.move_end_effector_to(
            ASSEMBLY_POS_LEFT, orientation=down_orn, steps=600,
            hold_companion=right if right_ok else None,
        )
        left.maintain_grasp_hold(steps=60)
        print("[Demo] left arm at assembly position.")
 
    # ══════════════════════════════════════════════════════
    # Step 2-R. Right arm: 조립 위치로 이동
    #           left hold_companion으로 보호
    # ══════════════════════════════════════════════════════
    if right_ok:
        right.move_end_effector_to(
            ASSEMBLY_POS_RIGHT, orientation=down_orn, steps=600,
            hold_companion=left if left_ok else None,
        )
        right.maintain_grasp_hold(steps=80)
        print("[Demo] right arm at assembly position.")
 
    # 수정 후
    # ══════════════════════════════════════════════════════
    # Step 3. Assembly — module3_output.json 계획 기반 실행
    # ══════════════════════════════════════════════════════
    assembly_manager = AssemblyManager()
 
    # JSON 계획 로드 (파일이 없으면 fallback으로 직접 attach)
    _plan_path = _MODULE3_JSON_PATH
    _plan_loaded = False
    if os.path.isfile(_plan_path):
        try:
            assembly_manager.load_plan_from_json(_plan_path)
            # JSON에 등장하는 모든 물체 label → pybullet body_id 매핑 등록
            # ycb_object_ids 키는 MODULE3_LABEL_TO_YCB의 urdf_label(=raw json label)과 일치
            _m3_all_labels = _load_module3_object_labels(_plan_path)
            _body_map = {lbl: ycb_object_ids[lbl] for lbl in _m3_all_labels if lbl in ycb_object_ids}
            if not _body_map:
                # fallback: 로드된 두 물체만 등록
                _body_map = {left_target_label: left_body_id, right_target_label: right_body_id}
            assembly_manager.register_bodies(_body_map)
            _plan_loaded = True
        except Exception as exc:
            print(f"[Demo][WARN] module3 plan load failed ({exc}), using fallback attach.")
    else:
        print(f"[Demo][WARN] module3_output.json not found at {_plan_path}, using fallback attach.")
 
    assembly_results = []
    if left_ok and right_ok:
        # ── attach 전 두 물체 간 거리 확인 및 nudge ─────────────────────────
        main_pos, main_orn = p.getBasePositionAndOrientation(left_body_id)
        aux_pos,  _        = p.getBasePositionAndOrientation(right_body_id)
        body_dist = float(np.linalg.norm(np.array(aux_pos) - np.array(main_pos)))
        print(f"[Demo] pre-attach body distance: {body_dist:.3f} m")
 
        if body_dist > 0.35:
            print("[Demo] bodies too far apart — nudging right arm closer...")
            left_aabb_min, left_aabb_max   = p.getAABB(left_body_id)
            right_aabb_min, right_aabb_max = p.getAABB(right_body_id)
            left_top_z     = float(left_aabb_max[2])
            right_half_z   = (float(right_aabb_max[2]) - float(right_aabb_min[2])) / 2.0
            nudge_pos = [float(main_pos[0]), float(main_pos[1]), left_top_z + right_half_z + 0.01]
            right.move_end_effector_to(nudge_pos, orientation=down_orn, steps=400, hold_companion=left)
            right.maintain_grasp_hold(steps=60)
            main_pos, main_orn = p.getBasePositionAndOrientation(left_body_id)
            aux_pos,  _        = p.getBasePositionAndOrientation(right_body_id)
            body_dist = float(np.linalg.norm(np.array(aux_pos) - np.array(main_pos)))
            print(f"[Demo] post-nudge body distance: {body_dist:.3f} m")
 
        print("[Demo] assembling parts...")
 
        if _plan_loaded:
            # ── module3 JSON 계획 실행 ──────────────────────────────────────
            # step1(position-only)은 배치 기록만, step2 이후 attach 실행
            assembly_results = assembly_manager.execute_plan(settle_steps=60, max_force=500)
            attach_results = [r for r in assembly_results if r.get("constraint_id") is not None]
            if attach_results:
                print(f"[Demo] assembly successful via module3 plan "
                      f"({len(attach_results)} constraint(s) created).")
            else:
                print("[Demo][WARN] module3 plan produced no constraints — falling back.")
                _plan_loaded = False   # fallback으로 전환
 
        if not _plan_loaded:
            # ── fallback: 현재 실제 위치 기반 직접 attach ──────────────────
            main_orn_inv = p.invertTransform([0, 0, 0], list(main_orn))[1]
            contact_offset, _ = p.multiplyTransforms(
                [0, 0, 0], main_orn_inv,
                (np.array(aux_pos) - np.array(main_pos)).tolist(), [0, 0, 0, 1],
            )
            cid = assembly_manager.attach(
                main_body_id=left_body_id,
                aux_body_id=right_body_id,
                contact_offset=list(contact_offset),
                label=f"{left_target_label}_{right_target_label}",
                settle_steps=60,
                max_force=500,
            )
            assembly_results = [{"step": 1, "ok": cid is not None, "constraint_id": cid}]
            if cid is not None:
                print(f"[Demo] fallback assembly successful (constraint={cid}, "
                      f"offset=[{contact_offset[0]:.3f}, {contact_offset[1]:.3f}, {contact_offset[2]:.3f}], "
                      f"dist={body_dist:.3f} m)")
            else:
                print("[Demo][WARN] fallback assembly also failed.")
 
        # 안정화
        for _ in range(DEMO_HOLD_STEPS):
            left._tick_gripper_hold()
            right._tick_gripper_hold()
            p.stepSimulation()
            time.sleep(SIM_TIMESTEP)
 
    else:
        print("[Demo][WARN] skipping assembly (one or both grasps failed).")
 
    # ══════════════════════════════════════════════════════
    # Step 4. 결합체 내려놓기 & release
    # ══════════════════════════════════════════════════════
    if left_ok:
        left.move_end_effector_to(
            PLACE_POS, orientation=down_orn, steps=400,
            hold_companion=right if right_ok else None,
        )
        left.maintain_grasp_hold(steps=60)
 
    # (carry_constraint 없음 — gripper force 유지로 이동)
 
    if left_ok:
        left.release_grasp(open_after=True, steps=120)
 
    if right_ok:
        right.move_end_effector_to(
            [PLACE_POS[0], PLACE_POS[1], PLACE_POS[2] + 0.15],
            orientation=down_orn, steps=400,
        )
        right.maintain_grasp_hold(steps=60)
        right.release_grasp(open_after=True, steps=120)
 
    # assembly constraint는 release 후에도 두 물체가 붙어있게 유지
    # 분리하려면: p.removeConstraint(assembly_constraint)
 
    print("[Demo] return to home")
    left.reset_to_home(steps=600)
    right.reset_to_home(steps=600)
 
    for _ in range(DEMO_HOLD_STEPS):
        p.stepSimulation()
        time.sleep(SIM_TIMESTEP)
 
def keep_gui_alive() -> None:
    print("[Boot] simulation running. Press Ctrl+C to exit.")
    while True:
        p.stepSimulation()
        time.sleep(SIM_TIMESTEP)
 
 
def main() -> None:
    print(f"[Boot] R1 source (HF): {R1_HF_REPO}")
    enable_affordance_r1 = env_flag(
        "ENABLE_AFFORDANCE_R1",
        default=ENABLE_AFFORDANCE_R1_DEFAULT,
    )
    enable_sam2_refinement = env_flag(
        "ENABLE_SAM2_REFINEMENT",
        default=ENABLE_SAM2_REFINEMENT_DEFAULT,
    )
    enable_module1_dynamics = env_flag(
        "ENABLE_MODULE1_DYNAMICS",
        default=ENABLE_MODULE1_DYNAMICS_DEFAULT,
    )
    enable_module1_profile_inference = env_flag(
        "ENABLE_MODULE1_PROFILE_INFERENCE",
        default=ENABLE_MODULE1_PROFILE_INFERENCE_DEFAULT,
    )
    print(f"[Boot] optional Affordance-R1 enabled: {enable_affordance_r1}")
    print(f"[Boot] optional SAM2 refinement enabled: {enable_sam2_refinement}")
    print(f"[Boot] module1 dynamics enabled: {enable_module1_dynamics}")
    print(f"[Boot] module1 profile inference enabled: {enable_module1_profile_inference}")
    print("[Boot] grasp mode: contact-based (no fixed constraint)")
    configure_simulation()
    scene_ids = load_static_scene()
    ycb_object_ids = load_ycb_objects(table_body_id=scene_ids.get("table_id"))
    controllers, robot_ids = create_dual_arm_controllers()
 
    if enable_module1_dynamics:
        map_cfg = load_module1_map()
        applied_dynamics = apply_module1_dynamics_to_loaded_objects(
            ycb_object_ids=ycb_object_ids,
            map_cfg=map_cfg,
            infer_profile_from_body=enable_module1_profile_inference,
        )
        print(f"[Boot] module1 dynamics applied (summary): {summarize_applied_dynamics(applied_dynamics)}")
    else:
        print("[Boot] module1 dynamics skipped by flag.")
 
    print(f"[Boot] scene IDs: {scene_ids}")
    print(f"[Boot] ycb objects: {ycb_object_ids}")
    print(f"[Boot] robot IDs: {robot_ids}")
 
    # 수정 후
    stabilize_scene()

     # 안정화 후 YCB 물체들의 실제 시뮬레이션 좌표 출력
    print("[Boot] === YCB 물체 실제 좌표 (안정화 후) ===")
    for label, body_id in ycb_object_ids.items():
        pos, orn = p.getBasePositionAndOrientation(body_id)
        aabb_min, aabb_max = p.getAABB(body_id)
        aabb_center = [
            (aabb_min[0] + aabb_max[0]) / 2,
            (aabb_min[1] + aabb_max[1]) / 2,
            (aabb_min[2] + aabb_max[2]) / 2,
        ]
        print(
            f"  [{label}] body_id={body_id} "
            f"pos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] "
            f"aabb_center=[{aabb_center[0]:.4f}, {aabb_center[1]:.4f}, {aabb_center[2]:.4f}] "
            f"size=[{aabb_max[0]-aabb_min[0]:.4f}, {aabb_max[1]-aabb_min[1]:.4f}, {aabb_max[2]-aabb_min[2]:.4f}]"
        )
    print("[Boot] ==========================================")

    # module3 대상 물체 labels를 R1 probe에 전달
    _m3_labels = _load_module3_object_labels(_MODULE3_JSON_PATH)
    r1_hints = run_optional_affordance_probe(
        enable_affordance_r1=enable_affordance_r1,
        enable_sam2_refinement=enable_sam2_refinement,
        controllers=controllers,
        ycb_object_ids=ycb_object_ids,
        target_labels=_m3_labels if _m3_labels else None,
    )

    run_sequential_demo(
        controllers=controllers,
        ycb_object_ids=ycb_object_ids,
        r1_hints=r1_hints,
    )
    keep_gui_alive()
 
 
if __name__ == "__main__":
    main()