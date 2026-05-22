"""
main_simulation_3.py
Boot-only dual-arm Panda simulation orchestration.

Notes:
- R1 model assets should be obtained from Hugging Face:
  https://huggingface.co/hqking/affordance-r1
- This boot path intentionally avoids GPT / OpenAI / VL-Grasp / open3d imports.
"""

import os
import re
import time
import math

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
AFFORDANCE_CAPTURE_WIDTH = 1280
AFFORDANCE_CAPTURE_HEIGHT = 960
MODULE1_MAP_PATH = "/workspace/KCC-2026-VLM/module1-2B/configs/module1_to_pybullet_map.yaml"

LEFT_BASE_POSITION = [0.6, -0.35, 0.65]
RIGHT_BASE_POSITION = [0.6, 0.35, 0.65]
TABLE_BASE_POSITION = [0.6, 0.0, 0.0]
YCB_DIR = "/workspace/KCC-2026-VLM/data/object2urdf/examples/ycb"

YCB_OBJECT_SPECS = [
    ("large_marker", "040_large_marker.urdf", [1.2, 0.3, 0.82]),
    ("cracker_box", "003_cracker_box.urdf", [1.2, 0.10, 0.82]),
    ("pudding_box", "008_pudding_box.urdf", [0.5, -0.10, 0.82]),
    ("gelatin_box", "009_gelatin_box.urdf", [1.2, -0.3, 0.82]),
    ("bowl", "024_bowl.urdf", [1.0, 0.2, 0.82]),
    ("mug", "025_mug.urdf", [1.0, 0.0, 0.82]),
    ("sponge", "026_sponge.urdf", [1.0, -0.2, 0.82]),
    ("plate", "029_plate.urdf", [0.7, 0.3, 0.82]),   
    ("spatula", "033_spatula.urdf", [0.7, 0.10, 0.82]),
    ("foam_brick", "061_foam_brick.urdf", [0.7, -0.1, 0.82]),
    ("dice", "062_dice.urdf", [0.7, -0.3, 0.82]),
    ("marbles", "063-a_marbles.urdf", [0.45, 0.2, 0.82]),
    ("cups", "065-a_cups.urdf", [0.45, 0.0, 0.82]),
    ("timer", "076_timer.urdf", [0.45, -0.2, 0.82]),
    ("wood_block", "036_wood_block.urdf", [0.45, -0.35, 0.82]),
]

def _load_module3_object_labels(json_path: str) -> list[str]:
    """
    Parse object labels used in module3_output.json.
    Collect both base_object and attach_object while preserving order.
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
    Build YCB_OBJECT_SPECS dynamically from module3_output.json labels.
    Skip unknown labels with a warning.
    """
    labels = _load_module3_object_labels(json_path)
    if not labels:
        # Fallback: load full default YCB set.
        return [
            ("wood_block", "036_wood_block.urdf", [1.2, 0.3, 0.82]),
            ("cracker_box", "003_cracker_box.urdf", [1.2, 0.10, 0.82]),
            ("plate", "029_plate.urdf", [1.2, -0.10, 0.82]),
            ("gelatin_box", "009_gelatin_box.urdf", [1.2, -0.3, 0.82]),
            ("bowl", "024_bowl.urdf", [1.0, 0.2, 0.82]),
            ("mug", "025_mug.urdf", [1.0, 0.0, 0.82]),
            ("sponge", "026_sponge.urdf", [1.0, -0.2, 0.82]),
            ("timer", "076_timer.urdf", [0.7, 0.3, 0.82]),   
            ("spatula", "033_spatula.urdf", [0.7, 0.10, 0.82]),
            ("foam_brick", "061_foam_brick.urdf", [0.7, -0.1, 0.82]),
            ("dice", "062_dice.urdf", [0.7, -0.3, 0.82]),
            ("marbles", "063-a_marbles.urdf", [0.45, 0.2, 0.82]),
            ("cups", "065-a_cups.urdf", [0.45, 0.0, 0.82]),
            ("pudding_box", "008_pudding_box.urdf", [0.45, -0.2, 0.82]),
            ("large_marker", "040_large_marker.urdf", [0.45, -0.35, 0.82]),
        ]
    specs = []
    for raw_label in labels:
        entry = _YCB_SPEC_BY_LABEL.get(raw_label)
        if entry is None:
            print(f"[Boot][WARN] '{raw_label}' is not in YCB_OBJECT_SPECS - skipping load.")
            continue
        specs.append(entry)
    return list(YCB_OBJECT_SPECS)


# Resolve JSON path once for reuse before main().
_MODULE3_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "module3_balloon_output.json")

# Dynamically resolved YCB object specs.
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
    "sponge": {"surface_friction": "medium", "slip_tendency": "low", "mass_category": "light", "size_relative": "medium"},
    "flat_screwdriver": {"surface_friction": "medium", "slip_tendency": "low", "mass_category": "light", "size_relative": "medium"},
}
YCB_DYNAMICS_OVERRIDE_BY_LABEL = {
    
}

def env_flag(env_name: str, default: bool = False) -> bool:
    raw = os.getenv(env_name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _normalize_label_token(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(label).lower())


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
    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.0,
        cameraYaw=45,
        cameraPitch=-45,
        cameraTargetPosition=[1.0, 0.0, 0.6],
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
    """?뚯씠釉?AABB ?곷떒 z瑜?諛섑솚. 紐⑤Ⅴ硫?湲곕낯媛??ъ슜."""
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
 
    # globalScaling ?쒓굅: YCB URDF???대? 誘명꽣 ?⑥쐞 ?ㅻЪ ?ш린.
    flags = p.URDF_USE_INERTIA_FROM_FILE
    loaded = {}

    _SPAWN_ORIENTATION_OVERRIDE_RPY_DEG = {
        "fork": [0.0, 0.0, 180.0],
        "phillips_screwdriver": [0.0, 0.0, 180.0],
    }
 
    for label, urdf_name, base_position in YCB_OBJECT_SPECS:
        urdf_path = os.path.join(ycb_dir, urdf_name)
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"Missing URDF: {urdf_path}")
        spawn_pos = [base_position[0], base_position[1], base_position[2]]
        _rpy_deg = _SPAWN_ORIENTATION_OVERRIDE_RPY_DEG.get(label, [0.0, 0.0, 0.0])
        _rpy_rad = [math.radians(v) for v in _rpy_deg]
        body_id = p.loadURDF(
            urdf_path,
            basePosition=spawn_pos,
            baseOrientation=p.getQuaternionFromEuler(_rpy_rad),
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
        cameraEyePosition=[1.92, -0.35, 2.08],    # 카메라 본체의 위치
        cameraTargetPosition=[0.28, 0.02, 0.76], # 카메라가 바라보는 지점
        cameraUpVector=[0, 0, 1]                 # 하늘 방향 (Z축 위쪽)
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


# ?섏젙 ??
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
    2D ?대?吏 ?쎌? (px, py) + PyBullet depth buffer ??world 3D 醫뚰몴 蹂??
    depth_buf: getCameraImage??depthBuffer (float32 [H, W], 0~1 NDC)
    """
    ix = int(np.clip(px, 0, img_w - 1))
    iy = int(np.clip(py, 0, img_h - 1))
    depth_ndc = float(depth_buf[iy, ix])
    if depth_ndc >= 0.9999:   # 諛곌꼍/臾댄븳?
        return None

    # NDC ??clip ??view ??world
    # PyBullet projection? OpenGL 而⑤깽??(column-major, z [-1,1])
    proj = np.array(proj_matrix).reshape(4, 4).T
    view = np.array(view_matrix).reshape(4, 4).T

    # pixel ??NDC
    ndc_x = (px / img_w) * 2.0 - 1.0
    ndc_y = 1.0 - (py / img_h) * 2.0   # y異?諛섏쟾
    ndc_z = 2.0 * depth_ndc - 1.0

    clip = np.array([ndc_x, ndc_y, ndc_z, 1.0])
    view_inv = np.linalg.inv(proj)
    view_coord = view_inv @ clip
    view_coord /= view_coord[3]

    world_inv = np.linalg.inv(view)
    world = world_inv @ view_coord
    return world[:3]


def _pixel_depth_to_world_with_local_search(
    px: float,
    py: float,
    depth_buf: np.ndarray,
    proj_matrix: tuple,
    view_matrix: tuple,
    img_w: int,
    img_h: int,
    max_radius: int = 10,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    """
    Robust world-point lookup around a target pixel.

    This keeps the R1 path (pixel/depth based) while reducing misses when the
    exact pixel falls on background for thin objects.
    """
    ix0 = int(np.clip(round(px), 0, img_w - 1))
    iy0 = int(np.clip(round(py), 0, img_h - 1))
    direct_allowed = valid_mask is None or bool(valid_mask[iy0, ix0])
    if direct_allowed:
        direct = _pixel_depth_to_world(
            px=px,
            py=py,
            depth_buf=depth_buf,
            proj_matrix=proj_matrix,
            view_matrix=view_matrix,
            img_w=img_w,
            img_h=img_h,
        )
        if direct is not None:
            return direct

    best = None
    best_depth = None
    cx = ix0
    cy = iy0
    for r in range(1, max_radius + 1):
        x1 = max(0, cx - r)
        x2 = min(img_w - 1, cx + r)
        y1 = max(0, cy - r)
        y2 = min(img_h - 1, cy + r)
        for iy in range(y1, y2 + 1):
            for ix in range(x1, x2 + 1):
                if valid_mask is not None and not bool(valid_mask[iy, ix]):
                    continue
                depth_ndc = float(depth_buf[iy, ix])
                if depth_ndc >= 0.9999:
                    continue
                world = _pixel_depth_to_world(
                    px=float(ix),
                    py=float(iy),
                    depth_buf=depth_buf,
                    proj_matrix=proj_matrix,
                    view_matrix=view_matrix,
                    img_w=img_w,
                    img_h=img_h,
                )
                if world is None:
                    continue
                if best_depth is None or depth_ndc < best_depth:
                    best_depth = depth_ndc
                    best = world
        if best is not None:
            return best
    return None


def _crop_object_rgb(
    rgb: np.ndarray,
    depth: np.ndarray,
    bbox_pixel: list[int],
    pad: int = 8,
) -> tuple[np.ndarray, list[int]]:
    """Return padded crop image and crop bounds for a bbox."""
    h, w = rgb.shape[:2]
    x1 = max(0, bbox_pixel[0] - pad)
    y1 = max(0, bbox_pixel[1] - pad)
    x2 = min(w, bbox_pixel[2] + pad)
    y2 = min(h, bbox_pixel[3] + pad)
    if x2 <= x1 or y2 <= y1:
        return rgb, [0, 0, w, h]
    return rgb[y1:y2, x1:x2].copy(), [x1, y1, x2, y2]


def _binary_dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Simple binary dilation without external dependencies."""
    out = np.asarray(mask, dtype=bool)
    if radius <= 0 or not bool(np.any(out)):
        return out
    for _ in range(int(radius)):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        out = (
            padded[:-2, :-2] | padded[:-2, 1:-1] | padded[:-2, 2:] |
            padded[1:-1, :-2] | padded[1:-1, 1:-1] | padded[1:-1, 2:] |
            padded[2:, :-2] | padded[2:, 1:-1] | padded[2:, 2:]
        )
    return out


def _mask_bbox_xyxy(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if xs.size == 0 or ys.size == 0:
        return None
    # xyxy (x2, y2 are exclusive)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _crop_object_rgb_by_mask(
    rgb: np.ndarray,
    object_mask: np.ndarray,
    pad: int = 8,
    mask_only: bool = True,
) -> tuple[np.ndarray, list[int], np.ndarray]:
    """
    Build a tight crop around object mask.
    If mask_only=True, keep target object pixels and zero background.
    """
    h, w = rgb.shape[:2]
    mask = np.asarray(object_mask, dtype=bool)
    bbox = _mask_bbox_xyxy(mask)
    if bbox is None:
        full_bounds = [0, 0, w, h]
        full_mask = np.ones((h, w), dtype=bool)
        return rgb.copy(), full_bounds, full_mask

    x1 = max(0, int(bbox[0]) - int(pad))
    y1 = max(0, int(bbox[1]) - int(pad))
    x2 = min(w, int(bbox[2]) + int(pad))
    y2 = min(h, int(bbox[3]) + int(pad))
    if x2 <= x1 or y2 <= y1:
        full_bounds = [0, 0, w, h]
        full_mask = np.ones((h, w), dtype=bool)
        return rgb.copy(), full_bounds, full_mask

    crop_img = rgb[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]
    if mask_only:
        crop_img[~crop_mask] = 0
    return crop_img, [x1, y1, x2, y2], crop_mask


def _save_binary_mask_png(mask: np.ndarray, out_path: str) -> tuple[bool, str | None]:
    try:
        from PIL import Image as _PILImage
        mask_u8 = (np.asarray(mask, dtype=np.uint8) * 255)
        _PILImage.fromarray(mask_u8, mode="L").save(out_path)
        return True, None
    except Exception as exc:
        return False, str(exc)


def _save_r1_view_debug_pngs(
    rgb: np.ndarray,
    depth_buf: np.ndarray,
    seg_obj_id: np.ndarray,
    out_dir: str,
) -> tuple[bool, list[str], str | None]:
    """Save the exact R1 camera view as RGB, depth, and segmentation PNGs."""
    try:
        from PIL import Image as _PILImage

        os.makedirs(out_dir, exist_ok=True)
        saved_paths = []

        rgb_path = os.path.join(out_dir, "r1_view_rgb.png")
        _PILImage.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(rgb_path)
        saved_paths.append(rgb_path)

        depth = np.asarray(depth_buf, dtype=np.float32)
        depth_vis = np.clip((1.0 - depth) * 255.0, 0.0, 255.0).astype(np.uint8)
        depth_path = os.path.join(out_dir, "r1_view_depth.png")
        _PILImage.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(rgb_path, format="PNG", compress_level=1  # 기본 6 → 1로 낮춤 (용량 크지만 빠르고 선명))
        )
        saved_paths.append(depth_path)

        ids = np.asarray(seg_obj_id, dtype=np.int32)
        seg_rgb = np.zeros((*ids.shape, 3), dtype=np.uint8)
        seg_rgb[:, :, 0] = ((ids * 37) % 255).astype(np.uint8)
        seg_rgb[:, :, 1] = ((ids * 67) % 255).astype(np.uint8)
        seg_rgb[:, :, 2] = ((ids * 97) % 255).astype(np.uint8)
        seg_rgb[ids <= 0] = 0
        seg_path = os.path.join(out_dir, "r1_view_seg.png")
        _PILImage.fromarray(seg_rgb, mode="RGB").save(seg_path)
        saved_paths.append(seg_path)

        return True, saved_paths, None
    except Exception as exc:
        return False, [], str(exc)


def _project_body_to_pixel(
    body_id: int,
    proj_matrix: tuple,
    view_matrix: tuple,
    img_w: int,
    img_h: int,
) -> list[int] | None:
    """PyBullet body??world pos瑜?移대찓???쎌? 醫뚰몴濡??ъ쁺."""
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


def _normalize_angle_rad(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _estimate_topdown_grasp_orientation(body_id: int) -> tuple[list[float], dict]:
    """
    Estimate top-down grasp orientation from current PyBullet object state.

    Before grasping, this computes object in-plane yaw (or long/short axis cue)
    and returns a gripper orientation aligned to object width.
    """
    aabb_min, aabb_max = p.getAABB(body_id)
    dx = float(aabb_max[0] - aabb_min[0])
    dy = float(aabb_max[1] - aabb_min[1])
    _, body_orn = p.getBasePositionAndOrientation(body_id)
    pose_yaw = float(p.getEulerFromQuaternion(body_orn)[2])

    # Panda hand-frame mapping:
    # - yaw=0   -> finger opening axis aligns with +Y
    # - yaw=90  -> finger opening axis aligns with +X
    # Therefore, to keep finger opening perpendicular to AABB long axis:
    # - x_long -> yaw=0
    # - y_long -> yaw=90deg
    if dx > dy * 1.2:
        grasp_yaw = 0.0
        axis_mode = "x_long"
    elif dy > dx * 1.2:
        grasp_yaw = np.pi / 2.0
        axis_mode = "y_long"
    else:
        # Near-square: no strong AABB long-axis cue, keep pose yaw.
        grasp_yaw = pose_yaw
        axis_mode = "near_square"

    grasp_yaw = _normalize_angle_rad(grasp_yaw)
    grasp_orientation = list(p.getQuaternionFromEuler([np.pi, 0.0, grasp_yaw]))
    return grasp_orientation, {
        "dx": dx,
        "dy": dy,
        "pose_yaw_deg": float(np.degrees(pose_yaw)),
        "grasp_yaw_deg": float(np.degrees(grasp_yaw)),
        "axis_mode": axis_mode,
    }


def _offset_orientation_yaw(orientation: list[float], yaw_offset_deg: float) -> list[float]:
    roll, pitch, yaw = p.getEulerFromQuaternion(orientation)
    yaw_next = _normalize_angle_rad(float(yaw) + float(np.radians(yaw_offset_deg)))
    return list(p.getQuaternionFromEuler([float(roll), float(pitch), yaw_next]))


def _grasp_with_orientation_sweep(
    controller,
    body_id: int,
    object_label: str,
    seed_world_pos: list[float],
    seed_orientation: list[float],
    axis_mode: str,
    hold_companion=None,
) -> bool:
    """
    Try grasp with yaw sweep if initial grasp fails.

    Stage-wise offsets broaden angle ranges progressively:
    1) narrow around estimated yaw
    2) medium offsets
    3) wide offsets (including opposite direction)
    """
    if axis_mode == "near_square":
        yaw_offset_stages = [
            [0.0, 45.0, -45.0, 90.0, -90.0],
            [22.5, -22.5, 67.5, -67.5, 135.0, -135.0],
            [180.0],
        ]
    else:
        yaw_offset_stages = [
            [0.0, 12.0, -12.0, 24.0, -24.0],
            [36.0, -36.0, 50.0, -50.0, 70.0, -70.0],
            [90.0, -90.0, 135.0, -135.0, 180.0],
        ]

    attempt_idx = 0
    for stage_idx, stage_offsets in enumerate(yaw_offset_stages):
        for yaw_offset_deg in stage_offsets:
            attempt_idx += 1
            try_orientation = _offset_orientation_yaw(seed_orientation, yaw_offset_deg)
            print(
                f"[Grasp-Retry] {object_label} attempt={attempt_idx} "
                f"stage={stage_idx + 1} yaw_offset={yaw_offset_deg:+.1f} deg"
            )
            ok = controller.grasp_body(
                body_id=body_id,
                object_label=object_label,
                r1_world_pos=seed_world_pos,
                orientation=try_orientation,
                hold_companion=hold_companion,
            )
            if ok:
                print(
                    f"[Grasp-Retry] {object_label} success "
                    f"(stage={stage_idx + 1}, yaw_offset={yaw_offset_deg:+.1f} deg)"
                )
                return True
            try:
                controller.release_grasp(open_after=True, steps=60)
            except Exception:
                pass

    return False


def _compute_aabb_grasp_pose(body_id: int) -> tuple[list[float], list[float], dict]:
    """Fallback grasp pose derived from PyBullet AABB.

    Returns (world_pos, orientation, info_dict) where world_pos is the AABB
    center (xy) at AABB center z, and orientation is a top-down gripper
    aligned to the object's long/short axis (yaw).

    NOTE: world_pos.z is the grasp *contact* point (object center height).
    grasp_body() in robot_controller adds the EE finger-tip offset on top of
    this to compute the actual EE descend target.  Do NOT clamp cz upward
    toward aabb_max[2] here: that shifts the EE target above the object by
    roughly (finger_offset + half_object_height), causing contact=0.
    """
    aabb_min, aabb_max = p.getAABB(body_id)
    cx = (float(aabb_min[0]) + float(aabb_max[0])) / 2.0
    cy = (float(aabb_min[1]) + float(aabb_max[1])) / 2.0
    cz_center = (float(aabb_min[2]) + float(aabb_max[2])) / 2.0
    cz_top = float(aabb_max[2])
    # Use AABB center z as the grasp contact point.  grasp_body() adds the
    # EE finger offset, so this correctly lands the fingers on the object body.
    # The old `max(cz_center, cz_top - 0.02)` clamp placed the EE ~4-5 cm
    # above the object top (nominal z >> aabb_max), giving contact=0.
    cz = cz_center
    world_pos = [cx, cy, cz]
    orientation, info = _estimate_topdown_grasp_orientation(body_id)
    info = dict(info)
    info["aabb_min"] = [float(v) for v in aabb_min]
    info["aabb_max"] = [float(v) for v in aabb_max]
    info["world_pos"] = world_pos
    return world_pos, orientation, info


def _grasp_with_r1_then_aabb_fallback(
    controller,
    body_id: int,
    object_label: str,
    r1_hint: dict | None,
    hold_companion=None,
) -> tuple[bool, str]:
    """Try R1 grasp pose first; on failure, retry with AABB-based pose.

    Returns (success, source) where source is one of
    {"r1", "aabb_fallback", "none"}.
    """
    r1_world_pos: list[float] | None = None
    r1_orientation: list[float] | None = None
    if isinstance(r1_hint, dict):
        wp = r1_hint.get("world_pos")
        orn = r1_hint.get("orientation")
        if (
            isinstance(wp, (list, tuple))
            and len(wp) >= 3
            and isinstance(orn, (list, tuple))
            and len(orn) >= 4
        ):
            r1_world_pos = [float(v) for v in wp[:3]]
            r1_orientation = [float(v) for v in orn[:4]]

    if r1_world_pos is not None and r1_orientation is not None:
        print(
            f"[Grasp] {object_label}: trying R1 pose "
            f"world_pos={[round(v, 4) for v in r1_world_pos]}, "
            f"orn={[round(v, 4) for v in r1_orientation]}"
        )
        # Limit R1 attempts to 3 small yaw offsets, then bail out to AABB
        # fallback. R1 coords often miss by a small offset that yaw alone
        # cannot recover, so wide yaw sweeps just waste time.
        r1_yaw_offsets_deg = [0.0, 12.0, -12.0]
        r1_success = False
        for r1_attempt_idx, yaw_offset_deg in enumerate(r1_yaw_offsets_deg, start=1):
            try_orientation = _offset_orientation_yaw(r1_orientation, yaw_offset_deg)
            print(
                f"[Grasp-R1] {object_label} attempt={r1_attempt_idx}/"
                f"{len(r1_yaw_offsets_deg)} yaw_offset={yaw_offset_deg:+.1f} deg"
            )
            try:
                ok_attempt = controller.grasp_body(
                    body_id=body_id,
                    object_label=object_label,
                    r1_world_pos=r1_world_pos,
                    orientation=try_orientation,
                    hold_companion=hold_companion,
                )
            except Exception as exc:
                print(f"[Grasp-R1][WARN] attempt failed with exception: {exc}")
                ok_attempt = False
            if ok_attempt:
                print(
                    f"[Grasp-R1] {object_label} success "
                    f"(attempt={r1_attempt_idx}, yaw_offset={yaw_offset_deg:+.1f} deg)"
                )
                r1_success = True
                break
            try:
                controller.release_grasp(open_after=True, steps=60)
            except Exception:
                pass
        if r1_success:
            return True, "r1"
        print(
            f"[Grasp] {object_label}: R1 pose failed after "
            f"{len(r1_yaw_offsets_deg)} attempts; falling back to AABB-based pose."
        )
    else:
        print(
            f"[Grasp] {object_label}: R1 hint unavailable; "
            "using AABB-based pose directly."
        )

    # Make sure gripper is fully open before retrying.
    try:
        controller.release_grasp(open_after=True, steps=60)
    except Exception:
        pass

    aabb_world_pos, aabb_orientation, aabb_info = _compute_aabb_grasp_pose(body_id)
    print(
        f"[Grasp-Fallback] {object_label}: AABB pose "
        f"world_pos={[round(v, 4) for v in aabb_world_pos]}, "
        f"axis_mode={aabb_info.get('axis_mode')}, "
        f"yaw={float(aabb_info.get('grasp_yaw_deg', 0.0)):.1f} deg"
    )
    ok = _grasp_with_orientation_sweep(
        controller=controller,
        body_id=body_id,
        object_label=object_label,
        seed_world_pos=aabb_world_pos,
        seed_orientation=aabb_orientation,
        axis_mode=str(aabb_info.get("axis_mode", "near_square")),
        hold_companion=hold_companion,
    )
    if ok:
        return True, "aabb_fallback"
    return False, "none"


def run_optional_affordance_probe(
    enable_affordance_r1: bool,
    enable_sam2_refinement: bool = False,
    controllers: dict | None = None,
    ycb_object_ids: dict | None = None,
    target_labels: list[str] | None = None,
) -> dict[str, dict]:
    """
    Run R1 per target object and inject 3D grasp hints into controllers.

    Returns:
        { label: { "world_pos": [x,y,z], "orientation": quaternion, "bbox_pixel": [...] } }
    """
    results: dict[str, dict] = {}

    if not enable_affordance_r1:
        print("[R1] optional affordance probe disabled (set ENABLE_AFFORDANCE_R1=1 to enable)")
        return results

    print("[R1] optional affordance probe enabled")
    print(
        "[R1] safeguards enabled: "
        "target-label validation, object-mask depth sampling, sim-pose outlier correction"
    )

    try:
        from affordancegrasp_r1_adapter import AffordanceGraspR1Adapter
    except Exception as exc:
        print(f"[R1][WARN] adapter import failed: {exc}")
        return results

    try:
        # Step 1. Camera setup (RGB + depth + segmentation).
        IMG_W = AFFORDANCE_CAPTURE_WIDTH
        IMG_H = AFFORDANCE_CAPTURE_HEIGHT
        view_matrix = p.computeViewMatrix(
        cameraEyePosition=[1.92, -0.35, 2.08],    # 카메라 본체의 위치
        cameraTargetPosition=[0.28, 0.02, 0.76], # 카메라가 바라보는 지점
        cameraUpVector=[0, 0, 1]                 # 하늘 방향 (Z축 위쪽)
    )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=IMG_W / IMG_H,
            nearVal=0.01,
            farVal=5.0,
        )
        _, _, rgba_raw, depth_raw, seg_raw = p.getCameraImage(
            width=IMG_W,
            height=IMG_H,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER,
        )
        rgb_full = np.asarray(rgba_raw, dtype=np.uint8).reshape(IMG_H, IMG_W, 4)[:, :, :3]
        depth_buf = np.asarray(depth_raw, dtype=np.float32).reshape(IMG_H, IMG_W)
        seg_buf = np.asarray(seg_raw, dtype=np.int32).reshape(IMG_H, IMG_W)
        seg_obj_id = np.bitwise_and(seg_buf, (1 << 24) - 1)

        # Step 2. Load R1 adapter with selectable runtime device.
        # Do not force CPU here: keep model inputs on the same device as the model.
        import os as _os
        def _safe_env_int(env_name: str, default_value: int) -> int:
            raw = _os.getenv(env_name)
            if raw is None:
                return int(default_value)
            try:
                return int(raw)
            except Exception:
                return int(default_value)

        save_mask_png = env_flag("SAVE_R1_MASK_PNG", default=True)
        save_view_png = env_flag("SAVE_R1_VIEW_PNG", default=True)
        mask_debug_dir = _os.getenv("R1_MASK_DEBUG_DIR") or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "r1_mask_debug"
        )
        mask_crop_pad_px = max(0, _safe_env_int("R1_MASK_CROP_PAD_PX", 8))
        thin_mask_dilate_px = max(0, _safe_env_int("R1_THIN_MASK_DILATION_PX", 2))
        default_mask_dilate_px = max(0, _safe_env_int("R1_MASK_DILATION_PX", 0))
        thin_object_tokens = {
            "knife",
            "largemarker",
            "fork",
            "spoon",
            "phillipsscrewdriver",
            "flatscrewdriver",
        }
        if save_mask_png or save_view_png:
            try:
                os.makedirs(mask_debug_dir, exist_ok=True)
                print(f"[R1] debug PNGs: {mask_debug_dir}")
            except Exception as _mk_exc:
                print(f"[R1][WARN] could not create mask debug dir ({_mk_exc})")
                save_mask_png = False
                save_view_png = False

        if save_view_png:
            ok_view, view_paths, view_err = _save_r1_view_debug_pngs(
                rgb=rgb_full,
                depth_buf=depth_buf,
                seg_obj_id=seg_obj_id,
                out_dir=mask_debug_dir,
            )
            if ok_view:
                print(f"[R1] view PNGs saved: {view_paths}")
            elif view_err:
                print(f"[R1][WARN] view PNG save failed: {view_err}")

        device_env = (_os.getenv("AFFORDANCE_R1_DEVICE") or "cuda").strip().lower()
        if device_env in {"", "cuda", "gpu"}:
            adapter_device = "cuda"
        elif device_env in {"auto", "default"}:
            adapter_device = None
        else:
            adapter_device = device_env
        print(f"[R1] adapter device request: {device_env or 'cuda'}")

        adapter = AffordanceGraspR1Adapter(
            model_id=AFFORDANCE_MODEL_ID,
            device=adapter_device,
            local_model_dir=_os.getenv("AFFORDANCE_R1_LOCAL_DIR"),
            local_files_only=env_flag("AFFORDANCE_R1_LOCAL_ONLY", default=False),
        )
        adapter.load()
        if not adapter.is_available():
            print(f"[R1][WARN] model not available: {adapter._last_error}")
            return results
        print(f"[R1] adapter runtime device: {adapter.device}")

        # Step 3. Per-object R1 inference.
        labels_to_probe = target_labels or (list(ycb_object_ids.keys()) if ycb_object_ids else [])
        if not labels_to_probe:
            print("[R1][WARN] no target labels to probe.")
            return results

        for label in labels_to_probe:
            body_id = (ycb_object_ids or {}).get(label)
            if body_id is None:
                print(f"[R1][WARN] label '{label}' not in ycb_object_ids, skipping.")
                continue
            label_norm = _normalize_label_token(label)
            file_label = label_norm or re.sub(r"[^a-zA-Z0-9_-]+", "_", str(label))
            object_mask_raw = (seg_obj_id == int(body_id))
            if not bool(np.any(object_mask_raw)):
                object_mask = None
                dilation_px = 0
            else:
                dilation_px = thin_mask_dilate_px if label_norm in thin_object_tokens else default_mask_dilate_px
                if dilation_px > 0:
                    object_mask = _binary_dilate(object_mask_raw, radius=dilation_px)
                else:
                    object_mask = object_mask_raw

            print(f"[R1] probing '{label}' (body_id={body_id})...")

            # 3-a. Build object-only mask, then create tight crop for R1.
            if object_mask is not None and bool(np.any(object_mask)):
                crop_img, crop_bounds, crop_mask = _crop_object_rgb_by_mask(
                    rgb=rgb_full,
                    object_mask=object_mask,
                    pad=mask_crop_pad_px,
                    mask_only=True,
                )
                crop_bbox = list(crop_bounds)
                print(
                    f"[R1] '{label}' mask pixels={int(np.count_nonzero(object_mask))}, "
                    f"dilation_px={int(dilation_px)}, crop_bounds={crop_bounds}"
                )
                if save_mask_png:
                    raw_path = os.path.join(mask_debug_dir, f"{file_label}_mask_raw.png")
                    ok_raw, err_raw = _save_binary_mask_png(object_mask_raw, raw_path)
                    if ok_raw:
                        print(f"[R1] '{label}' mask saved: {raw_path}")
                    elif err_raw:
                        print(f"[R1][WARN] '{label}' raw mask save failed: {err_raw}")

                    if dilation_px > 0:
                        dil_path = os.path.join(mask_debug_dir, f"{file_label}_mask_dilated.png")
                        ok_dil, err_dil = _save_binary_mask_png(object_mask, dil_path)
                        if ok_dil:
                            print(f"[R1] '{label}' dilated mask saved: {dil_path}")
                        elif err_dil:
                            print(f"[R1][WARN] '{label}' dilated mask save failed: {err_dil}")

                    crop_path = os.path.join(mask_debug_dir, f"{file_label}_mask_crop.png")
                    ok_crop, err_crop = _save_binary_mask_png(crop_mask, crop_path)
                    if ok_crop:
                        print(f"[R1] '{label}' crop mask saved: {crop_path}")
                    elif err_crop:
                        print(f"[R1][WARN] '{label}' crop mask save failed: {err_crop}")
            else:
                # Fallback: if segmentation mask is missing, keep previous bbox-based crop.
                center_px = _project_body_to_pixel(
                    body_id, proj_matrix, view_matrix, IMG_W, IMG_H
                )
                if center_px is not None:
                    try:
                        aabb_min, aabb_max = p.getAABB(body_id)
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
                    crop_img, crop_bounds = _crop_object_rgb(rgb_full, depth_buf, crop_bbox)
                else:
                    crop_img = rgb_full
                    crop_bbox = [0, 0, IMG_W, IMG_H]
                    crop_bounds = [0, 0, IMG_W, IMG_H]
                print(f"[R1][WARN] '{label}' object mask unavailable; using bbox fallback crop.")

            # 3-b. R1 inference with explicit target label in prompt.
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

            # Always log raw_output so per-object behavior is comparable across runs.
            _raw_out = str(r1_result.get("raw_output") or "").replace("\n", " ").strip()
            if len(_raw_out) > 240:
                _raw_out = _raw_out[:240] + "...<truncated>"
            print(f"[R1] '{label}' raw_output: {_raw_out}")

            candidates = r1_result.get("grasp_candidates") or []
            if not candidates:
                print(f"[R1][INFO] '{label}' no candidate parsed from output: {r1_result.get('raw_output')}")
                continue

            target_norm = label_norm
            # Generic / stop-word / sub-part labels that should NOT be treated
            # as a mismatched object. R1 frequently emits either demonstratives
            # ("this is the mug" -> "this") or semantic sub-parts of the target
            # ("handle" for a knife, "rim" for a mug). Both are valid grasp
            # candidates, not wrong-object detections, so accept them as
            # generic when the target label is not echoed verbatim.
            _GENERIC_PART_TOKENS = {
                # filler / parser noise
                "", "object", "part", "region", "area", "item", "thing",
                "this", "that", "these", "those", "it", "its",
                "they", "them", "their",
                "a", "an", "the", "some", "any",
                "is", "are", "was", "were", "be", "been", "being",
                "of", "for", "with", "and", "or", "to", "in", "on", "at",
                "robot", "gripper", "grasp",
                # generic graspable sub-parts of common YCB objects
                "handle", "grip", "shaft", "stem", "neck", "knob",
                "body", "side", "top", "bottom", "base", "back", "front",
                "edge", "rim", "lip", "mouth", "opening", "head",
                "tip", "end", "tail", "center", "middle", "surface",
                "blade", "prong", "tine",
            }
            matched = []
            generic = []
            mismatched_parts = []
            for cand in candidates:
                part_raw = str(cand.get("part", "")).strip().lower()
                part_norm = _normalize_label_token(part_raw)
                if part_norm in _GENERIC_PART_TOKENS:
                    generic.append(cand)
                    continue
                if part_norm == target_norm:
                    matched.append(cand)
                else:
                    mismatched_parts.append(part_raw)

            if matched:
                top = matched[0]
            elif generic:
                top = generic[0]
                if mismatched_parts:
                    print(
                        f"[R1][WARN] '{label}' candidate labels mismatched {mismatched_parts}; "
                        "using generic candidate after validation."
                    )
            else:
                print(
                    f"[R1][WARN] '{label}' candidates rejected by target-label validation: "
                    f"{mismatched_parts}"
                )
                continue
            print(f"[R1] '{label}' top candidate: {top}")
            top_part_raw = str(top.get("part", "")).strip().lower()
            top_part_norm = _normalize_label_token(top_part_raw)

            # 3-c. Map crop pixel coordinates back to full-image pixels.
            cx_crop, cy_crop = top["center_pixel"]
            # IMPORTANT: use the actual padded crop bounds for reverse mapping.
            crop_w = crop_bounds[2] - crop_bounds[0]
            crop_h = crop_bounds[3] - crop_bounds[1]
            if crop_w > 0 and crop_h > 0:
                cx_full = crop_bounds[0] + cx_crop * (crop_w / max(crop_img.shape[1], 1))
                cy_full = crop_bounds[1] + cy_crop * (crop_h / max(crop_img.shape[0], 1))
            else:
                cx_full, cy_full = cx_crop, cy_crop

            # 3-d. Convert 2D pixels to world 3D points.
            # Use 3x3 bbox samples first; fallback to center search if needed.
            _sample_pts = []
            _bpx = top["bbox_pixel"]  # crop-space bbox
            for _sy in [0.25, 0.5, 0.75]:
                for _sx in [0.25, 0.5, 0.75]:
                    _scx = _bpx[0] + (_bpx[2] - _bpx[0]) * _sx
                    _scy = _bpx[1] + (_bpx[3] - _bpx[1]) * _sy
                    # crop -> full-image coordinate mapping
                    if crop_w > 0 and crop_h > 0:
                        _fx = crop_bounds[0] + _scx * (crop_w / max(crop_img.shape[1], 1))
                        _fy = crop_bounds[1] + _scy * (crop_h / max(crop_img.shape[0], 1))
                    else:
                        _fx, _fy = _scx, _scy
                    _wp = _pixel_depth_to_world_with_local_search(
                        px=_fx, py=_fy,
                        depth_buf=depth_buf,
                        proj_matrix=proj_matrix,
                        view_matrix=view_matrix,
                        img_w=IMG_W,
                        img_h=IMG_H,
                        max_radius=8,
                        valid_mask=object_mask,
                    )
                    if _wp is not None:
                        _sample_pts.append(_wp)

            if _sample_pts:
                # Use median of valid samples for robustness.
                world_pos = np.median(np.array(_sample_pts), axis=0)
            else:
                # If all samples fail, retry around center pixel.
                world_pos = _pixel_depth_to_world_with_local_search(
                    px=float(cx_full), py=float(cy_full),
                    depth_buf=depth_buf,
                    proj_matrix=proj_matrix,
                    view_matrix=view_matrix,
                    img_w=IMG_W,
                    img_h=IMG_H,
                    max_radius=20,
                    valid_mask=object_mask,
                )
                if world_pos is None and object_mask is not None:
                    # Last-resort: pick any valid-depth pixel that lies inside the
                    # object segmentation mask. Prevents silently dropping the
                    # whole object hint just because the R1 bbox barely missed
                    # the mask. Pose validation downstream still corrects outliers.
                    try:
                        depth_valid = depth_buf < 0.999
                        candidate_mask = np.logical_and(object_mask, depth_valid)
                        if bool(np.any(candidate_mask)):
                            ys, xs = np.where(candidate_mask)
                            cy_mask = float(np.mean(ys))
                            cx_mask = float(np.mean(xs))
                            world_pos = _pixel_depth_to_world_with_local_search(
                                px=cx_mask, py=cy_mask,
                                depth_buf=depth_buf,
                                proj_matrix=proj_matrix,
                                view_matrix=view_matrix,
                                img_w=IMG_W,
                                img_h=IMG_H,
                                max_radius=4,
                                valid_mask=object_mask,
                            )
                            if world_pos is not None:
                                print(
                                    f"[R1][INFO] '{label}' world_pos recovered from object-mask "
                                    f"centroid pixel=({cx_mask:.1f}, {cy_mask:.1f})"
                                )
                    except Exception as _mc_exc:
                        print(f"[R1][WARN] '{label}' mask-centroid fallback failed: {_mc_exc}")
                if world_pos is None:
                    print(
                        f"[R1][WARN] '{label}' could not compute world_pos from R1 pixel/depth, skipping "
                        "(PyBullet pose fallback disabled)."
                    )
                    continue
            actual_pos, _ = p.getBasePositionAndOrientation(body_id)
            aabb_min, aabb_max = p.getAABB(body_id)
            aabb_center = np.array(
                [
                    (float(aabb_min[0]) + float(aabb_max[0])) / 2.0,
                    (float(aabb_min[1]) + float(aabb_max[1])) / 2.0,
                    (float(aabb_min[2]) + float(aabb_max[2])) / 2.0,
                ],
                dtype=float,
            )
            world_pos = np.array(world_pos, dtype=float)
            dist_to_sim = float(np.linalg.norm(world_pos - np.array(actual_pos, dtype=float)))
            in_expanded_aabb = (
                (float(aabb_min[0]) - 0.08) <= float(world_pos[0]) <= (float(aabb_max[0]) + 0.08)
                and (float(aabb_min[1]) - 0.08) <= float(world_pos[1]) <= (float(aabb_max[1]) + 0.08)
                and (float(aabb_min[2]) - 0.05) <= float(world_pos[2]) <= (float(aabb_max[2]) + 0.12)
            )

            if top_part_norm not in _GENERIC_PART_TOKENS and top_part_norm != target_norm and dist_to_sim > 0.25:
                print(
                    f"[R1][WARN] '{label}' candidate part '{top_part_raw}' mismatches target and "
                    f"is far from sim object (dist={dist_to_sim:.3f}m). Skipping."
                )
                continue

            if (not in_expanded_aabb) or dist_to_sim > 0.40:
                corrected = aabb_center.copy()
                corrected[2] = max(float(aabb_center[2]), float(aabb_min[2]) + 0.01)
                print(
                    f"[R1][WARN] '{label}' world_pos outlier detected "
                    f"(dist={dist_to_sim:.3f}m, in_aabb={in_expanded_aabb}). "
                    f"Correcting to sim-aabb center {corrected.tolist()}."
                )
                world_pos = corrected
                dist_to_sim = float(np.linalg.norm(world_pos - np.array(actual_pos, dtype=float)))

            grasp_yaw = 0.0
            for _yaw_key in ("grasp_yaw_deg", "yaw_deg", "angle_deg", "theta_deg", "yaw", "angle", "theta"):
                if _yaw_key not in top:
                    continue
                try:
                    _raw_yaw = float(top[_yaw_key])
                except Exception:
                    continue
                if _yaw_key.endswith("_deg"):
                    grasp_yaw = float(np.radians(_raw_yaw))
                else:
                    # Heuristic: small magnitude values are likely radians.
                    if abs(_raw_yaw) <= (2.0 * np.pi + 1e-6):
                        grasp_yaw = _raw_yaw
                    else:
                        grasp_yaw = float(np.radians(_raw_yaw))
                break
            grasp_orientation = p.getQuaternionFromEuler([np.pi, 0.0, grasp_yaw])

            print(
                f"[R1] '{label}' world_pos={[round(v, 4) for v in world_pos.tolist()]}, "
                f"sim_pos={[round(float(v), 4) for v in actual_pos]}, "
                f"sim_aabb_center={[round(float(v), 4) for v in aabb_center.tolist()]}, "
                f"delta={dist_to_sim:.4f}m, "
                f"yaw={np.degrees(grasp_yaw):.1f} deg"
            )

            hint_payload = {
                "world_pos": world_pos.tolist(),
                "orientation": list(grasp_orientation),
                "bbox_pixel": top.get("bbox_pixel", []),
                "part": top.get("part", ""),
                "score": float(top.get("score", 1.0)),
            }
            results[label] = hint_payload

            # SAM2 refinement (?좏깮)
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

            # 3-f. controller??hint 二쇱엯
            if isinstance(controllers, dict):
                for arm_name, controller in controllers.items():
                    # ??label???대떦 arm???寃잛씤吏??run_sequential_demo?먯꽌 寃곗젙
                    # ?ш린?쒕뒗 ?꾩껜 inference_result??world_pos ?ы븿?댁꽌 ???
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
    r1_hints: dict | None = None,   # ??run_optional_affordance_probe 寃곌낵
) -> None:
    left = controllers["left"]
    right = controllers["right"]
    down_orn = p.getQuaternionFromEuler([np.pi, 0.0, 0.0])
    r1_hints = r1_hints or {}
    print(f"[R1] hint labels available: {sorted(list(r1_hints.keys()))}")
 
    CAM_CONFIG = {
    "cam_target":   [1.0, 0.0, 0.6],
    "cam_distance": 1.2,
    "cam_yaw":      45,
    "cam_pitch":    -45,
    }
 
    print("\n[3] 媛??移대찓???뚮뜑留?以?..")
    rgb, depth, proj_matrix, view_matrix = render_camera(**CAM_CONFIG)
 
    try:
        from PIL import Image as PILImage
        PILImage.fromarray(rgb).save("scene_capture.png")
        print("[3] scene_capture.png ????꾨즺")
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
 
    # ?? module3_output.json?먯꽌 grasp ???寃곗젙 ????????????????????????????
    # 臾쇱껜 y 醫뚰몴 湲곗??쇰줈 ???좊떦:
    #   left  base y=-0.35 ??spawn y媛 ???묒?(?뚯닔??媛源뚯슫) 臾쇱껜 ?대떦
    #   right base y=+0.35 ??spawn y媛 ?????묒닔??媛源뚯슫) 臾쇱껜 ?대떦
    # ?대젃寃??섎㈃ ?붿씠 ?쒕줈 援먯감?섏? ?딆쓬.
    _m3_labels = _load_module3_object_labels(_MODULE3_JSON_PATH)
    _loaded_labels = list(ycb_object_ids.keys())
 
    # YCB_OBJECT_SPECS?먯꽌 label?뭩pawn_y 留ㅽ븨
    _label_to_y = {lbl: pos[1] for lbl, _, pos in YCB_OBJECT_SPECS}
 
    # JSON 臾쇱껜 以??ㅼ젣 濡쒕뱶??寃껊쭔 ?꾨낫
    _candidates = [lbl for lbl in _m3_labels if lbl in ycb_object_ids]
    if not _candidates:
        _candidates = _loaded_labels[:2]
 
    if len(_candidates) >= 2:
        # y 湲곗? ?뺣젹: ?묒? y ??left, ??y ??right
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
 
    print(f"[Demo] grasp targets ??left: '{left_target_label}', right: '{right_target_label}'")
 
    # 議곕┰쨌諛곗튂 ?꾩튂 (???붿씠 ?쒕줈 ?ㅻⅨ Y 諛⑺뼢?먯꽌 ?묎렐)
    ASSEMBLY_POS_LEFT  = [0.62, -0.10, 0.97]
    ASSEMBLY_POS_RIGHT = [0.62,  0.10, 1.13]
    PLACE_POS          = [0.62,  0.00, 0.85]

    # Parse module3 assembly plan once so we can reuse JSON joint/assembly pose
    # for assembly movement (not for grasping).
    assembly_manager = AssemblyManager()
    _plan_path = _MODULE3_JSON_PATH
    _plan_loaded = False
    _transport_targets = {}
    if os.path.isfile(_plan_path):
        try:
            assembly_manager.load_plan_from_json(_plan_path)
            _m3_all_labels = _load_module3_object_labels(_plan_path)
            _body_map = {lbl: ycb_object_ids[lbl] for lbl in _m3_all_labels if lbl in ycb_object_ids}
            if not _body_map:
                _body_map = {left_target_label: left_body_id, right_target_label: right_body_id}
            assembly_manager.register_bodies(_body_map)
            _transport_targets = assembly_manager.get_object_transport_targets()
            _plan_loaded = True
            print(f"[Demo] assembly transport targets: {_transport_targets}")
        except Exception as exc:
            print(f"[Demo][WARN] module3 plan load failed ({exc}), using fallback attach.")
    else:
        print(f"[Demo][WARN] module3_output.json not found at {_plan_path}, using fallback attach.")

    _left_transport_target = _transport_targets.get(left_target_label, {})
    _right_transport_target = _transport_targets.get(right_target_label, {})
    left_assembly_pos = _left_transport_target.get("position", ASSEMBLY_POS_LEFT)
    left_assembly_orn = _left_transport_target.get("orientation", down_orn)
    right_assembly_pos = _right_transport_target.get("position", ASSEMBLY_POS_RIGHT)
    right_assembly_orn = _right_transport_target.get("orientation", down_orn)
 
    # ?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧
    # Step 1-L. Left arm: base_object ?뚯? ???쒖옄由??湲?
    # ?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧
    # ?섏젙 ??
    print(f"[Demo] left-arm grasp target: {left_target_label}")
    _left_hint = r1_hints.get(left_target_label, {})
    left_ok, left_grasp_src = _grasp_with_r1_then_aabb_fallback(
        controller=left,
        body_id=left_body_id,
        object_label=left_target_label,
        r1_hint=_left_hint,
    )
    if not left_ok:
        print(
            f"[Demo][WARN] left-arm grasp failed for '{left_target_label}' "
            "(R1 + AABB fallback both failed)"
        )
    else:
        print(f"[Demo] left-arm grasp succeeded via source={left_grasp_src}")
        left.maintain_grasp_hold(steps=120)
        print("[Demo] left arm holding ??waiting for right arm grasp.")
 
    # ?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧
    # Step 1-R. Right arm: attach_object ?뚯? ???쒖옄由??湲?
    #           left媛 ?ㅺ퀬 ?덈뒗 ?숈븞 hold_companion?쇰줈 蹂댄샇
    # ?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧
    print(f"[Demo] right-arm grasp target: {right_target_label}")
    _right_hint = r1_hints.get(right_target_label, {})
    right_ok, right_grasp_src = _grasp_with_r1_then_aabb_fallback(
        controller=right,
        body_id=right_body_id,
        object_label=right_target_label,
        r1_hint=_right_hint,
        hold_companion=left if left_ok else None,
    )
    if not right_ok:
        print(
            f"[Demo][WARN] right-arm grasp failed for '{right_target_label}' "
            "(R1 + AABB fallback both failed)"
        )
    else:
        print(f"[Demo] right-arm grasp succeeded via source={right_grasp_src}")
        right.maintain_grasp_hold(steps=120)
        print("[Demo] right arm holding ??both arms ready.")
 
    # ?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧
    # Step 2-L. Left arm: 議곕┰ ?꾩튂濡??대룞
    #           right媛 ?ㅺ퀬 ?덈뒗 ?숈븞 hold_companion?쇰줈 蹂댄샇
    # ?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧
    # move_end_effector_to ?대??먯꽌 留??ㅽ뀦 _update_grasp_hold_feedback() +
    # _set_gripper_target()???몄텧?섎?濡?slip 媛먯? ???먮룞?쇰줈 force媛 利앷???
    # EE z compensation: when a top-down gripper holds a body, the body's
    # AABB center sits below the EE by (FINGER_TIP_OFFSET + half body z).
    # If we want the BODY to be at JSON target z, the EE must be that much
    # higher. Without this, the body ends up 5-7cm below the JSON target.
    FINGER_TIP_OFFSET = 0.058

    def _ee_pose_with_grip_z_comp(target_pos: list[float], body_id: int) -> list[float]:
        try:
            aabb_min, aabb_max = p.getAABB(body_id)
            half_z = max(1e-3, float(aabb_max[2] - aabb_min[2]) / 2.0)
        except Exception:
            half_z = 0.02
        return [
            float(target_pos[0]),
            float(target_pos[1]),
            float(target_pos[2]) + half_z + FINGER_TIP_OFFSET,
        ]

    if left_ok:
        left_assembly_ee = _ee_pose_with_grip_z_comp(left_assembly_pos, left_body_id)
        print(
            f"[Demo] left assembly move: object_target={[round(v, 4) for v in left_assembly_pos]}, "
            f"ee_pose={[round(v, 4) for v in left_assembly_ee]} (z-comp for grip)"
        )
        left.move_end_effector_to(
            left_assembly_ee, orientation=left_assembly_orn, steps=600,
            hold_companion=right if right_ok else None,
        )
        left.maintain_grasp_hold(steps=60)
        print("[Demo] left arm at assembly position.")
 
    # ?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧
    # Step 2-R. Right arm: 議곕┰ ?꾩튂濡??대룞
    #           left hold_companion?쇰줈 蹂댄샇
    # ?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧
    if right_ok:
        right_assembly_ee = _ee_pose_with_grip_z_comp(right_assembly_pos, right_body_id)
        print(
            f"[Demo] right assembly move: object_target={[round(v, 4) for v in right_assembly_pos]}, "
            f"ee_pose={[round(v, 4) for v in right_assembly_ee]} (z-comp for grip)"
        )
        right.move_end_effector_to(
            right_assembly_ee, orientation=right_assembly_orn, steps=600,
            hold_companion=left if left_ok else None,
        )
        right.maintain_grasp_hold(steps=80)
        print("[Demo] right arm at assembly position.")
 
    # ?섏젙 ??
    # ?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧
    # Step 3. Assembly ??module3_output.json 怨꾪쉷 湲곕컲 ?ㅽ뻾
    # ?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧
    # assembly_manager/_plan_loaded are prepared before grasp so JSON
    # joint_position/joint_orientation can also be used for assembly movement.
 
    assembly_results = []
    if left_ok and right_ok:
        # ?? attach ????臾쇱껜 媛?嫄곕━ ?뺤씤 諛?nudge ?????????????????????????
        main_pos, main_orn = p.getBasePositionAndOrientation(left_body_id)
        aux_pos,  _        = p.getBasePositionAndOrientation(right_body_id)
        body_dist = float(np.linalg.norm(np.array(aux_pos) - np.array(main_pos)))
        print(f"[Demo] pre-attach body distance: {body_dist:.3f} m")
 
        if body_dist > 0.35:
            print("[Demo] bodies too far apart ??nudging right arm closer...")
            ee_before, _ = right.get_end_effector_pose()
            delta = np.array(main_pos, dtype=float) - np.array(aux_pos, dtype=float)
            delta_xy = np.array([delta[0], delta[1]], dtype=float)
            delta_xy_norm = float(np.linalg.norm(delta_xy))
            if delta_xy_norm > 1e-6:
                max_xy_nudge = 0.18
                scale = min(1.0, max_xy_nudge / delta_xy_norm)
                delta_xy = delta_xy * scale
            delta_z = float(np.clip(delta[2], -0.03, 0.03))
            nudge_pos = [
                float(ee_before[0] + delta_xy[0]),
                float(ee_before[1] + delta_xy[1]),
                float(ee_before[2] + delta_z),
            ]
            right.move_end_effector_to(
                nudge_pos,
                orientation=right_assembly_orn,
                steps=320,
                hold_companion=left,
            )
            right.maintain_grasp_hold(steps=60)
            main_pos, main_orn = p.getBasePositionAndOrientation(left_body_id)
            aux_pos,  _        = p.getBasePositionAndOrientation(right_body_id)
            body_dist_after = float(np.linalg.norm(np.array(aux_pos) - np.array(main_pos)))
            print(f"[Demo] post-nudge body distance: {body_dist_after:.3f} m")
            if body_dist_after > body_dist + 0.02:
                print("[Demo][WARN] nudge increased distance; restoring right EE pre-nudge pose.")
                right.move_end_effector_to(
                    [float(ee_before[0]), float(ee_before[1]), float(ee_before[2])],
                    orientation=right_assembly_orn,
                    steps=220,
                    hold_companion=left,
                )
                right.maintain_grasp_hold(steps=40)
                main_pos, main_orn = p.getBasePositionAndOrientation(left_body_id)
                aux_pos,  _        = p.getBasePositionAndOrientation(right_body_id)
                body_dist = float(np.linalg.norm(np.array(aux_pos) - np.array(main_pos)))
                print(f"[Demo] restored body distance: {body_dist:.3f} m")
            else:
                body_dist = body_dist_after
 
        # Pose-snap stage: gravity stays ON throughout. With EE z-compensation
        # already applied, both bodies sit at JSON target z held by their
        # respective closed grippers (fingers at body top, no clipping). A
        # small snap brings AABB centers exactly onto target. The constraints
        # plus closed grippers hold everything in place against gravity.
        if _plan_loaded:
            print("[Demo] Phase A snap (gravity ON, grippers closed)...")
            # Phase A snap: ONLY phillips_screwdriver + fork (the two bodies
            # we just grasped). Sponge stays on the table for Phase B, where
            # right arm will pick it up and bring it to the assembly.
            phase_a_labels = {left_target_label, right_target_label}
            snapped = assembly_manager.snap_objects_to_targets(
                settle_steps=0,
                use_aabb_offset=True,
                labels=phase_a_labels,
            )
            print(f"[Demo] Phase A snapped {len(snapped)} bodies: {sorted(list(snapped.keys()))}")

        print("[Demo] assembling parts...")

        if _plan_loaded:
            # ?? module3 JSON 怨꾪쉷 ?ㅽ뻾 ??????????????????????????????????????
            # step1(position-only)? 諛곗튂 湲곕줉留? step2 ?댄썑 attach ?ㅽ뻾
            # Phase A only handles steps whose objects are in {left, right}
            # grasp set (phillips + fork). Step 3 (sponge) is intentionally
            # skipped here and run in Phase B after sponge is picked up.
            _active_labels = {left_target_label, right_target_label}
            _selected_steps = []
            _plan_obj = getattr(assembly_manager, "_plan", None)
            if _plan_obj is not None and getattr(_plan_obj, "steps", None):
                for _step in _plan_obj.steps:
                    if not _step.has_attach():
                        if _step.base_object in _active_labels:
                            _selected_steps.append(int(_step.step))
                        else:
                            print(
                                f"[Demo] skip plan step {_step.step}: "
                                f"position-only base '{_step.base_object}' is outside active grasp set."
                            )
                        continue

                    _step_objects = {_step.base_object, _step.attach_object}
                    if _step_objects.issubset(_active_labels):
                        _selected_steps.append(int(_step.step))
                    else:
                        print(
                            f"[Demo] skip plan step {_step.step}: "
                            f"objects {sorted(list(_step_objects))} are outside active grasp set "
                            f"{sorted(list(_active_labels))}."
                        )

            if _selected_steps:
                print(f"[Demo] Phase A executing module3 plan steps: {_selected_steps}")
                assembly_results = []
                for _step_no in _selected_steps:
                    assembly_results.append(
                        assembly_manager.execute_step(
                            step_number=_step_no,
                            settle_steps=60,
                            max_force=500,
                            suspend_gravity=False,
                        )
                    )
            else:
                print("[Demo][WARN] no compatible module3 steps for current two-object grasp; using fallback attach.")
                _plan_loaded = False
            attach_results = [r for r in assembly_results if r.get("constraint_id") is not None]
            failed_attach_steps = [
                r for r in assembly_results
                if r.get("constraint_id") is None and r.get("reason") not in {"position_only"}
            ]
            if attach_results and not failed_attach_steps:
                print(f"[Demo] Phase A assembly successful "
                      f"({len(attach_results)} constraint(s) created).")
            else:
                if failed_attach_steps:
                    print(f"[Demo][WARN] Phase A attach validation failed: {failed_attach_steps}")
                else:
                    print("[Demo][WARN] Phase A produced no constraints; falling back.")
                _plan_loaded = False

        # World-pin combo: after Step 2 attach, fix phillips (and via the
        # constraint, fork) to the static plane. This keeps the assembled
        # combo at its JSON pose during Phase B even if the right arm
        # brushes against it while transporting sponge. We use a strong
        # max_force so the pin survives accidental impacts. The pin is
        # removed after Phase B step 3 attach.
        world_pin_id = None
        if _plan_loaded:
            try:
                _phillips_pos, _phillips_orn = p.getBasePositionAndOrientation(left_body_id)
                world_pin_id = p.createConstraint(
                    parentBodyUniqueId=0,             # static plane (body_id=0)
                    parentLinkIndex=-1,
                    childBodyUniqueId=left_body_id,
                    childLinkIndex=-1,
                    jointType=p.JOINT_FIXED,
                    jointAxis=[0, 0, 0],
                    parentFramePosition=list(_phillips_pos),
                    childFramePosition=[0, 0, 0],
                    parentFrameOrientation=list(_phillips_orn),
                    childFrameOrientation=[0, 0, 0, 1],
                )
                p.changeConstraint(world_pin_id, maxForce=10000)
                print(f"[Demo] world-pinned combo at {[round(v,4) for v in _phillips_pos]} (pin id={world_pin_id})")
            except Exception as exc:
                print(f"[Demo][WARN] world pin failed: {exc}")
                world_pin_id = None

        # =========================================================
        # Phase B: pick sponge with right arm and attach to fork
        # =========================================================
        # Conditions: Phase A succeeded (_plan_loaded still True) AND
        # plan has a step 3 with sponge as attach object.
        _phase_b_done = False
        if _plan_loaded:
            _plan_obj_b = getattr(assembly_manager, "_plan", None)
            _step3 = None
            if _plan_obj_b is not None:
                for _s in _plan_obj_b.steps:
                    if _s.has_attach() and _s.attach_object == "sponge":
                        _step3 = _s
                        break
            sponge_id = ycb_object_ids.get("sponge")
            if _step3 is not None and sponge_id is not None:
                print("[Demo] === Phase B: pick sponge and attach to fork (gravity ON) ===")
                # Right arm currently holds fork at right_assembly_ee. Fork is
                # already constrained to phillips (Step 2), and combo is
                # world-pinned, so right's grip is no longer needed.
                try:
                    right.open_gripper(steps=30)
                except Exception as exc:
                    print(f"[Demo][WARN] right.open_gripper (phaseB pre) failed: {exc}")
                # Reset right arm to home before grasping sponge. After Phase
                # A, right was at the edge of reach (~0.84 m from base, near
                # Panda's 0.855 m limit) with stretched/twisted joints. IK
                # called from that seed diverges (EE shoots to z=1.4+ instead
                # of descending). reset_to_home gives IK a clean starting
                # joint configuration close to PANDA_HOME_JOINTS, which is
                # what `restPoses` biases toward. Combo stays put thanks to
                # the world pin (max_force=10000) even if right arm body
                # brushes against it during the home move.
                try:
                    right.reset_to_home(steps=240)
                except Exception as exc:
                    print(f"[Demo][WARN] right.reset_to_home (phaseB pre) failed: {exc}")
                # Grasp sponge with right arm (R1 hint may be empty - the
                # routine falls back to AABB-based pose automatically).
                _sponge_hint = r1_hints.get("sponge", {})
                sponge_ok, sponge_grasp_src = _grasp_with_r1_then_aabb_fallback(
                    controller=right,
                    body_id=sponge_id,
                    object_label="sponge",
                    r1_hint=_sponge_hint,
                )
                if not sponge_ok:
                    print("[Demo][WARN] Phase B sponge grasp failed; skipping step 3.")
                else:
                    print(f"[Demo] sponge grasp succeeded via source={sponge_grasp_src}")
                    right.maintain_grasp_hold(steps=60)
                    # Move sponge to step 3 attach pose with EE z-comp so
                    # sponge body lands at JSON target_pose_world.
                    sponge_target_pos = list(_step3.target_position)
                    sponge_target_orn = list(_step3.target_orientation)
                    sponge_assembly_ee = _ee_pose_with_grip_z_comp(sponge_target_pos, sponge_id)
                    print(
                        f"[Demo] sponge assembly move: object_target={[round(v, 4) for v in sponge_target_pos]}, "
                        f"ee_pose={[round(v, 4) for v in sponge_assembly_ee]}"
                    )
                    right.move_end_effector_to(
                        sponge_assembly_ee,
                        orientation=sponge_target_orn,
                        steps=600,
                        hold_companion=left if left_ok else None,
                    )
                    right.maintain_grasp_hold(steps=60)
                    print("[Demo] right arm at sponge assembly position.")
                    # Gravity stays ON. Snap sponge with closed gripper for
                    # precision, then create the constraint to fork.
                    snapped_b = assembly_manager.snap_objects_to_targets(
                        settle_steps=0,
                        use_aabb_offset=True,
                        labels={"sponge"},
                    )
                    print(f"[Demo] Phase B snapped {len(snapped_b)} body: {sorted(list(snapped_b.keys()))}")
                    step3_result = assembly_manager.execute_step(
                        step_number=int(_step3.step),
                        settle_steps=60,
                        max_force=500,
                        suspend_gravity=False,
                    )
                    if step3_result.get("constraint_id") is not None:
                        print("[Demo] Phase B assembly successful (sponge attached to fork).")
                        _phase_b_done = True
                        # Sponge is now constrained to fork; right's grip
                        # on sponge is redundant. Release it so Phase 4
                        # can move the combo without right's grip fighting
                        # the rigid constraint chain.
                        try:
                            right.open_gripper(steps=30)
                        except Exception as exc:
                            print(f"[Demo][WARN] right.open_gripper (post-step3) failed: {exc}")
                    else:
                        print(f"[Demo][WARN] Phase B step 3 failed: {step3_result}")
            else:
                print("[Demo] Phase B skipped (no step 3 or sponge missing).")

        # Remove the world pin so the assembled tool is free to be carried
        # or released. This must happen before the place/release/home stage.
        if world_pin_id is not None:
            try:
                p.removeConstraint(world_pin_id)
                print(f"[Demo] world pin removed (id={world_pin_id})")
            except Exception as exc:
                print(f"[Demo][WARN] removing world pin failed: {exc}")
 
        if not _plan_loaded:
            # ?? fallback: ?꾩옱 ?ㅼ젣 ?꾩튂 湲곕컲 吏곸젒 attach ??????????????????
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
 
        # ?덉젙??
        for _ in range(DEMO_HOLD_STEPS):
            left._tick_gripper_hold()
            right._tick_gripper_hold()
            p.stepSimulation()
            time.sleep(SIM_TIMESTEP)
 
    else:
        print("[Demo][WARN] skipping assembly (one or both grasps failed).")
 
    # ?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧
    # Step 4. 寃고빀泥??대젮?볤린 & release
    # ?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧
    if left_ok:
        left.move_end_effector_to(
            PLACE_POS, orientation=down_orn, steps=400,
            hold_companion=right if right_ok else None,
        )
        left.maintain_grasp_hold(steps=60)
 
    # (carry_constraint ?놁쓬 ??gripper force ?좎?濡??대룞)
 
    if left_ok:
        left.release_grasp(open_after=True, steps=120)
 
    if right_ok:
        right.move_end_effector_to(
            [PLACE_POS[0], PLACE_POS[1], PLACE_POS[2] + 0.15],
            orientation=down_orn, steps=400,
        )
        right.maintain_grasp_hold(steps=60)
        right.release_grasp(open_after=True, steps=120)
 
    # assembly constraint??release ?꾩뿉????臾쇱껜媛 遺숈뼱?덇쾶 ?좎?
    # 遺꾨━?섎젮硫? p.removeConstraint(assembly_constraint)
 
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
    print(
        "[Boot] grasp safety policy: "
        "R1 target validation + sim outlier guard + descend z-floor clamp"
    )
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

    # module3 object labels are passed to R1 probe.
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