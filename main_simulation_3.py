import os
import time

import numpy as np
import pybullet as p
import pybullet_data

from robot_controller_3 import PandaController


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
    ("chips_can", "001_chips_can.urdf", [0.80, -0.15, 0.82]),
    ("apple", "013_apple.urdf", [0.55, -0.25, 0.82]),
    ("cracker_box", "003_cracker_box.urdf", [0.70, 0.10, 0.82]),
    ("mug", "025_mug.urdf", [0.62, -0.05, 0.82]),
    ("mustard_bottle", "006_mustard_bottle.urdf", [0.47, -0.12, 0.82]),
    ("large_clamp", "051_large_clamp.urdf", [0.76, 0.22, 0.82]),
]

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

# Optional per-label overrides. Base profile can be inferred from loaded body features.
YCB_MODULE1_PROFILE_BY_LABEL = {
    "chips_can": {
        "surface_friction": "medium",
        "slip_tendency": "medium",
        "mass_category": "light",
        "density_category": "medium",
        "size_relative": "small",
    },
    "apple": {
        "surface_friction": "medium",
        "slip_tendency": "low",
        "mass_category": "light",
        "density_category": "medium",
        "size_relative": "small",
        "deformability": "medium",
    },
    "cracker_box": {
        "surface_friction": "medium",
        "slip_tendency": "medium",
        "mass_category": "light",
        "density_category": "medium",
        "size_relative": "medium",
    },
    "mug": {
        "surface_friction": "medium",
        "slip_tendency": "medium",
        "mass_category": "medium",
        "density_category": "high",
        "size_relative": "small",
    },
    "mustard_bottle": {
        "surface_friction": "medium",
        "slip_tendency": "low",
        "mass_category": "light",
        "density_category": "high",
        "size_relative": "medium",
    },
    "large_clamp": {
        "surface_friction": "high",
        "slip_tendency": "low",
        "mass_category": "medium",
        "density_category": "high",
        "size_relative": "small",
    },
}

YCB_DYNAMICS_OVERRIDE_BY_LABEL = {
    # Real-apple test profile (temporary): if grasp degrades, revert to stable profile.
    "apple": {
        "mass_kg": 0.18,
        "lateral_friction": 0.65,
    },
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


def configure_simulation() -> None:
    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.6,
        cameraYaw=45,
        cameraPitch=-40,
        cameraTargetPosition=[0.6, 0.0, 0.8],
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


def load_ycb_objects(ycb_dir: str = YCB_DIR) -> dict:
    if not os.path.isdir(ycb_dir):
        raise FileNotFoundError(
            f"YCB directory not found: {ycb_dir}. "
            "Extract data.zip first."
        )

    flags = p.URDF_USE_INERTIA_FROM_FILE
    loaded = {}

    for label, urdf_name, base_position in YCB_OBJECT_SPECS:
        urdf_path = os.path.join(ycb_dir, urdf_name)
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"Missing URDF: {urdf_path}")
        body_id = p.loadURDF(
            urdf_path,
            basePosition=base_position,
            baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
            globalScaling=0.1,
            flags=flags,
        )
        loaded[label] = body_id
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
        cameraTargetPosition=[0.60, 0.0, 0.83],
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


def run_optional_affordance_probe(
    enable_affordance_r1: bool,
    enable_sam2_refinement: bool = False,
    controllers: dict | None = None,
) -> None:
    if not enable_affordance_r1:
        print("[R1] optional affordance probe disabled (set ENABLE_AFFORDANCE_R1=1 to enable)")
        return

    print("[R1] optional affordance probe enabled")

    try:
        from affordancegrasp_r1_adapter import AffordanceGraspR1Adapter
    except Exception as exc:
        print(f"[R1][WARN] adapter import failed: {exc}")
        return

    try:
        rgb = capture_affordance_rgb()
        adapter = AffordanceGraspR1Adapter(
            model_id=AFFORDANCE_MODEL_ID,
            local_model_dir=os.getenv("AFFORDANCE_R1_LOCAL_DIR"),
            local_files_only=env_flag("AFFORDANCE_R1_LOCAL_ONLY", default=False),
        )

        result = adapter.predict(
            rgb,
            prompt=(
                "Find one robust grasp affordance for tabletop robot pick and place. "
                "Return [x1, y1, x2, y2] and part name."
            ),
            extra_context={
                "scene": "dual-panda-tabletop",
                "objects": [spec[0] for spec in YCB_OBJECT_SPECS],
            },
        )

        if result.get("success"):
            candidates = result.get("grasp_candidates") or []
            if enable_sam2_refinement and candidates:
                print("[SAM2] refinement enabled")
                try:
                    from sam2_segmenter_adapter import SAM2SegmentationAdapter

                    sam2_adapter = SAM2SegmentationAdapter(
                        model_id=os.getenv("SAM2_MODEL_ID", SAM2_MODEL_ID),
                        local_model_dir=os.getenv("SAM2_LOCAL_DIR"),
                        local_files_only=env_flag("SAM2_LOCAL_ONLY", default=False),
                    )
                    refined = sam2_adapter.refine_result(
                        image=rgb,
                        affordance_result=result,
                        top_k=1,
                    )
                    if refined.get("success"):
                        result = refined.get("result", result)
                        candidates = result.get("grasp_candidates") or candidates
                        print("[SAM2] top-1 candidate refined with mask.")
                    else:
                        print(f"[SAM2][WARN] refinement skipped: {refined.get('error')}")
                except Exception as exc:
                    print(f"[SAM2][WARN] refinement failed, using R1 candidate only: {exc}")
            elif enable_sam2_refinement:
                print("[SAM2][INFO] no R1 candidates available to refine.")

            print(f"[R1] summary: {result.get('affordance_summary')}")
            if candidates:
                print(f"[R1] top candidate: {candidates[0]}")
            else:
                print("[R1][INFO] inference succeeded but no parsed candidate was found.")
            if isinstance(controllers, dict):
                for arm_name, controller in controllers.items():
                    hint = controller.set_affordance_hint(result)
                    print(f"[R1] {arm_name} controller hint: {hint}")
            print("[R1][TODO] map 2D affordance candidates to object-level 3D pick waypoints.")
            return

        print(f"[R1][WARN] inference unavailable: {result.get('error')}")
    except Exception as exc:
        print(f"[R1][WARN] optional affordance probe failed, continuing simulation: {exc}")


def run_sequential_demo(
    controllers: dict,
    ycb_object_ids: dict,
) -> None:
    left = controllers["left"]
    right = controllers["right"]
    down_orn = p.getQuaternionFromEuler([np.pi, 0.0, 0.0])

    print("[Boot] reset both robots to home")
    left.reset_to_home(steps=600)
    right.reset_to_home(steps=600)

    print("[Demo] open both grippers")
    left.open_gripper(steps=120)
    right.open_gripper(steps=120)

    if not ycb_object_ids:
        print("[Demo][WARN] no YCB objects were loaded, skipping grasp demo.")
        return

    # ── 파지 대상 선택 ─────────────────────────────────────────────────────────
    left_target_label = "apple" if "apple" in ycb_object_ids else next(iter(ycb_object_ids))
    right_target_label = (
        "mustard_bottle" if "mustard_bottle" in ycb_object_ids else next(iter(ycb_object_ids))
    )

    # ── Left arm: 첫 번째 물체 파지 후 조립 위치로 이동 ────────────────────────
    print(f"[Demo] left-arm grasp target: {left_target_label}")
    left_ok = left.grasp_body(
        body_id=ycb_object_ids[left_target_label],
        object_label=left_target_label,
        orientation=down_orn,
    )

    # 조립 위치: 테이블 중앙 위 (두 팔이 만나는 지점)
    ASSEMBLY_POS = [0.60, 0.0, 1.00]

    if left_ok:
        left.maintain_grasp_hold(steps=120)
        # 조립 위치로 운반 (main 파트 역할 — 고정 위치 유지)
        left.move_end_effector_to(ASSEMBLY_POS, orientation=down_orn, steps=600)
        left.maintain_grasp_hold(steps=80)
    else:
        print(f"[Demo][WARN] left-arm grasp failed for '{left_target_label}'")

    # ── Right arm: 두 번째 물체 파지 후 조립 위치로 접근 ──────────────────────
    # 수정 후
    print(f"[Demo] right-arm grasp target: {right_target_label}")
    right_ok = right.grasp_body(
        body_id=ycb_object_ids[right_target_label],
        object_label=right_target_label,
        orientation=down_orn,
    )

    if not right_ok:
        print(f"[Demo][WARN] right-arm grasp failed for '{right_target_label}', skipping right arm.")
    else:
        right.maintain_grasp_hold(steps=120)
        # left_ok인 경우에만 조립 위치로 이동, 아니면 제자리 hold
        if left_ok:
            aux_approach = [ASSEMBLY_POS[0], ASSEMBLY_POS[1], ASSEMBLY_POS[2] + 0.12]
            right.move_end_effector_to(aux_approach, orientation=down_orn, steps=600)
        right.maintain_grasp_hold(steps=80)


    # ── Assembly: 두 물체 고정 결합 ────────────────────────────────────────────
    assembly_constraint = None
    if left_ok and right_ok:
        print("[Demo] assembling parts...")
        # left가 쥔 물체(main)에 right가 쥔 물체(aux)를 붙임
        # attach_to_body는 PandaController 인스턴스에 추가된 메서드
        assembly_constraint = left.attach_to_body(
            main_body_id=ycb_object_ids[left_target_label],
            aux_body_id=ycb_object_ids[right_target_label],
        )
        if assembly_constraint is not None:
            print("[Demo] assembly successful — holding assembled structure.")
            # 결합 후 두 팔 모두 hold 유지하며 안정화
            for _ in range(DEMO_HOLD_STEPS):
                p.stepSimulation()
                time.sleep(SIM_TIMESTEP)
        else:
            print("[Demo][WARN] assembly constraint failed.")
    else:
        print("[Demo][WARN] skipping assembly (one or both grasps failed).")

    # ── 결합체 내려놓기 ────────────────────────────────────────────────────────
    PLACE_POS = [0.60, 0.0, 0.85]
    if left_ok:
        left.move_end_effector_to(PLACE_POS, orientation=down_orn, steps=400)
        left.maintain_grasp_hold(steps=60)
        left.release_grasp(open_after=True, steps=140)

    if right_ok:
        right.move_end_effector_to(
            [PLACE_POS[0], PLACE_POS[1], PLACE_POS[2] + 0.12],
            orientation=down_orn,
            steps=400,
        )
        right.maintain_grasp_hold(steps=60)
        right.release_grasp(open_after=True, steps=140)

    # constraint는 release 후에도 물체끼리 붙어있게 유지 (필요시 detach 가능)
    # left.detach_body(assembly_constraint)  # 분리하려면 이 줄 활성화

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
    ycb_object_ids = load_ycb_objects()
    controllers, robot_ids = create_dual_arm_controllers()

    if enable_module1_dynamics:
        map_cfg = load_module1_map()
        applied_dynamics = apply_module1_dynamics_to_loaded_objects(
            ycb_object_ids=ycb_object_ids,
            map_cfg=map_cfg,
            infer_profile_from_body=enable_module1_profile_inference,
        )
        print(f"[Boot] module1 dynamics applied: {applied_dynamics}")
    else:
        print("[Boot] module1 dynamics skipped by flag.")

    print(f"[Boot] scene IDs: {scene_ids}")
    print(f"[Boot] ycb objects: {ycb_object_ids}")
    print(f"[Boot] robot IDs: {robot_ids}")

    stabilize_scene()
    run_optional_affordance_probe(
        enable_affordance_r1=enable_affordance_r1,
        enable_sam2_refinement=enable_sam2_refinement,
        controllers=controllers,
    )
    run_sequential_demo(
        controllers=controllers,
        ycb_object_ids=ycb_object_ids,
    )
    keep_gui_alive()


if __name__ == "__main__":
    main()