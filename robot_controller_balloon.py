"""
robot_controller_balloon.py
Lightweight Panda-arm + Robotiq 2F-140 controller module for simulation boot/demo.

Notes:
- This module keeps dependencies minimal on purpose:
  pybullet, numpy, time
- Affordance-R1 model weights are expected to come from:
  https://huggingface.co/hqking/affordance-r1
"""

import os
import time
import numpy as np
import pybullet as p


ROBOT_MODEL_NAME = "Panda arm + Robotiq 2F-140 gripper"
ROBOT_URDF = "franka_panda/panda.urdf"
ROBOTIQ_GRIPPER_URDF = os.path.join(
    os.path.dirname(__file__),
    "assets",
    "robots",
    "robotiq_2f140_box.urdf",
)

# Panda remains the arm carrier; a lightweight box-finger Robotiq 2F-140
# surrogate is fixed to the wrist and used for gripper contacts/control.
END_EFFECTOR_INDEX = 11
NUM_JOINTS = 7
GRIPPER_JOINT_INDICES = (9, 10)
GRIPPER_CONTACT_LINK_INDICES = GRIPPER_JOINT_INDICES
PANDA_HAND_LINK_INDICES = (8, 9, 10, 11)
ROBOTIQ_GRIPPER_JOINT_INDICES = (0, 1)
ROBOTIQ_GRIPPER_CONTACT_LINK_INDICES = (0, 1)

ROBOTIQ_2F140_MAX_OPENING = 0.140
ROBOTIQ_2F140_BOX_FINGER_THICKNESS = 0.014
ROBOTIQ_2F140_SURROGATE_FINGER_MAX = (
    ROBOTIQ_2F140_MAX_OPENING + ROBOTIQ_2F140_BOX_FINGER_THICKNESS
) / 2.0
GRIPPER_MAX_TARGET = ROBOTIQ_2F140_SURROGATE_FINGER_MAX
GRIPPER_MIN_TARGET = 0.0
GRIPPER_FINGER_TIP_OFFSET = 0.085

PANDA_HOME_JOINTS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

PANDA_LOWER = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
PANDA_UPPER = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
PANDA_RANGE = [5.7946, 3.5256, 5.7946, 3.0020, 5.7946, 3.7700, 5.7946]

ARM_FORCE = 500
ARM_MAX_VELOCITY = 1.0
GRIPPER_FORCE = 160
GRIPPER_HOLD_FORCE = 500
GRIPPER_HOLD_FORCE_MAX = 900
RUBBER_PAD_LATERAL_FRICTION = 4.5
RUBBER_PAD_ROLLING_FRICTION = 0.08
RUBBER_PAD_SPINNING_FRICTION = 0.08
RUBBER_PAD_CONTACT_STIFFNESS = 60000
RUBBER_PAD_CONTACT_DAMPING = 1800
HOLD_TIGHTEN_STEP = 0.003
HOLD_SLIP_DISTANCE_MARGIN = 0.05
SIM_TIMESTEP = 1.0 / 240.0
NO_CONSTRAINT_CLOSE_STEPS = 220
NO_CONSTRAINT_MIN_LIFT_GAIN = 0.02

DEFAULT_GRASP_PROFILE = {
    "gripper_force": GRIPPER_FORCE,
    "hold_force": GRIPPER_HOLD_FORCE,
    "hold_force_max": GRIPPER_HOLD_FORCE_MAX,
    "tighten_step": HOLD_TIGHTEN_STEP,
    "slip_distance_margin": HOLD_SLIP_DISTANCE_MARGIN,
    "close_steps": NO_CONSTRAINT_CLOSE_STEPS,
    "min_lift_gain": NO_CONSTRAINT_MIN_LIFT_GAIN,
    "approach_height": 0.16,
    "grasp_clearance": 0.004,
    "lift_height": 0.12,
    "grasp_z_ratio": 0.85,
    "slip_force_boost": 8.0,
    "hold_force_decay": 1.0,
}

OBJECT_GRASP_PROFILE_TABLE = {
    "chips_can": {
        "gripper_force": 175,
        "hold_force": 350,
        "hold_force_max": 560,
        "tighten_step": 0.0020,
        "slip_distance_margin": 0.055,
        "close_steps": 230,
        "grasp_clearance": 0.0035,
    },
    "apple": {
        "gripper_force": 190,
        "hold_force": 420,
        "hold_force_max": 680,
        "tighten_step": 0.0026,
        "slip_distance_margin": 0.060,
        "close_steps": 270,
        "grasp_clearance": 0.0020,
        "approach_height": 0.17,
        "lift_height": 0.10,
        "grasp_z_ratio": 0.62,
    },
    "cracker_box": {
        "gripper_force": 165,
        "hold_force": 335,
        "hold_force_max": 540,
        "tighten_step": 0.0018,
        "slip_distance_margin": 0.050,
        "close_steps": 220,
        "grasp_clearance": 0.0035,
    },
    "mug": {
        "gripper_force": 170,
        "hold_force": 360,
        "hold_force_max": 570,
        "tighten_step": 0.0019,
        "slip_distance_margin": 0.055,
        "close_steps": 235,
        "grasp_clearance": 0.0030,
    },
    "mustard_bottle": {
        "gripper_force": 180,
        "hold_force": 380,
        "hold_force_max": 590,
        "tighten_step": 0.0021,
        "slip_distance_margin": 0.058,
        "close_steps": 240,
        "grasp_clearance": 0.0030,
    },
    "large_clamp": {
        "gripper_force": 200,
        "hold_force": 420,
        "hold_force_max": 640,
        "tighten_step": 0.0023,
        "slip_distance_margin": 0.060,
        "close_steps": 250,
        "grasp_clearance": 0.0030,
    },
}

# ── 카메라 파라미터 ───────────────────────────────────────────────────────────
CAM_WIDTH  = 1280
CAM_HEIGHT = 960
CAM_FOV    = 60
CAM_NEAR   = 0.1
CAM_FAR    = 5.0

def render_camera(cam_target=[0.55, -0.35, 0.8], cam_distance=1.0,
                  cam_yaw=45, cam_pitch=-45):
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
        CAM_WIDTH, CAM_HEIGHT,       # ← 해상도를 1280x960 등으로 키우세요
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,  # ← TINY → OpenGL
        shadow=1,
        lightDirection=[1, 1, 1],
        lightColor=[1, 1, 1],
        lightDistance=2,
        lightAmbientCoeff=0.8,
        lightDiffuseCoeff=0.8,
        lightSpecularCoeff=0.1,
    )
    rgb = np.array(rgb_raw, dtype=np.uint8).reshape(CAM_HEIGHT, CAM_WIDTH, 4)[:, :, :3]
    depth_buf = np.array(depth_raw, dtype=np.float32).reshape(CAM_HEIGHT, CAM_WIDTH)
    depth = CAM_FAR * CAM_NEAR / (CAM_FAR - (CAM_FAR - CAM_NEAR) * depth_buf)
    return rgb, depth, proj_matrix, view_matrix


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _normalize_angle_rad(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _step_simulation(steps: int, hold_companion: "PandaController | None" = None) -> None:
    for _ in range(steps):
        if hold_companion is not None:
            hold_companion._tick_gripper_hold()
        p.stepSimulation()
        time.sleep(SIM_TIMESTEP)


def _opening_to_joint_target(opening_width: float) -> float:
    """Map desired inner-finger opening width to one prismatic finger joint."""
    opening = _clamp(opening_width, 0.0, ROBOTIQ_2F140_MAX_OPENING)
    return (opening + ROBOTIQ_2F140_BOX_FINGER_THICKNESS) / 2.0


def _joint_target_to_opening(joint_target: float) -> float:
    return max(0.0, 2.0 * float(joint_target) - ROBOTIQ_2F140_BOX_FINGER_THICKNESS)


def load_robot(base_position, use_fixed_base: bool = True) -> int:
    return p.loadURDF(
        ROBOT_URDF,
        useFixedBase=use_fixed_base,
        basePosition=base_position,
    )


def load_panda(base_position, use_fixed_base: bool = True) -> int:
    return load_robot(base_position=base_position, use_fixed_base=use_fixed_base)


def reset_to_home(panda_id, steps=800):
    for joint_idx in range(NUM_JOINTS):
        p.setJointMotorControl2(
            panda_id,
            joint_idx,
            p.POSITION_CONTROL,
            targetPosition=PANDA_HOME_JOINTS[joint_idx],
            force=ARM_FORCE,
            maxVelocity=ARM_MAX_VELOCITY,
        )
    _step_simulation(steps)


def move_end_effector_to(panda_id, position, orientation=None, steps=1000):
    if orientation is None:
        orientation = p.getQuaternionFromEuler([np.pi, 0.0, 0.0])

    joint_poses = p.calculateInverseKinematics(
        panda_id,
        END_EFFECTOR_INDEX,
        position,
        targetOrientation=orientation,
        lowerLimits=PANDA_LOWER,
        upperLimits=PANDA_UPPER,
        jointRanges=PANDA_RANGE,
        restPoses=PANDA_HOME_JOINTS,
        maxNumIterations=1000,
        residualThreshold=1e-6,
    )

    for joint_idx in range(NUM_JOINTS):
        p.setJointMotorControl2(
            panda_id,
            joint_idx,
            p.POSITION_CONTROL,
            targetPosition=joint_poses[joint_idx],
            force=ARM_FORCE,
            maxVelocity=ARM_MAX_VELOCITY,
        )
    _step_simulation(steps)


def open_gripper(panda_id: int, steps: int = 100):
    target = _opening_to_joint_target(ROBOTIQ_2F140_MAX_OPENING)
    for finger_joint in GRIPPER_JOINT_INDICES:
        p.setJointMotorControl2(
            panda_id,
            finger_joint,
            p.POSITION_CONTROL,
            targetPosition=target,
            force=GRIPPER_FORCE,
        )
    _step_simulation(steps)


def close_gripper(panda_id: int, steps: int = 100):
    for finger_joint in GRIPPER_JOINT_INDICES:
        p.setJointMotorControl2(
            panda_id,
            finger_joint,
            p.POSITION_CONTROL,
            targetPosition=0.0,
            force=GRIPPER_FORCE,
        )
    _step_simulation(steps)


def select_top_grasp_candidate(inference_result: dict) -> dict | None:
    """
    Extract one normalized grasp candidate from Affordance-R1 style output.
    This helper is intentionally lightweight so controller code can consume
    inference output without importing heavy model dependencies.
    """
    if not isinstance(inference_result, dict):
        return None

    candidates = inference_result.get("grasp_candidates")
    if not isinstance(candidates, list) or not candidates:
        return None

    top = candidates[0]
    if not isinstance(top, dict):
        return None

    bbox_norm = top.get("bbox_norm")
    if (
        isinstance(bbox_norm, list)
        and len(bbox_norm) == 4
        and all(isinstance(v, (int, float)) for v in bbox_norm)
    ):
        x1, y1, x2, y2 = bbox_norm
        center_norm = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
    else:
        center_norm = top.get("center_norm")

    return {
        "part": top.get("part", "object"),
        "bbox_norm": bbox_norm,
        "center_norm": center_norm,
        "score": top.get("score", 1.0),
    }


class Robotiq2F140Controller:
    def __init__(self, name: str, base_position, use_fixed_base: bool = True):
        self.name = name
        self.base_position = list(base_position)
        self.use_fixed_base = use_fixed_base
        self.panda_id = None
        self.gripper_id = None
        self.gripper_constraint_id = None
        self.last_target_position = None
        self.last_target_orientation = None
        self.last_affordance_hint = None
        self._grasp_profile_table = {
            key: value.copy() for key, value in OBJECT_GRASP_PROFILE_TABLE.items()
        }
        self._active_grasp_profile = DEFAULT_GRASP_PROFILE.copy()
        self._active_body_features = None
        self._active_gripper_hold_target = None
        self._active_gripper_hold_force = self._active_grasp_profile["hold_force"]
        self._held_body_id = None
        self._held_reference_ee_distance = None
        self._hold_slip_events = 0
        self._last_grasp_failure = None

    def _ensure_loaded(self) -> None:
        if self.panda_id is None:
            raise RuntimeError(f"[{self.name}] robot is not loaded yet.")

    def load_robot(self):
        self.panda_id = load_robot(
            base_position=self.base_position,
            use_fixed_base=self.use_fixed_base,
        )
        self._hide_builtin_panda_hand()
        self._load_and_attach_robotiq_gripper()
        self.configure_gripper_contact_dynamics()
        return self.panda_id

    def load_panda(self):
        return self.load_robot()

    def reset_to_home(self, steps=800):
        self._ensure_loaded()
        reset_to_home(self.panda_id, steps=steps)

    def _hide_builtin_panda_hand(self) -> None:
        for link_idx in PANDA_HAND_LINK_INDICES:
            try:
                p.changeVisualShape(self.panda_id, link_idx, rgbaColor=[1, 1, 1, 0])
                p.setCollisionFilterGroupMask(self.panda_id, link_idx, 0, 0)
            except Exception:
                pass

    def _load_and_attach_robotiq_gripper(self) -> None:
        if not os.path.isfile(ROBOTIQ_GRIPPER_URDF):
            raise FileNotFoundError(f"Missing Robotiq surrogate URDF: {ROBOTIQ_GRIPPER_URDF}")

        ee_pos, ee_orn = self.get_end_effector_pose()
        self.gripper_id = p.loadURDF(
            ROBOTIQ_GRIPPER_URDF,
            basePosition=ee_pos.tolist(),
            baseOrientation=ee_orn.tolist(),
            useFixedBase=False,
        )
        self.gripper_constraint_id = p.createConstraint(
            parentBodyUniqueId=self.panda_id,
            parentLinkIndex=END_EFFECTOR_INDEX,
            childBodyUniqueId=self.gripper_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, -0.1],
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=[0, 0, 0, 1],
            childFrameOrientation=[0, 0, 0, 1],
        )
        p.changeConstraint(self.gripper_constraint_id, maxForce=10000)
        self._disable_arm_gripper_self_collision()

    def _disable_arm_gripper_self_collision(self) -> None:
        if self.gripper_id is None:
            return
        for panda_link in range(-1, p.getNumJoints(self.panda_id)):
            for gripper_link in range(-1, p.getNumJoints(self.gripper_id)):
                try:
                    p.setCollisionFilterPair(
                        self.panda_id,
                        self.gripper_id,
                        panda_link,
                        gripper_link,
                        enableCollision=0,
                    )
                except Exception:
                    pass

    def move_end_effector_to(self, position, orientation=None, steps=1000, hold_companion: "PandaController | None" = None):
        self._ensure_loaded()
        self.last_target_position = list(position)
        self.last_target_orientation = list(orientation) if orientation is not None else None
        if orientation is None:
            orientation = p.getQuaternionFromEuler([np.pi, 0.0, 0.0])

        joint_poses = p.calculateInverseKinematics(
            self.panda_id,
            END_EFFECTOR_INDEX,
            position,
            targetOrientation=orientation,
            lowerLimits=PANDA_LOWER,
            upperLimits=PANDA_UPPER,
            jointRanges=PANDA_RANGE,
            restPoses=PANDA_HOME_JOINTS,
            maxNumIterations=1000,
            residualThreshold=1e-6,
        )
        for _ in range(steps):
            for joint_idx in range(NUM_JOINTS):
                p.setJointMotorControl2(
                    self.panda_id,
                    joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=joint_poses[joint_idx],
                    force=ARM_FORCE,
                    maxVelocity=ARM_MAX_VELOCITY,
                )
            if self._active_gripper_hold_target is not None:
                self._update_grasp_hold_feedback()
                self._set_gripper_target(
                    target_position=self._active_gripper_hold_target,
                    force=self._active_gripper_hold_force,
                    max_velocity=0.12,
                )
            if hold_companion is not None:
                hold_companion._tick_gripper_hold()
            p.stepSimulation()
            time.sleep(SIM_TIMESTEP)

    def open_gripper(self, steps: int = 100):
        self._ensure_loaded()
        self._clear_hold_control()
        self._set_gripper_target(
            target_position=GRIPPER_MAX_TARGET,
            force=GRIPPER_FORCE,
            max_velocity=0.25,
        )
        _step_simulation(steps)

    def close_gripper(self, steps: int = 100):
        self._ensure_loaded()
        self._active_gripper_hold_target = 0.0
        self._active_gripper_hold_force = self._active_grasp_profile["gripper_force"]
        self._set_gripper_target(
            target_position=GRIPPER_MIN_TARGET,
            force=self._active_gripper_hold_force,
            max_velocity=0.25,
        )
        _step_simulation(steps)

    def set_affordance_hint(self, inference_result: dict) -> dict | None:
        """
        Store latest grasp hint from external inference (optional).
        Keeping this separate avoids any direct model dependency here.
        """
        hint = select_top_grasp_candidate(inference_result)
        self.last_affordance_hint = hint
        return hint

    def _read_body_features(self, body_id: int) -> dict:
        aabb_min, aabb_max = p.getAABB(body_id)
        size_x = max(1e-4, float(aabb_max[0] - aabb_min[0]))
        size_y = max(1e-4, float(aabb_max[1] - aabb_min[1]))
        size_z = max(1e-4, float(aabb_max[2] - aabb_min[2]))
        span_xy = max(size_x, size_y)
        min_xy = min(size_x, size_y)
        volume = size_x * size_y * size_z
        info = p.getDynamicsInfo(body_id, -1)
        mass = float(info[0]) if len(info) > 0 else 0.2
        lateral_friction = float(info[1]) if len(info) > 1 else 0.5
        restitution = float(info[5]) if len(info) > 5 else 0.0

        # Heuristic shape cues for profile adaptation.
        height_ratio = size_z / max(span_xy, 1e-4)
        roundness_xy = min_xy / max(span_xy, 1e-4)
        round_like = roundness_xy > 0.82 and height_ratio > 0.75
        slender_like = height_ratio > 1.25
        flat_like = height_ratio < 0.55

        return {
            "size_x": size_x,
            "size_y": size_y,
            "size_z": size_z,
            "span_xy": span_xy,
            "volume": volume,
            "aabb_min": aabb_min,
            "aabb_max": aabb_max,
            "mass": max(0.02, mass),
            "lateral_friction": max(0.01, lateral_friction),
            "restitution": restitution,
            "round_like": round_like,
            "slender_like": slender_like,
            "flat_like": flat_like,
        }

    def _infer_grasp_profile_from_body(
        self,
        body_id: int,
        features: dict | None = None,
    ) -> dict:
        if features is None:
            features = self._read_body_features(body_id)
        mass = features["mass"]
        friction = features["lateral_friction"]
        span_xy = features["span_xy"]
        size_z = features["size_z"]

        # Normalize rough scales.
        mass_factor = _clamp((mass - 0.08) / 0.8, 0.0, 1.8)
        low_friction_factor = _clamp((0.7 - friction) / 0.7, 0.0, 1.0)
        width_factor = _clamp((span_xy - 0.04) / 0.10, 0.0, 1.0)

        grip_force = 135.0 + 80.0 * mass_factor + 95.0 * low_friction_factor + 25.0 * width_factor
        if features["round_like"]:
            grip_force += 18.0
        if features["flat_like"]:
            grip_force -= 10.0
        grip_force = _clamp(grip_force, 120.0, 260.0)

        hold_force = _clamp(
            grip_force * (1.85 + 0.35 * low_friction_factor),
            260.0,
            560.0,
        )
        hold_force_max = _clamp(
            hold_force * (1.45 + 0.25 * low_friction_factor),
            hold_force + 60.0,
            760.0,
        )
        tighten_step = _clamp(
            0.0012 + 0.0016 * low_friction_factor + 0.0003 * width_factor,
            0.0010,
            0.0035,
        )
        slip_distance_margin = _clamp(
            0.040 + 0.020 * low_friction_factor + 0.005 * width_factor,
            0.035,
            0.075,
        )
        close_steps = int(_clamp(190 + 120 * width_factor + 50 * low_friction_factor, 180, 320))
        grasp_clearance = _clamp(size_z * 0.032, 0.0015, 0.0060)
        if features["round_like"]:
            grasp_clearance = _clamp(grasp_clearance - 0.0007, 0.0012, 0.0050)
        grasp_z_ratio = 0.85
        if features.get("slender_like"):
            grasp_z_ratio = 0.62
        elif features.get("flat_like"):
            grasp_z_ratio = 0.85   # 0.72 → 0.85: flat 물체도 top 근처를 잡아야 함
        elif features.get("round_like") and not features.get("slender_like"):
            grasp_z_ratio = 0.92
        approach_height = _clamp(0.12 + size_z * 0.65, 0.12, 0.24)
        lift_height = _clamp(0.08 + 0.045 * mass_factor + 0.02 * width_factor, 0.08, 0.18)
        min_lift_gain = _clamp(size_z * 0.22, 0.012, 0.030)
        slip_force_boost = _clamp(6.0 + 5.0 * low_friction_factor, 5.0, 14.0)
        hold_force_decay = _clamp(0.8 + 0.4 * (1.0 - low_friction_factor), 0.6, 1.4)

        profile = DEFAULT_GRASP_PROFILE.copy()
        profile.update(
            {
                "gripper_force": round(grip_force, 3),
                "hold_force": round(hold_force, 3),
                "hold_force_max": round(hold_force_max, 3),
                "tighten_step": round(tighten_step, 6),
                "slip_distance_margin": round(slip_distance_margin, 6),
                "close_steps": close_steps,
                "min_lift_gain": round(min_lift_gain, 6),
                "approach_height": round(approach_height, 6),
                "grasp_clearance": round(grasp_clearance, 6),
                "lift_height": round(lift_height, 6),
                "grasp_z_ratio": round(grasp_z_ratio, 6),
                "slip_force_boost": round(slip_force_boost, 6),
                "hold_force_decay": round(hold_force_decay, 6),
            }
        )
        return profile

    def _activate_grasp_profile(self, body_id: int, object_label: str | None) -> dict:
        features = self._read_body_features(body_id)
        profile = self._infer_grasp_profile_from_body(
            body_id=body_id,
            features=features,
        )
        if object_label and object_label in self._grasp_profile_table:
            # Label-specific table acts as an optional fallback override.
            profile.update(self._grasp_profile_table[object_label])
        self._active_body_features = features
        self._active_grasp_profile = profile
        self._active_gripper_hold_force = profile["hold_force"]
        return profile

    def _build_retry_offsets(self, max_attempts: int) -> list[tuple[float, float]]:
        span_xy = 0.08
        if isinstance(self._active_body_features, dict):
            span_xy = float(self._active_body_features.get("span_xy", span_xy))
        jitter = _clamp(span_xy * 0.18, 0.008, 0.020)
        offsets = [
            (0.0, 0.0),
            (jitter, -jitter * 0.65),
            (-jitter, jitter * 0.65),
            (0.0, jitter),
        ]
        attempts = max(1, min(max_attempts, len(offsets)))
        return offsets[:attempts]

    def _extract_roll_pitch_from_orientation(self, orientation) -> tuple[float, float]:
        if orientation is None:
            return np.pi, 0.0
        euler = p.getEulerFromQuaternion(orientation)
        return float(euler[0]), float(euler[1])

    def _estimate_planar_grasp_geometry(self, body_id: int) -> dict:
        """Estimate top-view major axis and the gripped width of an object."""
        aabb_min, aabb_max = p.getAABB(body_id)
        dx = max(1e-4, float(aabb_max[0] - aabb_min[0]))
        dy = max(1e-4, float(aabb_max[1] - aabb_min[1]))
        _, body_orn = p.getBasePositionAndOrientation(body_id)
        pose_yaw = float(p.getEulerFromQuaternion(body_orn)[2])

        geometry = {
            "grasp_yaw": pose_yaw,
            "grasp_width": min(dx, dy),
            "long_extent": max(dx, dy),
            "aspect_ratio": max(dx, dy) / min(dx, dy),
            "source": "aabb",
        }
        if dx > dy * 1.12:
            geometry["grasp_yaw"] = 0.0
        elif dy > dx * 1.12:
            geometry["grasp_yaw"] = np.pi / 2.0

        try:
            mesh_data = p.getMeshData(
                body_id,
                -1,
                flags=p.MESH_DATA_SIMULATION_MESH,
            )
            local_vertices = mesh_data[1] if mesh_data and len(mesh_data) > 1 else []
            if len(local_vertices) < 3:
                return geometry

            body_pos, body_orn = p.getBasePositionAndOrientation(body_id)
            world_xy = []
            for vertex in local_vertices:
                world_pos, _ = p.multiplyTransforms(
                    body_pos,
                    body_orn,
                    vertex,
                    [0.0, 0.0, 0.0, 1.0],
                )
                world_xy.append([float(world_pos[0]), float(world_pos[1])])
            points = np.asarray(world_xy, dtype=float)
            centered = points - np.mean(points, axis=0)
            covariance = np.cov(centered, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            long_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
            short_axis = np.array([-long_axis[1], long_axis[0]], dtype=float)
            along_long = centered @ long_axis
            along_short = centered @ short_axis
            long_extent = max(1e-4, float(np.ptp(along_long)))
            short_extent = max(1e-4, float(np.ptp(along_short)))
            if short_extent > long_extent:
                long_axis, short_axis = short_axis, long_axis
                long_extent, short_extent = short_extent, long_extent

            aspect_ratio = long_extent / short_extent
            if aspect_ratio > 1.08:
                grasp_yaw = float(np.arctan2(long_axis[1], long_axis[0]))
            else:
                grasp_yaw = pose_yaw
            return {
                "grasp_yaw": _normalize_angle_rad(grasp_yaw),
                "grasp_width": short_extent,
                "long_extent": long_extent,
                "aspect_ratio": aspect_ratio,
                "source": "collision_mesh_pca",
            }
        except Exception:
            return geometry

    def _estimate_object_inplane_angle(self, body_id: int) -> float:
        """Return wrist yaw whose finger opening axis crosses the short axis."""
        try:
            return float(self._estimate_planar_grasp_geometry(body_id)["grasp_yaw"])
        except Exception:
            return 0.0

    def _build_grasp_yaw_candidates(
        self,
        object_angle: float,
        orientation_hint,
    ) -> list[float]:
        # Prioritize measured object geometry before optional vision yaw hints.
        base = [
            object_angle,
            object_angle + np.pi,
            object_angle + np.pi / 2.0,
            object_angle - np.pi / 2.0,
        ]

        if orientation_hint is not None:
            hinted_yaw = float(p.getEulerFromQuaternion(orientation_hint)[2])
            # Keep R1 yaw as secondary options, after axis-rule candidates.
            base.extend([
                hinted_yaw,
                hinted_yaw + np.pi / 2.0,
                hinted_yaw - np.pi / 2.0,
                hinted_yaw + np.pi,
            ])

        unique = []
        min_gap = np.deg2rad(12.0)
        for angle in base:
            norm = _normalize_angle_rad(angle)
            if all(abs(_normalize_angle_rad(norm - prev)) > min_gap for prev in unique):
                unique.append(norm)
        return unique

    def _compose_grasp_orientation(self, yaw: float, orientation_hint):
        roll, pitch = self._extract_roll_pitch_from_orientation(orientation_hint)
        return p.getQuaternionFromEuler([roll, pitch, _normalize_angle_rad(yaw)])

    def _build_object_grasp_frame(
        self,
        body_id: int,
        grasp_clearance: float,
        xy_offset: tuple[float, float] = (0.0, 0.0),
        r1_world_pos: list[float] | None = None,
    ) -> dict:
        aabb_min, aabb_max = p.getAABB(body_id)
        aabb_center_xy = np.array([
            (float(aabb_min[0]) + float(aabb_max[0])) / 2.0,
            (float(aabb_min[1]) + float(aabb_max[1])) / 2.0,
        ], dtype=float)
        size_x = max(1e-4, float(aabb_max[0] - aabb_min[0]))
        size_y = max(1e-4, float(aabb_max[1] - aabb_min[1]))
        span_xy = max(size_x, size_y)

        # R1 world_pos가 있으면 grasp seed를 R1 기준으로 사용 (AABB 강제 보정 제거)
        # AABB fallback은 r1_world_pos=None일 때만 발동
        if r1_world_pos is not None:
            r1_xy = np.array([float(r1_world_pos[0]), float(r1_world_pos[1])], dtype=float)
            base_xy = r1_xy
            print(f"[{self.name}] grasp xy from R1 world_pos: {base_xy.tolist()}")
        else:
            base_xy = aabb_center_xy
        target_xy = base_xy + np.array(xy_offset, dtype=float)
        top_z = float(aabb_max[2])
        bottom_z = float(aabb_min[2])
        size_z = max(1e-4, top_z - bottom_z)
        # approach 기준점: top_z 바로 위
        # 실제 contact z는 _descend_until_precontact가 contact 감지로 결정
        nominal_grasp_z = top_z + GRIPPER_FINGER_TIP_OFFSET + grasp_clearance
        target_grasp_z = nominal_grasp_z
        if r1_world_pos is not None and len(r1_world_pos) >= 3:
            requested_grasp_z = float(r1_world_pos[2])
            # Safety clamp:
            # keep R1 z usable but never allow deep downward targets that can drive
            # the gripper below the table/object region.
            z_low = nominal_grasp_z - 0.015
            z_high = nominal_grasp_z + 0.060
            target_grasp_z = _clamp(requested_grasp_z, z_low, z_high)
            print(
                f"[{self.name}] grasp z from R1 world_pos: {requested_grasp_z:.4f} "
                f"-> clamped {target_grasp_z:.4f} (nominal={nominal_grasp_z:.4f})"
            )
        safe_descend_floor_z = float(bottom_z + 0.006)
        return {
            "target_xy": target_xy,
            "top_z": top_z,
            "bottom_z": bottom_z,
            "size_z": size_z,
            "span_xy": span_xy,
            "target_grasp_z": target_grasp_z,
            "safe_descend_floor_z": safe_descend_floor_z,
        }

    def _compose_grasp_candidate(
        self,
        grasp_frame: dict,
        yaw: float,
        orientation_hint,
        approach_height: float,
        lift_height: float,
    ) -> dict:
        target_xy = np.array(grasp_frame["target_xy"], dtype=float)
        span_xy = float(grasp_frame["span_xy"])
        top_z = float(grasp_frame["top_z"])
        # 수정 후
        target_grasp_z = float(grasp_frame["target_grasp_z"])

        width_axis = np.array([np.cos(yaw), np.sin(yaw)], dtype=float)
        pregrasp_backoff = _clamp(0.30 * span_xy, 0.010, 0.028)
        safety_z_lift = _clamp(0.20 * grasp_frame["size_z"], 0.010, 0.030)

        pregrasp_xy = target_xy - width_axis * pregrasp_backoff
        approach = [
            float(pregrasp_xy[0]),
            float(pregrasp_xy[1]),
            float(top_z + approach_height + safety_z_lift),
        ]
        # grasp = 물체 xy 정중앙, z = target_grasp_z (top 바로 위)
        # 실제 하강은 _descend_until_precontact가 담당
        grasp = [
            float(target_xy[0]),
            float(target_xy[1]),
            float(target_grasp_z),
        ]
        lift = [float(target_xy[0]), float(target_xy[1]), float(target_grasp_z + lift_height)]

        return {
            "yaw": _normalize_angle_rad(yaw),
            "orientation": self._compose_grasp_orientation(yaw=yaw, orientation_hint=orientation_hint),
            "approach": approach,
            "grasp": grasp,
            "lift": lift,
            "size_z": float(grasp_frame["size_z"]),
            "safe_descend_floor_z": float(grasp_frame.get("safe_descend_floor_z", grasp_frame["bottom_z"] + 0.006)),
            "probe": [
                float(target_xy[0]),
                float(target_xy[1]),
                float(target_grasp_z + _clamp(0.45 * approach_height, 0.025, 0.055)),
            ],
            "pregrasp_backoff": float(pregrasp_backoff),
        }

    def _snapshot_arm_joint_positions(self) -> list[float]:
        num_joints = p.getNumJoints(self.panda_id)
        return [float(p.getJointState(self.panda_id, joint_idx)[0]) for joint_idx in range(num_joints)]

    def _restore_arm_joint_positions(self, joint_positions: list[float]) -> None:
        for joint_idx, joint_pos in enumerate(joint_positions):
            p.resetJointState(self.panda_id, joint_idx, joint_pos)

    def _preview_set_arm_pose(self, position, orientation) -> bool:
        joint_poses = p.calculateInverseKinematics(
            self.panda_id,
            END_EFFECTOR_INDEX,
            position,
            targetOrientation=orientation,
            lowerLimits=PANDA_LOWER,
            upperLimits=PANDA_UPPER,
            jointRanges=PANDA_RANGE,
            restPoses=PANDA_HOME_JOINTS,
            maxNumIterations=240,
            residualThreshold=1e-4,
        )
        if joint_poses is None or len(joint_poses) < NUM_JOINTS:
            return False
        for joint_idx in range(NUM_JOINTS):
            p.resetJointState(self.panda_id, joint_idx, float(joint_poses[joint_idx]))
        return True

    def _count_nonfinger_object_contacts(self, body_id: int) -> int:
        contacts = p.getContactPoints(bodyA=self.panda_id, bodyB=body_id)
        blocked = 0
        for contact in contacts:
            link_a = int(contact[3])
            if link_a in PANDA_HAND_LINK_INDICES:
                continue
            blocked += 1
        if self.gripper_id is not None:
            for contact in p.getContactPoints(bodyA=self.gripper_id, bodyB=body_id):
                link_a = int(contact[3])
                if link_a in ROBOTIQ_GRIPPER_CONTACT_LINK_INDICES:
                    continue
                blocked += 1
        return blocked

    def _compute_nonfinger_min_distance(
        self,
        body_id: int,
        max_distance: float = 0.08,
    ) -> float:
        min_distance = float(max_distance)
        num_links = p.getNumJoints(self.panda_id)
        for link_idx in range(-1, num_links):
            if link_idx in PANDA_HAND_LINK_INDICES:
                continue
            points = p.getClosestPoints(
                bodyA=self.panda_id,
                bodyB=body_id,
                distance=max_distance,
                linkIndexA=link_idx,
            )
            for pt in points:
                min_distance = min(min_distance, float(pt[8]))
        if self.gripper_id is not None:
            num_gripper_links = p.getNumJoints(self.gripper_id)
            for link_idx in range(-1, num_gripper_links):
                if link_idx in ROBOTIQ_GRIPPER_CONTACT_LINK_INDICES:
                    continue
                points = p.getClosestPoints(
                    bodyA=self.gripper_id,
                    bodyB=body_id,
                    distance=max_distance,
                    linkIndexA=link_idx,
                )
                for pt in points:
                    min_distance = min(min_distance, float(pt[8]))
        return float(min_distance)

    def _evaluate_grasp_candidate_collision(self, body_id: int, candidate: dict) -> dict:
        # Fast, non-animated preview to avoid repeating full approach motions per candidate.
        saved_joints = self._snapshot_arm_joint_positions()
        ik_ok = self._preview_set_arm_pose(
            position=candidate["probe"],
            orientation=candidate["orientation"],
        )
        if ik_ok:
            p.stepSimulation()
            blocked_contacts = self._count_nonfinger_object_contacts(body_id=body_id)
            min_distance = self._compute_nonfinger_min_distance(body_id=body_id, max_distance=0.08)
        else:
            blocked_contacts = 999
            min_distance = 0.0

        self._restore_arm_joint_positions(saved_joints)
        p.stepSimulation()

        min_safe_clearance = _clamp(0.25 * float(candidate["pregrasp_backoff"]), 0.004, 0.012)
        collision_risk = (not ik_ok) or blocked_contacts > 0 or min_distance < min_safe_clearance
        return {
            "ik_ok": bool(ik_ok),
            "blocked_contacts": int(blocked_contacts),
            "min_distance": float(min_distance),
            "min_safe_clearance": float(min_safe_clearance),
            "collision_risk": bool(collision_risk),
        }

    def _select_grasp_candidate(
        self,
        body_id: int,
        orientation_hint,
        approach_height: float,
        grasp_clearance: float,
        lift_height: float,
        xy_offset: tuple[float, float] = (0.0, 0.0),
        r1_world_pos: list[float] | None = None,
    ) -> dict:
        grasp_frame = self._build_object_grasp_frame(
            body_id=body_id,
            grasp_clearance=grasp_clearance,
            xy_offset=xy_offset,
            r1_world_pos=r1_world_pos,
        )
        geometry = self._estimate_planar_grasp_geometry(body_id=body_id)
        object_angle = float(geometry["grasp_yaw"])
        grasp_frame["grasp_width"] = float(geometry["grasp_width"])
        grasp_frame["geometry_source"] = str(geometry["source"])
        hinted_yaw = None
        if orientation_hint is not None:
            hinted_yaw = float(p.getEulerFromQuaternion(orientation_hint)[2])
        yaw_candidates = self._build_grasp_yaw_candidates(
            object_angle=object_angle,
            orientation_hint=orientation_hint,
        )
        if hinted_yaw is not None:
            print(
                f"[{self.name}] seed grasp angle={np.degrees(object_angle):.1f} deg "
                f"({geometry['source']}, width={geometry['grasp_width']:.4f}m), "
                f"r1_hint_yaw={np.degrees(hinted_yaw):.1f} deg"
            )
        else:
            print(
                f"[{self.name}] seed grasp angle={np.degrees(object_angle):.1f} deg "
                f"({geometry['source']}, width={geometry['grasp_width']:.4f}m)"
            )

        best = None
        best_rank = None
        for idx, yaw in enumerate(yaw_candidates):
            candidate = self._compose_grasp_candidate(
                grasp_frame=grasp_frame,
                yaw=yaw,
                orientation_hint=orientation_hint,
                approach_height=approach_height,
                lift_height=lift_height,
            )
            collision_eval = self._evaluate_grasp_candidate_collision(
                body_id=body_id,
                candidate=candidate,
            )
            candidate.update(collision_eval)
            print(
                f"[{self.name}] yaw-candidate[{idx}]={np.degrees(candidate['yaw']):.1f} deg "
                f"ik_ok={candidate.get('ik_ok')} "
                f"collision={candidate['collision_risk']} "
                f"contacts={candidate['blocked_contacts']} "
                f"clearance={candidate['min_distance']:.4f}"
            )

            rank = (
                1 if candidate["collision_risk"] else 0,
                0 if candidate.get("ik_ok", False) else 1,
                candidate["blocked_contacts"],
                -candidate["min_distance"],
                idx,
            )
            if best is None or rank < best_rank:
                best = candidate
                best_rank = rank

        if best is None:
            # Extremely defensive fallback; should never happen.
            fallback = self._compose_grasp_candidate(
                grasp_frame=grasp_frame,
                yaw=object_angle,
                orientation_hint=orientation_hint,
                approach_height=approach_height,
                lift_height=lift_height,
            )
            fallback.update(
                {
                    "blocked_contacts": 0,
                    "min_distance": 0.0,
                    "min_safe_clearance": 0.0,
                    "collision_risk": False,
                }
            )
            best = fallback

        print(
            f"[{self.name}] selected grasp yaw={np.degrees(best['yaw']):.1f} deg "
            f"(collision={best['collision_risk']}, contacts={best['blocked_contacts']})"
        )
        return best

    def configure_gripper_contact_dynamics(
        self,
        lateral_friction: float = RUBBER_PAD_LATERAL_FRICTION,
        rolling_friction: float = RUBBER_PAD_ROLLING_FRICTION,
        spinning_friction: float = RUBBER_PAD_SPINNING_FRICTION,
        restitution: float = 0.0,
        contact_stiffness: float = RUBBER_PAD_CONTACT_STIFFNESS,
        contact_damping: float = RUBBER_PAD_CONTACT_DAMPING,
    ) -> None:
        self._ensure_loaded()
        if self.gripper_id is None:
            return
        for finger_link in ROBOTIQ_GRIPPER_CONTACT_LINK_INDICES:
            p.changeDynamics(
                self.gripper_id,
                finger_link,
                lateralFriction=lateral_friction,
                rollingFriction=rolling_friction,
                spinningFriction=spinning_friction,
                restitution=restitution,
                contactStiffness=contact_stiffness,
                contactDamping=contact_damping,
            )

    def _set_gripper_target(
        self,
        target_position: float,
        force: float,
        max_velocity: float = 0.25,
    ) -> None:
        if self.gripper_id is None:
            return
        for finger_joint in ROBOTIQ_GRIPPER_JOINT_INDICES:
            p.setJointMotorControl2(
                self.gripper_id,
                finger_joint,
                p.POSITION_CONTROL,
                targetPosition=target_position,
                force=force,
                maxVelocity=max_velocity,
            )

    def _tick_gripper_hold(self) -> None:
        """
        hold feedback + gripper 명령을 1스텝 실행한다.
        외부 루프(다른 팔 동작 중)에서 호출해 hold를 유지할 때 사용.
        """
        if self._active_gripper_hold_target is None:
            return
        self._update_grasp_hold_feedback()
        self._set_gripper_target(
            target_position=self._active_gripper_hold_target,
            force=self._active_gripper_hold_force,
            max_velocity=0.10,
        )

    def _step_with_gripper_hold(
        self,
        target_position: float,
        force: float,
        steps: int,
        max_velocity: float = 0.25,
        hold_companion: "PandaController | None" = None,
    ) -> None:
        for _ in range(steps):
            self._update_grasp_hold_feedback()
            self._set_gripper_target(
                target_position=self._active_gripper_hold_target
                if self._active_gripper_hold_target is not None
                else target_position,
                force=self._active_gripper_hold_force
                if self._active_gripper_hold_target is not None
                else force,
                max_velocity=max_velocity,
            )
            if hold_companion is not None:
                hold_companion._tick_gripper_hold()
            p.stepSimulation()
            time.sleep(SIM_TIMESTEP)

    def _update_grasp_hold_feedback(self) -> None:
        """
        Keep grasp stable after pickup:
        - detect slip from contact loss / EE-object distance growth
        - tighten target opening and raise hold force when needed
        """
        if self._held_body_id is None or self._active_gripper_hold_target is None:
            return

        summary = self._finger_contact_summary(self._held_body_id)
        ee_pos, _ = self.get_end_effector_pose()
        obj_pos = np.array(p.getBasePositionAndOrientation(self._held_body_id)[0])
        ee_to_object = float(np.linalg.norm(obj_pos - ee_pos))

        if self._held_reference_ee_distance is None:
            self._held_reference_ee_distance = ee_to_object

        slipping = (
            summary["total"] <= 0
            or ee_to_object
            > self._held_reference_ee_distance + self._active_grasp_profile["slip_distance_margin"]
        )
        if slipping:
            self._hold_slip_events += 1
            self._active_gripper_hold_target = max(
                0.0,
                self._active_gripper_hold_target - self._active_grasp_profile["tighten_step"],
            )
            self._active_gripper_hold_force = min(
                self._active_grasp_profile["hold_force_max"],
                self._active_gripper_hold_force + self._active_grasp_profile["slip_force_boost"],
            )
            return

        self._active_gripper_hold_force = max(
            self._active_grasp_profile["hold_force"],
            self._active_gripper_hold_force - self._active_grasp_profile["hold_force_decay"],
        )
        self._held_reference_ee_distance = (
            0.9 * self._held_reference_ee_distance + 0.1 * ee_to_object
        )

    def _finger_contact_summary(self, body_id: int) -> dict:
        if self.gripper_id is None or body_id is None:
            return {"left": 0, "right": 0, "hand": 0, "total": 0}
        left = len(
            p.getContactPoints(
                bodyA=self.gripper_id,
                bodyB=body_id,
                linkIndexA=ROBOTIQ_GRIPPER_CONTACT_LINK_INDICES[0],
            ) or []
        )
        right = len(
            p.getContactPoints(
                bodyA=self.gripper_id,
                bodyB=body_id,
                linkIndexA=ROBOTIQ_GRIPPER_CONTACT_LINK_INDICES[1],
            ) or []
        )
        hand = len(
            p.getContactPoints(
                bodyA=self.gripper_id,
                bodyB=body_id,
                linkIndexA=-1,
            ) or []
        )
        return {
            "left": left,
            "right": right,
            "hand": hand,
            "total": left + right + hand,
        }

    def _finger_contact_count(self, summary: dict) -> int:
        return int(summary.get("left", 0)) + int(summary.get("right", 0))

    def _close_until_contact(
        self,
        body_id: int,
        start_open: float = GRIPPER_MAX_TARGET,
        end_closed: float = GRIPPER_MIN_TARGET,
        steps: int | None = None,
        close_force: float | None = None,
        hold_companion: "PandaController | None" = None,
    ) -> tuple[bool, float, dict]:
        if steps is None:
            steps = int(self._active_grasp_profile["close_steps"])
        if close_force is None:
            close_force = float(self._active_grasp_profile["hold_force"])

        for step_idx in range(steps):
            ratio = (step_idx + 1) / float(steps)
            target = start_open + (end_closed - start_open) * ratio
            self._step_with_gripper_hold(
                target_position=target,
                force=close_force,
                steps=1,
                max_velocity=0.20,
                hold_companion=hold_companion,
            )
            summary = self._finger_contact_summary(body_id)
            finger_contacts = self._finger_contact_count(summary)
            bilateral_contact = summary["left"] > 0 and summary["right"] > 0
            # Hand-only contact is not treated as valid grasp contact.
            if bilateral_contact or finger_contacts >= 2:
                return True, target, summary

        summary = self._finger_contact_summary(body_id)
        return self._finger_contact_count(summary) > 0, end_closed, summary

    # 수정 후
    def _descend_until_precontact(
        self,
        body_id: int,
        orientation,
        drop_step: float = 0.004,
        max_drop: float = 0.25,
        min_ee_z: float | None = None,
        hold_companion: "PandaController | None" = None,
    ) -> None:
        checks = int(max_drop / drop_step)
        for _ in range(checks):
            summary = self._finger_contact_summary(body_id)
            if self._finger_contact_count(summary) > 0:
                return
            ee_pos, _ = self.get_end_effector_pose()
            next_z = float(ee_pos[2] - drop_step)
            if min_ee_z is not None and next_z <= float(min_ee_z):
                print(
                    f"[{self.name}] descend stop at safety floor "
                    f"(next_z={next_z:.4f} <= min_ee_z={float(min_ee_z):.4f})"
                )
                return
            next_pos = [float(ee_pos[0]), float(ee_pos[1]), next_z]
            self.move_end_effector_to(next_pos, orientation=orientation, steps=50, hold_companion=hold_companion)

    def get_end_effector_pose(self) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_loaded()
        link_state = p.getLinkState(
            self.panda_id,
            END_EFFECTOR_INDEX,
            computeForwardKinematics=True,
        )
        return np.array(link_state[0]), np.array(link_state[1])

    def _enable_hold_control(
        self,
        body_id: int,
        hold_target: float,
        hold_force: float | None = None,
        reference_ee_distance: float | None = None,
    ) -> None:
        if hold_force is None:
            hold_force = self._active_grasp_profile["hold_force"]
        self._held_body_id = body_id
        self._active_gripper_hold_target = max(0.0, hold_target)
        self._active_gripper_hold_force = hold_force
        self._held_reference_ee_distance = reference_ee_distance
        self._hold_slip_events = 0

    def _clear_hold_control(self) -> None:
        self._held_body_id = None
        self._held_reference_ee_distance = None
        self._active_gripper_hold_target = None
        self._active_gripper_hold_force = self._active_grasp_profile["hold_force"]

    def maintain_grasp_hold(self, steps: int = 120) -> None:
        """
        Run hold control while keeping the current pose.
        Useful right after lift or after transport before release.
        """
        self._ensure_loaded()
        if self._active_gripper_hold_target is None:
            return
        self._step_with_gripper_hold(
            target_position=self._active_gripper_hold_target,
            force=self._active_gripper_hold_force,
            steps=steps,
            max_velocity=0.10,
        )

    def _grasp_body_no_constraint(
        self,
        body_id: int,
        orientation,
        approach_height: float,
        grasp_clearance: float,
        lift_height: float,
        xy_offset: tuple[float, float] = (0.0, 0.0),
        hold_companion: "PandaController | None" = None,
        r1_world_pos: list[float] | None = None,
    ) -> bool:
        start_obj_pos = np.array(p.getBasePositionAndOrientation(body_id)[0])
        self._last_grasp_failure = None

        # Set opening from the same width axis used for wrist yaw selection.
        try:
            _geometry = self._estimate_planar_grasp_geometry(body_id)
            _obj_width = float(_geometry["grasp_width"])
            desired_opening = _clamp(
                _obj_width + 0.014,
                0.025,
                ROBOTIQ_2F140_MAX_OPENING,
            )
            _finger_open = _opening_to_joint_target(desired_opening)
            print(
                f"[{self.name}] dynamic Robotiq 2F-140 opening={desired_opening:.4f}m "
                f"(surrogate_joint={_finger_open:.4f}) "
                f"for object width={_obj_width:.4f}m ({_geometry['source']})"
            )
        except Exception:
            _finger_open = GRIPPER_MAX_TARGET
        self._set_gripper_target(
            target_position=_finger_open,
            force=GRIPPER_FORCE,
            max_velocity=0.25,
        )
        _step_simulation(90)
        candidate = self._select_grasp_candidate(
            body_id=body_id,
            orientation_hint=orientation,
            approach_height=approach_height,
            grasp_clearance=grasp_clearance,
            lift_height=lift_height,
            xy_offset=xy_offset,
            r1_world_pos=r1_world_pos,
        )

        selected_orientation = candidate["orientation"]
        approach = candidate["approach"]
        grasp = candidate["grasp"]
        lift = candidate["lift"]
        size_z = float(candidate.get("size_z", 0.10))
        safe_descend_floor_z = float(candidate.get("safe_descend_floor_z", 0.0))

        print(f"[{self.name}] grasp coords: approach={[round(v,4) for v in approach]}, grasp={[round(v,4) for v in grasp]}")
        aabb_min_dbg, aabb_max_dbg = p.getAABB(body_id)
        print(f"[{self.name}] object AABB: min={[round(v,4) for v in aabb_min_dbg]}, max={[round(v,4) for v in aabb_max_dbg]}")
        # 수정 후
        self.move_end_effector_to(approach, orientation=selected_orientation, steps=420, hold_companion=hold_companion)
        self.move_end_effector_to(grasp, orientation=selected_orientation, steps=360, hold_companion=hold_companion)
        ee_pos_dbg, _ = self.get_end_effector_pose()
        print(f"[{self.name}] EE after grasp move: {[round(v,4) for v in ee_pos_dbg]}")
        # grasp 위치 도달 직후 이미 contact인지 확인 — 있으면 descend 생략
        _pre_summary = self._finger_contact_summary(body_id)
        if self._finger_contact_count(_pre_summary) == 0:
            self._descend_until_precontact(
                body_id=body_id,
                orientation=selected_orientation,
                min_ee_z=safe_descend_floor_z,
                hold_companion=hold_companion,
            )
        ee_pos_dbg2, _ = self.get_end_effector_pose()
        print(f"[{self.name}] EE after descend: {[round(v,4) for v in ee_pos_dbg2]}")

        # _finger_open: AABB 기반 계산값을 start_open으로 전달해
        # _close_until_contact가 다시 0.04로 벌리지 않도록 함
        contact_found, hold_target, summary = self._close_until_contact(
            body_id=body_id, start_open=_finger_open, hold_companion=hold_companion)
        if not contact_found:
            extra_drop = _clamp(0.18 * size_z, 0.008, 0.025)
            fallback_z = max(float(grasp[2] - extra_drop), safe_descend_floor_z)
            fallback_grasp = [grasp[0], grasp[1], fallback_z]
            self.move_end_effector_to(fallback_grasp, orientation=selected_orientation, steps=140, hold_companion=hold_companion)
            contact_found, hold_target, summary = self._close_until_contact(
                body_id=body_id, start_open=_finger_open, hold_companion=hold_companion)
        if not contact_found:
            print(
                f"[{self.name}] no-contact close "
                f"(finger_contacts={self._finger_contact_count(summary)}, total_contacts={summary['total']})"
            )
            end_obj_pos = np.array(p.getBasePositionAndOrientation(body_id)[0])
            self._last_grasp_failure = {
                "reason": "no_contact_close",
                "contacts": int(self._finger_contact_count(summary)),
                "ee_dist": None,
                "object_shift": float(np.linalg.norm(end_obj_pos - start_obj_pos)),
            }
            self._clear_hold_control()
            return False

        hold_target = max(0.0, hold_target - 0.007)
        self._enable_hold_control(
            body_id=body_id,
            hold_target=hold_target,
            hold_force=self._active_grasp_profile["hold_force"],
        )
        self._step_with_gripper_hold(
            target_position=hold_target,
            force=self._active_grasp_profile["hold_force"],
            steps=120,
            max_velocity=0.12,
            hold_companion=hold_companion,
        )

        before_lift_pos = np.array(p.getBasePositionAndOrientation(body_id)[0])
        self.move_end_effector_to(lift, orientation=selected_orientation, steps=520, hold_companion=hold_companion)
        self._step_with_gripper_hold(
            target_position=hold_target,
            force=self._active_grasp_profile["hold_force"],
            steps=80,
            max_velocity=0.10,
            hold_companion=hold_companion,
        )

        after_lift_pos = np.array(p.getBasePositionAndOrientation(body_id)[0])
        ee_pos, _ = self.get_end_effector_pose()
        ee_to_object = float(np.linalg.norm(after_lift_pos - ee_pos))
        summary_after = self._finger_contact_summary(body_id)
        finger_contacts_after = self._finger_contact_count(summary_after)

        lifted = after_lift_pos[2] > before_lift_pos[2] + self._active_grasp_profile["min_lift_gain"]
        maintained = finger_contacts_after > 0 or ee_to_object < 0.14
        if not (lifted and maintained):
            print(
                f"[{self.name}] no-constraint grasp unstable "
                f"(lifted={lifted}, finger_contacts={finger_contacts_after}, "
                f"total_contacts={summary_after['total']}, ee_dist={ee_to_object:.3f})"
            )
            self._last_grasp_failure = {
                "reason": "unstable",
                "lifted": bool(lifted),
                "contacts": int(finger_contacts_after),
                "ee_dist": float(ee_to_object),
                "object_shift": float(np.linalg.norm(after_lift_pos - start_obj_pos)),
            }
            self._clear_hold_control()
            return False
        self._held_reference_ee_distance = ee_to_object
        self._last_grasp_failure = {"reason": "success"}
        return True

    def _should_retry_after_failure(self, attempt_idx: int, attempts: int) -> bool:
        if attempt_idx >= attempts - 1:
            return False
        info = self._last_grasp_failure
        if not isinstance(info, dict):
            return True

        reason = info.get("reason")
        contacts = int(info.get("contacts", 0) or 0)
        object_shift = float(info.get("object_shift", 0.0) or 0.0)
        ee_dist = info.get("ee_dist")
        ee_dist = None if ee_dist is None else float(ee_dist)

        # Retry only when there was meaningful interaction/progress.
        if reason == "no_contact_close" and contacts == 0:
            return False
        if reason == "unstable":
            if contacts == 0 and object_shift < 0.008 and (ee_dist is not None and ee_dist > 0.22):
                return False
        return True

    def release_grasp(self, open_after: bool = True, steps: int = 120) -> None:
        self._ensure_loaded()
        self._clear_hold_control()
        if open_after:
            self.open_gripper(steps=steps)

    def grasp_body(
        self,
        body_id: int,
        object_label: str | None = None,
        orientation=None,
        approach_height: float | None = None,
        grasp_clearance: float | None = None,
        lift_height: float | None = None,
        use_constraint: bool = False,
        max_attempts: int = 3,
        hold_companion: "PandaController | None" = None,
        r1_world_pos: list[float] | None = None,
    ) -> bool:
        """
        Contact-based grasp primitive.
        hold_companion: 이 팔이 grasp하는 동안 hold를 유지해야 하는 다른 PandaController.
        """
        self._ensure_loaded()
        profile = self._activate_grasp_profile(
            body_id=body_id,
            object_label=object_label,
        )

        # Mandatory mode: use only R1 planner grasp pose.
        # JSON/PyBullet object pose fallback is intentionally disabled.
        if orientation is None or len(orientation) < 4:
            raise ValueError(
                f"[{self.name}] R1 grasp orientation is required; "
                "PyBullet object-state fallback is disabled."
            )
        if r1_world_pos is None or len(r1_world_pos) < 3:
            raise ValueError(
                f"[{self.name}] R1 grasp world_pos is required; "
                "PyBullet object-state fallback is disabled."
            )
        orientation = [float(v) for v in orientation[:4]]
        r1_world_pos = [float(v) for v in r1_world_pos[:3]]
        print(
            f"[{self.name}] mandatory R1 grasp pose: "
            f"world_pos={[round(v,4) for v in r1_world_pos]}, "
            f"orientation={[round(v,4) for v in orientation]}"
        )
        if use_constraint:
            print(f"[{self.name}] use_constraint=True is deprecated; using contact-based grasp.")

        if approach_height is None:
            approach_height = float(profile["approach_height"])
        if grasp_clearance is None:
            grasp_clearance = float(profile["grasp_clearance"])
        if lift_height is None:
            lift_height = float(profile["lift_height"])
        profile_name = (
            f"{object_label}+feature"
            if object_label in self._grasp_profile_table
            else "feature-inferred"
        )
        feature_mass = None
        feature_friction = None
        if isinstance(self._active_body_features, dict):
            feature_mass = self._active_body_features.get("mass")
            feature_friction = self._active_body_features.get("lateral_friction")
        print(
            f"[{self.name}] grasp profile={profile_name} "
            f"(grip={profile['gripper_force']}, hold={profile['hold_force']}, "
            f"hold_max={profile['hold_force_max']}, "
            f"mass={feature_mass}, friction={feature_friction})"
        )

        retry_offsets = self._build_retry_offsets(max_attempts=max_attempts)
        attempts = len(retry_offsets)
        for attempt_idx in range(attempts):
            # Keep using the R1 grasp pose across retries.
            _r1_pos = r1_world_pos
            ok = self._grasp_body_no_constraint(
                body_id=body_id,
                orientation=orientation,
                approach_height=approach_height,
                grasp_clearance=grasp_clearance,
                lift_height=lift_height,
                xy_offset=retry_offsets[attempt_idx],
                hold_companion=hold_companion,
                r1_world_pos=_r1_pos,
            )
            if ok:
                return True
            if attempt_idx < attempts - 1:
                if not self._should_retry_after_failure(attempt_idx=attempt_idx, attempts=attempts):
                    print(f"[{self.name}] retry skipped (no progress from previous attempt)")
                    break
                self.release_grasp(open_after=True, steps=70)
                self.move_end_effector_to(
                    [self.base_position[0] + 0.55, self.base_position[1], 1.02],
                    orientation=orientation,
                    steps=260,
                    hold_companion=hold_companion,
                )
        return False


PandaController = Robotiq2F140Controller