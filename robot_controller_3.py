"""
robot_controller_3.py
Lightweight Panda controller module for simulation boot/demo.

Notes:
- This module keeps dependencies minimal on purpose:
  pybullet, numpy, time
- Affordance-R1 model weights are expected to come from:
  https://huggingface.co/hqking/affordance-r1
"""

import time

import numpy as np
import pybullet as p


END_EFFECTOR_INDEX = 11
NUM_JOINTS = 7
GRIPPER_JOINT_INDICES = (9, 10)

PANDA_HOME_JOINTS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

PANDA_LOWER = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
PANDA_UPPER = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
PANDA_RANGE = [5.7946, 3.5256, 5.7946, 3.0020, 5.7946, 3.7700, 5.7946]

ARM_FORCE = 500
ARM_MAX_VELOCITY = 1.0
GRIPPER_FORCE = 160
GRIPPER_HOLD_FORCE = 320
GRIPPER_HOLD_FORCE_MAX = 520
HOLD_TIGHTEN_STEP = 0.0018
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


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _step_simulation(steps: int) -> None:
    for _ in range(steps):
        p.stepSimulation()
        time.sleep(SIM_TIMESTEP)


def load_panda(base_position, use_fixed_base: bool = True) -> int:
    return p.loadURDF(
        "franka_panda/panda.urdf",
        useFixedBase=use_fixed_base,
        basePosition=base_position,
    )


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
    for finger_joint in GRIPPER_JOINT_INDICES:
        p.setJointMotorControl2(
            panda_id,
            finger_joint,
            p.POSITION_CONTROL,
            targetPosition=0.04,
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


class PandaController:
    def __init__(self, name: str, base_position, use_fixed_base: bool = True):
        self.name = name
        self.base_position = list(base_position)
        self.use_fixed_base = use_fixed_base
        self.panda_id = None
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

    def _ensure_loaded(self) -> None:
        if self.panda_id is None:
            raise RuntimeError(f"[{self.name}] panda is not loaded yet.")

    def load_panda(self):
        self.panda_id = load_panda(
            base_position=self.base_position,
            use_fixed_base=self.use_fixed_base,
        )
        self.configure_gripper_contact_dynamics()
        return self.panda_id

    def reset_to_home(self, steps=800):
        self._ensure_loaded()
        reset_to_home(self.panda_id, steps=steps)

    def move_end_effector_to(self, position, orientation=None, steps=1000):
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
            p.stepSimulation()
            time.sleep(SIM_TIMESTEP)

    def open_gripper(self, steps: int = 100):
        self._ensure_loaded()
        self._clear_hold_control()
        open_gripper(self.panda_id, steps=steps)

    def close_gripper(self, steps: int = 100):
        self._ensure_loaded()
        self._active_gripper_hold_target = 0.0
        self._active_gripper_hold_force = self._active_grasp_profile["gripper_force"]
        close_gripper(self.panda_id, steps=steps)

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
        if features["round_like"]:
            grasp_z_ratio = 0.62
        elif features["slender_like"]:
            grasp_z_ratio = 0.82
        elif features["flat_like"]:
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

    def configure_gripper_contact_dynamics(
        self,
        lateral_friction: float = 1.6,
        rolling_friction: float = 0.003,
        spinning_friction: float = 0.003,
        restitution: float = 0.0,
    ) -> None:
        self._ensure_loaded()
        for finger_joint in GRIPPER_JOINT_INDICES:
            p.changeDynamics(
                self.panda_id,
                finger_joint,
                lateralFriction=lateral_friction,
                rollingFriction=rolling_friction,
                spinningFriction=spinning_friction,
                restitution=restitution,
            )

    def _set_gripper_target(
        self,
        target_position: float,
        force: float,
        max_velocity: float = 0.25,
    ) -> None:
        for finger_joint in GRIPPER_JOINT_INDICES:
            p.setJointMotorControl2(
                self.panda_id,
                finger_joint,
                p.POSITION_CONTROL,
                targetPosition=target_position,
                force=force,
                maxVelocity=max_velocity,
            )

    def _step_with_gripper_hold(
        self,
        target_position: float,
        force: float,
        steps: int,
        max_velocity: float = 0.25,
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
        left = len(
            p.getContactPoints(
                bodyA=self.panda_id,
                bodyB=body_id,
                linkIndexA=GRIPPER_JOINT_INDICES[0],
            )
        )
        right = len(
            p.getContactPoints(
                bodyA=self.panda_id,
                bodyB=body_id,
                linkIndexA=GRIPPER_JOINT_INDICES[1],
            )
        )
        hand = len(
            p.getContactPoints(
                bodyA=self.panda_id,
                bodyB=body_id,
                linkIndexA=END_EFFECTOR_INDEX,
            )
        )
        return {
            "left": left,
            "right": right,
            "hand": hand,
            "total": left + right + hand,
        }

    def _close_until_contact(
        self,
        body_id: int,
        start_open: float = 0.04,
        end_closed: float = 0.0,
        steps: int | None = None,
        close_force: float | None = None,
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
            )
            summary = self._finger_contact_summary(body_id)
            bilateral_contact = summary["left"] > 0 and summary["right"] > 0
            if bilateral_contact or summary["total"] >= 2:
                return True, target, summary

        summary = self._finger_contact_summary(body_id)
        return summary["total"] > 0, end_closed, summary
# 수정 후
    def _descend_until_precontact(
        self,
        body_id: int,
        orientation,
        drop_step: float = 0.010,    # 0.015 → 0.010: 더 정밀하게
        max_drop: float = 0.30,      # 0.18 → 0.30: 더 넓은 탐색 범위
    ) -> None:
        checks = int(max_drop / drop_step)
        for _ in range(checks):
            summary = self._finger_contact_summary(body_id)
            if summary["total"] > 0:
                return
            ee_pos, _ = self.get_end_effector_pose()
            # 물체 AABB bottom보다 아래로는 내려가지 않도록 안전 제한
            aabb_min, _ = p.getAABB(body_id)
            safe_floor = float(aabb_min[2]) - 0.02
            next_z = float(ee_pos[2]) - drop_step
            if next_z < safe_floor:
                return
            next_pos = [float(ee_pos[0]), float(ee_pos[1]), next_z]
            self.move_end_effector_to(next_pos, orientation=orientation, steps=60)

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
    def _compute_grasp_orientation(self, body_id: int, base_orientation) -> tuple:
        """
        물체의 AABB 단면 긴 축에 그리퍼 yaw를 정렬한다.
        - 단면이 원형에 가까우면 base_orientation 그대로 반환
        - 타원형/직사각형이면 긴 축 방향으로 yaw 회전
        """
        try:
            aabb_min, aabb_max = p.getAABB(body_id)
            size_x = float(aabb_max[0] - aabb_min[0])
            size_y = float(aabb_max[1] - aabb_min[1])
            # 단면이 거의 원형이면 회전 불필요
            ratio = min(size_x, size_y) / max(size_x, size_y, 1e-6)
            if ratio > 0.88:
                return base_orientation, 0.0

            # 긴 축 방향 yaw 계산 (그리퍼 핑거가 긴 축에 수직이 되도록)
            if size_x >= size_y:
                yaw = 0.0        # 긴 축이 X축 → 핑거를 Y 방향으로
            else:
                yaw = np.pi / 2  # 긴 축이 Y축 → 핑거를 X 방향으로

            base_euler = p.getEulerFromQuaternion(base_orientation)
            aligned_orn = p.getQuaternionFromEuler(
                [base_euler[0], base_euler[1], base_euler[2] + yaw]
            )
            return aligned_orn, yaw
        except Exception:
            return base_orientation, 0.0
    
    def _grasp_body_no_constraint(
        self,
        body_id: int,
        orientation,
        approach_height: float,
        grasp_clearance: float,
        lift_height: float,
        xy_offset: tuple[float, float] = (0.0, 0.0),
    ) -> bool:
        aabb_min, aabb_max = p.getAABB(body_id)
        body_pos, _ = p.getBasePositionAndOrientation(body_id)
        target_xy = np.array(body_pos[:2], dtype=float)
        target_xy += np.array(xy_offset, dtype=float)
        top_z = float(aabb_max[2])
        bottom_z = float(aabb_min[2])
        size_z = max(1e-4, top_z - bottom_z)
        grasp_z_ratio = float(self._active_grasp_profile.get("grasp_z_ratio", 0.85))
        grasp_z_ratio = _clamp(grasp_z_ratio, 0.45, 0.98)
        target_grasp_z = bottom_z + size_z * grasp_z_ratio

        # ── 물체 단면 긴 축에 그리퍼 yaw 정렬 ──────────────────────────────
        aligned_orn, yaw_applied = self._compute_grasp_orientation(body_id, orientation)
        if yaw_applied != 0.0:
            print(f"[{self.name}] grasp orientation aligned: yaw={np.degrees(yaw_applied):.1f}°")

        # approach는 물체 정상보다 충분히 위 → 내려오면서 물체를 밀지 않도록
        approach_z = top_z + approach_height
        approach = [target_xy[0], target_xy[1], approach_z]
        grasp = [target_xy[0], target_xy[1], target_grasp_z + grasp_clearance]
        lift = [target_xy[0], target_xy[1], grasp[2] + lift_height]

        self.open_gripper(steps=90)
        self.move_end_effector_to(approach, orientation=aligned_orn, steps=500)
        self.move_end_effector_to(grasp, orientation=aligned_orn, steps=360)
        self._descend_until_precontact(body_id=body_id, orientation=aligned_orn)

        contact_found, hold_target, summary = self._close_until_contact(body_id=body_id)
        if not contact_found:
            print(f"[{self.name}] no-contact close (contacts={summary['total']})")
            self._clear_hold_control()
            return False

        hold_target = max(0.0, hold_target - 0.003)
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
        )

        before_lift_pos = np.array(p.getBasePositionAndOrientation(body_id)[0])
        self.move_end_effector_to(lift, orientation=orientation, steps=520)
        self._step_with_gripper_hold(
            target_position=hold_target,
            force=self._active_grasp_profile["hold_force"],
            steps=80,
            max_velocity=0.10,
        )

        after_lift_pos = np.array(p.getBasePositionAndOrientation(body_id)[0])
        ee_pos, _ = self.get_end_effector_pose()
        ee_to_object = float(np.linalg.norm(after_lift_pos - ee_pos))
        summary_after = self._finger_contact_summary(body_id)

        lifted = after_lift_pos[2] > before_lift_pos[2] + self._active_grasp_profile["min_lift_gain"]
        maintained = summary_after["total"] > 0 or ee_to_object < 0.14
        if not (lifted and maintained):
            print(
                f"[{self.name}] no-constraint grasp unstable "
                f"(lifted={lifted}, contacts={summary_after['total']}, ee_dist={ee_to_object:.3f})"
            )
            self._clear_hold_control()
            return False
        self._held_reference_ee_distance = ee_to_object
        return True


    def release_grasp(self, open_after: bool = True, steps: int = 120) -> None:
        self._ensure_loaded()
        self._clear_hold_control()
        if open_after:
            self.open_gripper(steps=steps)

    # ── Assembly attach / detach ──────────────────────────────────────────────

    def attach_to_body(
        self,
        main_body_id: int,
        aux_body_id: int,
        contact_offset: list | None = None,
        settle_steps: int = 60,
    ) -> int | None:
        """
        두 물체를 고정 constraint로 결합한다 (affordance_r1_assembly.py assemble_objects 이식).

        main_body_id 위에 aux_body_id를 붙인다.
        contact_offset: main_body 기준 프레임 내 부착 위치 ([x, y, z]).
                        None이면 AABB 기반으로 자동 계산 (위쪽 면 중심).
        Returns: constraint id (성공) or None (실패).
        """
        if contact_offset is None:
            try:
                main_aabb_min, main_aabb_max = p.getAABB(main_body_id)
                aux_aabb_min,  aux_aabb_max  = p.getAABB(aux_body_id)
                main_half_z = (main_aabb_max[2] - main_aabb_min[2]) / 2.0
                aux_half_z  = (aux_aabb_max[2]  - aux_aabb_min[2])  / 2.0
                contact_offset = [0.0, 0.0, main_half_z + aux_half_z]
            except Exception:
                contact_offset = [0.0, 0.0, 0.05]

        try:
            constraint_id = p.createConstraint(
                parentBodyUniqueId=main_body_id,
                parentLinkIndex=-1,
                childBodyUniqueId=aux_body_id,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=contact_offset,
                childFramePosition=[0, 0, 0],
            )
            p.changeConstraint(constraint_id, maxForce=500)
            _step_simulation(settle_steps)
            print(
                f"[{self.name}] attach: body {aux_body_id} → body {main_body_id} "
                f"(constraint={constraint_id}, offset={[round(v,4) for v in contact_offset]})"
            )
            return constraint_id
        except Exception as exc:
            print(f"[{self.name}][WARN] attach_to_body failed: {exc}")
            return None

    def detach_body(self, constraint_id: int) -> bool:
        """attach_to_body로 만든 constraint를 제거한다."""
        try:
            p.removeConstraint(constraint_id)
            print(f"[{self.name}] detach: constraint {constraint_id} removed.")
            return True
        except Exception as exc:
            print(f"[{self.name}][WARN] detach_body failed: {exc}")
            return False

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
    ) -> bool:
        """
        Contact-based grasp primitive.
        This path relies on contact + friction only (no fixed constraint).
        """
        self._ensure_loaded()
        if orientation is None:
            orientation = p.getQuaternionFromEuler([np.pi, 0.0, 0.0])
        profile = self._activate_grasp_profile(
            body_id=body_id,
            object_label=object_label,
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
            # 튕겨나간 물체를 새 위치에서 다시 추적
            self._active_body_features = self._read_body_features(body_id)

            ok = self._grasp_body_no_constraint(
                body_id=body_id,
                orientation=orientation,
                approach_height=approach_height,
                grasp_clearance=grasp_clearance,
                lift_height=lift_height,
                xy_offset=retry_offsets[attempt_idx],
            )
            if ok:
                return True
            if attempt_idx < attempts - 1:
                self.release_grasp(open_after=True, steps=70)
                # 후퇴 전 물체 안정화 대기
                for _ in range(120):
                    p.stepSimulation()
                    time.sleep(SIM_TIMESTEP)
                self.move_end_effector_to(
                    [self.base_position[0] + 0.55, self.base_position[1], 1.02],
                    orientation=orientation,
                    steps=260,
                )
                # 후퇴 후 물체가 완전히 멈출 때까지 추가 대기
                for _ in range(180):
                    p.stepSimulation()
                    time.sleep(SIM_TIMESTEP)
        return False