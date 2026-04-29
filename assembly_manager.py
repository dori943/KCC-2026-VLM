"""
assembly_manager.py
??臾쇱껜瑜?PyBullet fixed constraint濡?寃고빀/遺꾨━?섎뒗 ?좏떥由ы떚.
module3_output.json??assembly_steps瑜??뚯떛???먮룞?쇰줈 ?ㅽ뻾 怨꾪쉷???앹꽦?쒕떎.

?ъ슜 ??
    from assembly_manager import AssemblyManager
    asm = AssemblyManager()

    # JSON?먯꽌 怨꾪쉷 濡쒕뱶
    plan = asm.load_plan_from_json("module3_output.json")

    # label ??pybullet body_id 留ㅽ븨 ?깅줉
    asm.register_body("ruler",        body_id=2)
    asm.register_body("tweezers",     body_id=3)
    asm.register_body("sticky notes", body_id=4)

    # 怨꾪쉷 ?꾩껜 ?ㅽ뻾
    results = asm.execute_plan()

    # 媛쒕퀎 detach
    asm.detach_by_label("step2_ruler_tweezers")
    asm.detach_all()
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import pybullet as p

SIM_TIMESTEP = 1.0 / 240.0


def _step_simulation(steps: int) -> None:
    for _ in range(steps):
        p.stepSimulation()
        time.sleep(SIM_TIMESTEP)


def _rpy_deg_to_quaternion(rpy_deg: list[float]) -> list[float]:
    """[roll, pitch, yaw] (degrees) ??pybullet quaternion [x,y,z,w]."""
    rpy_rad = [math.radians(v) for v in rpy_deg]
    return list(p.getQuaternionFromEuler(rpy_rad))


def _world_pos_to_parent_local(
    joint_pos_world: list[float],
    parent_body_id: int,
) -> list[float]:
    """
    world 醫뚰몴怨꾩쓽 joint_pos_world瑜?parent body 濡쒖뺄 ?꾨젅?꾩쑝濡?蹂??
    pybullet createConstraint??parentFramePosition???ｌ쓣 媛?
    """
    parent_pos, parent_orn = p.getBasePositionAndOrientation(parent_body_id)
    parent_orn_inv = p.invertTransform([0, 0, 0], list(parent_orn))[1]
    diff = (np.array(joint_pos_world) - np.array(parent_pos)).tolist()
    local_pos, _ = p.multiplyTransforms([0, 0, 0], parent_orn_inv, diff, [0, 0, 0, 1])
    return list(local_pos)


def _world_orientation_to_body_local(
    orientation_world: list[float],
    body_id: int,
) -> list[float]:
    """Convert world-frame orientation to body-local frame orientation."""
    _, body_orn = p.getBasePositionAndOrientation(body_id)
    body_orn_inv = p.invertTransform([0, 0, 0], list(body_orn))[1]
    _, local_orn = p.multiplyTransforms([0, 0, 0], body_orn_inv, [0, 0, 0], list(orientation_world))
    return list(local_orn)


def _body_anchor_radius_limit(body_id: int, extra_margin: float = 0.03) -> float:
    """
    Conservative bound for plausible local anchor distance from body origin.
    Uses half of world AABB diagonal + margin.
    """
    aabb_min, aabb_max = p.getAABB(body_id)
    dx = float(aabb_max[0] - aabb_min[0])
    dy = float(aabb_max[1] - aabb_min[1])
    dz = float(aabb_max[2] - aabb_min[2])
    half_diag = 0.5 * float(np.linalg.norm([dx, dy, dz]))
    return float(half_diag + extra_margin)


def _coerce_vec3(value, default: list[float]) -> list[float]:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return [float(value[0]), float(value[1]), float(value[2])]
        except (TypeError, ValueError):
            return list(default)
    return list(default)


def _coerce_orientation_quaternion(value, default_rpy_deg: list[float] | None = None) -> list[float]:
    if isinstance(value, (list, tuple)):
        if len(value) >= 4:
            try:
                return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
            except (TypeError, ValueError):
                pass
        if len(value) >= 3:
            try:
                return _rpy_deg_to_quaternion([float(value[0]), float(value[1]), float(value[2])])
            except (TypeError, ValueError):
                pass
    if isinstance(value, dict):
        for key in ("quaternion", "orientation_quaternion", "joint_orientation"):
            if key in value:
                return _coerce_orientation_quaternion(value.get(key), default_rpy_deg=default_rpy_deg)
        for key in ("rpy_deg", "orientation_rpy_deg"):
            if key in value:
                return _coerce_orientation_quaternion(value.get(key), default_rpy_deg=default_rpy_deg)
    if default_rpy_deg is None:
        default_rpy_deg = [0.0, 0.0, 0.0]
    return _rpy_deg_to_quaternion(default_rpy_deg)


def parse_assembly_instruction(instruction: dict) -> dict:
    """
    Parse assembly-related information from module3_task1_output.json.

    joint/assembly fields are preserved for assembly movement and relation
    application. This parser does not provide grasp pose inputs.
    """
    source_object = instruction.get("source_object") or instruction.get("attach_object")
    target_object = instruction.get("target_object") or instruction.get("base_object")

    joint_position = instruction.get("joint_position")
    joint_orientation_raw = instruction.get("joint_orientation")

    joint_position_world = instruction.get("joint_position_world", {})
    if joint_position is None and isinstance(joint_position_world, dict):
        joint_position = joint_position_world.get("position")
    if joint_orientation_raw is None and isinstance(joint_position_world, dict):
        joint_orientation_raw = (
            joint_position_world.get("orientation")
            or joint_position_world.get("orientation_quaternion")
        )

    target_pose_world = instruction.get("target_pose_world", {})
    target_position = None
    target_orientation_raw = None
    target_rpy_deg = [0.0, 0.0, 0.0]
    if isinstance(target_pose_world, dict):
        target_position = target_pose_world.get("position")
        target_orientation_raw = (
            target_pose_world.get("orientation")
            or target_pose_world.get("orientation_quaternion")
            or target_pose_world.get("orientation_rpy_deg")
        )
        target_rpy_deg = _coerce_vec3(target_pose_world.get("orientation_rpy_deg"), [0.0, 0.0, 0.0])

    joint_position = _coerce_vec3(
        joint_position if joint_position is not None else target_position,
        [0.0, 0.0, 0.0],
    )
    joint_orientation = _coerce_orientation_quaternion(
        joint_orientation_raw if joint_orientation_raw is not None else target_orientation_raw,
        default_rpy_deg=target_rpy_deg,
    )

    attachment_position = instruction.get("attachment_position")
    if attachment_position is None:
        attachment_position = instruction.get("relative_offset_from_base")
    attachment_position = _coerce_vec3(attachment_position, [0.0, 0.0, 0.0])

    attachment_orientation = _coerce_orientation_quaternion(
        instruction.get("attachment_orientation")
        or instruction.get("attachment_orientation_quaternion")
        or instruction.get("attachment_orientation_rpy_deg"),
        default_rpy_deg=[0.0, 0.0, 0.0],
    )

    connection_relationship = (
        instruction.get("connection_relationship")
        or instruction.get("contact_type")
    )
    assembly_relationship = (
        instruction.get("assembly_relationship")
        or instruction.get("functional_roles")
    )

    return {
        "source_object": source_object,
        "target_object": target_object,
        "joint_position": joint_position,
        "joint_orientation": joint_orientation,
        "target_position": _coerce_vec3(target_position, joint_position),
        "target_orientation": _coerce_orientation_quaternion(
            target_orientation_raw,
            default_rpy_deg=target_rpy_deg,
        ),
        "attachment_position": attachment_position,
        "attachment_orientation": attachment_orientation,
        "connection_relationship": connection_relationship,
        "assembly_relationship": assembly_relationship,
        "joint_calc_basis": (
            joint_position_world.get("calculation_basis", "")
            if isinstance(joint_position_world, dict)
            else ""
        ),
    }


class AssemblyStep:
    """module3_output.json??assembly_steps ??ぉ ?섎굹瑜??쒗쁽."""

    def __init__(self, raw: dict):
        parsed = parse_assembly_instruction(raw)
        self.step: int                  = int(raw["step"])
        self.base_object: str           = str(raw["base_object"])
        self.attach_object: str | None  = raw.get("attach_object")  # null 媛??
        self.attach_region_base: str    = str(raw.get("attach_region_base", ""))
        self.attach_region_object: str | None = raw.get("attach_region_object")
        self.contact_type: str          = str(raw.get("contact_type", "surface"))
        self.functional_roles: dict     = raw.get("functional_roles", {})

        self.source_object: str | None = parsed["source_object"]
        self.target_object: str | None = parsed["target_object"]
        self.joint_pos_world: list[float] = list(parsed["joint_position"])
        self.joint_orientation: list[float] = list(parsed["joint_orientation"])
        self.joint_calc_basis: str      = str(parsed["joint_calc_basis"])
        self.target_position: list[float] = list(parsed["target_position"])
        self.target_orientation: list[float] = list(parsed["target_orientation"])
        self.target_rpy_deg: list[float]  = list(p.getEulerFromQuaternion(self.target_orientation))
        self.attachment_position: list[float] = list(parsed["attachment_position"])
        self.attachment_orientation: list[float] = list(parsed["attachment_orientation"])
        self.connection_relationship = parsed["connection_relationship"]
        self.assembly_relationship = parsed["assembly_relationship"]

        self.reason: str                = str(raw.get("reason", ""))
        self.expected_function: str     = str(raw.get("expected_function_after_step", ""))

    def has_attach(self) -> bool:
        return self.attach_object is not None

    def label(self) -> str:
        if self.has_attach():
            return f"step{self.step}_{self.base_object}_{self.attach_object}"
        return f"step{self.step}_{self.base_object}_position"

    def __repr__(self) -> str:
        return (
            f"<AssemblyStep {self.step}: base={self.base_object!r} "
            f"attach={self.attach_object!r} "
            f"joint_world={[round(v, 4) for v in self.joint_pos_world]}>"
        )


class AssemblyPlan:
    """module3_output.json ?꾩껜瑜??뚯떛???ㅽ뻾 怨꾪쉷."""

    def __init__(self, raw: dict):
        self.strategy_summary: str  = raw.get("assembly_strategy", {}).get("strategy_summary", "")
        self.base_object: str       = raw.get("assembly_strategy", {}).get("base_object", "")
        self.is_valid: bool         = raw.get("verification", {}).get("is_valid", False)
        self.steps: list[AssemblyStep] = [
            AssemblyStep(s) for s in raw.get("assembly_steps", [])
        ]

    def __repr__(self) -> str:
        return (
            f"<AssemblyPlan base={self.base_object!r} "
            f"steps={len(self.steps)} valid={self.is_valid}>"
        )


class AssemblyManager:
    """
    module3_output.json 湲곕컲 議곕┰ 怨꾪쉷??濡쒕뱶?섍퀬
    PyBullet constraint濡??ㅽ뻾쨌愿由ы븳??
    """

    def __init__(self):
        self._plan: AssemblyPlan | None = None
        # label ??body_id  (?ъ슜?먭? register_body濡??깅줉)
        self._body_map: dict[str, int] = {}
        # label ??constraint_id  (attach ???깅줉)
        self._registry: dict[str, int] = {}

    # ?? 怨꾪쉷 濡쒕뱶 ?????????????????????????????????????????????????????????????

    def load_plan_from_json(self, path: str) -> AssemblyPlan:
        """
        module3_output.json???쎌뼱 AssemblyPlan??諛섑솚?섍퀬 ?대??????
        """
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self._plan = AssemblyPlan(raw)
        print(f"[Assembly] plan loaded: {self._plan}")
        for step in self._plan.steps:
            print(f"  {step}")
        return self._plan

    def load_plan_from_dict(self, raw: dict) -> AssemblyPlan:
        """dict?먯꽌 吏곸젒 濡쒕뱶 (?뚯뒪?맞룹씤?쇱씤 ?ъ슜)."""
        self._plan = AssemblyPlan(raw)
        return self._plan

    def parse_assembly_instruction(self, instruction: dict) -> dict:
        """
        Parse one assembly step and preserve joint/assembly fields.
        """
        return parse_assembly_instruction(instruction)

    def get_object_transport_targets(self) -> dict[str, dict]:
        """
        Build per-object transport targets for the assembly move stage.

        Returns:
            {
                object_label: {
                    "position": [x, y, z],
                    "orientation": [x, y, z, w],
                    "step": int,
                    "role": "base" | "attach",
                },
                ...
            }
        """
        targets: dict[str, dict] = {}
        if self._plan is None:
            return targets

        for step in self._plan.steps:
            if not step.has_attach():
                targets[step.base_object] = {
                    "position": list(step.joint_pos_world),
                    "orientation": list(step.joint_orientation),
                    "step": int(step.step),
                    "role": "base",
                }
                continue

            # For attach steps, move attach object to its assembled center
            # (target_pose_world), not the joint contact point on the base.
            # joint_pos_world is the contact point on the base's region;
            # the attach object's CENTER must end up at target_pose_world,
            # which equals joint_pos_world + relative_offset_from_base.
            targets[step.attach_object] = {
                "position": list(step.target_position),
                "orientation": list(step.target_orientation),
                "step": int(step.step),
                "role": "attach",
            }

            # Keep a base target as well if one was not set by standalone step.
            if step.base_object not in targets:
                targets[step.base_object] = {
                    "position": list(step.target_position),
                    "orientation": list(step.target_orientation),
                    "step": int(step.step),
                    "role": "base",
                }
        return targets

    def snap_objects_to_targets(
        self,
        settle_steps: int = 20,
        use_aabb_offset: bool = True,
    ) -> dict[str, dict]:
        """
        Teleport every registered body to its assembly target pose.

        Call this right before execute_plan() to bypass grasp slip
        during transport and guarantee the constraint is created at the
        exact JSON-specified geometry.

        When use_aabb_offset is True, the URDF origin offset from the
        body's current AABB center is preserved, so the visible AABB
        center lands exactly on target_pose_world.
        """
        if self._plan is None:
            return {}
        targets = self.get_object_transport_targets()
        snapped: dict[str, dict] = {}
        for label, target in targets.items():
            body_id = self._body_map.get(label)
            if body_id is None:
                continue

            snap_pos = list(target["position"])
            snap_orn = list(target["orientation"])
            urdf_offset = [0.0, 0.0, 0.0]
            if use_aabb_offset:
                cur_pos, _ = p.getBasePositionAndOrientation(body_id)
                aabb_min, aabb_max = p.getAABB(body_id)
                aabb_center = [
                    (aabb_min[i] + aabb_max[i]) / 2.0 for i in range(3)
                ]
                urdf_offset = [cur_pos[i] - aabb_center[i] for i in range(3)]
                snap_pos = [snap_pos[i] + urdf_offset[i] for i in range(3)]

            p.resetBasePositionAndOrientation(body_id, snap_pos, snap_orn)
            p.resetBaseVelocity(body_id, [0, 0, 0], [0, 0, 0])
            snapped[label] = {
                "target_aabb_center": list(target["position"]),
                "snap_urdf_pos": snap_pos,
                "urdf_offset_from_aabb_center": urdf_offset,
                "step": int(target.get("step", -1)),
                "role": target.get("role"),
            }
            print(
                f"[Assembly] snap '{label}' "
                f"(step={target.get('step')}, role={target.get('role')}): "
                f"aabb_center->{[round(v, 4) for v in target['position']]}, "
                f"urdf_pos={[round(v, 4) for v in snap_pos]}"
            )

        if settle_steps > 0:
            _step_simulation(settle_steps)

        return snapped

    # ?? body 留ㅽ븨 ?깅줉 ????????????????????????????????????????????????????????

    def register_body(self, label: str, body_id: int) -> None:
        """
        臾쇱껜 ?대쫫(JSON??base_object / attach_object)怨?
        pybullet body_id瑜?留ㅽ븨?쒕떎.
        """
        self._body_map[label] = body_id
        print(f"[Assembly] register: '{label}' ??body_id={body_id}")

    def register_bodies(self, mapping: dict[str, int]) -> None:
        """dict濡???踰덉뿉 ?깅줉."""
        for label, body_id in mapping.items():
            self.register_body(label, body_id)

    # ?? 怨꾪쉷 ?ㅽ뻾 ?????????????????????????????????????????????????????????????

    def execute_plan(
        self,
        settle_steps: int = 60,
        max_force: float = 500.0,
        suspend_gravity: bool = True,
        restore_gravity: tuple[float, float, float] = (0.0, 0.0, -9.8),
    ) -> list[dict]:
        """
        濡쒕뱶??AssemblyPlan??紐⑤뱺 step???쒖꽌?濡??ㅽ뻾?쒕떎.

        - attach_object媛 null??step? 諛곗튂(positioning) 湲곕줉留??④?.
        - attach_object媛 ?덈뒗 step? joint_position_world瑜?
          parent 濡쒖뺄 ?꾨젅?꾩쑝濡?蹂????createConstraint ?ㅽ뻾.

        Returns:
            step蹂?寃곌낵 dict 紐⑸줉.
        """
        if self._plan is None:
            raise RuntimeError("[Assembly] no plan loaded. call load_plan_from_json() first.")

        if suspend_gravity:
            p.setGravity(0.0, 0.0, 0.0)
            print("[Assembly] gravity suspended for plan execution")

        try:
            results = []
            for step in self._plan.steps:
                result = self._execute_step(step, settle_steps=settle_steps, max_force=max_force)
                results.append(result)
                if not result["ok"] and step.has_attach():
                    print(f"[Assembly][WARN] step {step.step} failed, continuing...")
            return results
        finally:
            if suspend_gravity:
                p.setGravity(*restore_gravity)
                print(f"[Assembly] gravity restored to {restore_gravity}")

    def execute_step(
        self,
        step_number: int,
        settle_steps: int = 60,
        max_force: float = 500.0,
    ) -> dict:
        """?뱀젙 step 踰덊샇留??ㅽ뻾."""
        if self._plan is None:
            raise RuntimeError("[Assembly] no plan loaded.")
        for step in self._plan.steps:
            if step.step == step_number:
                return self._execute_step(step, settle_steps=settle_steps, max_force=max_force)
        raise ValueError(f"[Assembly] step {step_number} not found in plan.")

    def _snap_step_targets(
        self,
        step: "AssemblyStep",
        use_aabb_offset: bool = True,
    ) -> None:
        """
        Re-position THIS step's relevant body to its JSON target right
        before validation/constraint creation. Position-only step snaps
        the base; attach step snaps the attach object (base may already
        be constrained from a prior step, so it is not touched).
        """
        if not step.has_attach():
            label = step.base_object
            target_pos = list(step.joint_pos_world)
            target_orn = list(step.joint_orientation)
        else:
            label = step.attach_object
            target_pos = list(step.target_position)
            target_orn = list(step.target_orientation)

        body_id = self._body_map.get(label)
        if body_id is None:
            return

        snap_pos = list(target_pos)
        snap_orn = list(target_orn)
        if use_aabb_offset:
            cur_pos, _ = p.getBasePositionAndOrientation(body_id)
            aabb_min, aabb_max = p.getAABB(body_id)
            aabb_center = [
                (aabb_min[i] + aabb_max[i]) / 2.0 for i in range(3)
            ]
            urdf_offset = [cur_pos[i] - aabb_center[i] for i in range(3)]
            snap_pos = [snap_pos[i] + urdf_offset[i] for i in range(3)]

        p.resetBasePositionAndOrientation(body_id, snap_pos, snap_orn)
        p.resetBaseVelocity(body_id, [0, 0, 0], [0, 0, 0])
        print(
            f"[Assembly] step{step.step} pre-snap '{label}': "
            f"urdf={[round(v, 4) for v in snap_pos]}"
        )

    def _execute_step(
        self,
        step: AssemblyStep,
        settle_steps: int,
        max_force: float,
    ) -> dict:
        # Re-snap this step's relevant body right before validation so
        # any drift from earlier physics steps does not push it past the
        # joint anchor limits.
        self._snap_step_targets(step)
        base_id = self._body_map.get(step.base_object)
        if base_id is None:
            msg = f"base_object '{step.base_object}' not registered."
            print(f"[Assembly][WARN] step {step.step}: {msg}")
            return {"step": step.step, "ok": False, "reason": msg, "constraint_id": None}

        # attach_object媛 ?녿뒗 step ??諛곗튂 湲곕줉留?
        if not step.has_attach():
            print(
                f"[Assembly] step {step.step}: position-only '{step.base_object}' "
                f"at {[round(v, 4) for v in step.joint_pos_world]} "
                f"(role={step.functional_roles.get('base_role')})"
            )
            return {
                "step": step.step,
                "ok": True,
                "reason": "position_only",
                "constraint_id": None,
                "label": step.label(),
                "joint_pos_world": step.joint_pos_world,
                "joint_orientation": step.joint_orientation,
                "target_position": step.target_position,
                "target_orientation": step.target_orientation,
            }

        # attach_object媛 ?덈뒗 step ??constraint ?앹꽦
        aux_id = self._body_map.get(step.attach_object)
        if aux_id is None:
            msg = f"attach_object '{step.attach_object}' not registered."
            print(f"[Assembly][WARN] step {step.step}: {msg}")
            return {"step": step.step, "ok": False, "reason": msg, "constraint_id": None}

        # joint_position_world ??parent(base) 濡쒖뺄 ?꾨젅?꾩쑝濡?蹂??
        parent_frame_pos = _world_pos_to_parent_local(
            joint_pos_world=step.joint_pos_world,
            parent_body_id=base_id,
        )
        child_frame_pos = _world_pos_to_parent_local(
            joint_pos_world=step.joint_pos_world,
            parent_body_id=aux_id,
        )
        parent_frame_orn = _world_orientation_to_body_local(
            orientation_world=step.joint_orientation,
            body_id=base_id,
        )
        child_frame_orn = _world_orientation_to_body_local(
            orientation_world=step.joint_orientation,
            body_id=aux_id,
        )

        aux_pos_world, _ = p.getBasePositionAndOrientation(aux_id)
        joint_gap_m = float(np.linalg.norm(np.array(aux_pos_world) - np.array(step.joint_pos_world)))
        base_anchor_norm = float(np.linalg.norm(np.array(parent_frame_pos, dtype=float)))
        child_anchor_norm = float(np.linalg.norm(np.array(child_frame_pos, dtype=float)))
        base_anchor_limit = _body_anchor_radius_limit(base_id)
        child_anchor_limit = _body_anchor_radius_limit(aux_id)
        anchors_plausible = (
            base_anchor_norm <= base_anchor_limit
            and child_anchor_norm <= child_anchor_limit
        )
        if not anchors_plausible:
            print(
                f"[Assembly][WARN] step {step.step}: joint pose is too far from current body geometry "
                f"(base_anchor={base_anchor_norm:.3f}>{base_anchor_limit:.3f}, "
                f"child_anchor={child_anchor_norm:.3f}>{child_anchor_limit:.3f}, "
                f"joint_gap={joint_gap_m:.3f} m). Skipping attach."
            )
            return {
                "step": step.step,
                "ok": False,
                "reason": "joint_anchor_out_of_bounds",
                "constraint_id": None,
                "label": step.label(),
                "base_object": step.base_object,
                "attach_object": step.attach_object,
                "joint_pos_world": step.joint_pos_world,
                "joint_orientation": step.joint_orientation,
                "joint_gap_m": joint_gap_m,
                "parent_anchor_norm": base_anchor_norm,
                "child_anchor_norm": child_anchor_norm,
                "parent_anchor_limit": base_anchor_limit,
                "child_anchor_limit": child_anchor_limit,
            }

        cid = self.attach(
            main_body_id=base_id,
            aux_body_id=aux_id,
            contact_offset=parent_frame_pos,
            child_frame_offset=child_frame_pos,
            parent_frame_orientation=parent_frame_orn,
            child_frame_orientation=child_frame_orn,
            label=step.label(),
            settle_steps=settle_steps,
            max_force=max_force,
        )

        print(
            f"[Assembly] step {step.step}: '{step.base_object}' ??'{step.attach_object}' "
            f"| region: {step.attach_region_base} / {step.attach_region_object} "
            f"| contact: {step.contact_type} "
            f"| role: {step.functional_roles.get('attach_role')} "
            f"| expected: {step.expected_function}"
        )

        return {
            "step": step.step,
            "ok": cid is not None,
            "reason": "attached" if cid is not None else "constraint_failed",
            "constraint_id": cid,
            "label": step.label(),
            "base_object": step.base_object,
            "attach_object": step.attach_object,
            "joint_pos_world": step.joint_pos_world,
            "joint_orientation": step.joint_orientation,
            "target_position": step.target_position,
            "target_orientation": step.target_orientation,
            "connection_relationship": step.connection_relationship,
            "assembly_relationship": step.assembly_relationship,
            "contact_offset_local": parent_frame_pos,
            "child_contact_offset_local": child_frame_pos,
            "parent_frame_orientation_local": parent_frame_orn,
            "child_frame_orientation_local": child_frame_orn,
            "offset_policy": "joint_world_dual_anchor",
            "joint_gap_m": joint_gap_m,
        }

    # ?? ?듭떖 attach / detach API ??????????????????????????????????????????????

    def attach(
        self,
        main_body_id: int,
        aux_body_id: int,
        contact_offset: list[float] | None = None,
        child_frame_offset: list[float] | None = None,
        parent_frame_orientation: list[float] | None = None,
        child_frame_orientation: list[float] | None = None,
        label: str | None = None,
        settle_steps: int = 60,
        max_force: float = 500.0,
    ) -> int | None:
        """
        aux_body_id瑜?main_body_id??怨좎젙 寃고빀?쒕떎.
        contact_offset??None?대㈃ AABB 湲곕컲 ?먮룞 怨꾩궛.
        """
        if contact_offset is None:
            contact_offset = self._auto_contact_offset(main_body_id, aux_body_id)
        if child_frame_offset is None:
            child_frame_offset = [0.0, 0.0, 0.0]
        if parent_frame_orientation is None:
            parent_frame_orientation = [0.0, 0.0, 0.0, 1.0]
        if child_frame_orientation is None:
            child_frame_orientation = [0.0, 0.0, 0.0, 1.0]

        try:
            pre_main_pos, pre_main_orn = p.getBasePositionAndOrientation(main_body_id)
            pre_aux_pos, pre_aux_orn = p.getBasePositionAndOrientation(aux_body_id)
            pre_parent_anchor_world, pre_parent_anchor_orn = p.multiplyTransforms(
                pre_main_pos,
                pre_main_orn,
                contact_offset,
                parent_frame_orientation,
            )
            pre_child_anchor_world, pre_child_anchor_orn = p.multiplyTransforms(
                pre_aux_pos,
                pre_aux_orn,
                child_frame_offset,
                child_frame_orientation,
            )
            pre_anchor_gap = float(
                np.linalg.norm(np.array(pre_parent_anchor_world) - np.array(pre_child_anchor_world))
            )

            constraint_id = p.createConstraint(
                parentBodyUniqueId=main_body_id,
                parentLinkIndex=-1,
                childBodyUniqueId=aux_body_id,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=contact_offset,
                childFramePosition=child_frame_offset,
                parentFrameOrientation=parent_frame_orientation,
                childFrameOrientation=child_frame_orientation,
            )
            p.changeConstraint(constraint_id, maxForce=max_force)
            _step_simulation(settle_steps)

            post_main_pos, post_main_orn = p.getBasePositionAndOrientation(main_body_id)
            post_aux_pos, post_aux_orn = p.getBasePositionAndOrientation(aux_body_id)
            post_parent_anchor_world, _ = p.multiplyTransforms(
                post_main_pos,
                post_main_orn,
                contact_offset,
                parent_frame_orientation,
            )
            post_child_anchor_world, _ = p.multiplyTransforms(
                post_aux_pos,
                post_aux_orn,
                child_frame_offset,
                child_frame_orientation,
            )
            post_anchor_gap = float(
                np.linalg.norm(np.array(post_parent_anchor_world) - np.array(post_child_anchor_world))
            )
            max_allowed_anchor_gap = 0.012
            if post_anchor_gap > max_allowed_anchor_gap:
                print(
                    f"[Assembly][WARN] attach validation failed: "
                    f"anchor_gap={post_anchor_gap:.4f} m "
                    f"(pre_gap={pre_anchor_gap:.4f}, limit={max_allowed_anchor_gap:.4f}). "
                    "Removing constraint."
                )
                try:
                    p.removeConstraint(constraint_id)
                except Exception:
                    pass
                return None

            key = label or f"main{main_body_id}_aux{aux_body_id}"
            self._registry[key] = constraint_id

            print(
                f"[Assembly] attach '{key}': "
                f"body {aux_body_id} ??body {main_body_id} "
                f"(constraint={constraint_id}, "
                f"offset=[{contact_offset[0]:.4f}, {contact_offset[1]:.4f}, {contact_offset[2]:.4f}], "
                f"child_offset=[{child_frame_offset[0]:.4f}, {child_frame_offset[1]:.4f}, {child_frame_offset[2]:.4f}], "
                f"pre_anchor_gap={pre_anchor_gap:.4f}, post_anchor_gap={post_anchor_gap:.4f})"
            )
            return constraint_id

        except Exception as exc:
            print(f"[Assembly][WARN] attach failed: {exc}")
            return None

    def detach(self, constraint_id: int) -> bool:
        try:
            p.removeConstraint(constraint_id)
            self._registry = {k: v for k, v in self._registry.items() if v != constraint_id}
            print(f"[Assembly] detach: constraint {constraint_id} removed.")
            return True
        except Exception as exc:
            print(f"[Assembly][WARN] detach failed: {exc}")
            return False

    def detach_by_label(self, label: str) -> bool:
        """label濡??깅줉??constraint瑜??댁젣?쒕떎."""
        cid = self._registry.get(label)
        if cid is None:
            print(f"[Assembly][WARN] detach_by_label: label '{label}' not found.")
            return False
        return self.detach(cid)

    def detach_all(self) -> None:
        """?깅줉??紐⑤뱺 constraint瑜??댁젣?쒕떎."""
        for label, cid in list(self._registry.items()):
            self.detach(cid)

    def is_attached(self, label: str) -> bool:
        return label in self._registry

    def list_attached(self) -> dict[str, int]:
        """?꾩옱 ?쒖꽦 constraint 紐⑸줉 諛섑솚."""
        return dict(self._registry)

    # ?? ?대? ?좏떥 ????????????????????????????????????????????????????????????

    def _auto_contact_offset(self, main_body_id: int, aux_body_id: int) -> list[float]:
        """AABB 湲곕컲 contact_offset ?먮룞 怨꾩궛."""
        try:
            main_aabb_min, main_aabb_max = p.getAABB(main_body_id)
            aux_aabb_min,  aux_aabb_max  = p.getAABB(aux_body_id)
            main_half_z = (main_aabb_max[2] - main_aabb_min[2]) / 2.0
            aux_half_z  = (aux_aabb_max[2]  - aux_aabb_min[2])  / 2.0
            return [0.0, 0.0, float(main_half_z + aux_half_z)]
        except Exception:
            return [0.0, 0.0, 0.05]
