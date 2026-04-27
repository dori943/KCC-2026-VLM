"""
assembly_manager.py
두 물체를 PyBullet fixed constraint로 결합/분리하는 유틸리티.
module3_output.json의 assembly_steps를 파싱해 자동으로 실행 계획을 생성한다.

사용 예:
    from assembly_manager import AssemblyManager
    asm = AssemblyManager()

    # JSON에서 계획 로드
    plan = asm.load_plan_from_json("module3_output.json")

    # label → pybullet body_id 매핑 등록
    asm.register_body("ruler",        body_id=2)
    asm.register_body("tweezers",     body_id=3)
    asm.register_body("sticky notes", body_id=4)

    # 계획 전체 실행
    results = asm.execute_plan()

    # 개별 detach
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
    """[roll, pitch, yaw] (degrees) → pybullet quaternion [x,y,z,w]."""
    rpy_rad = [math.radians(v) for v in rpy_deg]
    return list(p.getQuaternionFromEuler(rpy_rad))


def _world_pos_to_parent_local(
    joint_pos_world: list[float],
    parent_body_id: int,
) -> list[float]:
    """
    world 좌표계의 joint_pos_world를 parent body 로컬 프레임으로 변환.
    pybullet createConstraint의 parentFramePosition에 넣을 값.
    """
    parent_pos, parent_orn = p.getBasePositionAndOrientation(parent_body_id)
    parent_orn_inv = p.invertTransform([0, 0, 0], list(parent_orn))[1]
    diff = (np.array(joint_pos_world) - np.array(parent_pos)).tolist()
    local_pos, _ = p.multiplyTransforms([0, 0, 0], parent_orn_inv, diff, [0, 0, 0, 1])
    return list(local_pos)


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
    """module3_output.json의 assembly_steps 항목 하나를 표현."""

    def __init__(self, raw: dict):
        parsed = parse_assembly_instruction(raw)
        self.step: int                  = int(raw["step"])
        self.base_object: str           = str(raw["base_object"])
        self.attach_object: str | None  = raw.get("attach_object")  # null 가능
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
    """module3_output.json 전체를 파싱한 실행 계획."""

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
    module3_output.json 기반 조립 계획을 로드하고
    PyBullet constraint로 실행·관리한다.
    """

    def __init__(self):
        self._plan: AssemblyPlan | None = None
        # label → body_id  (사용자가 register_body로 등록)
        self._body_map: dict[str, int] = {}
        # label → constraint_id  (attach 후 등록)
        self._registry: dict[str, int] = {}

    # ── 계획 로드 ─────────────────────────────────────────────────────────────

    def load_plan_from_json(self, path: str) -> AssemblyPlan:
        """
        module3_output.json을 읽어 AssemblyPlan을 반환하고 내부에 저장.
        """
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self._plan = AssemblyPlan(raw)
        print(f"[Assembly] plan loaded: {self._plan}")
        for step in self._plan.steps:
            print(f"  {step}")
        return self._plan

    def load_plan_from_dict(self, raw: dict) -> AssemblyPlan:
        """dict에서 직접 로드 (테스트·인라인 사용)."""
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

            # For attach steps, move attach object to joint pose from JSON.
            targets[step.attach_object] = {
                "position": list(step.joint_pos_world),
                "orientation": list(step.joint_orientation),
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

    # ── body 매핑 등록 ────────────────────────────────────────────────────────

    def register_body(self, label: str, body_id: int) -> None:
        """
        물체 이름(JSON의 base_object / attach_object)과
        pybullet body_id를 매핑한다.
        """
        self._body_map[label] = body_id
        print(f"[Assembly] register: '{label}' → body_id={body_id}")

    def register_bodies(self, mapping: dict[str, int]) -> None:
        """dict로 한 번에 등록."""
        for label, body_id in mapping.items():
            self.register_body(label, body_id)

    # ── 계획 실행 ─────────────────────────────────────────────────────────────

    def execute_plan(
        self,
        settle_steps: int = 60,
        max_force: float = 500.0,
    ) -> list[dict]:
        """
        로드된 AssemblyPlan의 모든 step을 순서대로 실행한다.

        - attach_object가 null인 step은 배치(positioning) 기록만 남김.
        - attach_object가 있는 step은 joint_position_world를
          parent 로컬 프레임으로 변환 후 createConstraint 실행.

        Returns:
            step별 결과 dict 목록.
        """
        if self._plan is None:
            raise RuntimeError("[Assembly] no plan loaded. call load_plan_from_json() first.")

        results = []
        for step in self._plan.steps:
            result = self._execute_step(step, settle_steps=settle_steps, max_force=max_force)
            results.append(result)
            if not result["ok"] and step.has_attach():
                print(f"[Assembly][WARN] step {step.step} failed, continuing...")
        return results

    def execute_step(
        self,
        step_number: int,
        settle_steps: int = 60,
        max_force: float = 500.0,
    ) -> dict:
        """특정 step 번호만 실행."""
        if self._plan is None:
            raise RuntimeError("[Assembly] no plan loaded.")
        for step in self._plan.steps:
            if step.step == step_number:
                return self._execute_step(step, settle_steps=settle_steps, max_force=max_force)
        raise ValueError(f"[Assembly] step {step_number} not found in plan.")

    def _execute_step(
        self,
        step: AssemblyStep,
        settle_steps: int,
        max_force: float,
    ) -> dict:
        base_id = self._body_map.get(step.base_object)
        if base_id is None:
            msg = f"base_object '{step.base_object}' not registered."
            print(f"[Assembly][WARN] step {step.step}: {msg}")
            return {"step": step.step, "ok": False, "reason": msg, "constraint_id": None}

        # attach_object가 없는 step → 배치 기록만
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

        # attach_object가 있는 step → constraint 생성
        aux_id = self._body_map.get(step.attach_object)
        if aux_id is None:
            msg = f"attach_object '{step.attach_object}' not registered."
            print(f"[Assembly][WARN] step {step.step}: {msg}")
            return {"step": step.step, "ok": False, "reason": msg, "constraint_id": None}

        # joint_position_world → parent(base) 로컬 프레임으로 변환
        contact_offset = _world_pos_to_parent_local(
            joint_pos_world=step.joint_pos_world,
            parent_body_id=base_id,
        )

        cid = self.attach(
            main_body_id=base_id,
            aux_body_id=aux_id,
            contact_offset=contact_offset,
            label=step.label(),
            settle_steps=settle_steps,
            max_force=max_force,
        )

        print(
            f"[Assembly] step {step.step}: '{step.base_object}' ← '{step.attach_object}' "
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
            "contact_offset_local": contact_offset,
        }

    # ── 핵심 attach / detach API ──────────────────────────────────────────────

    def attach(
        self,
        main_body_id: int,
        aux_body_id: int,
        contact_offset: list[float] | None = None,
        label: str | None = None,
        settle_steps: int = 60,
        max_force: float = 500.0,
    ) -> int | None:
        """
        aux_body_id를 main_body_id에 고정 결합한다.
        contact_offset이 None이면 AABB 기반 자동 계산.
        """
        if contact_offset is None:
            contact_offset = self._auto_contact_offset(main_body_id, aux_body_id)

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
            p.changeConstraint(constraint_id, maxForce=max_force)
            _step_simulation(settle_steps)

            key = label or f"main{main_body_id}_aux{aux_body_id}"
            self._registry[key] = constraint_id

            print(
                f"[Assembly] attach '{key}': "
                f"body {aux_body_id} → body {main_body_id} "
                f"(constraint={constraint_id}, "
                f"offset=[{contact_offset[0]:.4f}, {contact_offset[1]:.4f}, {contact_offset[2]:.4f}])"
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
        """label로 등록된 constraint를 해제한다."""
        cid = self._registry.get(label)
        if cid is None:
            print(f"[Assembly][WARN] detach_by_label: label '{label}' not found.")
            return False
        return self.detach(cid)

    def detach_all(self) -> None:
        """등록된 모든 constraint를 해제한다."""
        for label, cid in list(self._registry.items()):
            self.detach(cid)

    def is_attached(self, label: str) -> bool:
        return label in self._registry

    def list_attached(self) -> dict[str, int]:
        """현재 활성 constraint 목록 반환."""
        return dict(self._registry)

    # ── 내부 유틸 ────────────────────────────────────────────────────────────

    def _auto_contact_offset(self, main_body_id: int, aux_body_id: int) -> list[float]:
        """AABB 기반 contact_offset 자동 계산."""
        try:
            main_aabb_min, main_aabb_max = p.getAABB(main_body_id)
            aux_aabb_min,  aux_aabb_max  = p.getAABB(aux_body_id)
            main_half_z = (main_aabb_max[2] - main_aabb_min[2]) / 2.0
            aux_half_z  = (aux_aabb_max[2]  - aux_aabb_min[2])  / 2.0
            return [0.0, 0.0, float(main_half_z + aux_half_z)]
        except Exception:
            return [0.0, 0.0, 0.05]
