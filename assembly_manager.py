"""
assembly_manager.py
두 물체를 PyBullet fixed constraint로 결합/분리하는 유틸리티.

사용 예:
    from assembly_manager import AssemblyManager
    asm = AssemblyManager()
    cid = asm.attach(main_body_id=3, aux_body_id=6)
    asm.detach(cid)
"""

import pybullet as p
import time

SIM_TIMESTEP = 1.0 / 240.0


def _step_simulation(steps: int) -> None:
    for _ in range(steps):
        p.stepSimulation()
        time.sleep(SIM_TIMESTEP)


class AssemblyManager:
    """
    물체 간 attach / detach 를 관리한다.
    constraint id를 내부 레지스트리에 보관하므로
    label 기반으로 detach 하거나 전체 해제도 가능하다.
    """

    def __init__(self):
        # { label: constraint_id }
        self._registry: dict[str, int] = {}

    # ── 핵심 API ──────────────────────────────────────────────────────────────

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
        aux_body_id 를 main_body_id 에 고정 결합한다.

        Args:
            main_body_id  : 기준 물체 (base).
            aux_body_id   : 붙일 물체.
            contact_offset: main 기준 프레임 내 부착 위치 [x, y, z].
                            None 이면 AABB 기반으로 자동 계산.
            label         : 레지스트리 키 (나중에 label로 detach 가능).
                            None 이면 "main{main_body_id}_aux{aux_body_id}" 로 자동 생성.
            settle_steps  : 결합 후 물리 안정화 스텝 수.
            max_force     : constraint 최대 허용 힘 (N).

        Returns:
            constraint_id (int) 또는 None (실패 시).
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
        """
        constraint_id 로 결합을 해제한다.

        Returns:
            True (성공) / False (실패).
        """
        try:
            p.removeConstraint(constraint_id)
            # 레지스트리에서도 제거
            self._registry = {
                k: v for k, v in self._registry.items() if v != constraint_id
            }
            print(f"[Assembly] detach: constraint {constraint_id} removed.")
            return True
        except Exception as exc:
            print(f"[Assembly][WARN] detach failed: {exc}")
            return False

    def detach_by_label(self, label: str) -> bool:
        """label 로 등록된 constraint 를 해제한다."""
        cid = self._registry.get(label)
        if cid is None:
            print(f"[Assembly][WARN] detach_by_label: label '{label}' not found.")
            return False
        return self.detach(cid)

    def detach_all(self) -> None:
        """등록된 모든 constraint 를 해제한다."""
        for label, cid in list(self._registry.items()):
            self.detach(cid)

    def is_attached(self, label: str) -> bool:
        return label in self._registry

    # ── 내부 유틸 ────────────────────────────────────────────────────────────

    def _auto_contact_offset(
        self,
        main_body_id: int,
        aux_body_id: int,
    ) -> list[float]:
        """
        AABB 기반으로 contact_offset 자동 계산.
        main 물체 상단 + aux 물체 절반 높이 위치를 부착점으로 사용.
        """
        try:
            main_aabb_min, main_aabb_max = p.getAABB(main_body_id)
            aux_aabb_min,  aux_aabb_max  = p.getAABB(aux_body_id)
            main_half_z = (main_aabb_max[2] - main_aabb_min[2]) / 2.0
            aux_half_z  = (aux_aabb_max[2]  - aux_aabb_min[2])  / 2.0
            return [0.0, 0.0, main_half_z + aux_half_z]
        except Exception:
            return [0.0, 0.0, 0.05]