"""Typed data models for Module 3 assembly pose calculation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TargetPoseWorld:
    position: list[float]
    orientation_rpy_deg: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "orientation_rpy_deg": self.orientation_rpy_deg,
        }


@dataclass(slots=True)
class JointPositionWorld:
    position: list[float]
    calculation_basis: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "calculation_basis": self.calculation_basis,
            "description": self.description,
        }


@dataclass(slots=True)
class AssemblyStep:
    step: int
    partial_assembly_state_before: list[str]
    base_object: str
    attach_object: str
    attach_region_base: str
    attach_region_object: str
    relative_offset_from_base: list[float]
    target_pose_world: TargetPoseWorld
    joint_position_world: JointPositionWorld
    contact_type: str
    expected_function_after_step: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "partial_assembly_state_before": self.partial_assembly_state_before,
            "base_object": self.base_object,
            "attach_object": self.attach_object,
            "attach_region_base": self.attach_region_base,
            "attach_region_object": self.attach_region_object,
            "relative_offset_from_base": self.relative_offset_from_base,
            "target_pose_world": self.target_pose_world.to_dict(),
            "joint_position_world": self.joint_position_world.to_dict(),
            "contact_type": self.contact_type,
            "expected_function_after_step": self.expected_function_after_step,
            "reason": self.reason,
        }


@dataclass(slots=True)
class AssemblyStrategy:
    base_object: str
    strategy_summary: str
    sequence_reason: str
    filter_result_reflection: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_object": self.base_object,
            "strategy_summary": self.strategy_summary,
            "sequence_reason": self.sequence_reason,
            "filter_result_reflection": self.filter_result_reflection,
        }


@dataclass(slots=True)
class FinalStructure:
    description: str
    functional_end: str
    handle_end: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "functional_end": self.functional_end,
            "handle_end": self.handle_end,
        }


@dataclass(slots=True)
class VerificationCheck:
    item: str
    result: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "item": self.item,
            "result": self.result,
            "reason": self.reason,
        }


@dataclass(slots=True)
class Verification:
    is_valid: bool
    checks: list[VerificationCheck]

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "checks": [c.to_dict() for c in self.checks],
        }


@dataclass(slots=True)
class Feedback:
    """논문 3.1.4: 피드백은 3.1.1 (Scene Resource Parser = module2a)로 전달.
    최대 2회 수행 후에도 실패 시 task_abandoned=True."""
    need_feedback_to_module2a: bool
    repair_type: str
    suggested_action: str
    feedback_target: str | None = None
    feedback_iteration: int = 0
    task_abandoned: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "need_feedback_to_module2a": self.need_feedback_to_module2a,
            "feedback_target":          self.feedback_target,
            "repair_type":              self.repair_type,
            "suggested_action":         self.suggested_action,
            "feedback_iteration":       self.feedback_iteration,
            "task_abandoned":           self.task_abandoned,
        }


@dataclass(slots=True)
class Module3Input:
    task: str
    scene_context: dict[str, Any]
    object_physical_properties: list[dict[str, Any]]
    tool_constraints: dict[str, Any]
    selected_candidate: dict[str, Any]
    filter_result: dict[str, Any]
    feedback_iteration: int = 0  # 논문 3.1.4: 최대 2회 피드백 회차 추적

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Module3Input":
        return cls(
            task=data["task"],
            scene_context=data["scene_context"],
            object_physical_properties=data.get("object_physical_properties", []),
            tool_constraints=data["tool_constraints"],
            selected_candidate=data["selected_candidate"],
            filter_result=data["filter_result"],
            feedback_iteration=int(data.get("feedback_iteration", 0)),
        )


@dataclass(slots=True)
class Module3Output:
    assembly_strategy: AssemblyStrategy
    assembly_steps: list[AssemblyStep]
    final_structure: FinalStructure
    verification: Verification
    feedback: Feedback
    generation_trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": "module3_output",
            "schema_version": "0.1",
            "assembly_strategy": self.assembly_strategy.to_dict(),
            "assembly_steps": [s.to_dict() for s in self.assembly_steps],
            "final_structure": self.final_structure.to_dict(),
            "verification": self.verification.to_dict(),
            "feedback": self.feedback.to_dict(),
        }
