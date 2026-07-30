"""Input/output validators for Module 3."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from app.feedback.verification_items import ALL_ITEMS


@dataclass(slots=True)
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "errors": self.errors, "warnings": self.warnings}


class Module3InputValidator:
    def validate(self, data: dict[str, Any]) -> ValidationResult:
        errors, warnings = [], []
        if not data.get("task"):
            errors.append("task 필드가 없거나 비어 있습니다.")
        if "scene_context" not in data:
            errors.append("scene_context 필드가 없습니다.")
        elif "scene_objects" not in data["scene_context"]:
            errors.append("scene_context.scene_objects 필드가 없습니다.")
        if "tool_constraints" not in data:
            errors.append("tool_constraints 필드가 없습니다.")
        if "selected_candidate" not in data:
            errors.append("selected_candidate 필드가 없습니다.")
        else:
            for key in ["candidate_id", "used_objects", "structure_description"]:
                if key not in data["selected_candidate"]:
                    errors.append(f"selected_candidate.{key} 필드가 없습니다.")
        if "filter_result" not in data:
            errors.append("filter_result 필드가 없습니다.")
        if "object_physical_properties" not in data:
            warnings.append("object_physical_properties 필드가 없습니다.")
        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


class Module3OutputValidator:
    # 논문 §3.4 8항목 검증 — 단일 정의 소스(app/feedback/verification_items.py)에서 참조.
    REQUIRED_CHECK_ITEMS = set(ALL_ITEMS)

    def validate(self, data: dict[str, Any]) -> ValidationResult:
        errors, warnings = [], []
        for key in ["assembly_strategy", "assembly_steps", "final_structure",
                    "verification", "feedback"]:
            if key not in data:
                errors.append(f"{key} 필드가 없습니다.")

        steps = data.get("assembly_steps", [])
        if not isinstance(steps, list) or len(steps) == 0:
            errors.append("assembly_steps가 비어 있습니다.")
        for i, step in enumerate(steps):
            if "joint_position_world" not in step:
                errors.append(f"assembly_steps[{i}].joint_position_world 필드가 없습니다.")
            elif "position" not in step["joint_position_world"]:
                errors.append(f"assembly_steps[{i}].joint_position_world.position 필드가 없습니다.")

        checks = data.get("verification", {}).get("checks", [])
        if len(checks) < 5:
            warnings.append(f"verification.checks가 {len(checks)}개입니다. 최소 5개 권장.")
        check_items = {c.get("item") for c in checks}
        missing = self.REQUIRED_CHECK_ITEMS - check_items
        if missing:
            warnings.append(f"verification.checks 누락 항목: {missing}")

        fb = data.get("feedback", {})
        # 논문 3.1.4 기준: need_feedback_to_module2a (구 need_feedback_to_module2c 허용)
        if "need_feedback_to_module2a" not in fb and "need_feedback_to_module2c" not in fb:
            errors.append("feedback.need_feedback_to_module2a 필드가 없습니다.")

        if "reasoning_trace" not in data:
            warnings.append("reasoning_trace 필드가 없습니다. 추론 과정을 확인할 수 없습니다.")

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
