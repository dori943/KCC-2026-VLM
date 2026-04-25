"""Input/output validators for Module 2-C."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "errors": self.errors, "warnings": self.warnings}


class Module2CInputValidator:
    def validate(self, data: dict[str, Any]) -> ValidationResult:
        errors, warnings = [], []
        if not data.get("task"):
            errors.append("task 필드가 없거나 비어 있습니다.")
        if "tool_constraints" not in data:
            errors.append("tool_constraints 필드가 없습니다.")
        else:
            for key in ["global_constraints", "subgoal_constraints"]:
                if key not in data["tool_constraints"]:
                    errors.append(f"tool_constraints.{key} 필드가 없습니다.")
        if not isinstance(data.get("scene_objects"), list):
            errors.append("scene_objects 필드가 없거나 리스트가 아닙니다.")
        elif len(data["scene_objects"]) == 0:
            warnings.append("scene_objects가 비어 있습니다.")
        if "object_physical_properties" not in data:
            warnings.append("object_physical_properties 필드가 없습니다.")
        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


class Module2COutputValidator:
    def validate(self, data: dict[str, Any]) -> ValidationResult:
        errors, warnings = [], []
        candidates = data.get("candidate_tools", [])
        if not isinstance(candidates, list):
            errors.append("candidate_tools가 리스트가 아닙니다.")
            return ValidationResult(valid=False, errors=errors)
        if len(candidates) == 0:
            warnings.append("후보가 0개입니다.")
        seen_ids: set[str] = set()
        for i, c in enumerate(candidates):
            for key in ["candidate_id", "used_objects", "structure_description",
                        "function_mapping", "subgoal_coverage", "failure_modes"]:
                if key not in c:
                    errors.append(f"candidate_tools[{i}].{key} 필드가 없습니다.")
            cid = c.get("candidate_id", "")
            if cid in seen_ids:
                errors.append(f"candidate_id 중복: {cid}")
            seen_ids.add(cid)
        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
