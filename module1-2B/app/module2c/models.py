"""Typed data models for Module 2-C candidate generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class FunctionMapping:
    object: str
    function: str
    related_physics: str


@dataclass(slots=True)
class SubgoalCoverage:
    subgoal_id: str
    covered: bool
    method: str


@dataclass(slots=True)
class FailureMode:
    subgoal_id: str
    failure_type: str
    cause: str
    related_constraint: str


@dataclass(slots=True)
class CandidateTool:
    candidate_id: str
    used_objects: list[str]
    structure_description: str
    function_mapping: list[FunctionMapping]
    subgoal_coverage: list[SubgoalCoverage]
    failure_modes: list[FailureMode]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "used_objects": self.used_objects,
            "structure_description": self.structure_description,
            "function_mapping": [
                {"object": fm.object, "function": fm.function, "related_physics": fm.related_physics}
                for fm in self.function_mapping
            ],
            "subgoal_coverage": [
                {"subgoal_id": sc.subgoal_id, "covered": sc.covered, "method": sc.method}
                for sc in self.subgoal_coverage
            ],
            "failure_modes": [
                {"subgoal_id": fm.subgoal_id, "failure_type": fm.failure_type,
                 "cause": fm.cause, "related_constraint": fm.related_constraint}
                for fm in self.failure_modes
            ],
        }


@dataclass(slots=True)
class Module2CInput:
    task: str
    tool_constraints: dict[str, Any]
    scene_objects: list[dict[str, Any]]
    object_physical_properties: list[dict[str, Any]]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Module2CInput":
        return cls(
            task=data["task"],
            tool_constraints=data["tool_constraints"],
            scene_objects=data["scene_objects"],
            object_physical_properties=data.get("object_physical_properties", []),
        )


@dataclass(slots=True)
class Module2COutput:
    candidate_tools: list[CandidateTool]
    generation_trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": "module2c_output",
            "schema_version": "0.1",
            "candidate_tools": [c.to_dict() for c in self.candidate_tools],
        }
