"""Typed data models for Module 2-D candidate filtering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class StageScores:
    geometry: float
    physics: float
    commonsense: float
    task_fit: float = 0.0
    emergence: float = 0.0
    necessity: float = 0.0
    role_contribution: float = 0.0

    def total(self) -> float:
        """Weighted score used for candidate selection.

        The final score intentionally favors emergent, scenario-necessary,
        role-complementary combinations over merely feasible assemblies.
        """
        return round(
            (
                self.geometry
                + self.physics
                + self.commonsense
                + 1.5 * self.task_fit
                + 1.5 * self.emergence
                + 1.2 * self.necessity
                + 1.3 * self.role_contribution
            )
            / 8.5,
            4,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "geometry": self.geometry,
            "physics": self.physics,
            "commonsense": self.commonsense,
            "task_fit": self.task_fit,
            "emergence": self.emergence,
            "necessity": self.necessity,
            "role_contribution": self.role_contribution,
        }


@dataclass(slots=True)
class WeakPoint:
    stage: str
    item: str
    score: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "item": self.item,
            "score": self.score,
            "reason": self.reason,
        }


@dataclass(slots=True)
class RepairAnalysis:
    repair_type: str
    target_issue: str
    reason: str
    repair_strategy: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "repair_type": self.repair_type,
            "target_issue": self.target_issue,
            "reason": self.reason,
            "repair_strategy": self.repair_strategy,
        }


@dataclass(slots=True)
class EvaluatedCandidate:
    candidate_id: str
    stage_scores: StageScores
    total_score: float
    passed: bool
    failed_stage: str | None
    weak_points: list[WeakPoint]
    repair_analysis: RepairAnalysis
    environment_filter: dict[str, Any] | None = field(default=None)
    assembly_filter: dict[str, Any] | None = field(default=None)
    checklist: dict[str, Any] | None = field(default=None)
    pre_analysis: dict[str, Any] | None = field(default=None)
    emergence_analysis: dict[str, Any] | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "pre_analysis": self.pre_analysis,
            "environment_filter": self.environment_filter,
            "assembly_filter": self.assembly_filter,
            "checklist": self.checklist,
            "emergence_analysis": self.emergence_analysis,
            "stage_scores": self.stage_scores.to_dict(),
            "total_score": self.total_score,
            "pass": self.passed,
            "failed_stage": self.failed_stage,
            "weak_points": [wp.to_dict() for wp in self.weak_points],
            "repair_analysis": self.repair_analysis.to_dict(),
        }


@dataclass(slots=True)
class FeedbackDecision:
    need_feedback: bool
    feedback_target: str | None
    reason: str
    dominant_failure_pattern: list[str]
    suggested_relaxations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "need_feedback": self.need_feedback,
            "feedback_target": self.feedback_target,
            "reason": self.reason,
            "dominant_failure_pattern": self.dominant_failure_pattern,
            "suggested_relaxations": self.suggested_relaxations,
        }


@dataclass(slots=True)
class Module2DInput:
    task: str
    tool_constraints: dict[str, Any]
    candidate_tools: list[dict[str, Any]]
    scene_objects: list[dict[str, Any]]
    object_physical_properties: list[dict[str, Any]]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Module2DInput":
        return cls(
            task=data["task"],
            tool_constraints=data["tool_constraints"],
            candidate_tools=data["candidate_tools"],
            scene_objects=data["scene_objects"],
            object_physical_properties=data.get("object_physical_properties", []),
        )


@dataclass(slots=True)
class Module2DOutput:
    evaluated_candidates: list[EvaluatedCandidate]
    selected_candidate_id: str | None
    feedback_decision: FeedbackDecision

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": "module2d_output",
            "schema_version": "0.1",
            "evaluated_candidates": [c.to_dict() for c in self.evaluated_candidates],
            "selected_candidate_id": self.selected_candidate_id,
            "feedback_decision": self.feedback_decision.to_dict(),
        }
