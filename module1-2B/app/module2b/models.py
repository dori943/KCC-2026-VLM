"""Typed normalized data models for Module 2-B pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SceneRelation:
    """Normalized relation record attached to one inventory object."""

    relation: str
    object_ref: str | None
    relation_note: str
    provenance_id: str


@dataclass(slots=True)
class FunctionalPart:
    """Normalized functional part for one inventory object."""

    part_name: str
    role_canonical: str
    contact_profile: str
    local_property_tags: list[str]


@dataclass(slots=True)
class InventoryObject:
    """Normalized inventory object from module2_common_input."""

    index: int
    object_id: str
    object_name: str
    object_type_canonical: str
    coarse_location_hint: str
    visibility: str
    accessibility: str
    pose_class: str
    support_context: str
    scene_relations: list[SceneRelation]
    geometry_cues: dict[str, Any]
    functional_parts: list[FunctionalPart]
    target_mode_numeric_summary: dict[str, Any]
    uncertainty_overall: float


@dataclass(slots=True)
class Subgoal:
    """Reduced Module 2-A subgoal form used by Module 2-B."""

    order: int
    subgoal_id: str
    objective: str
    success_condition: str


@dataclass(slots=True)
class NormalizedContext:
    """Layer 2 typed context model for Module 2-B reasoning."""

    schema_name: str
    schema_version: str
    task_id: str
    task_brief: dict[str, Any]
    task_model: dict[str, Any]
    task_text_corpus: list[str]
    inventory: list[InventoryObject]
    subgoals: list[Subgoal]
    object_id_order: list[str]
    object_lookup: dict[str, int] = field(default_factory=dict)
    normalization_log: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass model into JSON-safe dictionary."""
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "task_brief": self.task_brief,
            "task_model": self.task_model,
            "task_text_corpus": list(self.task_text_corpus),
            "inventory": [
                {
                    "index": item.index,
                    "object_id": item.object_id,
                    "object_name": item.object_name,
                    "object_type_canonical": item.object_type_canonical,
                    "coarse_location_hint": item.coarse_location_hint,
                    "visibility": item.visibility,
                    "accessibility": item.accessibility,
                    "pose_class": item.pose_class,
                    "support_context": item.support_context,
                    "scene_relations": [
                        {
                            "relation": rel.relation,
                            "object_ref": rel.object_ref,
                            "relation_note": rel.relation_note,
                            "provenance_id": rel.provenance_id,
                        }
                        for rel in item.scene_relations
                    ],
                    "geometry_cues": item.geometry_cues,
                    "functional_parts": [
                        {
                            "part_name": part.part_name,
                            "role_canonical": part.role_canonical,
                            "contact_profile": part.contact_profile,
                            "local_property_tags": list(part.local_property_tags),
                        }
                        for part in item.functional_parts
                    ],
                    "target_mode_numeric_summary": item.target_mode_numeric_summary,
                    "uncertainty_overall": item.uncertainty_overall,
                }
                for item in self.inventory
            ],
            "subgoals": [
                {
                    "order": sg.order,
                    "subgoal_id": sg.subgoal_id,
                    "objective": sg.objective,
                    "success_condition": sg.success_condition,
                }
                for sg in self.subgoals
            ],
            "object_id_order": list(self.object_id_order),
            "normalization_log": list(self.normalization_log),
        }
