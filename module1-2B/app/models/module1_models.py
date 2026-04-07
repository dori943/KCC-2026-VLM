"""Typed internal models for Module 1 normalized data."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class PropertyObservation:
    """One qualitative property with confidence and short evidence."""

    label: str
    confidence: float
    evidence: str


@dataclass
class TargetModeNumeric:
    """Numeric affordance cues emitted by Module 1."""

    point_score: float | None
    edge_score: float | None
    face_score: float | None
    rim_score: float | None
    cavity_score: float | None
    axis_score: float | None
    hook_gap_score: float | None
    exposure_ratio: float | None
    clearance_ratio: float | None
    usable_span_m: float | None
    local_thickness_m: float | None
    tip_radius_m: float | None
    flat_patch_m2: float | None
    approach_directions_count: int | None


@dataclass
class UsablePartNormalized:
    """Normalized affordance part card."""

    part_name: str
    affordance_scores: dict[str, float]
    interaction_primitives: dict[str, float]
    target_mode_numeric: TargetModeNumeric


@dataclass
class FunctionalPartNormalized:
    """Normalized functional part entry."""

    part_name: str
    role: str
    role_canonical: str
    contact_profile: str
    local_material: str
    local_property_note: str
    local_property_tags: list[str]


@dataclass
class Module1ObjectNormalized:
    """One normalized object preserving source traceability."""

    raw_object_id: str
    normalized_object_key: str
    raw_order_index: int
    object_name: str
    object_type_canonical: str
    grouped: bool
    quantity_estimate: int
    coarse_location_hint: str
    visibility: str
    accessibility: str
    state: dict[str, Any]
    scene_relations: list[dict[str, Any]]
    observed_vs_inferred: dict[str, list[str]]
    geometry: dict[str, Any]
    scale_anchor_status: str
    material_hypotheses: list[dict[str, Any]]
    functional_parts: list[FunctionalPartNormalized]
    usable_parts: list[UsablePartNormalized]
    affordance_card_meta: dict[str, Any]
    physical: dict[str, PropertyObservation]
    uncertainty: dict[str, float]
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class Module1Normalized:
    """Root normalized payload used by bridges and simulators."""

    schema_name: str
    schema_version: str
    source_schema_name: str
    source_schema_version: str
    ordering_rule: str
    objects: list[Module1ObjectNormalized]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass tree into JSON-friendly dictionary."""
        return asdict(self)
