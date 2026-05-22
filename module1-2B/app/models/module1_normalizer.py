"""Normalize validated Module 1 raw output into typed internal model."""

from __future__ import annotations

from typing import Any

from app.models.module1_models import (
    FunctionalPartNormalized,
    Module1Normalized,
    Module1ObjectNormalized,
    PropertyObservation,
    TargetModeNumeric,
    UsablePartNormalized,
)


def normalize_module1_raw(raw: dict[str, Any]) -> Module1Normalized:
    """Normalize strict raw payload into internal typed model.

    The normalizer does not infer task-level meaning. It only creates a typed
    object model and preserves traceability back to raw paths.
    """
    objects: list[Module1ObjectNormalized] = []
    warnings: list[str] = []

    raw_objects = raw.get("objects", [])
    if not raw_objects:
        warnings.append(
            "No objects found in Module 1 raw output. Downstream bridges will have empty inventories."
        )

    for index, raw_obj in enumerate(raw_objects):
        physical_raw: dict[str, dict[str, Any]] = raw_obj["physical_properties"]
        physical: dict[str, PropertyObservation] = {
            key: PropertyObservation(
                label=value["label"],
                confidence=float(value["confidence"]),
                evidence=value["evidence"],
            )
            for key, value in physical_raw.items()
        }

        functional_parts: list[FunctionalPartNormalized] = []
        for part_idx, part in enumerate(raw_obj["functional_parts"]):
            functional_parts.append(
                FunctionalPartNormalized(
                    part_name=part["part_name"],
                    role=part["role"],
                    role_canonical=part["role_canonical"],
                    contact_profile=part["contact_profile"],
                    local_material=part["local_material"],
                    local_property_note=part["local_property_note"],
                    local_property_tags=list(part["local_property_tags"]),
                )
            )
            if not part["part_name"]:
                warnings.append(
                    f"{raw_obj['object_id']} functional_parts[{part_idx}] has empty part_name."
                )

        usable_parts: list[UsablePartNormalized] = []
        for part_idx, part in enumerate(raw_obj["affordance_card"]["usable_parts"]):
            target = part["target_mode_numeric"]
            target_numeric = TargetModeNumeric(
                point_score=target.get("point_score"),
                edge_score=target.get("edge_score"),
                face_score=target.get("face_score"),
                rim_score=target.get("rim_score"),
                cavity_score=target.get("cavity_score"),
                axis_score=target.get("axis_score"),
                hook_gap_score=target.get("hook_gap_score"),
                exposure_ratio=target.get("exposure_ratio"),
                clearance_ratio=target.get("clearance_ratio"),
                usable_span_m=target.get("usable_span_m"),
                local_thickness_m=target.get("local_thickness_m"),
                tip_radius_m=target.get("tip_radius_m"),
                flat_patch_m2=target.get("flat_patch_m2"),
                approach_directions_count=target.get("approach_directions_count"),
            )
            usable_parts.append(
                UsablePartNormalized(
                    part_name=part["part_name"],
                    affordance_scores=dict(part["affordance_scores"]),
                    interaction_primitives=dict(part["interaction_primitives"]),
                    target_mode_numeric=target_numeric,
                )
            )
            if not part["part_name"]:
                warnings.append(
                    f"{raw_obj['object_id']} affordance_card.usable_parts[{part_idx}] has empty part_name."
                )

        object_normalized = Module1ObjectNormalized(
            raw_object_id=raw_obj["object_id"],
            normalized_object_key=raw_obj["object_id"],
            raw_order_index=index,
            object_name=raw_obj["object_name"],
            object_type_canonical=raw_obj["object_type_canonical"],
            grouped=bool(raw_obj["grouped"]),
            quantity_estimate=int(raw_obj["quantity_estimate"]),
            coarse_location_hint=raw_obj["coarse_location_hint"],
            visibility=raw_obj["visibility"],
            accessibility=raw_obj["accessibility"],
            state=dict(raw_obj["state"]),
            scene_relations=[dict(item) for item in raw_obj["scene_relations"]],
            observed_vs_inferred={
                "observed_cues": list(raw_obj["observed_vs_inferred"]["observed_cues"]),
                "inferred_aspects": list(raw_obj["observed_vs_inferred"]["inferred_aspects"]),
                "assumed_aspects": list(raw_obj["observed_vs_inferred"]["assumed_aspects"]),
            },
            geometry=dict(raw_obj["geometry_cues"]),
            scale_anchor_status=raw_obj["scale_anchor_status"],
            material_hypotheses=[dict(item) for item in raw_obj["material_hypotheses"]],
            functional_parts=functional_parts,
            usable_parts=usable_parts,
            affordance_card_meta={
                "object_name": raw_obj["affordance_card"]["object_name"],
                "observed_visual_features": list(
                    raw_obj["affordance_card"]["observed_visual_features"]
                ),
                "inferred_physical_properties": list(
                    raw_obj["affordance_card"]["inferred_physical_properties"]
                ),
                "connection_modes": [
                    dict(mode) for mode in raw_obj["affordance_card"]["connection_modes"]
                ],
                "weaknesses_or_risks": list(raw_obj["affordance_card"]["weaknesses_or_risks"]),
                "uncertain_points": list(raw_obj["affordance_card"]["uncertain_points"]),
                "confidence": float(raw_obj["affordance_card"]["confidence"]),
            },
            physical=physical,
            uncertainty={k: float(v) for k, v in raw_obj["uncertainty"].items()},
            provenance={
                "raw_path": f"objects[{index}]",
                "raw_object_id": raw_obj["object_id"],
                "assumptions": [],
                "defaults_applied": [],
                "clamps_applied": [],
            },
        )
        objects.append(object_normalized)

    return Module1Normalized(
        schema_name="module1_normalized_internal",
        schema_version="0.1",
        source_schema_name=raw["schema_name"],
        source_schema_version=raw["schema_version"],
        ordering_rule=raw["scene_summary"]["ordering_rule"],
        objects=objects,
        warnings=warnings,
    )
