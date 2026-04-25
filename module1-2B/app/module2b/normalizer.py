"""Normalizer for Module 2-B upstream bundle into typed internal context."""

from __future__ import annotations

from typing import Any

from app.module2b.models import (
    FunctionalPart,
    InventoryObject,
    NormalizedContext,
    SceneRelation,
    Subgoal,
)
from app.module2b.utils import clamp01, dedupe_keep_order, normalize_text


_VISIBILITY_ALLOWED = {"full", "partial", "heavily_occluded"}
_ACCESS_ALLOWED = {"clear", "partial", "occluded", "nested", "entangled"}


def normalize_module2b_bundle(bundle: dict[str, Any]) -> NormalizedContext:
    """Normalize raw bundle to Module 2-B internal typed context model."""
    module2_common_input = bundle.get("module2_common_input", {})
    module2a_output = bundle.get("module2a_output", {})

    log: list[dict[str, Any]] = []

    task_id = str(module2_common_input.get("task_id", "")).strip()
    if not task_id:
        task_id = "task_unknown"
        log.append(
            {
                "kind": "default",
                "path": "module2_common_input.task_id",
                "reason": "missing_task_id",
                "value": task_id,
            }
        )

    task_brief_raw = module2_common_input.get("task_brief", {})
    task_brief = {
        "user_goal": task_brief_raw.get("user_goal"),
        "success_criteria": [str(item) for item in task_brief_raw.get("success_criteria", [])],
        "task_notes": [str(item) for item in task_brief_raw.get("task_notes", [])],
    }

    task_model_raw = module2a_output.get("task_model", {})
    task_model = {
        "task_restatement": str(task_model_raw.get("task_restatement", "")),
        "primary_success_condition": str(task_model_raw.get("primary_success_condition", "")),
    }

    text_corpus = _build_task_text_corpus(
        task_brief=task_brief,
        task_model=task_model,
        subgoals_raw=module2a_output.get("subgoals", []),
    )

    inventory_raw = (
        module2_common_input.get("scene_resources", {})
        .get("resource_inventory", [])
    )
    inventory: list[InventoryObject] = []
    object_id_order: list[str] = []
    object_lookup: dict[str, int] = {}

    for idx, raw_obj in enumerate(inventory_raw):
        object_id = str(raw_obj.get("object_id", "")).strip()
        if not object_id:
            object_id = f"obj_missing_{idx:02d}"
            log.append(
                {
                    "kind": "default",
                    "path": f"module2_common_input.scene_resources.resource_inventory[{idx}].object_id",
                    "reason": "missing_object_id",
                    "value": object_id,
                }
            )

        object_name = str(raw_obj.get("object_name", "")).strip()
        object_type = str(raw_obj.get("object_type_canonical", "")).strip()
        coarse_location_hint = str(raw_obj.get("coarse_location_hint", "")).strip()

        visibility = str(raw_obj.get("visibility", "partial")).strip()
        if visibility not in _VISIBILITY_ALLOWED:
            log.append(
                {
                    "kind": "default",
                    "path": f"object[{idx}].visibility",
                    "reason": "invalid_visibility_value",
                    "value": visibility,
                    "applied": "partial",
                }
            )
            visibility = "partial"

        accessibility = str(raw_obj.get("accessibility", "partial")).strip()
        if accessibility not in _ACCESS_ALLOWED:
            log.append(
                {
                    "kind": "default",
                    "path": f"object[{idx}].accessibility",
                    "reason": "invalid_accessibility_value",
                    "value": accessibility,
                    "applied": "partial",
                }
            )
            accessibility = "partial"

        state_raw = raw_obj.get("state", {})
        pose_class = str(state_raw.get("pose_class", "unknown")).strip() or "unknown"
        support_context = str(state_raw.get("support_context", "unknown")).strip() or "unknown"

        relations: list[SceneRelation] = []
        for rel_idx, rel in enumerate(raw_obj.get("scene_relations", [])):
            relation = str(rel.get("relation", "unknown")).strip() or "unknown"
            object_ref_raw = rel.get("object_ref")
            object_ref = str(object_ref_raw).strip() if isinstance(object_ref_raw, str) else None
            relation_note = str(rel.get("relation_note", "")).strip()
            relations.append(
                SceneRelation(
                    relation=relation,
                    object_ref=object_ref,
                    relation_note=relation_note,
                    provenance_id=f"obj:{object_id}:rel:{rel_idx:02d}",
                )
            )

        geometry_raw = raw_obj.get("geometry_cues", {})
        geometry = {
            "shape_class": str(geometry_raw.get("shape_class", "unknown")),
            "aspect_ratio_hint": str(geometry_raw.get("aspect_ratio_hint", "unknown")),
            "size_relative": str(geometry_raw.get("size_relative", "unknown")),
            "thickness_class": str(geometry_raw.get("thickness_class", "unknown")),
            "primary_contact_profile": str(
                geometry_raw.get("primary_contact_profile", "unknown")
            ),
            "has_pointed_or_thin_end": bool(geometry_raw.get("has_pointed_or_thin_end", False)),
            "has_flat_contact_face": bool(geometry_raw.get("has_flat_contact_face", False)),
            "has_open_cavity": bool(geometry_raw.get("has_open_cavity", False)),
            "roll_risk_source": str(geometry_raw.get("roll_risk_source", "none")),
        }

        parts: list[FunctionalPart] = []
        for part in raw_obj.get("functional_parts", []):
            parts.append(
                FunctionalPart(
                    part_name=str(part.get("part_name", "unknown")),
                    role_canonical=str(part.get("role_canonical", "unknown")),
                    contact_profile=str(part.get("contact_profile", "unknown")),
                    local_property_tags=[str(tag) for tag in part.get("local_property_tags", [])],
                )
            )

        target_summary_raw = raw_obj.get("target_mode_numeric_summary", {})
        target_summary = {
            "exposure_ratio": _to_optional_float(target_summary_raw.get("exposure_ratio")),
            "clearance_ratio": _to_optional_float(target_summary_raw.get("clearance_ratio")),
            "usable_span_m": _to_optional_float(target_summary_raw.get("usable_span_m")),
            "local_thickness_m": _to_optional_float(target_summary_raw.get("local_thickness_m")),
            "tip_radius_m": _to_optional_float(target_summary_raw.get("tip_radius_m")),
            "flat_patch_m2": _to_optional_float(target_summary_raw.get("flat_patch_m2")),
            "approach_directions_count": _to_optional_int(
                target_summary_raw.get("approach_directions_count")
            ),
        }

        uncertainty_raw = raw_obj.get("uncertainty", {}).get("overall", 0.5)
        uncertainty_value = _to_optional_float(uncertainty_raw)
        if uncertainty_value is None:
            uncertainty_value = 0.5
            log.append(
                {
                    "kind": "default",
                    "path": f"object[{idx}].uncertainty.overall",
                    "reason": "non_numeric_uncertainty",
                    "value": uncertainty_raw,
                    "applied": uncertainty_value,
                }
            )
        clamped_uncertainty = clamp01(uncertainty_value)
        if clamped_uncertainty != uncertainty_value:
            log.append(
                {
                    "kind": "clamp",
                    "path": f"object[{idx}].uncertainty.overall",
                    "reason": "uncertainty_out_of_range",
                    "value": uncertainty_value,
                    "applied": clamped_uncertainty,
                }
            )

        inventory_obj = InventoryObject(
            index=idx,
            object_id=object_id,
            object_name=object_name,
            object_type_canonical=object_type,
            coarse_location_hint=coarse_location_hint,
            visibility=visibility,
            accessibility=accessibility,
            pose_class=pose_class,
            support_context=support_context,
            scene_relations=relations,
            geometry_cues=geometry,
            functional_parts=parts,
            target_mode_numeric_summary=target_summary,
            uncertainty_overall=clamped_uncertainty,
        )
        inventory.append(inventory_obj)
        object_id_order.append(object_id)
        object_lookup[object_id] = idx

    subgoals: list[Subgoal] = []
    for idx, subgoal_raw in enumerate(module2a_output.get("subgoals", []), start=1):
        subgoals.append(
            Subgoal(
                order=idx,
                subgoal_id=str(subgoal_raw.get("subgoal_id", f"sg_{idx:02d}")),
                objective=str(subgoal_raw.get("objective", "")),
                success_condition=str(subgoal_raw.get("success_condition", "")),
            )
        )

    return NormalizedContext(
        schema_name="module2b_normalized_context",
        schema_version="0.1",
        task_id=task_id,
        task_brief=task_brief,
        task_model=task_model,
        task_text_corpus=text_corpus,
        inventory=inventory,
        subgoals=subgoals,
        object_id_order=object_id_order,
        object_lookup=object_lookup,
        normalization_log=log,
    )


def _build_task_text_corpus(
    task_brief: dict[str, Any],
    task_model: dict[str, Any],
    subgoals_raw: list[dict[str, Any]],
) -> list[str]:
    corpus = []

    goal = task_brief.get("user_goal")
    if isinstance(goal, str) and goal.strip():
        corpus.append(goal)

    for key in ("success_criteria", "task_notes"):
        for item in task_brief.get(key, []):
            value = str(item).strip()
            if value:
                corpus.append(value)

    restatement = str(task_model.get("task_restatement", "")).strip()
    if restatement:
        corpus.append(restatement)

    primary_success = str(task_model.get("primary_success_condition", "")).strip()
    if primary_success:
        corpus.append(primary_success)

    for subgoal in subgoals_raw:
        objective = str(subgoal.get("objective", "")).strip()
        success_condition = str(subgoal.get("success_condition", "")).strip()
        if objective:
            corpus.append(objective)
        if success_condition:
            corpus.append(success_condition)

    normalized = [normalize_text(item) for item in corpus if normalize_text(item)]
    return dedupe_keep_order(normalized)


def _to_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
