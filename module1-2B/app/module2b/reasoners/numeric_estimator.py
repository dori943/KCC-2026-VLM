"""Numeric estimate derivation engine for Module 2-B env-only baseline."""

from __future__ import annotations

from typing import Any

from app.module2b.models import NormalizedContext
from app.module2b.utils import clamp01, dedupe_keep_order, stable_round


def derive_numeric_estimates(
    context: NormalizedContext,
    target_binding: dict[str, Any],
    environment_result: dict[str, Any],
    rules_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Derive conservative environment numeric estimates and detailed trace."""
    structures = environment_result.get("relevant_structures", [])
    tag_labels = {item["label"] for item in environment_result.get("topology_tags", [])}
    access_profile = environment_result.get("access_path_profile", {})

    structure_ids = [item["environment_structure_id"] for item in structures]
    structure_by_role: dict[str, list[str]] = {}
    for structure in structures:
        role = structure["structure_role"]
        structure_by_role.setdefault(role, []).append(structure["environment_structure_id"])

    target_obj = _pick_target_object(context=context, target_binding=target_binding)
    uncertainty = target_obj.uncertainty_overall if target_obj is not None else 0.55
    target_summary = target_obj.target_mode_numeric_summary if target_obj is not None else {}
    target_ref = target_obj.object_id if target_obj is not None else None

    candidates: list[dict[str, Any]] = []
    omissions: list[dict[str, Any]] = []

    def append_measurement(
        parameter_name: str,
        unit: str,
        bound_type: str,
        lower_value: float | None,
        upper_value: float | None,
        estimate_basis: str,
        related_roles: list[str],
        rule_hit: str,
    ) -> None:
        related_env_ids = _collect_related_env_ids(structure_by_role, related_roles, structure_ids)
        if not related_env_ids:
            omissions.append(
                {
                    "item": parameter_name,
                    "reason": "no_related_environment_structure",
                    "source_refs": [target_binding["binding_id"]],
                }
            )
            return

        confidence = _estimate_confidence(
            estimate_basis=estimate_basis,
            uncertainty=uncertainty,
            unit=unit,
        )

        source_refs = dedupe_keep_order(
            related_env_ids
            + ([target_ref] if target_ref else [])
            + [target_binding["binding_id"]]
        )

        candidates.append(
            {
                "parameter_name": parameter_name,
                "unit": unit,
                "bound_type": bound_type,
                "lower_value": _round_or_none(lower_value),
                "upper_value": _round_or_none(upper_value),
                "estimate_basis": estimate_basis,
                "related_environment_structure_ids": related_env_ids,
                "source_refs": source_refs,
                "confidence": stable_round(confidence, 4),
                "rule_hit": rule_hit,
            }
        )

    local_thickness = _as_float(target_summary.get("local_thickness_m"))
    usable_span = _as_float(target_summary.get("usable_span_m"))
    exposure_ratio = _as_float(target_summary.get("exposure_ratio"))
    flat_patch = _as_float(target_summary.get("flat_patch_m2"))

    if {"partial_opening", "narrow_gap", "container_neck"} & tag_labels:
        anchor = local_thickness or (usable_span * 0.45 if usable_span is not None else None)
        if anchor is not None:
            append_measurement(
                parameter_name="opening_width",
                unit="m",
                bound_type="range",
                lower_value=max(0.005, anchor * 0.65),
                upper_value=max(0.007, anchor * 1.40),
                estimate_basis="estimated_from_anchor",
                related_roles=["partial_opening", "narrow_gap", "container_neck"],
                rule_hit="opening_from_local_thickness",
            )
        else:
            append_measurement(
                parameter_name="opening_width",
                unit="level_1_to_5",
                bound_type="range",
                lower_value=2.0,
                upper_value=4.0,
                estimate_basis="task_text_prior",
                related_roles=["partial_opening", "narrow_gap", "container_neck"],
                rule_hit="opening_ordinal_fallback",
            )

        anchor_h = usable_span or (local_thickness * 1.5 if local_thickness is not None else None)
        if anchor_h is not None:
            append_measurement(
                parameter_name="opening_height",
                unit="m",
                bound_type="range",
                lower_value=max(0.007, anchor_h * 0.55),
                upper_value=max(0.009, anchor_h * 1.35),
                estimate_basis="relative_geometry",
                related_roles=["partial_opening", "container_neck"],
                rule_hit="opening_height_from_span",
            )
        else:
            append_measurement(
                parameter_name="opening_height",
                unit="level_1_to_5",
                bound_type="range",
                lower_value=2.0,
                upper_value=4.0,
                estimate_basis="task_text_prior",
                related_roles=["partial_opening", "container_neck"],
                rule_hit="opening_height_ordinal_fallback",
            )

    if "container_neck" in tag_labels:
        anchor = local_thickness * 1.25 if local_thickness is not None else None
        if anchor is not None:
            append_measurement(
                parameter_name="neck_inner_diameter",
                unit="m",
                bound_type="upper_bound",
                lower_value=None,
                upper_value=max(0.006, anchor * 1.15),
                estimate_basis="estimated_from_anchor",
                related_roles=["container_neck"],
                rule_hit="neck_from_local_thickness",
            )
        else:
            append_measurement(
                parameter_name="neck_inner_diameter",
                unit="m",
                bound_type="upper_bound",
                lower_value=None,
                upper_value=0.080,
                estimate_basis="task_text_prior",
                related_roles=["container_neck"],
                rule_hit="neck_prior_upper",
            )

    if {"deep_recess", "container_cavity", "confined_channel"} & tag_labels:
        anchor = usable_span * 1.20 if usable_span is not None else None
        if anchor is not None:
            append_measurement(
                parameter_name="recess_depth",
                unit="m",
                bound_type="range",
                lower_value=max(0.020, anchor * 0.80),
                upper_value=max(0.030, anchor * 1.80),
                estimate_basis="relative_geometry",
                related_roles=["deep_recess", "container_cavity", "confined_channel"],
                rule_hit="recess_from_span",
            )
            append_measurement(
                parameter_name="reachable_depth",
                unit="m",
                bound_type="range",
                lower_value=max(0.020, anchor * 0.70),
                upper_value=max(0.035, anchor * 1.90),
                estimate_basis="relative_geometry",
                related_roles=["deep_recess", "container_cavity", "confined_channel"],
                rule_hit="reachable_depth_from_recess",
            )
        else:
            append_measurement(
                parameter_name="recess_depth",
                unit="level_1_to_5",
                bound_type="range",
                lower_value=3.0,
                upper_value=5.0,
                estimate_basis="task_text_prior",
                related_roles=["deep_recess", "container_cavity", "confined_channel"],
                rule_hit="recess_ordinal_fallback",
            )
            append_measurement(
                parameter_name="reachable_depth",
                unit="level_1_to_5",
                bound_type="range",
                lower_value=3.0,
                upper_value=5.0,
                estimate_basis="task_text_prior",
                related_roles=["deep_recess", "container_cavity", "confined_channel"],
                rule_hit="reachable_depth_ordinal_fallback",
            )

    if {"narrow_gap", "constraining_surface_pair", "confined_channel"} & tag_labels:
        anchor = local_thickness * 0.90 if local_thickness is not None else None
        if anchor is not None:
            append_measurement(
                parameter_name="lateral_clearance",
                unit="m",
                bound_type="upper_bound",
                lower_value=None,
                upper_value=max(0.003, anchor),
                estimate_basis="estimated_from_anchor",
                related_roles=["narrow_gap", "constraining_surface_pair", "confined_channel"],
                rule_hit="lateral_clearance_from_thickness",
            )
        else:
            append_measurement(
                parameter_name="lateral_clearance",
                unit="level_1_to_5",
                bound_type="upper_bound",
                lower_value=None,
                upper_value=3.0,
                estimate_basis="task_text_prior",
                related_roles=["narrow_gap", "constraining_surface_pair", "confined_channel"],
                rule_hit="lateral_clearance_ordinal_upper",
            )

    if {"under_overhang", "occluding_edge"} & tag_labels:
        anchor = local_thickness * 1.15 if local_thickness is not None else None
        if anchor is not None:
            append_measurement(
                parameter_name="vertical_clearance",
                unit="m",
                bound_type="upper_bound",
                lower_value=None,
                upper_value=max(0.004, anchor),
                estimate_basis="estimated_from_anchor",
                related_roles=["under_overhang", "occluding_edge"],
                rule_hit="vertical_clearance_from_thickness",
            )
        else:
            append_measurement(
                parameter_name="vertical_clearance",
                unit="level_1_to_5",
                bound_type="upper_bound",
                lower_value=None,
                upper_value=3.0,
                estimate_basis="task_text_prior",
                related_roles=["under_overhang", "occluding_edge"],
                rule_hit="vertical_clearance_ordinal_upper",
            )

    rotation_clearance = access_profile.get("rotation_clearance", "sufficient")
    if rotation_clearance == "severely_limited":
        angle_low, angle_high = 5.0, 35.0
    elif rotation_clearance == "limited":
        angle_low, angle_high = 10.0, 60.0
    else:
        angle_low, angle_high = 15.0, 85.0
    append_measurement(
        parameter_name="available_entry_angle_deg",
        unit="deg",
        bound_type="range",
        lower_value=angle_low,
        upper_value=angle_high,
        estimate_basis="observed",
        related_roles=[
            "partial_opening",
            "container_neck",
            "narrow_gap",
            "under_overhang",
            "occluding_edge",
        ],
        rule_hit="entry_angle_from_rotation_profile",
    )

    if exposure_ratio is not None or usable_span is not None:
        if exposure_ratio is not None and usable_span is not None:
            edge_len = max(0.003, usable_span * max(0.1, min(1.0, exposure_ratio)))
            append_measurement(
                parameter_name="target_exposed_edge_length",
                unit="m",
                bound_type="lower_bound",
                lower_value=edge_len,
                upper_value=None,
                estimate_basis="estimated_from_anchor",
                related_roles=["occluding_edge", "partial_opening", "narrow_gap", "support_surface"],
                rule_hit="exposed_edge_from_exposure_ratio",
            )
        else:
            append_measurement(
                parameter_name="target_exposed_edge_length",
                unit="level_1_to_5",
                bound_type="lower_bound",
                lower_value=2.0,
                upper_value=None,
                estimate_basis="task_text_prior",
                related_roles=["occluding_edge", "partial_opening", "narrow_gap", "support_surface"],
                rule_hit="exposed_edge_ordinal_lower",
            )

    if access_profile.get("available_support_surface", False) or {"support_surface", "contact_plane"} & tag_labels:
        if flat_patch is not None and flat_patch > 0:
            span = flat_patch ** 0.5
            append_measurement(
                parameter_name="support_surface_span",
                unit="m",
                bound_type="lower_bound",
                lower_value=max(0.008, span * 0.55),
                upper_value=None,
                estimate_basis="estimated_from_anchor",
                related_roles=["support_surface", "contact_plane"],
                rule_hit="support_span_from_flat_patch",
            )
        elif usable_span is not None:
            append_measurement(
                parameter_name="support_surface_span",
                unit="m",
                bound_type="lower_bound",
                lower_value=max(0.008, usable_span * 0.40),
                upper_value=None,
                estimate_basis="relative_geometry",
                related_roles=["support_surface", "contact_plane"],
                rule_hit="support_span_from_usable_span",
            )
        else:
            append_measurement(
                parameter_name="support_surface_span",
                unit="level_1_to_5",
                bound_type="lower_bound",
                lower_value=2.0,
                upper_value=None,
                estimate_basis="task_text_prior",
                related_roles=["support_surface", "contact_plane"],
                rule_hit="support_span_ordinal_lower",
            )

    deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for cand in candidates:
        key = (
            tuple(cand["related_environment_structure_ids"]),
            cand["parameter_name"],
            cand["bound_type"],
            cand["unit"],
            cand["lower_value"],
            cand["upper_value"],
        )
        if key not in deduped:
            deduped[key] = cand
            continue
        merged = deduped[key]
        merged["source_refs"] = dedupe_keep_order(merged["source_refs"] + cand["source_refs"])
        merged["confidence"] = stable_round(max(float(merged["confidence"]), float(cand["confidence"])), 4)

    ordered = sorted(
        deduped.values(),
        key=lambda item: (
            item["related_environment_structure_ids"][0],
            item["parameter_name"],
            item["bound_type"],
        ),
    )

    out_measurements: list[dict[str, Any]] = []
    for idx, item in enumerate(ordered, start=1):
        out_measurements.append(
            {
                "measurement_id": f"m_{idx:02d}",
                "parameter_name": item["parameter_name"],
                "unit": item["unit"],
                "bound_type": item["bound_type"],
                "lower_value": item["lower_value"],
                "upper_value": item["upper_value"],
                "estimate_basis": item["estimate_basis"],
                "related_environment_structure_ids": item["related_environment_structure_ids"],
                "source_refs": item["source_refs"],
                "confidence": item["confidence"],
            }
        )

    parameter_set = {item["parameter_name"] for item in out_measurements}
    required_supported = {
        "opening_width",
        "opening_height",
        "neck_inner_diameter",
        "recess_depth",
        "reachable_depth",
        "lateral_clearance",
        "vertical_clearance",
        "available_entry_angle_deg",
        "target_exposed_edge_length",
        "support_surface_span",
    }
    for parameter_name in sorted(required_supported - parameter_set):
        omissions.append(
            {
                "item": parameter_name,
                "reason": "insufficient_environment_evidence",
                "source_refs": [target_binding["binding_id"]],
            }
        )

    trace = {
        "raw_candidates": candidates,
        "deduped_candidates": ordered,
        "omissions": omissions,
        "formula": {
            "confidence": "basis_score - uncertainty_penalty - ordinal_penalty (clamped 0..1)",
            "basis_scores": {
                "observed": 0.82,
                "estimated_from_anchor": 0.68,
                "relative_geometry": 0.58,
                "task_text_prior": 0.42,
            },
        },
    }

    return out_measurements, trace, omissions


def _pick_target_object(context: NormalizedContext, target_binding: dict[str, Any]) -> Any | None:
    object_lookup = {obj.object_id: obj for obj in context.inventory}
    primary_targets = target_binding.get("primary_targets", [])
    if primary_targets:
        primary_id = primary_targets[0].get("object_id")
        return object_lookup.get(primary_id)
    candidate_ids = target_binding.get("candidate_ids_ranked", [])
    if candidate_ids:
        return object_lookup.get(candidate_ids[0])
    return context.inventory[0] if context.inventory else None


def _collect_related_env_ids(
    structure_by_role: dict[str, list[str]],
    related_roles: list[str],
    fallback_structure_ids: list[str],
) -> list[str]:
    env_ids: list[str] = []
    for role in related_roles:
        env_ids.extend(structure_by_role.get(role, []))
    env_ids = dedupe_keep_order(env_ids)
    if env_ids:
        return env_ids
    if fallback_structure_ids:
        return [fallback_structure_ids[0]]
    return []


def _estimate_confidence(estimate_basis: str, uncertainty: float, unit: str) -> float:
    basis_scores = {
        "observed": 0.82,
        "estimated_from_anchor": 0.68,
        "relative_geometry": 0.58,
        "task_text_prior": 0.42,
    }
    base = basis_scores.get(estimate_basis, 0.40)
    uncertainty_penalty = 0.30 * max(0.0, min(1.0, uncertainty))
    ordinal_penalty = 0.08 if unit == "level_1_to_5" else 0.0
    return clamp01(base - uncertainty_penalty - ordinal_penalty)


def _round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return stable_round(value, 5)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
