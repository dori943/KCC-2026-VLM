"""Module 3 handoff builder for Module 2-B env-only output."""

from __future__ import annotations

from typing import Any

from app.module2b.utils import dedupe_keep_order


def build_module3_handoff(
    target_binding: dict[str, Any],
    measurements: list[dict[str, Any]],
    derived_constraints: dict[str, Any],
    pending_merge_sources: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build strict module3_handoff plus human-readable preview artifact."""
    pending_sources = pending_merge_sources or ["material_reasoner"]
    omitted_families = [
        "risk_limit",
        "target_material_state",
        "damage_sensitivity",
        "contact_style_preference",
    ]

    constraint_ids = [
        item["constraint_id"]
        for item in derived_constraints.get("constraint_catalog", [])
        if isinstance(item, dict)
    ]
    handoff_constraint_ids = dedupe_keep_order(
        derived_constraints.get("global_constraint_ids", []) + constraint_ids
    )

    unit_policy = _select_constraint_units_policy(measurements)
    handoff_status = _resolve_handoff_status(
        target_binding_status=target_binding.get("binding_status", "deferred"),
        constraint_count=len(constraint_ids),
        pending_sources=pending_sources,
    )

    handoff = {
        "handoff_status": handoff_status,
        "constraint_units_policy": unit_policy,
        "handoff_constraint_ids": handoff_constraint_ids,
        "pending_merge_sources": dedupe_keep_order(pending_sources),
        "omitted_constraint_families": omitted_families,
        "notes": [
            "Module 2-B output is env-only and excludes material/state/damage merge.",
            "Module 3 should merge pending sources before final tool ranking/planning.",
        ],
    }

    preview = {
        "schema_name": "module3_handoff_preview",
        "schema_version": "0.1",
        "handoff_status": handoff_status,
        "constraint_units_policy": unit_policy,
        "handoff_constraint_count": len(handoff_constraint_ids),
        "pending_merge_sources": dedupe_keep_order(pending_sources),
        "omitted_constraint_families": omitted_families,
        "target_binding_status": target_binding.get("binding_status", "deferred"),
    }

    return handoff, preview


def _select_constraint_units_policy(measurements: list[dict[str, Any]]) -> str:
    if not measurements:
        return "ordinal_preferred"

    units = [item.get("unit") for item in measurements]
    has_ordinal = any(unit == "level_1_to_5" for unit in units)
    has_metric = any(unit in {"m", "deg"} for unit in units)

    basis_scores = {
        "observed": 3,
        "estimated_from_anchor": 2,
        "relative_geometry": 1,
        "task_text_prior": 0,
    }
    basis_mean = sum(
        basis_scores.get(str(item.get("estimate_basis")), 0) for item in measurements
    ) / float(len(measurements))

    if has_metric and not has_ordinal and basis_mean >= 1.5:
        return "metric_strict"
    if has_metric and has_ordinal:
        return "mixed_metric_and_ordinal"
    return "ordinal_preferred"


def _resolve_handoff_status(
    target_binding_status: str,
    constraint_count: int,
    pending_sources: list[str],
) -> str:
    if target_binding_status == "deferred" and constraint_count == 0:
        return "blocked"
    if pending_sources:
        return "partial"
    if target_binding_status in {"resolved", "partially_resolved"} and constraint_count > 0:
        return "ready"
    return "partial"
