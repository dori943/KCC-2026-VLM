"""Minimal downstream Module 3 handoff reader for readiness checks."""

from __future__ import annotations

from typing import Any


def read_module3_handoff_payload(
    preview_payload: dict[str, Any],
    module2b_output: dict[str, Any],
) -> dict[str, Any]:
    """Parse Module 2-B handoff artifacts into a downstream-ready payload.

    This is intentionally strict enough to act as the "Module 3 reader"
    readiness probe used by Module 2-B evaluation.
    """
    if not isinstance(preview_payload, dict):
        raise ValueError("preview_payload must be object")
    if preview_payload.get("schema_name") != "module3_handoff_preview":
        raise ValueError("module3_handoff_preview schema_name mismatch")

    handoff = module2b_output.get("module3_handoff")
    if not isinstance(handoff, dict):
        raise ValueError("module2b_output.module3_handoff missing")

    derived_constraints = module2b_output.get("derived_constraints")
    if not isinstance(derived_constraints, dict):
        raise ValueError("module2b_output.derived_constraints missing")

    constraint_catalog = derived_constraints.get("constraint_catalog", [])
    if not isinstance(constraint_catalog, list):
        raise ValueError("constraint_catalog must be list")

    constraint_lookup: dict[str, dict[str, Any]] = {}
    for item in constraint_catalog:
        if not isinstance(item, dict):
            continue
        constraint_id = item.get("constraint_id")
        if isinstance(constraint_id, str) and constraint_id:
            constraint_lookup[constraint_id] = item

    handoff_ids = handoff.get("handoff_constraint_ids", [])
    if not isinstance(handoff_ids, list):
        raise ValueError("handoff_constraint_ids must be list")

    resolved_constraints: list[dict[str, Any]] = []
    for raw_id in handoff_ids:
        if not isinstance(raw_id, str) or not raw_id:
            raise ValueError("handoff_constraint_ids must contain non-empty strings")
        if raw_id not in constraint_lookup:
            raise ValueError(f"unknown handoff constraint id: {raw_id}")
        resolved_constraints.append(constraint_lookup[raw_id])

    return {
        "handoff_status": handoff.get("handoff_status"),
        "constraint_units_policy": handoff.get("constraint_units_policy"),
        "constraint_ids": list(handoff_ids),
        "resolved_constraints": resolved_constraints,
        "pending_merge_sources": handoff.get("pending_merge_sources", []),
        "omitted_constraint_families": preview_payload.get("omitted_constraint_families", []),
    }
