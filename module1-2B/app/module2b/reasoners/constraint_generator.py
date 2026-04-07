"""Derived env-only constraint generator for Module 2-B baseline."""

from __future__ import annotations

from typing import Any

from app.module2b.models import NormalizedContext
from app.module2b.utils import clamp01, dedupe_keep_order, stable_round


def generate_constraints(
    context: NormalizedContext,
    target_binding: dict[str, Any],
    environment_result: dict[str, Any],
    measurements: list[dict[str, Any]],
    rules_cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate deterministic env-only constraints from numeric + topology evidence."""
    subgoal_ids = [subgoal.subgoal_id for subgoal in context.subgoals]
    subgoal_order = {subgoal_id: idx for idx, subgoal_id in enumerate(subgoal_ids, start=1)}

    mapping_cfg = rules_cfg.get("measurement_to_constraints", {})
    topology_cfg = rules_cfg.get("topology_to_constraints", {})
    priority_order = rules_cfg.get("priority_order", {"high": 1, "medium": 2, "low": 3})

    candidates: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []

    for measurement in measurements:
        parameter_name = measurement["parameter_name"]
        mappings = mapping_cfg.get(parameter_name, [])
        for mapping in mappings:
            transformed = _transform_bounds(
                measurement=measurement,
                transform_name=str(mapping["bound_transform"]),
            )
            if transformed is None:
                trace_rows.append(
                    {
                        "type": "measurement_mapping_skipped",
                        "parameter_name": parameter_name,
                        "mapping_parameter": mapping["parameter_name"],
                        "reason": "unsupported_bound_for_transform",
                        "measurement_id": measurement["measurement_id"],
                    }
                )
                continue

            subgoal_scope = _resolve_subgoal_scope(mapping["applies_to"], subgoal_ids)
            source_refs = dedupe_keep_order(
                [measurement["measurement_id"]] + measurement.get("source_refs", [])
            )
            target_binding_ids = (
                []
                if target_binding.get("binding_status") == "deferred"
                else [target_binding["binding_id"]]
            )

            confidence = clamp01(float(measurement["confidence"]) - 0.05)

            candidates.append(
                {
                    "category": mapping["category"],
                    "parameter_name": mapping["parameter_name"],
                    "applies_to": mapping["applies_to"],
                    "hardness": mapping["hardness"],
                    "priority": mapping["priority"],
                    "bound_type": transformed["bound_type"],
                    "unit": transformed["unit"],
                    "lower_value": transformed["lower_value"],
                    "upper_value": transformed["upper_value"],
                    "measurement_ids": [measurement["measurement_id"]],
                    "target_binding_ids": target_binding_ids,
                    "source_refs": source_refs,
                    "subgoal_ids": subgoal_scope,
                    "confidence": stable_round(confidence, 4),
                    "sort_subgoal_order": min((subgoal_order[s] for s in subgoal_scope), default=999),
                    "trace_rule": f"measurement:{parameter_name}->{mapping['parameter_name']}",
                }
            )

    tag_lookup = {
        item["label"]: item
        for item in environment_result.get("topology_tags", [])
        if isinstance(item, dict)
    }
    for label, topo_rule in topology_cfg.items():
        tag = tag_lookup.get(label)
        if tag is None:
            continue

        subgoal_scope = _resolve_subgoal_scope(topo_rule["applies_to"], subgoal_ids)
        source_refs = dedupe_keep_order([tag["tag_id"]] + tag.get("source_refs", []))
        target_binding_ids = (
            [] if target_binding.get("binding_status") == "deferred" else [target_binding["binding_id"]]
        )

        confidence = clamp01(float(tag.get("confidence", 0.5)) * 0.85)

        candidates.append(
            {
                "category": topo_rule["category"],
                "parameter_name": topo_rule["parameter_name"],
                "applies_to": topo_rule["applies_to"],
                "hardness": topo_rule["hardness"],
                "priority": topo_rule["priority"],
                "bound_type": topo_rule["bound_type"],
                "unit": topo_rule["unit"],
                "lower_value": _value_or_none(topo_rule.get("lower_value")),
                "upper_value": _value_or_none(topo_rule.get("upper_value")),
                "measurement_ids": [],
                "target_binding_ids": target_binding_ids,
                "source_refs": source_refs,
                "subgoal_ids": subgoal_scope,
                "confidence": stable_round(confidence, 4),
                "sort_subgoal_order": min((subgoal_order[s] for s in subgoal_scope), default=999),
                "trace_rule": f"topology:{label}->{topo_rule['parameter_name']}",
            }
        )

    deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for candidate in candidates:
        key = (
            candidate["category"],
            candidate["parameter_name"],
            candidate["applies_to"],
            candidate["hardness"],
            candidate["priority"],
            candidate["bound_type"],
            candidate["unit"],
            candidate["lower_value"],
            candidate["upper_value"],
            tuple(candidate["subgoal_ids"]),
        )
        if key not in deduped:
            deduped[key] = candidate
            continue
        existing = deduped[key]
        existing["measurement_ids"] = dedupe_keep_order(
            existing["measurement_ids"] + candidate["measurement_ids"]
        )
        existing["source_refs"] = dedupe_keep_order(existing["source_refs"] + candidate["source_refs"])
        existing["target_binding_ids"] = dedupe_keep_order(
            existing["target_binding_ids"] + candidate["target_binding_ids"]
        )
        existing["confidence"] = stable_round(
            max(float(existing["confidence"]), float(candidate["confidence"])),
            4,
        )

    ordered = sorted(
        deduped.values(),
        key=lambda item: (
            item["sort_subgoal_order"],
            priority_order.get(item["priority"], 999),
            item["category"],
            item["parameter_name"],
            item["applies_to"],
        ),
    )

    constraint_catalog: list[dict[str, Any]] = []
    for idx, item in enumerate(ordered, start=1):
        constraint_catalog.append(
            {
                "constraint_id": f"c_{idx:02d}",
                "category": item["category"],
                "parameter_name": item["parameter_name"],
                "applies_to": item["applies_to"],
                "hardness": item["hardness"],
                "priority": item["priority"],
                "bound_type": item["bound_type"],
                "unit": item["unit"],
                "lower_value": item["lower_value"],
                "upper_value": item["upper_value"],
                "measurement_ids": item["measurement_ids"],
                "target_binding_ids": item["target_binding_ids"],
                "source_refs": item["source_refs"],
                "subgoal_ids": item["subgoal_ids"],
                "confidence": item["confidence"],
            }
        )

    global_constraint_ids = [
        item["constraint_id"]
        for item in constraint_catalog
        if item["priority"] == "high" and item["hardness"] == "hard"
    ]

    subgoal_bindings: list[dict[str, Any]] = []
    for subgoal_id in subgoal_ids:
        ids = [
            item["constraint_id"]
            for item in constraint_catalog
            if subgoal_id in item["subgoal_ids"]
        ]
        subgoal_bindings.append(
            {
                "subgoal_id": subgoal_id,
                "constraint_ids": ids,
            }
        )

    catalog_by_rule = {
        item["constraint_id"]: item
        for item in constraint_catalog
    }
    for item in constraint_catalog:
        trace_rows.append(
            {
                "type": "constraint_generated",
                "constraint_id": item["constraint_id"],
                "source_refs": item["source_refs"],
                "measurement_ids": item["measurement_ids"],
                "priority": item["priority"],
            }
        )

    trace = {
        "candidate_count_before_dedup": len(candidates),
        "constraint_count_after_dedup": len(constraint_catalog),
        "rows": trace_rows,
        "ordering_rule": "(subgoal_order, priority, category, parameter_name, applies_to)",
        "constraint_catalog_snapshot": catalog_by_rule,
    }

    return {
        "constraint_catalog": constraint_catalog,
        "global_constraint_ids": global_constraint_ids,
        "subgoal_bindings": subgoal_bindings,
    }, trace


def _resolve_subgoal_scope(applies_to: str, subgoal_ids: list[str]) -> list[str]:
    if not subgoal_ids:
        return []
    if applies_to == "placement_strategy":
        return [subgoal_ids[-1]]
    return list(subgoal_ids)


def _transform_bounds(measurement: dict[str, Any], transform_name: str) -> dict[str, Any] | None:
    lower = _value_or_none(measurement.get("lower_value"))
    upper = _value_or_none(measurement.get("upper_value"))
    unit = measurement.get("unit")

    if transform_name == "upper_minus_margin":
        base = upper if upper is not None else lower
        if base is None:
            return None
        margin = 0.15
        if unit == "level_1_to_5":
            out_upper = max(1.0, base - 1.0)
        else:
            out_upper = base * (1.0 - margin)
        return {
            "bound_type": "upper_bound",
            "unit": unit,
            "lower_value": None,
            "upper_value": _value_or_none(out_upper),
        }

    if transform_name == "lower_plus_margin":
        base = lower if lower is not None else upper
        if base is None:
            return None
        margin = 0.10
        if unit == "level_1_to_5":
            out_lower = min(5.0, base + 1.0)
        else:
            out_lower = base * (1.0 + margin)
        return {
            "bound_type": "lower_bound",
            "unit": unit,
            "lower_value": _value_or_none(out_lower),
            "upper_value": None,
        }

    if transform_name == "lower_no_change":
        if lower is None:
            return None
        return {
            "bound_type": "lower_bound",
            "unit": unit,
            "lower_value": lower,
            "upper_value": None,
        }

    if transform_name == "upper_no_change":
        if upper is None:
            return None
        return {
            "bound_type": "upper_bound",
            "unit": unit,
            "lower_value": None,
            "upper_value": upper,
        }

    if transform_name == "span_to_stability_level":
        span = lower if lower is not None else upper
        if span is None:
            return None
        if unit == "m":
            if span >= 0.08:
                level = 4.0
            elif span >= 0.04:
                level = 3.0
            else:
                level = 2.0
        else:
            level = min(5.0, max(1.0, span))
        return {
            "bound_type": "lower_bound",
            "unit": "level_1_to_5",
            "lower_value": level,
            "upper_value": None,
        }

    return None


def _value_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return stable_round(float(value), 5)
    except (TypeError, ValueError):
        return None
