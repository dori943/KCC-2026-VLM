"""Deterministic bridge from Module 1 object-centric output to Module 2-A scene resources."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from app.models.module1_models import Module1Normalized, Module1ObjectNormalized, UsablePartNormalized
from app.utils import get_path, to_float


def build_module2_bridge_package(
    normalized: Module1Normalized,
    rule_cfg: dict[str, Any],
    vocab_registry: dict[str, Any],
) -> dict[str, Any]:
    """Create scene_resources and derived module2_common_input template.

    This does not run Module 2-A reasoning. It only prepares handoff artifacts.
    """
    atom_threshold = float(rule_cfg["atom_activation_threshold"])
    risk_threshold = float(rule_cfg["risk_activation_threshold"])

    affordance_histogram: dict[str, int] = {}
    risk_histogram: dict[str, int] = {}
    primitive_histogram: dict[str, int] = {}
    inventory: list[dict[str, Any]] = []

    atom_activation: list[dict[str, Any]] = []
    risk_activation: list[dict[str, Any]] = []
    skipped_objects: list[dict[str, str]] = []
    mapping_provenance: list[dict[str, str]] = []
    warnings: list[str] = []

    unique_atom_units: set[tuple[str, str, str]] = set()
    unique_risk_units: set[tuple[str, str, str]] = set()
    unique_primitive_units: set[tuple[str, str, str]] = set()

    for atom_name, atom_spec in rule_cfg["affordance_atoms"].items():
        for rule in atom_spec["rules"]:
            mapping_provenance.append(
                {
                    "type": "affordance_atom_rule",
                    "rule_id": rule["id"],
                    "description": f"{atom_name}:{rule['id']}",
                }
            )
    for atom_name, atom_spec in rule_cfg["risk_atoms"].items():
        for rule in atom_spec["rules"]:
            mapping_provenance.append(
                {
                    "type": "risk_atom_rule",
                    "rule_id": rule["id"],
                    "description": f"{atom_name}:{rule['id']}",
                }
            )

    for obj in normalized.objects:
        if not obj.usable_parts:
            skipped_objects.append(
                {
                    "object_id": obj.raw_object_id,
                    "reason": "no_usable_parts_in_affordance_card",
                }
            )
            continue

        part_lookup = {part.part_name: part for part in obj.functional_parts}

        for usable_part in obj.usable_parts:
            functional_part = part_lookup.get(usable_part.part_name)
            context = _build_context(
                obj=obj,
                usable_part=usable_part,
                functional_part=functional_part,
            )

            # Primitive histogram is not Module 2-A core input but useful diagnostics.
            for primitive, score in usable_part.interaction_primitives.items():
                if to_float(score) >= 0.15:
                    unit = (obj.raw_object_id, usable_part.part_name, primitive)
                    if unit not in unique_primitive_units:
                        unique_primitive_units.add(unit)
                        primitive_histogram[primitive] = primitive_histogram.get(primitive, 0) + 1

            active_atom_records = _evaluate_items(
                item_specs=rule_cfg["affordance_atoms"],
                context=context,
                threshold=atom_threshold,
            )
            for record in active_atom_records:
                unit = (obj.raw_object_id, usable_part.part_name, record["item"])
                if unit in unique_atom_units:
                    continue
                unique_atom_units.add(unit)
                affordance_histogram[record["item"]] = affordance_histogram.get(record["item"], 0) + 1
                atom_activation.append(
                    {
                        "object_id": obj.raw_object_id,
                        "part_name": usable_part.part_name,
                        "item": record["item"],
                        "score": round(record["score"], 4),
                        "threshold": atom_threshold,
                        "matched_rule_ids": record["matched_rule_ids"],
                        "uncertainty_overall": obj.uncertainty["overall"],
                    }
                )
                inventory.append(
                    {
                        "capability_unit_id": f"{obj.raw_object_id}::{usable_part.part_name}::{record['item']}",
                        "object_id": obj.raw_object_id,
                        "part_name": usable_part.part_name,
                        "atom": record["item"],
                        "atom_code": rule_cfg["affordance_atoms"][record["item"]]["code"],
                        "activation_score": round(record["score"], 4),
                        "uncertainty_overall": obj.uncertainty["overall"],
                        "supporting_primitives": sorted(
                            [
                                primitive
                                for primitive, score in usable_part.interaction_primitives.items()
                                if to_float(score) >= 0.15
                            ]
                        ),
                        "evidence": {
                            "matched_rule_ids": record["matched_rule_ids"],
                            "role_canonical": context["part"]["role_canonical"],
                            "contact_profile": context["part"]["contact_profile"],
                        },
                    }
                )

            active_risk_records = _evaluate_items(
                item_specs=rule_cfg["risk_atoms"],
                context=context,
                threshold=risk_threshold,
            )
            for record in active_risk_records:
                unit = (obj.raw_object_id, usable_part.part_name, record["item"])
                if unit in unique_risk_units:
                    continue
                unique_risk_units.add(unit)
                risk_histogram[record["item"]] = risk_histogram.get(record["item"], 0) + 1
                risk_activation.append(
                    {
                        "object_id": obj.raw_object_id,
                        "part_name": usable_part.part_name,
                        "item": record["item"],
                        "score": round(record["score"], 4),
                        "threshold": risk_threshold,
                        "matched_rule_ids": record["matched_rule_ids"],
                        "uncertainty_overall": obj.uncertainty["overall"],
                    }
                )

    if not normalized.objects:
        warnings.append("No objects in normalized model. scene_resources will be empty.")

    uncertainty_values = [obj.uncertainty["overall"] for obj in normalized.objects]
    if uncertainty_values:
        mean_uncertainty = round(sum(uncertainty_values) / len(uncertainty_values), 4)
        max_uncertainty = round(max(uncertainty_values), 4)
        high_uncertainty_object_ids = [
            obj.raw_object_id for obj in normalized.objects if obj.uncertainty["overall"] >= 0.36
        ]
    else:
        mean_uncertainty = 0.0
        max_uncertainty = 0.0
        high_uncertainty_object_ids = []

    scene_resources = {
        "schema_name": "scene_resources_from_module1",
        "schema_version": "0.1",
        "source_schema_name": normalized.source_schema_name,
        "source_schema_version": normalized.source_schema_version,
        "bridge_rule_version": rule_cfg["bridge_rule_version"],
        "counting_rule": "unique (object_id, part_name, atom) capability unit",
        "resource_summary": {
            "affordance_histogram": _sorted_histogram(affordance_histogram),
            "risk_histogram": _sorted_histogram(risk_histogram),
            "primitive_histogram": _sorted_histogram(primitive_histogram),
        },
        "resource_inventory": inventory,
    }

    module2_template = {
        "schema_name": "module2_common_input_template_derived_min",
        "schema_version": "0.1",
        "task_brief": {
            "user_goal": None,
            "success_criteria": [],
            "task_notes": [],
        },
        "scene_resources": {
            "resource_summary": {
                "affordance_histogram": _sorted_histogram(affordance_histogram),
                "risk_histogram": _sorted_histogram(risk_histogram),
                "primitive_histogram": _sorted_histogram(primitive_histogram),
            },
            "resource_inventory": inventory,
        },
    }

    diagnostics = {
        "schema_name": "module2_bridge_diagnostics",
        "schema_version": "0.1",
        "bridge_rule_version": rule_cfg["bridge_rule_version"],
        "counting_rule": rule_cfg["counting_rule"],
        "assumptions": [
            "task_brief is unknown at this stage; bridge exports scene capability pool only.",
            "capability count is based on unique (object_id, part_name, atom).",
            "primitive histogram is diagnostic only; Module 2-A core summary is atom-centric.",
        ],
        "warnings": warnings,
        "object_count": len(normalized.objects),
        "capability_unit_count": len(inventory),
        "atom_histogram": _sorted_histogram(affordance_histogram),
        "risk_histogram": _sorted_histogram(risk_histogram),
        "primitive_histogram": _sorted_histogram(primitive_histogram),
        "uncertainty_summary": {
            "mean_overall": mean_uncertainty,
            "max_overall": max_uncertainty,
            "high_uncertainty_object_ids": high_uncertainty_object_ids,
        },
        "atom_activation": atom_activation,
        "risk_activation": risk_activation,
        "skipped_objects": skipped_objects,
        "mapping_provenance": mapping_provenance,
        "vocab_registry_refs": {
            "interaction_primitives_count": len(
                vocab_registry["module2a"]["interaction_primitives"]
            ),
            "affordance_atom_count": len(vocab_registry["module2a"]["affordance_atoms"]),
            "risk_atom_count": len(vocab_registry["module2a"]["risk_atoms"]),
        },
    }

    return {
        "scene_resources_from_module1": scene_resources,
        "module2_common_input_template": module2_template,
        "module2_bridge_diagnostics": diagnostics,
    }


def _build_context(
    obj: Module1ObjectNormalized,
    usable_part: UsablePartNormalized,
    functional_part: Any,
) -> dict[str, Any]:
    part_dict = (
        {
            "part_name": functional_part.part_name,
            "role": functional_part.role,
            "role_canonical": functional_part.role_canonical,
            "contact_profile": functional_part.contact_profile,
            "local_material": functional_part.local_material,
            "local_property_note": functional_part.local_property_note,
            "local_property_tags": list(functional_part.local_property_tags),
        }
        if functional_part
        else {
            "part_name": usable_part.part_name,
            "role": "unknown",
            "role_canonical": "unknown",
            "contact_profile": "unknown",
            "local_material": "unknown",
            "local_property_note": "missing functional_part mapping",
            "local_property_tags": [],
        }
    )

    physical = {
        key: {"label": value.label, "confidence": value.confidence, "evidence": value.evidence}
        for key, value in obj.physical.items()
    }
    target_mode = asdict(usable_part.target_mode_numeric)
    return {
        "part": part_dict,
        "geometry": dict(obj.geometry),
        "physical": physical,
        "state": dict(obj.state),
        "target_mode": target_mode,
        "primitives": dict(usable_part.interaction_primitives),
        "affordances": dict(usable_part.affordance_scores),
    }


def _evaluate_items(
    item_specs: dict[str, Any],
    context: dict[str, Any],
    threshold: float,
) -> list[dict[str, Any]]:
    active: list[dict[str, Any]] = []
    for item_name, item_spec in item_specs.items():
        score = 0.0
        matched_rule_ids: list[str] = []
        for rule in item_spec["rules"]:
            if _match_all(rule["conditions"], context):
                score += float(rule["weight"])
                matched_rule_ids.append(rule["id"])
        score = min(score, 1.0)
        if score >= threshold:
            active.append(
                {
                    "item": item_name,
                    "score": score,
                    "matched_rule_ids": matched_rule_ids,
                }
            )
    return active


def _match_all(conditions: list[dict[str, Any]], context: dict[str, Any]) -> bool:
    for cond in conditions:
        if not _match_condition(cond, context):
            return False
    return True


def _match_condition(cond: dict[str, Any], context: dict[str, Any]) -> bool:
    path = cond["path"]
    op = cond["op"]
    actual = get_path(context, path)
    if op == "eq":
        return actual == cond.get("value")
    if op == "in":
        return actual in cond.get("values", [])
    if op == "gte":
        if actual is None:
            return False
        return to_float(actual, default=-1e9) >= to_float(cond.get("value"), default=0.0)
    if op == "lte":
        if actual is None:
            return False
        return to_float(actual, default=1e9) <= to_float(cond.get("value"), default=0.0)
    if op == "contains_any":
        text = str(actual or "").lower()
        return any(token.lower() in text for token in cond.get("values", []))
    return False


def _sorted_histogram(hist: dict[str, int]) -> dict[str, int]:
    return {k: hist[k] for k in sorted(hist.keys())}
