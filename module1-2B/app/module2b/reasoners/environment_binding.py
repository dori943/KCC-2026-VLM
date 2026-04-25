"""Environment structure synthesis engine for Module 2-B env-only baseline."""

from __future__ import annotations

from typing import Any

from app.module2b.models import NormalizedContext
from app.module2b.utils import clamp01, dedupe_keep_order, normalize_text, stable_round


def run_environment_binding(
    context: NormalizedContext,
    target_binding: dict[str, Any],
    rules_cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Synthesize environment structures, topology tags, and access path profile."""
    relation_to_tags = rules_cfg.get("relation_to_tags", {})
    relation_to_structures = rules_cfg.get("relation_to_structures", {})
    keyword_to_structure = rules_cfg.get("keyword_to_structure", {})
    support_context_keywords = rules_cfg.get("support_context_keywords", {})
    role_priority = {
        role: idx
        for idx, role in enumerate(rules_cfg.get("structure_role_priority", []), start=1)
    }

    object_by_id = {obj.object_id: obj for obj in context.inventory}
    anchor_ids = [item.get("object_id") for item in target_binding.get("primary_targets", [])]
    anchor_ids = [item for item in anchor_ids if isinstance(item, str)]
    if not anchor_ids:
        candidate_ids = target_binding.get("candidate_ids_ranked", [])
        if candidate_ids:
            anchor_ids = [candidate_ids[0]]

    task_text = " ".join(context.task_text_corpus)

    events: list[dict[str, Any]] = []
    order_counter = 0

    for anchor_id in anchor_ids:
        obj = object_by_id.get(anchor_id)
        if obj is None:
            continue

        for relation in obj.scene_relations:
            relation_name = relation.relation
            related_ids = [obj.object_id]
            if relation.object_ref:
                related_ids.append(relation.object_ref)
            related_ids = [rid for rid in related_ids if rid in object_by_id]

            tags = relation_to_tags.get(relation_name, [])
            structures = relation_to_structures.get(relation_name, [])
            for sidx, structure_role in enumerate(structures):
                order_counter += 1
                topology_tag = tags[sidx] if sidx < len(tags) else structure_role
                events.append(
                    {
                        "order": order_counter,
                        "structure_role": structure_role,
                        "topology_tag": topology_tag,
                        "related_object_ids": dedupe_keep_order(related_ids),
                        "evidence": f"relation:{relation_name}",
                        "source_refs": dedupe_keep_order([obj.object_id, target_binding["binding_id"]]),
                        "confidence": 0.78,
                    }
                )

            combined_text = normalize_text(relation.relation_note + " " + task_text)
            for keyword, mapped in keyword_to_structure.items():
                if keyword.lower() not in combined_text:
                    continue
                order_counter += 1
                events.append(
                    {
                        "order": order_counter,
                        "structure_role": mapped["role"],
                        "topology_tag": mapped["tag"],
                        "related_object_ids": dedupe_keep_order(related_ids),
                        "evidence": f"keyword:{keyword}",
                        "source_refs": dedupe_keep_order([obj.object_id, target_binding["binding_id"]]),
                        "confidence": 0.62,
                    }
                )

        if bool(obj.geometry_cues.get("has_open_cavity", False)):
            order_counter += 1
            events.append(
                {
                    "order": order_counter,
                    "structure_role": "container_cavity",
                    "topology_tag": "container_cavity",
                    "related_object_ids": [obj.object_id],
                    "evidence": "geometry:has_open_cavity",
                    "source_refs": [obj.object_id, target_binding["binding_id"]],
                    "confidence": 0.70,
                }
            )

        if bool(obj.geometry_cues.get("has_flat_contact_face", False)):
            order_counter += 1
            events.append(
                {
                    "order": order_counter,
                    "structure_role": "support_surface",
                    "topology_tag": "support_surface",
                    "related_object_ids": [obj.object_id],
                    "evidence": "geometry:has_flat_contact_face",
                    "source_refs": [obj.object_id, target_binding["binding_id"]],
                    "confidence": 0.64,
                }
            )
            order_counter += 1
            events.append(
                {
                    "order": order_counter,
                    "structure_role": "contact_plane",
                    "topology_tag": "contact_plane",
                    "related_object_ids": [obj.object_id],
                    "evidence": "geometry:flat_contact_implies_plane",
                    "source_refs": [obj.object_id, target_binding["binding_id"]],
                    "confidence": 0.58,
                }
            )

        if obj.accessibility in {"occluded", "nested", "entangled"}:
            order_counter += 1
            events.append(
                {
                    "order": order_counter,
                    "structure_role": "occluding_edge",
                    "topology_tag": "occluding_edge",
                    "related_object_ids": [obj.object_id],
                    "evidence": f"accessibility:{obj.accessibility}",
                    "source_refs": [obj.object_id, target_binding["binding_id"]],
                    "confidence": 0.68,
                }
            )

        if obj.accessibility in {"nested", "entangled"}:
            order_counter += 1
            events.append(
                {
                    "order": order_counter,
                    "structure_role": "deep_recess",
                    "topology_tag": "deep_recess",
                    "related_object_ids": [obj.object_id],
                    "evidence": f"accessibility:{obj.accessibility}",
                    "source_refs": [obj.object_id, target_binding["binding_id"]],
                    "confidence": 0.66,
                }
            )

        support_text = normalize_text(obj.support_context)
        for structure_role, keywords in support_context_keywords.items():
            if any(keyword.lower() in support_text for keyword in keywords):
                order_counter += 1
                events.append(
                    {
                        "order": order_counter,
                        "structure_role": structure_role,
                        "topology_tag": structure_role,
                        "related_object_ids": [obj.object_id],
                        "evidence": "support_context_keyword",
                        "source_refs": [obj.object_id, target_binding["binding_id"]],
                        "confidence": 0.60,
                    }
                )

    if not events and context.inventory:
        # Fallback: preserve env-only trace even with weak evidence.
        fallback_obj = context.inventory[0]
        events.append(
            {
                "order": 1,
                "structure_role": "obstacle",
                "topology_tag": "obstacle",
                "related_object_ids": [fallback_obj.object_id],
                "evidence": "fallback:weak_environment_evidence",
                "source_refs": [fallback_obj.object_id, target_binding["binding_id"]],
                "confidence": 0.30,
            }
        )

    topology_tags = _build_topology_tags(events)
    relevant_structures = _build_structures(events, role_priority)
    access_path_profile = _build_access_path_profile(
        context=context,
        anchor_ids=anchor_ids,
        topology_tags=topology_tags,
        defaults=rules_cfg.get("access_profile_defaults", {}),
        accessibility_to_rotation=rules_cfg.get("accessibility_to_rotation", {}),
    )

    trace = {
        "events": events,
        "topology_tags": topology_tags,
        "relevant_structures": relevant_structures,
        "access_path_profile_reasoning": {
            "anchor_ids": anchor_ids,
            "derived_from_tags": [item["label"] for item in topology_tags],
            "derived_profile": access_path_profile,
        },
        "confidence_components": {
            "event_confidences": [item["confidence"] for item in events],
            "mean_event_confidence": stable_round(
                sum(item["confidence"] for item in events) / float(len(events)),
                4,
            )
            if events
            else 0.0,
        },
    }

    return {
        "topology_tags": topology_tags,
        "relevant_structures": relevant_structures,
        "access_path_profile": access_path_profile,
    }, trace


def _build_topology_tags(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_label: dict[str, dict[str, Any]] = {}
    for event in events:
        label = event["topology_tag"]
        entry = by_label.get(label)
        if entry is None:
            by_label[label] = {
                "first_order": event["order"],
                "confidence_values": [float(event["confidence"])],
                "source_refs": list(event["source_refs"]),
            }
            continue
        entry["first_order"] = min(entry["first_order"], event["order"])
        entry["confidence_values"].append(float(event["confidence"]))
        entry["source_refs"] = dedupe_keep_order(entry["source_refs"] + list(event["source_refs"]))

    ordered = sorted(by_label.items(), key=lambda item: (item[1]["first_order"], item[0]))
    out: list[dict[str, Any]] = []
    for idx, (label, record) in enumerate(ordered, start=1):
        confidence = clamp01(
            sum(record["confidence_values"]) / float(len(record["confidence_values"]))
        )
        out.append(
            {
                "tag_id": f"tag_{idx:02d}",
                "label": label,
                "confidence": stable_round(confidence, 4),
                "source_refs": dedupe_keep_order(record["source_refs"]),
            }
        )
    return out


def _build_structures(
    events: list[dict[str, Any]],
    role_priority: dict[str, int],
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    for event in events:
        role = event["structure_role"]
        related_ids = tuple(sorted(event["related_object_ids"]))
        key = (role, related_ids)
        entry = by_key.get(key)
        if entry is None:
            by_key[key] = {
                "first_order": event["order"],
                "structure_role": role,
                "related_object_ids": list(related_ids),
                "topology_tags": [event["topology_tag"]],
                "evidence": [event["evidence"]],
                "source_refs": list(event["source_refs"]),
                "confidence_values": [float(event["confidence"])],
            }
            continue
        entry["first_order"] = min(entry["first_order"], event["order"])
        entry["topology_tags"] = dedupe_keep_order(entry["topology_tags"] + [event["topology_tag"]])
        entry["evidence"] = dedupe_keep_order(entry["evidence"] + [event["evidence"]])
        entry["source_refs"] = dedupe_keep_order(entry["source_refs"] + list(event["source_refs"]))
        entry["confidence_values"].append(float(event["confidence"]))

    ordered = sorted(
        by_key.values(),
        key=lambda item: (
            item["first_order"],
            role_priority.get(item["structure_role"], 999),
            item["structure_role"],
        ),
    )

    out: list[dict[str, Any]] = []
    for idx, item in enumerate(ordered, start=1):
        confidence = clamp01(sum(item["confidence_values"]) / float(len(item["confidence_values"])))
        out.append(
            {
                "environment_structure_id": f"env_{idx:02d}",
                "structure_role": item["structure_role"],
                "topology_tags": list(item["topology_tags"]),
                "related_object_ids": list(item["related_object_ids"]),
                "evidence": list(item["evidence"]),
                "source_refs": dedupe_keep_order(item["source_refs"]),
                "confidence": stable_round(confidence, 4),
            }
        )
    return out


def _build_access_path_profile(
    context: NormalizedContext,
    anchor_ids: list[str],
    topology_tags: list[dict[str, Any]],
    defaults: dict[str, Any],
    accessibility_to_rotation: dict[str, str],
) -> dict[str, Any]:
    tag_labels = {item["label"] for item in topology_tags}
    anchor_access = [
        obj.accessibility
        for obj in context.inventory
        if obj.object_id in set(anchor_ids)
    ]

    entry_mode = defaults.get("entry_mode", "top_entry")
    if {"narrow_gap", "under_overhang", "occluding_edge"} & tag_labels:
        entry_mode = "side_entry"
    elif {"partial_opening", "container_neck", "through_opening"} & tag_labels:
        entry_mode = "angled_entry"
    elif {"support_surface"} & tag_labels:
        entry_mode = "front_entry"

    rotation_clearance = defaults.get("rotation_clearance", "sufficient")
    for access in anchor_access:
        mapped = accessibility_to_rotation.get(access)
        if mapped == "severely_limited":
            rotation_clearance = "severely_limited"
            break
        if mapped == "limited" and rotation_clearance != "severely_limited":
            rotation_clearance = "limited"

    requires_pass_through_opening = bool(
        {"partial_opening", "container_neck", "through_opening"} & tag_labels
    )
    requires_deep_reach = bool({"deep_recess", "confined_channel"} & tag_labels)
    available_support_surface = bool({"support_surface", "contact_plane"} & tag_labels)

    slip_hazard_present = False
    for obj in context.inventory:
        if obj.object_id not in set(anchor_ids):
            continue
        source = normalize_text(str(obj.geometry_cues.get("roll_risk_source", "")))
        if source and source not in {"none", "low", "stable"}:
            slip_hazard_present = True
            break

    if {"narrow_gap", "confined_channel", "deep_recess"} & tag_labels:
        confinement_level = 3
    elif {"partial_opening", "occluding_edge", "under_overhang"} & tag_labels:
        confinement_level = 2
    else:
        confinement_level = 1

    for access in anchor_access:
        if access in {"nested", "entangled"}:
            confinement_level = max(confinement_level, 3)
        elif access in {"partial", "occluded"}:
            confinement_level = max(confinement_level, 2)

    return {
        "entry_mode": entry_mode,
        "rotation_clearance": rotation_clearance,
        "requires_pass_through_opening": requires_pass_through_opening,
        "requires_deep_reach": requires_deep_reach,
        "available_support_surface": available_support_surface,
        "slip_hazard_present": slip_hazard_present,
        "confinement_level": confinement_level,
    }
