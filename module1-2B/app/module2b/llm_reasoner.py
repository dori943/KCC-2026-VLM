"""LLM-only Module 2-B reasoner."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib import error as url_error
from urllib import request as url_request

from app.module2b.models import NormalizedContext
from app.utils import project_root

_ALLOWED_PARAMETER_UNITS: dict[str, tuple[str, ...]] = {
    "opening_width": ("m", "level_1_to_5"),
    "opening_height": ("m", "level_1_to_5"),
    "neck_inner_diameter": ("m",),
    "recess_depth": ("m", "level_1_to_5"),
    "reachable_depth": ("m", "level_1_to_5"),
    "lateral_clearance": ("m", "level_1_to_5"),
    "vertical_clearance": ("m", "level_1_to_5"),
    "available_entry_angle_deg": ("deg",),
    "target_exposed_edge_length": ("m", "level_1_to_5"),
    "support_surface_span": ("m", "level_1_to_5"),
}
_ALLOWED_BOUND_TYPES = {"range", "upper_bound", "lower_bound"}
_ALLOWED_PRIORITIES = {"high", "medium", "low"}
_ALLOWED_HARDNESS = {"hard", "soft"}


def generate_module2b_output_with_llm(
    raw_bundle: dict[str, Any],
    normalized_context: NormalizedContext,
    prompt_spec_path: Path | None = None,
    api_key: str | None = None,
    model: str | None = None,
    timeout_seconds: float = 180.0,
    api_base: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate Module 2-B output with an OpenAI LLM and normalize it."""
    root = project_root()
    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            "OPENAI_API_KEY is required for Module 2-B LLM reasoner. "
            "Set the environment variable and retry."
        )

    resolved_model = (
        model
        or os.getenv("MODULE2B_REASONER_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-4.1-mini"
    )
    base = api_base or os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"
    api_url = base.rstrip("/") + "/chat/completions"
    prompt_spec = _load_prompt_spec(
        path=prompt_spec_path or (root / "specs" / "module2b_prompt_spec.md")
    )

    system_prompt = (
        "You are Module 2-B env-only reasoner. "
        "Return JSON only. Do not do Module 3 planning. "
        "Produce a structurally complete module2b_output object."
    )
    user_payload = {
        "task": "Generate module2b_output_env_only from Module 2-B inputs.",
        "requirements": [
            "Use only inventory object ids that exist in the provided context.",
            "subgoal_bindings order must exactly match input subgoal order.",
            "Each constraint target_binding_ids must contain the same binding_id.",
            "Keep environment-only scope and leave assembly synthesis to downstream modules.",
        ],
        "module2b_prompt_spec": prompt_spec,
        "module2b_input_bundle": raw_bundle,
        "normalized_context": normalized_context.to_dict(),
        "output_contract": {
            "required_top_level": [
                "schema_name",
                "schema_version",
                "stage",
                "task_id",
                "target_binding",
                "environment_context",
                "derived_constraints",
                "module3_handoff",
                "deferred_items",
                "confidence_summary",
            ],
            "required_deferred_keys": [
                "unresolved_target_ambiguities",
                "unresolved_environment_ambiguities",
                "not_numericized_items",
                "needs_user_or_scale_anchor",
            ],
            "allowed_parameter_units": _ALLOWED_PARAMETER_UNITS,
        },
    }
    user_prompt = json.dumps(user_payload, ensure_ascii=False)

    body = {
        "model": resolved_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    response_payload = _post_json(
        url=api_url,
        api_key=resolved_api_key,
        body=body,
        timeout_seconds=timeout_seconds,
    )
    raw_content, parsed = _extract_json_content(response_payload=response_payload)
    output = _normalize_module2b_payload(
        parsed=parsed,
        normalized_context=normalized_context,
    )
    usage = _extract_usage(response_payload.get("usage"))
    return output, {
        "mode": "llm_openai",
        "model": resolved_model,
        "api_url": api_url,
        "api_usage": usage,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "raw_response_preview": raw_content[:1200],
    }


def _load_prompt_spec(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _post_json(
    url: str,
    api_key: str,
    body: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    req = url_request.Request(
        url=url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with url_request.urlopen(req, timeout=timeout_seconds) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except url_error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        raise ValueError(f"OpenAI API error ({exc.code}): {error_text[:1000]}") from exc
    except url_error.URLError as exc:
        raise ValueError(f"OpenAI API network error: {exc.reason}") from exc
    if not isinstance(payload, dict):
        raise ValueError("OpenAI API returned non-object JSON payload.")
    return payload


def _extract_json_content(response_payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    try:
        content = response_payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            "OpenAI response did not include a valid message content payload."
        ) from exc
    if not isinstance(content, str):
        raise ValueError("OpenAI response content is not a JSON string.")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("OpenAI response content is not valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("OpenAI response JSON root must be object.")
    return content, parsed


def _extract_usage(raw_usage: Any) -> dict[str, int]:
    usage_dict = raw_usage if isinstance(raw_usage, dict) else {}
    prompt_tokens = _safe_int(usage_dict.get("prompt_tokens"))
    completion_tokens = _safe_int(usage_dict.get("completion_tokens"))
    total_tokens = _safe_int(usage_dict.get("total_tokens"))
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "api_call_count": 1,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _normalize_module2b_payload(
    parsed: dict[str, Any],
    normalized_context: NormalizedContext,
) -> dict[str, Any]:
    payload = parsed.get("module2b_output")
    if not isinstance(payload, dict):
        payload = parsed
    if not isinstance(payload, dict):
        payload = {}

    context_dict = normalized_context.to_dict()
    task_id = str(context_dict.get("task_id", "")).strip() or "task_unknown"
    inventory_ids = _unique_keep_order(
        [str(item).strip() for item in context_dict.get("object_id_order", []) if str(item).strip()]
    )
    subgoal_ids = _unique_keep_order(
        [
            str(item.get("subgoal_id", "")).strip()
            for item in context_dict.get("subgoals", [])
            if isinstance(item, dict) and str(item.get("subgoal_id", "")).strip()
        ]
    )

    target_binding_raw = payload.get("target_binding")
    if not isinstance(target_binding_raw, dict):
        target_binding_raw = {}
    target_binding = _normalize_target_binding(
        raw=target_binding_raw,
        inventory_ids=inventory_ids,
    )

    env_raw = payload.get("environment_context")
    if not isinstance(env_raw, dict):
        env_raw = {}
    topology_tags = _normalize_topology_tags(
        raw_items=env_raw.get("topology_tags"),
    )
    relevant_structures = _normalize_relevant_structures(
        raw_items=env_raw.get("relevant_structures"),
        inventory_ids=inventory_ids,
        fallback_related_id=(target_binding["primary_targets"][0]["object_id"] if target_binding["primary_targets"] else None),
    )
    structure_ids = [item["environment_structure_id"] for item in relevant_structures]
    target_binding["context_refs"] = _sanitize_id_list(
        target_binding.get("context_refs"),
        structure_ids,
    )
    for target in target_binding["primary_targets"]:
        target["context_refs"] = list(target_binding["context_refs"])

    numeric_estimates = _normalize_numeric_estimates(
        raw_items=env_raw.get("numeric_estimates"),
        structure_ids=structure_ids,
    )
    measurement_ids = [item["measurement_id"] for item in numeric_estimates]

    derived_raw = payload.get("derived_constraints")
    if not isinstance(derived_raw, dict):
        derived_raw = {}
    constraint_catalog = _normalize_constraint_catalog(
        raw_items=derived_raw.get("constraint_catalog"),
        subgoal_ids=subgoal_ids,
        binding_id=target_binding["binding_id"],
        measurement_ids=measurement_ids,
        numeric_estimates=numeric_estimates,
    )
    constraint_ids = [item["constraint_id"] for item in constraint_catalog]
    global_constraint_ids = _sanitize_id_list(
        derived_raw.get("global_constraint_ids"),
        constraint_ids,
    )
    if not global_constraint_ids and constraint_ids:
        global_constraint_ids = [constraint_ids[0]]
    subgoal_bindings = _normalize_subgoal_bindings(
        raw_items=derived_raw.get("subgoal_bindings"),
        subgoal_ids=subgoal_ids,
        constraint_catalog=constraint_catalog,
        global_constraint_ids=global_constraint_ids,
    )

    module3_handoff = _normalize_module3_handoff(
        raw=payload.get("module3_handoff"),
        binding_id=target_binding["binding_id"],
        constraint_ids=constraint_ids,
        global_constraint_ids=global_constraint_ids,
    )

    deferred_items = _normalize_deferred_items(payload.get("deferred_items"))
    confidence_summary = _build_confidence_summary(
        raw=payload.get("confidence_summary"),
        target_binding=target_binding,
        topology_tags=topology_tags,
        relevant_structures=relevant_structures,
        constraint_catalog=constraint_catalog,
        deferred_items=deferred_items,
    )

    return {
        "schema_name": "module2b_output_env_only",
        "schema_version": "0.1",
        "stage": "target_object_and_environment_constraints_env_only",
        "task_id": task_id,
        "target_binding": target_binding,
        "environment_context": {
            "topology_tags": topology_tags,
            "relevant_structures": relevant_structures,
            "access_path_profile": _normalize_access_path_profile(
                raw=env_raw.get("access_path_profile")
            ),
            "numeric_estimates": numeric_estimates,
        },
        "derived_constraints": {
            "constraint_catalog": constraint_catalog,
            "global_constraint_ids": global_constraint_ids,
            "subgoal_bindings": subgoal_bindings,
        },
        "module3_handoff": module3_handoff,
        "deferred_items": deferred_items,
        "confidence_summary": confidence_summary,
    }


def _normalize_target_binding(
    raw: dict[str, Any],
    inventory_ids: list[str],
) -> dict[str, Any]:
    binding_id = _safe_str(raw.get("binding_id"), "tb_01")
    candidate_ids_ranked = _sanitize_id_list(raw.get("candidate_ids_ranked"), inventory_ids)
    if not candidate_ids_ranked:
        candidate_ids_ranked = list(inventory_ids)

    primary_targets: list[dict[str, Any]] = []
    raw_targets = raw.get("primary_targets")
    if isinstance(raw_targets, list):
        for item in raw_targets:
            if not isinstance(item, dict):
                continue
            object_id = _safe_str(item.get("object_id"), "")
            if object_id not in inventory_ids:
                continue
            primary_targets.append(
                {
                    "object_id": object_id,
                    "selection_rationale": _safe_str(
                        item.get("selection_rationale"),
                        "Selected by Module2B LLM reasoning.",
                    ),
                    "confidence": _clamp01(_safe_float(item.get("confidence"), 0.6)),
                    "context_refs": [],
                }
            )
    if not primary_targets and candidate_ids_ranked:
        primary_targets = [
            {
                "object_id": candidate_ids_ranked[0],
                "selection_rationale": "Top-ranked candidate from Module2B LLM output.",
                "confidence": 0.6,
                "context_refs": [],
            }
        ]

    if not candidate_ids_ranked and primary_targets:
        candidate_ids_ranked = [primary_targets[0]["object_id"]]

    target_mode_default = "single_object" if len(primary_targets) <= 1 else "multi_object"
    binding_status_default = "resolved" if primary_targets else "ambiguous"
    return {
        "binding_id": binding_id,
        "target_mode": _safe_str(raw.get("target_mode"), target_mode_default),
        "binding_status": _safe_str(raw.get("binding_status"), binding_status_default),
        "confidence": _clamp01(_safe_float(raw.get("confidence"), 0.6)),
        "primary_targets": primary_targets,
        "candidate_ids_ranked": candidate_ids_ranked,
        "deferred_reasons": _normalize_string_list(raw.get("deferred_reasons")),
        "context_refs": _normalize_string_list(raw.get("context_refs")),
    }


def _normalize_topology_tags(raw_items: Any) -> list[dict[str, Any]]:
    tags: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    items = raw_items if isinstance(raw_items, list) else []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        tag_id = _safe_str(item.get("tag_id"), f"tag_{idx:02d}")
        if tag_id in seen_ids:
            tag_id = f"{tag_id}_{idx:02d}"
        seen_ids.add(tag_id)
        tags.append(
            {
                "tag_id": tag_id,
                "label": _safe_str(item.get("label"), f"topology_tag_{idx:02d}"),
                "confidence": _clamp01(_safe_float(item.get("confidence"), 0.5)),
                "source_refs": [],
            }
        )
    return tags


def _normalize_relevant_structures(
    raw_items: Any,
    inventory_ids: list[str],
    fallback_related_id: str | None,
) -> list[dict[str, Any]]:
    structures: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    items = raw_items if isinstance(raw_items, list) else []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        structure_id = _safe_str(
            item.get("environment_structure_id"),
            f"es_{idx:02d}",
        )
        if structure_id in seen_ids:
            structure_id = f"{structure_id}_{idx:02d}"
        seen_ids.add(structure_id)

        related_ids = _sanitize_id_list(item.get("related_object_ids"), inventory_ids)
        if not related_ids and fallback_related_id and fallback_related_id in inventory_ids:
            related_ids = [fallback_related_id]

        structures.append(
            {
                "environment_structure_id": structure_id,
                "structure_role": _safe_str(item.get("structure_role"), "access_channel"),
                "description": _safe_str(item.get("description"), ""),
                "related_object_ids": related_ids,
                "confidence": _clamp01(_safe_float(item.get("confidence"), 0.5)),
                "source_refs": [],
            }
        )

    if not structures and fallback_related_id and fallback_related_id in inventory_ids:
        structures = [
            {
                "environment_structure_id": "es_01",
                "structure_role": "access_channel",
                "description": "Default environment structure inferred from target vicinity.",
                "related_object_ids": [fallback_related_id],
                "confidence": 0.5,
                "source_refs": [],
            }
        ]
    return structures


def _normalize_access_path_profile(raw: Any) -> dict[str, Any]:
    profile = raw if isinstance(raw, dict) else {}
    return {
        "entry_mode": _safe_str(profile.get("entry_mode"), "direct"),
        "confinement_level": _safe_str(profile.get("confinement_level"), "medium"),
        "requires_deep_reach": bool(profile.get("requires_deep_reach", False)),
        "collision_risk": _safe_str(profile.get("collision_risk"), "unknown"),
        "occlusion_level": _safe_str(profile.get("occlusion_level"), "unknown"),
    }


def _normalize_numeric_estimates(
    raw_items: Any,
    structure_ids: list[str],
) -> list[dict[str, Any]]:
    estimates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    items = raw_items if isinstance(raw_items, list) else []

    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        measurement_id = _safe_str(item.get("measurement_id"), f"me_{idx:02d}")
        if measurement_id in seen_ids:
            measurement_id = f"{measurement_id}_{idx:02d}"
        seen_ids.add(measurement_id)

        parameter_name = _safe_str(item.get("parameter_name"), "opening_width")
        if parameter_name not in _ALLOWED_PARAMETER_UNITS:
            parameter_name = "opening_width"

        unit = _safe_str(item.get("unit"), "")
        allowed_units = _ALLOWED_PARAMETER_UNITS[parameter_name]
        if unit not in allowed_units:
            unit = _preferred_unit(allowed_units)

        bound_type = _safe_str(item.get("bound_type"), "upper_bound")
        if bound_type not in _ALLOWED_BOUND_TYPES:
            bound_type = "upper_bound"

        lower_value = _to_optional_float(item.get("lower_value"))
        upper_value = _to_optional_float(item.get("upper_value"))

        default_value = _default_bound_value(parameter_name=parameter_name, unit=unit)
        if bound_type == "range":
            if lower_value is None and upper_value is None:
                lower_value = default_value * 0.8
                upper_value = default_value
            elif lower_value is None:
                lower_value = upper_value
            elif upper_value is None:
                upper_value = lower_value
            if lower_value is not None and upper_value is not None and lower_value > upper_value:
                lower_value, upper_value = upper_value, lower_value
        elif bound_type == "upper_bound":
            if upper_value is None:
                upper_value = lower_value if lower_value is not None else default_value
            lower_value = None
        else:
            if lower_value is None:
                lower_value = upper_value if upper_value is not None else default_value
            upper_value = None

        estimates.append(
            {
                "measurement_id": measurement_id,
                "parameter_name": parameter_name,
                "bound_type": bound_type,
                "lower_value": lower_value,
                "upper_value": upper_value,
                "unit": unit,
                "estimate_basis": _safe_str(item.get("estimate_basis"), "llm_inference"),
                "confidence": _clamp01(_safe_float(item.get("confidence"), 0.5)),
                "related_environment_structure_ids": _sanitize_id_list(
                    item.get("related_environment_structure_ids"),
                    structure_ids,
                ),
                "source_refs": [],
            }
        )

    if not estimates:
        default_related = [structure_ids[0]] if structure_ids else []
        estimates = [
            {
                "measurement_id": "me_01",
                "parameter_name": "opening_width",
                "bound_type": "upper_bound",
                "lower_value": None,
                "upper_value": 3.0,
                "unit": "level_1_to_5",
                "estimate_basis": "llm_inference",
                "confidence": 0.5,
                "related_environment_structure_ids": default_related,
                "source_refs": [],
            }
        ]
    return estimates


def _normalize_constraint_catalog(
    raw_items: Any,
    subgoal_ids: list[str],
    binding_id: str,
    measurement_ids: list[str],
    numeric_estimates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    constraints: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    items = raw_items if isinstance(raw_items, list) else []

    measurement_lookup = {
        item["measurement_id"]: item
        for item in numeric_estimates
        if isinstance(item, dict) and isinstance(item.get("measurement_id"), str)
    }
    default_measurement_id = measurement_ids[0] if measurement_ids else None

    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        constraint_id = _safe_str(item.get("constraint_id"), f"ct_{idx:02d}")
        if constraint_id in seen_ids:
            constraint_id = f"{constraint_id}_{idx:02d}"
        seen_ids.add(constraint_id)

        constraint_subgoal_ids = _sanitize_id_list(item.get("subgoal_ids"), subgoal_ids)
        if not constraint_subgoal_ids and subgoal_ids:
            constraint_subgoal_ids = list(subgoal_ids)

        constraint_measurement_ids = _sanitize_id_list(
            item.get("measurement_ids"),
            measurement_ids,
        )
        if not constraint_measurement_ids and default_measurement_id:
            constraint_measurement_ids = [default_measurement_id]

        primary_measurement = (
            measurement_lookup.get(constraint_measurement_ids[0])
            if constraint_measurement_ids
            else (measurement_lookup.get(default_measurement_id) if default_measurement_id else None)
        )
        parameter_name = _safe_str(item.get("parameter_name"), "")
        if parameter_name not in _ALLOWED_PARAMETER_UNITS:
            parameter_name = (
                _safe_str(primary_measurement.get("parameter_name"), "opening_width")
                if isinstance(primary_measurement, dict)
                else "opening_width"
            )
        if parameter_name not in _ALLOWED_PARAMETER_UNITS:
            parameter_name = "opening_width"

        allowed_units = _ALLOWED_PARAMETER_UNITS[parameter_name]
        unit = _safe_str(item.get("unit"), "")
        if unit not in allowed_units:
            if isinstance(primary_measurement, dict):
                unit = _safe_str(primary_measurement.get("unit"), "")
            if unit not in allowed_units:
                unit = _preferred_unit(allowed_units)

        bound_type = _safe_str(item.get("bound_type"), "")
        if bound_type not in _ALLOWED_BOUND_TYPES:
            if isinstance(primary_measurement, dict):
                bound_type = _safe_str(primary_measurement.get("bound_type"), "")
            if bound_type not in _ALLOWED_BOUND_TYPES:
                bound_type = "upper_bound"

        lower_value = _to_optional_float(item.get("lower_value"))
        upper_value = _to_optional_float(item.get("upper_value"))
        if isinstance(primary_measurement, dict):
            if lower_value is None:
                lower_value = _to_optional_float(primary_measurement.get("lower_value"))
            if upper_value is None:
                upper_value = _to_optional_float(primary_measurement.get("upper_value"))

        default_value = _default_bound_value(parameter_name=parameter_name, unit=unit)
        if bound_type == "range":
            if lower_value is None and upper_value is None:
                lower_value = default_value * 0.8
                upper_value = default_value
            elif lower_value is None:
                lower_value = upper_value
            elif upper_value is None:
                upper_value = lower_value
            if lower_value is not None and upper_value is not None and lower_value > upper_value:
                lower_value, upper_value = upper_value, lower_value
        elif bound_type == "upper_bound":
            if upper_value is None:
                upper_value = lower_value if lower_value is not None else default_value
            lower_value = None
        else:
            if lower_value is None:
                lower_value = upper_value if upper_value is not None else default_value
            upper_value = None

        priority = _safe_str(item.get("priority"), "medium")
        if priority not in _ALLOWED_PRIORITIES:
            priority = "medium"

        hardness = _safe_str(item.get("hardness"), "hard")
        if hardness not in _ALLOWED_HARDNESS:
            hardness = "hard"

        constraints.append(
            {
                "constraint_id": constraint_id,
                "subgoal_ids": constraint_subgoal_ids,
                "priority": priority,
                "hardness": hardness,
                "category": _safe_str(item.get("category"), "environment"),
                "parameter_name": parameter_name,
                "bound_type": bound_type,
                "lower_value": lower_value,
                "upper_value": upper_value,
                "unit": unit,
                "applies_to": _safe_str(item.get("applies_to"), "global"),
                "target_binding_ids": [binding_id],
                "measurement_ids": constraint_measurement_ids,
                "confidence": _clamp01(_safe_float(item.get("confidence"), 0.5)),
                "source_refs": [],
            }
        )

    if not constraints:
        default_measurement = numeric_estimates[0] if numeric_estimates else {}
        constraints = [
            {
                "constraint_id": "ct_01",
                "subgoal_ids": list(subgoal_ids),
                "priority": "medium",
                "hardness": "hard",
                "category": "environment",
                "parameter_name": _safe_str(default_measurement.get("parameter_name"), "opening_width"),
                "bound_type": _safe_str(default_measurement.get("bound_type"), "upper_bound"),
                "lower_value": default_measurement.get("lower_value"),
                "upper_value": default_measurement.get("upper_value"),
                "unit": _safe_str(default_measurement.get("unit"), "level_1_to_5"),
                "applies_to": "global",
                "target_binding_ids": [binding_id],
                "measurement_ids": [default_measurement.get("measurement_id")] if default_measurement.get("measurement_id") else [],
                "confidence": 0.5,
                "source_refs": [],
            }
        ]
    return constraints


def _normalize_subgoal_bindings(
    raw_items: Any,
    subgoal_ids: list[str],
    constraint_catalog: list[dict[str, Any]],
    global_constraint_ids: list[str],
) -> list[dict[str, Any]]:
    constraint_ids = [c["constraint_id"] for c in constraint_catalog]
    constraints_by_subgoal: dict[str, list[str]] = {sg: [] for sg in subgoal_ids}
    for constraint in constraint_catalog:
        cid = constraint["constraint_id"]
        for subgoal_id in constraint.get("subgoal_ids", []):
            if subgoal_id in constraints_by_subgoal and cid not in constraints_by_subgoal[subgoal_id]:
                constraints_by_subgoal[subgoal_id].append(cid)

    raw_map: dict[str, list[str]] = {}
    items = raw_items if isinstance(raw_items, list) else []
    for item in items:
        if not isinstance(item, dict):
            continue
        sid = _safe_str(item.get("subgoal_id"), "")
        if sid not in subgoal_ids:
            continue
        raw_map[sid] = _sanitize_id_list(item.get("constraint_ids"), constraint_ids)

    bindings: list[dict[str, Any]] = []
    for subgoal_id in subgoal_ids:
        chosen = raw_map.get(subgoal_id, [])
        if not chosen:
            chosen = constraints_by_subgoal.get(subgoal_id, [])
        if not chosen:
            chosen = [cid for cid in global_constraint_ids if cid in constraint_ids]
        if not chosen and constraint_ids:
            chosen = [constraint_ids[0]]
        bindings.append(
            {
                "subgoal_id": subgoal_id,
                "constraint_ids": _unique_keep_order(chosen),
            }
        )
    return bindings


def _normalize_module3_handoff(
    raw: Any,
    binding_id: str,
    constraint_ids: list[str],
    global_constraint_ids: list[str],
) -> dict[str, Any]:
    payload = raw if isinstance(raw, dict) else {}
    handoff_constraint_ids = _sanitize_id_list(
        payload.get("handoff_constraint_ids"),
        constraint_ids,
    )
    if not handoff_constraint_ids:
        handoff_constraint_ids = [cid for cid in global_constraint_ids if cid in constraint_ids]
    if not handoff_constraint_ids and constraint_ids:
        handoff_constraint_ids = [constraint_ids[0]]

    pending_merge_sources = _normalize_string_list(payload.get("pending_merge_sources"))
    if not pending_merge_sources:
        pending_merge_sources = ["material_reasoner"]

    return {
        "handoff_status": _safe_str(payload.get("handoff_status"), "ready"),
        "target_binding_id": binding_id,
        "handoff_constraint_ids": handoff_constraint_ids,
        "constraint_units_policy": _safe_str(payload.get("constraint_units_policy"), "mixed"),
        "pending_merge_sources": pending_merge_sources,
    }


def _normalize_deferred_items(raw: Any) -> dict[str, list[dict[str, Any]]]:
    payload = raw if isinstance(raw, dict) else {}
    result: dict[str, list[dict[str, Any]]] = {}
    keys = [
        "unresolved_target_ambiguities",
        "unresolved_environment_ambiguities",
        "not_numericized_items",
        "needs_user_or_scale_anchor",
    ]
    for key in keys:
        entries: list[dict[str, Any]] = []
        raw_items = payload.get(key)
        if isinstance(raw_items, list):
            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                entry = {
                    "item": _safe_str(item.get("item"), key),
                    "reason": _safe_str(item.get("reason"), "llm_deferred"),
                    "source_refs": [],
                }
                entries.append(entry)
        result[key] = entries
    return result


def _build_confidence_summary(
    raw: Any,
    target_binding: dict[str, Any],
    topology_tags: list[dict[str, Any]],
    relevant_structures: list[dict[str, Any]],
    constraint_catalog: list[dict[str, Any]],
    deferred_items: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    payload = raw if isinstance(raw, dict) else {}
    env_components = [
        _safe_float(item.get("confidence"), 0.0) for item in topology_tags + relevant_structures
    ]
    constraint_components = [
        _safe_float(item.get("confidence"), 0.0) for item in constraint_catalog
    ]
    environment_confidence = (
        sum(env_components) / len(env_components) if env_components else 0.5
    )
    constraint_confidence = (
        sum(constraint_components) / len(constraint_components) if constraint_components else 0.5
    )

    high_impact_uncertainties = _normalize_string_list(
        payload.get("high_impact_uncertainties")
    )
    if not high_impact_uncertainties:
        high_impact_uncertainties = _unique_keep_order(
            [
                _safe_str(item.get("reason"), "")
                for key in deferred_items
                for item in deferred_items.get(key, [])
                if _safe_str(item.get("reason"), "")
            ]
        )

    return {
        "target_binding_confidence": _clamp01(
            _safe_float(
                payload.get("target_binding_confidence"),
                _safe_float(target_binding.get("confidence"), 0.6),
            )
        ),
        "environment_binding_confidence": _clamp01(
            _safe_float(payload.get("environment_binding_confidence"), environment_confidence)
        ),
        "constraint_set_confidence": _clamp01(
            _safe_float(payload.get("constraint_set_confidence"), constraint_confidence)
        ),
        "high_impact_uncertainties": high_impact_uncertainties,
    }


def _preferred_unit(allowed_units: tuple[str, ...]) -> str:
    for preferred in ("level_1_to_5", "m", "deg"):
        if preferred in allowed_units:
            return preferred
    return allowed_units[0]


def _default_bound_value(parameter_name: str, unit: str) -> float:
    if unit == "level_1_to_5":
        return 3.0
    if unit == "deg":
        return 25.0
    if parameter_name in {"opening_width", "opening_height"}:
        return 0.05
    if parameter_name in {"recess_depth", "reachable_depth"}:
        return 0.08
    if parameter_name in {"lateral_clearance", "vertical_clearance"}:
        return 0.02
    if parameter_name in {"target_exposed_edge_length", "support_surface_span"}:
        return 0.04
    return 0.03


def _sanitize_id_list(raw: Any, allowed: list[str]) -> list[str]:
    allowed_set = set(allowed)
    values = raw if isinstance(raw, list) else []
    return _unique_keep_order(
        [str(item).strip() for item in values if str(item).strip() in allowed_set]
    )


def _normalize_string_list(raw: Any) -> list[str]:
    values = raw if isinstance(raw, list) else []
    return _unique_keep_order([str(item).strip() for item in values if str(item).strip()])


def _unique_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _safe_str(value: Any, default: str) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
    return default


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value
