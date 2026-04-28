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
# constraint_catalog.parameter_name 은 numeric_estimates.parameter_name 과
# 완전히 다른 enum (도구/접근/배치 요구사항). 별도 매핑 필요.
_ALLOWED_CONSTRAINT_PARAMETER_UNITS: dict[str, tuple[str, ...]] = {
    "min_effective_reach": ("m", "level_1_to_5"),
    "max_tool_body_width": ("m", "level_1_to_5"),
    "max_cross_section_width": ("m", "level_1_to_5"),
    "max_tip_thickness": ("m", "level_1_to_5"),
    "max_tip_diameter": ("m", "level_1_to_5"),
    "min_insert_depth": ("m", "level_1_to_5"),
    "min_contact_span": ("m", "level_1_to_5"),
    "min_tip_force_transmission_level": ("level_1_to_5",),
    "min_global_stiffness_level": ("level_1_to_5",),
    "max_allowed_compliance_level": ("level_1_to_5",),
    "min_surface_friction_level": ("level_1_to_5",),
    "preferred_contact_friction_level": ("level_1_to_5",),
    "min_placement_stability_level": ("level_1_to_5",),
    "max_required_entry_angle_deg": ("deg",),
    "max_allowed_roll_instability_level": ("level_1_to_5",),
}
_ALLOWED_HANDOFF_STATUS = {"ready", "partial", "blocked"}
_ALLOWED_CONSTRAINT_UNITS_POLICY = {
    "metric_strict", "mixed_metric_and_ordinal", "ordinal_preferred",
}
_ALLOWED_OMITTED_FAMILIES = {
    "risk_limit", "target_material_state",
    "damage_sensitivity", "contact_style_preference",
}
_ALLOWED_BOUND_TYPES = {"range", "upper_bound", "lower_bound"}
_ALLOWED_PRIORITIES = {"high", "medium", "low"}
_ALLOWED_HARDNESS = {"hard", "soft"}
_ALLOWED_CONSTRAINT_CATEGORIES = {
    "geometric",
    "mechanical",
    "surface_interaction",
    "stability_access",
}
_ALLOWED_CONSTRAINT_APPLIES_TO = {
    "tool_profile",
    "approach_path",
    "placement_strategy",
}
_ALLOWED_TOPOLOGY_LABELS = {
    "container_neck", "recess_wall", "partial_opening", "deep_recess",
    "through_opening", "narrow_gap", "constraining_surface_pair",
    "confined_channel", "under_overhang", "occluding_edge",
    "container_cavity", "support_surface", "contact_plane", "obstacle",
}
_ALLOWED_ENTRY_MODES = {"top_entry", "side_entry", "angled_entry", "front_entry"}
_ALLOWED_ROTATION_CLEARANCE = {"sufficient", "limited", "severely_limited"}
_ALLOWED_ESTIMATE_BASIS = {
    "observed", "estimated_from_anchor", "relative_geometry", "task_text_prior",
}


def generate_module2b_output_with_llm(
    raw_bundle: dict[str, Any],
    normalized_context: NormalizedContext,
    prompt_spec_path: Path | None = None,
    api_key: str | None = None,
    model: str | None = None,
    timeout_seconds: float = 180.0,
    api_base: str | None = None,
    task_scene_image_path: Path | None = None,
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

    # task_scene_image 보조 입력 사용 시 system prompt 에 가드 안내문 추가
    from app.utils import GUARD_TEXT_2B, build_user_content_with_image

    system_prompt = (
        "너는 Module 2-B: Target Object 및 환경 제약 반영기(env-only)이다. "
        "user 메시지의 module2b_prompt_spec(specs/module2b_prompt_spec.md)을 "
        "ground truth 지시문으로 따른다. 그 안의 §토폴로지 매핑 표, §수치 추정 가이드, "
        "§제약 카탈로그 가이드, §출력 규칙, §자기 점검을 모두 적용해 출력하라. "
        "특히: (1) module2b_input_bundle.task_brief.description(또는 user_goal) "
        "키워드를 반드시 읽고 topology_tags/numeric_estimates에 반영한다. "
        "(2) topology_tags ≥ 2, numeric_estimates ≥ 2, constraint_catalog ≥ 3 "
        "(parameter_name 서로 다름) 미만이면 규칙 위반이다. "
        "(3) 'c_01: min_effective_reach'만 있는 generic 출력은 절대 금지. "
        "(4) Module 3의 plan/assembly/실행은 절대 하지 않는다. 환경 제약만 본다. "
        "JSON only, JSON 바깥 설명 금지."
    )
    if task_scene_image_path is not None and Path(task_scene_image_path).exists():
        system_prompt = system_prompt + GUARD_TEXT_2B
    task_description_focus = _extract_task_focus(
        raw_bundle=raw_bundle,
        normalized_context=normalized_context,
    )
    user_payload = {
        "task": "Generate module2b_output_env_only from Module 2-B inputs.",
        "task_description_focus": task_description_focus,
        "requirements": [
            "task_description_focus.text의 키워드를 반드시 topology_tags/numeric_estimates로 변환하라.",
            "topology_tags ≥ 2, numeric_estimates ≥ 2, constraint_catalog ≥ 3 (parameter_name 모두 상이).",
            "'c_01: min_effective_reach' generic 단독 출력은 절대 금지.",
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
    user_content = build_user_content_with_image(
        text=user_prompt,
        task_scene_image_path=task_scene_image_path,
    )

    body = {
        "model": resolved_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
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


def _extract_task_focus(
    raw_bundle: dict[str, Any],
    normalized_context: NormalizedContext,
) -> dict[str, Any]:
    """Surface task description text up to top-level so LLM cannot miss it.

    LLM이 generic c_01 출력만 내는 패턴의 주된 원인은 task description이
    raw_bundle 깊은 곳에 묻혀 무시되는 것. 이를 top-level focus로 끌어올린다.
    """
    pieces: list[str] = []

    def _add(value: Any) -> None:
        if isinstance(value, str):
            text = value.strip()
            if text and text not in pieces:
                pieces.append(text)
        elif isinstance(value, list):
            for item in value:
                _add(item)

    bundle = raw_bundle if isinstance(raw_bundle, dict) else {}
    task_brief = bundle.get("task_brief") if isinstance(bundle.get("task_brief"), dict) else {}
    _add(task_brief.get("user_goal"))
    _add(task_brief.get("description"))
    _add(task_brief.get("success_criteria"))
    _add(task_brief.get("task_notes"))

    ctx_dict = normalized_context.to_dict()
    ctx_brief = ctx_dict.get("task_brief") if isinstance(ctx_dict.get("task_brief"), dict) else {}
    _add(ctx_brief.get("user_goal"))
    _add(ctx_brief.get("description"))
    _add(ctx_brief.get("success_criteria"))
    _add(ctx_brief.get("task_notes"))

    combined_text = " | ".join(pieces) if pieces else ""

    keyword_hints: list[str] = []
    keyword_map = [
        ("좁은 틈", "narrow_gap"),
        ("끼인", "narrow_gap"),
        ("틈새", "narrow_gap"),
        ("깊은 구멍", "deep_recess"),
        ("구멍 바닥", "deep_recess"),
        ("팔 길이", "reachable_depth lower_bound"),
        ("팔의 길이", "reachable_depth lower_bound"),
        ("길이의 절반", "reachable_depth lower_bound"),
        ("매달", "under_overhang"),
        ("공중", "under_overhang"),
        ("실에", "occluding_edge"),
        ("장력", "max_allowed_compliance_level"),
        ("장애물", "obstacle"),
        ("막힌", "obstacle"),
        ("문", "obstacle confined_channel"),
        ("손잡이", "min_tip_force_transmission_level"),
        ("회전", "max_required_entry_angle_deg"),
        ("유리", "fragility max_tip_thickness"),
        ("파편", "obstacle constraining_surface_pair"),
        ("날카로", "fragility"),
        ("직접 접촉", "no_direct_contact"),
        ("좁은 접근", "confined_channel"),
    ]
    for kw, hint in keyword_map:
        if kw in combined_text and hint not in keyword_hints:
            keyword_hints.append(hint)

    return {
        "text": combined_text,
        "keyword_hints": keyword_hints,
        "instruction": (
            "이 텍스트는 task의 핵심 물리 제약이다. 키워드와 hint를 "
            "topology_tags / numeric_estimates / constraint_catalog에 반드시 반영하라."
        ),
    }


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
    # object_id -> object_name 매핑 (primary_targets 의 object_name 채우기용)
    id_to_name: dict[str, str] = {}
    for inv_item in context_dict.get("inventory", []) or []:
        if isinstance(inv_item, dict):
            oid = str(inv_item.get("object_id", "")).strip()
            oname = str(inv_item.get("object_name", "")).strip()
            if oid and oname:
                id_to_name[oid] = oname
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
        id_to_name=id_to_name,
    )

    env_raw = payload.get("environment_context")
    if not isinstance(env_raw, dict):
        env_raw = {}
    topology_tags = _normalize_topology_tags(
        raw_items=env_raw.get("topology_tags"),
        inventory_ids=inventory_ids,
    )
    relevant_structures = _normalize_relevant_structures(
        raw_items=env_raw.get("relevant_structures"),
        inventory_ids=inventory_ids,
        fallback_related_id=(target_binding["primary_targets"][0]["object_id"] if target_binding["primary_targets"] else None),
        topology_tags=topology_tags,
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
    # LLM 이 global_constraint_ids 를 비우거나 c_01 하나만 넣는 경우가 많아서,
    # priority=high 인 모든 constraint 를 자동으로 global 로 승격한다.
    if not global_constraint_ids and constraint_catalog:
        high_priority_ids = [
            item["constraint_id"]
            for item in constraint_catalog
            if item.get("priority") == "high"
        ]
        global_constraint_ids = high_priority_ids or [constraint_ids[0]]
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
    id_to_name: dict[str, str] | None = None,
) -> dict[str, Any]:
    id_to_name = id_to_name or {}
    # binding_id 는 schema 가 ^tb_[0-9]{2}$ 패턴 강제 → 결정적 채번
    binding_id = "tb_01"
    candidate_ids_ranked = _sanitize_id_list(raw.get("candidate_ids_ranked"), inventory_ids)
    if not candidate_ids_ranked:
        candidate_ids_ranked = list(inventory_ids)

    def _build_target(object_id: str, item: dict[str, Any] | None) -> dict[str, Any]:
        item = item or {}
        raw_evidence = item.get("evidence_keys")
        evidence_keys: list[str] = []
        seen: set[str] = set()
        if isinstance(raw_evidence, list):
            for k in raw_evidence:
                if isinstance(k, str) and k and k not in seen:
                    evidence_keys.append(k)
                    seen.add(k)
        return {
            "object_id": object_id,
            "object_name": id_to_name.get(object_id) or object_id,
            "confidence": _clamp01(_safe_float(item.get("confidence"), 0.6)),
            "evidence_keys": evidence_keys,
            "context_refs": [],
        }

    primary_targets: list[dict[str, Any]] = []
    raw_targets = raw.get("primary_targets")
    if isinstance(raw_targets, list):
        for item in raw_targets:
            if not isinstance(item, dict):
                continue
            object_id = _safe_str(item.get("object_id"), "")
            if object_id not in inventory_ids:
                continue
            primary_targets.append(_build_target(object_id, item))
    if not primary_targets and candidate_ids_ranked:
        primary_targets = [_build_target(candidate_ids_ranked[0], None)]

    if not candidate_ids_ranked and primary_targets:
        candidate_ids_ranked = [primary_targets[0]["object_id"]]

    # target_mode 와 binding_status enum 검증
    target_mode = _safe_str(raw.get("target_mode"), "")
    valid_modes = {"single", "multiple", "implicit", "ambiguous", "none"}
    if target_mode not in valid_modes:
        target_mode = "single" if len(primary_targets) <= 1 else "multiple"

    binding_status = _safe_str(raw.get("binding_status"), "")
    valid_statuses = {"resolved", "partially_resolved", "ambiguous", "deferred"}
    if binding_status not in valid_statuses:
        binding_status = "resolved" if primary_targets else "ambiguous"

    return {
        "binding_id": binding_id,
        "target_mode": target_mode,
        "binding_status": binding_status,
        "primary_targets": primary_targets,
        "candidate_ids_ranked": candidate_ids_ranked,
        "context_refs": _normalize_string_list(raw.get("context_refs")),
        "deferred_reasons": _normalize_string_list(raw.get("deferred_reasons")),
        "confidence": _clamp01(_safe_float(raw.get("confidence"), 0.6)),
    }


def _normalize_topology_tags(
    raw_items: Any,
    inventory_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    tags: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    items = raw_items if isinstance(raw_items, list) else []
    inventory_ids = inventory_ids or []
    fallback_ref = inventory_ids[0] if inventory_ids else None
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        tag_id = f"tag_{idx:02d}"
        seen_ids.add(tag_id)
        label = _safe_str(item.get("label"), "")
        if label not in _ALLOWED_TOPOLOGY_LABELS:
            label = "obstacle"
        # source_refs 는 known_ref 집합(inventory/tag/env/m/c/sg/tb)에 속해야 함.
        # 가장 안전한 valid ref: 자기 자신의 tag_id (known_refs 에 포함됨).
        tag_source = [tag_id] if not fallback_ref else [fallback_ref]
        tags.append(
            {
                "tag_id": tag_id,
                "label": label,
                "confidence": _clamp01(_safe_float(item.get("confidence"), 0.5)),
                "source_refs": tag_source,
            }
        )
    return tags


def _normalize_relevant_structures(
    raw_items: Any,
    inventory_ids: list[str],
    fallback_related_id: str | None,
    topology_tags: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    structures: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    items = raw_items if isinstance(raw_items, list) else []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        structure_id = f"env_{idx:02d}"
        seen_ids.add(structure_id)

        related_ids = _sanitize_id_list(item.get("related_object_ids"), inventory_ids)
        if not related_ids and fallback_related_id and fallback_related_id in inventory_ids:
            related_ids = [fallback_related_id]

        role = _safe_str(item.get("structure_role"), "")
        if role not in _ALLOWED_TOPOLOGY_LABELS:
            role = "obstacle"

        # topology_tags 는 라벨 enum 의 부분집합이어야 함
        raw_tags = item.get("topology_tags") if isinstance(item.get("topology_tags"), list) else []
        valid_tags: list[str] = []
        seen_tags: set[str] = set()
        for t in raw_tags:
            if isinstance(t, str) and t in _ALLOWED_TOPOLOGY_LABELS and t not in seen_tags:
                valid_tags.append(t)
                seen_tags.add(t)
        if not valid_tags:
            valid_tags = [role]

        evidence = item.get("evidence") if isinstance(item.get("evidence"), list) else []
        evidence = [str(e) for e in evidence if isinstance(e, str)]

        # source_refs: related_object_ids 가 있으면 첫 번째, 없으면 inventory[0],
        # 그것도 없으면 자기 자신의 structure_id (known_refs 에 포함됨).
        if related_ids:
            structure_source = [related_ids[0]]
        elif inventory_ids:
            structure_source = [inventory_ids[0]]
        else:
            structure_source = [structure_id]
        structures.append(
            {
                "environment_structure_id": structure_id,
                "structure_role": role,
                "topology_tags": valid_tags,
                "related_object_ids": related_ids,
                "evidence": evidence,
                "source_refs": structure_source,
                "confidence": _clamp01(_safe_float(item.get("confidence"), 0.5)),
            }
        )

    # LLM 이 relevant_structures 를 비웠을 때:
    # topology_tags 기반으로 의미 있는 fallback 을 만든다.
    # 예전엔 무조건 obstacle 한 개만 만들어서 task1(narrow_gap) 일 때도
    # env_01 = obstacle 로 잘못 표기됐다.
    if not structures and topology_tags:
        related = [fallback_related_id] if (
            fallback_related_id and fallback_related_id in inventory_ids
        ) else (inventory_ids[:1] if inventory_ids else [])
        for idx, tag in enumerate(topology_tags, start=1):
            label = tag.get("label", "obstacle")
            if label not in _ALLOWED_TOPOLOGY_LABELS:
                label = "obstacle"
            sid = f"env_{idx:02d}"
            if related:
                src = [related[0]]
            else:
                src = [sid]
            structures.append(
                {
                    "environment_structure_id": sid,
                    "structure_role": label,
                    "topology_tags": [label],
                    "related_object_ids": list(related),
                    "evidence": [
                        f"Inferred from topology_tag {tag.get('tag_id', f'tag_{idx:02d}')} ({label})."
                    ],
                    "source_refs": src,
                    "confidence": _clamp01(_safe_float(tag.get("confidence"), 0.5)),
                }
            )

    if not structures and fallback_related_id and fallback_related_id in inventory_ids:
        structures = [
            {
                "environment_structure_id": "env_01",
                "structure_role": "obstacle",
                "topology_tags": ["obstacle"],
                "related_object_ids": [fallback_related_id],
                "evidence": ["Default environment structure inferred from target vicinity."],
                "source_refs": [fallback_related_id],
                "confidence": 0.5,
            }
        ]
    return structures


def _normalize_access_path_profile(raw: Any) -> dict[str, Any]:
    profile = raw if isinstance(raw, dict) else {}

    entry_mode = _safe_str(profile.get("entry_mode"), "")
    if entry_mode not in _ALLOWED_ENTRY_MODES:
        entry_mode = "top_entry"

    rotation_clearance = _safe_str(profile.get("rotation_clearance"), "")
    if rotation_clearance not in _ALLOWED_ROTATION_CLEARANCE:
        rotation_clearance = "limited"

    # confinement_level 은 정수 enum [1, 2, 3]
    raw_conf = profile.get("confinement_level")
    try:
        conf_int = int(raw_conf)
    except (TypeError, ValueError):
        conf_int = 2
    if conf_int not in (1, 2, 3):
        conf_int = 2

    return {
        "entry_mode": entry_mode,
        "rotation_clearance": rotation_clearance,
        "requires_pass_through_opening": bool(profile.get("requires_pass_through_opening", False)),
        "requires_deep_reach": bool(profile.get("requires_deep_reach", False)),
        "available_support_surface": bool(profile.get("available_support_surface", False)),
        "slip_hazard_present": bool(profile.get("slip_hazard_present", False)),
        "confinement_level": conf_int,
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
        measurement_id = f"m_{idx:02d}"
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

        estimate_basis = _safe_str(item.get("estimate_basis"), "")
        if estimate_basis not in _ALLOWED_ESTIMATE_BASIS:
            estimate_basis = "task_text_prior"

        related = _sanitize_id_list(
            item.get("related_environment_structure_ids"),
            structure_ids,
        )
        if not related and structure_ids:
            related = [structure_ids[0]]

        # source_refs: 관련 env_structure_id 사용 (없으면 measurement_id 자기참조)
        meas_source = related[:1] if related else [measurement_id]
        estimates.append(
            {
                "measurement_id": measurement_id,
                "parameter_name": parameter_name,
                "unit": unit,
                "bound_type": bound_type,
                "lower_value": lower_value,
                "upper_value": upper_value,
                "estimate_basis": estimate_basis,
                "related_environment_structure_ids": related,
                "source_refs": meas_source,
                "confidence": _clamp01(_safe_float(item.get("confidence"), 0.5)),
            }
        )

    if not estimates:
        default_related = [structure_ids[0]] if structure_ids else ["env_01"]
        estimates = [
            {
                "measurement_id": "m_01",
                "parameter_name": "opening_width",
                "unit": "level_1_to_5",
                "bound_type": "upper_bound",
                "lower_value": None,
                "upper_value": 3.0,
                "estimate_basis": "task_text_prior",
                "related_environment_structure_ids": default_related,
                "source_refs": default_related[:1],
                "confidence": 0.5,
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
        # constraint_id 는 schema 가 ^c_[0-9]{2}$ 패턴을 강제하므로 LLM 출력은
        # 무시하고 코드가 결정적으로 c_01, c_02 ... 로 채번한다.
        constraint_id = f"c_{idx:02d}"
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
        # constraint_catalog.parameter_name 은 측정값 enum 이 아닌
        # constraint enum 을 사용해야 한다.
        parameter_name = _safe_str(item.get("parameter_name"), "")
        if parameter_name not in _ALLOWED_CONSTRAINT_PARAMETER_UNITS:
            parameter_name = "min_effective_reach"

        allowed_units = _ALLOWED_CONSTRAINT_PARAMETER_UNITS[parameter_name]
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

        category = _safe_str(item.get("category"), "")
        if category not in _ALLOWED_CONSTRAINT_CATEGORIES:
            # parameter_name 으로부터 카테고리를 추정 (덜 임의적인 default)
            if parameter_name in {
                "min_effective_reach", "max_tool_body_width",
                "max_cross_section_width", "max_tip_thickness",
                "max_tip_diameter", "min_insert_depth",
                "min_contact_span", "max_required_entry_angle_deg",
            }:
                category = "geometric"
            elif parameter_name in {
                "min_tip_force_transmission_level",
                "min_global_stiffness_level",
                "max_allowed_compliance_level",
            }:
                category = "mechanical"
            elif parameter_name in {
                "min_surface_friction_level",
                "preferred_contact_friction_level",
            }:
                category = "surface_interaction"
            elif parameter_name in {
                "min_placement_stability_level",
                "max_allowed_roll_instability_level",
            }:
                category = "stability_access"
            else:
                category = "geometric"

        applies_to = _safe_str(item.get("applies_to"), "")
        if applies_to not in _ALLOWED_CONSTRAINT_APPLIES_TO:
            # parameter_name 으로부터 applies_to 를 추정
            if parameter_name in {
                "min_placement_stability_level",
                "max_allowed_roll_instability_level",
            }:
                applies_to = "placement_strategy"
            elif parameter_name in {
                "max_required_entry_angle_deg",
                "min_effective_reach",
            }:
                applies_to = "approach_path"
            else:
                applies_to = "tool_profile"

        constraints.append(
            {
                "constraint_id": constraint_id,
                "subgoal_ids": constraint_subgoal_ids,
                "priority": priority,
                "hardness": hardness,
                "category": category,
                "parameter_name": parameter_name,
                "bound_type": bound_type,
                "lower_value": lower_value,
                "upper_value": upper_value,
                "unit": unit,
                "applies_to": applies_to,
                "target_binding_ids": [binding_id],
                "measurement_ids": constraint_measurement_ids,
                "confidence": _clamp01(_safe_float(item.get("confidence"), 0.5)),
                # source_refs: 측정 id 우선, 없으면 subgoal id, 없으면 binding_id
                "source_refs": (
                    constraint_measurement_ids[:1]
                    if constraint_measurement_ids
                    else (constraint_subgoal_ids[:1] if constraint_subgoal_ids else [binding_id])
                ),
            }
        )

    if not constraints:
        default_measurement = numeric_estimates[0] if numeric_estimates else {}
        # constraint enum 으로 default 잡기 (측정값 enum 아님)
        default_param = "min_effective_reach"
        default_unit = _preferred_unit(_ALLOWED_CONSTRAINT_PARAMETER_UNITS[default_param])
        default_bound = "upper_bound"
        default_meas_ids = (
            [default_measurement.get("measurement_id")]
            if default_measurement.get("measurement_id")
            else []
        )
        fallback_source = (
            default_meas_ids[:1]
            if default_meas_ids
            else (subgoal_ids[:1] if subgoal_ids else [binding_id])
        )
        constraints = [
            {
                "constraint_id": "c_01",
                "subgoal_ids": list(subgoal_ids),
                "priority": "medium",
                "hardness": "hard",
                "category": "geometric",
                "parameter_name": default_param,
                "bound_type": default_bound,
                "lower_value": None,
                "upper_value": 3.0,
                "unit": default_unit,
                "applies_to": "tool_profile",
                "target_binding_ids": [binding_id],
                "measurement_ids": default_meas_ids,
                "confidence": 0.5,
                "source_refs": fallback_source,
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

    handoff_status = _safe_str(payload.get("handoff_status"), "")
    if handoff_status not in _ALLOWED_HANDOFF_STATUS:
        handoff_status = "ready"

    units_policy = _safe_str(payload.get("constraint_units_policy"), "")
    if units_policy not in _ALLOWED_CONSTRAINT_UNITS_POLICY:
        units_policy = "mixed_metric_and_ordinal"

    raw_omitted = payload.get("omitted_constraint_families")
    omitted: list[str] = []
    seen_omitted: set[str] = set()
    if isinstance(raw_omitted, list):
        for fam in raw_omitted:
            if isinstance(fam, str) and fam in _ALLOWED_OMITTED_FAMILIES and fam not in seen_omitted:
                omitted.append(fam)
                seen_omitted.add(fam)

    raw_notes = payload.get("notes")
    notes = [str(n) for n in raw_notes if isinstance(n, str)] if isinstance(raw_notes, list) else []

    # binding_id 는 schema 에 없는 필드라 제거.
    return {
        "handoff_status": handoff_status,
        "constraint_units_policy": units_policy,
        "handoff_constraint_ids": handoff_constraint_ids,
        "pending_merge_sources": pending_merge_sources,
        "omitted_constraint_families": omitted,
        "notes": notes,
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
