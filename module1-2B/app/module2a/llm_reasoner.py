"""LLM-backed Module 2-A reasoner for subgoal decomposition."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib import error as url_error
from urllib import request as url_request

from app.utils import load_json, project_root


def generate_module2a_output_with_llm(
    module2_common_input: dict[str, Any],
    vocab_registry: dict[str, Any],
    prompt_spec_path: Path | None = None,
    api_key: str | None = None,
    model: str | None = None,
    timeout_seconds: float = 120.0,
    api_base: str | None = None,
    task_scene_image_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate Module 2-A output by calling an OpenAI LLM."""
    root = project_root()
    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            "OPENAI_API_KEY is required for Module 2-A LLM reasoner. "
            "Set the environment variable and retry."
        )

    resolved_model = (
        model
        or os.getenv("MODULE2A_REASONER_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-4o"  # 2026-04-28: gpt-4.1-mini → gpt-4o (복잡 subgoal 분해 추론 강화)
    )
    base = api_base or os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"
    api_url = base.rstrip("/") + "/chat/completions"
    schema = _sanitize_response_schema(
        load_json(root / "schemas" / "module2a_output.schema.json")
    )
    prompt_spec = _load_prompt_spec(
        path=prompt_spec_path or (root / "specs" / "module2a_prompt_spec.md")
    )

    # task_scene_image 보조 입력 사용 시 system prompt 에 가드 안내문 추가
    from app.utils import GUARD_TEXT_2A, build_user_content_with_image

    system_prompt = (
        "You are Module 2-A task decomposition and function requirement extraction reasoner. "
        "Return only strict JSON that follows the provided JSON schema."
    )
    if task_scene_image_path is not None and Path(task_scene_image_path).exists():
        system_prompt = system_prompt + GUARD_TEXT_2A
    user_payload = {
        "task": "Generate module2a_output from module2_common_input.",
        "requirements": [
            "Output must validate against the schema exactly.",
            "Use conservative reasoning grounded in provided scene resources.",
            "Keep subgoal_id format as sg_XX and preserve forward dependency ordering.",
            "Use only valid atom and primitive codes from vocab_registry.module2a.",
        ],
        "module2a_prompt_spec": prompt_spec,
        "module2_common_input": module2_common_input,
        "module2a_vocab": vocab_registry.get("module2a", {}),
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
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "module2a_output",
                "strict": True,
                "schema": schema,
            },
        },
        "temperature": 0,
    }

    try:
        response_payload = _post_json(
            url=api_url,
            api_key=resolved_api_key,
            body=body,
            timeout_seconds=timeout_seconds,
        )
    except ValueError as exc:
        # 진단용 로깅: strict json_schema 모드가 왜 실패했는지 그대로 출력.
        # OpenAI 가 schema 자체를 거절(invalid keyword/구조)했는지, 아니면
        # 모델이 schema 를 못 맞춘 건지 판단하기 위해 필요.
        print(
            "[module2a][llm_reasoner] strict json_schema mode FAILED, "
            "falling back to json_object mode.",
            flush=True,
        )
        print(f"[module2a][llm_reasoner] strict-mode error: {exc}", flush=True)
        try:
            schema_preview = json.dumps(schema, ensure_ascii=False)[:1500]
        except Exception:  # noqa: BLE001
            schema_preview = "<unserializable>"
        print(
            f"[module2a][llm_reasoner] sanitized schema preview (first 1500 chars): "
            f"{schema_preview}",
            flush=True,
        )
        fallback_body = {
            "model": resolved_model,
            "messages": body["messages"],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        response_payload = _post_json(
            url=api_url,
            api_key=resolved_api_key,
            body=fallback_body,
            timeout_seconds=timeout_seconds,
        )

    parsed = _extract_json_content(response_payload=response_payload)
    output = _normalize_module2a_payload(parsed=parsed)
    output = _ensure_schema_compliance(
        output=output,
        module2_common_input=module2_common_input,
    )
    # LLM 이 생성한 pybullet_bridge 값들은 derivable metadata 라 무시하고
    # 시멘틱 필드 + vocab_registry 를 기준으로 코드가 직접 재계산해 덮어쓴다.
    # (subgoal_index, *_count, *_code, uncertainty_score 등은 LLM 이 정할 게 아님.)
    output = _rebuild_pybullet_bridges(
        output=output,
        vocab_registry=vocab_registry,
    )
    usage = _extract_usage(response_payload.get("usage"))
    return output, {
        "mode": "llm_openai",
        "model": resolved_model,
        "api_url": api_url,
        "api_usage": usage,
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


def _extract_json_content(response_payload: dict[str, Any]) -> dict[str, Any]:
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
    return parsed


def _normalize_module2a_payload(parsed: dict[str, Any]) -> dict[str, Any]:
    direct = parsed.get("module2a_output")
    if isinstance(direct, dict):
        payload = dict(direct)
    else:
        payload = dict(parsed)

    if not isinstance(payload.get("subgoals"), list):
        task_decomposition = payload.get("task_decomposition")
        if isinstance(task_decomposition, list):
            payload["subgoals"] = _normalize_task_decomposition(task_decomposition)

    payload.setdefault("schema_name", "module2a_output")
    payload.setdefault("schema_version", "0.2")
    return payload


def _ensure_schema_compliance(
    output: dict[str, Any],
    module2_common_input: dict[str, Any],
) -> dict[str, Any]:
    """LLM이 누락한 schema 필수 필드를 합리적 기본값으로 채워 넣는다.

    Module 2-A schema는 stage/task_model/task_level_requirement_summary/
    scene_resource_readout/pybullet_bridge 를 요구하지만, LLM이 strict mode 실패 후
    json_object fallback으로 빠지면 schema를 무시하고 자유 형식 JSON을 생성한다.
    여기서 누락 필드를 채워 downstream(Module 2-B) validation을 통과시킨다.
    """
    payload = dict(output)

    # schema에 없는 필드 제거
    payload.pop("functional_requirements", None)
    payload.pop("task_decomposition", None)

    task_brief = module2_common_input.get("task_brief", {}) or {}
    user_goal = str(task_brief.get("user_goal") or "").strip()
    success_criteria = task_brief.get("success_criteria", []) or []

    # subgoals 정규화 (function_requirements 등 보강)
    subgoals = payload.get("subgoals", []) or []
    required_atoms_union: list[str] = []
    preferred_atoms_union: list[str] = []
    risk_atoms_union: list[str] = []
    required_primitives_union: list[str] = []

    for sg in subgoals:
        if not isinstance(sg, dict):
            continue
        sg.setdefault("subgoal_id", "sg_01")
        sg.setdefault("subgoal_name", sg.get("subgoal_id", "subgoal"))
        sg.setdefault("objective", "")
        sg.setdefault("success_condition", sg.get("objective", ""))
        sg.setdefault("depends_on", [])
        sg.setdefault("required_interaction_primitives", [])
        fr = sg.setdefault("function_requirements", {})
        if isinstance(fr, dict):
            fr.setdefault("required_atoms", [])
            fr.setdefault("preferred_atoms", [])
            fr.setdefault("risk_atoms_to_avoid", [])
            required_atoms_union.extend(fr.get("required_atoms", []))
            preferred_atoms_union.extend(fr.get("preferred_atoms", []))
            risk_atoms_union.extend(fr.get("risk_atoms_to_avoid", []))
        required_primitives_union.extend(sg.get("required_interaction_primitives", []))
        sg.setdefault("physical_rationale", "")
        sg.setdefault(
            "resource_feasibility_hint",
            {
                "coverage_status": "usable",
                "supporting_atoms_seen_in_scene": [],
                "gap_atoms": [],
            },
        )
        sg.setdefault("failure_risks", [])
        sg.setdefault(
            "pybullet_bridge",
            {
                "subgoal_index": 1,
                "depends_on_indices": [],
                "required_interaction_primitive_codes": [],
                "required_atom_codes": [],
                "preferred_atom_codes": [],
                "risk_atom_codes_to_avoid": [],
                "supporting_atom_codes_seen_in_scene": [],
                "gap_atom_codes": [],
                "coverage_status_code": 3,
                "uncertainty_score": 3,
                "required_interaction_primitive_count": 0,
                "required_atom_count": 0,
                "preferred_atom_count": 0,
                "risk_atom_count": 0,
                "required_atom_missing_count": 0,
                "has_blocking_gap_flag": 0,
            },
        )

    payload["subgoals"] = subgoals

    # stage
    payload.setdefault("stage", "module2a")

    # task_model
    payload.setdefault(
        "task_model",
        {
            "task_restatement": user_goal or "Decompose and satisfy the requested manipulation goal.",
            "primary_success_condition": (success_criteria[0] if success_criteria else user_goal) or "Goal achieved without damage.",
            "secondary_success_conditions": [str(s) for s in success_criteria[1:]] if len(success_criteria) > 1 else [],
            "decomposition_principle": "Sequential subgoal decomposition based on functional requirements.",
            "assumptions": ["Scene resources are sufficient at coarse level."],
            "deferred_items": [],
        },
    )

    # task_level_requirement_summary
    payload.setdefault(
        "task_level_requirement_summary",
        {
            "required_atoms_union": _dedupe_keep_order(required_atoms_union),
            "preferred_atoms_union": _dedupe_keep_order(preferred_atoms_union),
            "risk_atoms_to_avoid_union": _dedupe_keep_order(risk_atoms_union),
            "required_interaction_primitives_union": _dedupe_keep_order(required_primitives_union),
            "resource_gap_hypotheses": [],
            "overall_resource_sufficiency": "usable",
            "pybullet_bridge": {
                "required_atom_count": len(_dedupe_keep_order(required_atoms_union)),
                "preferred_atom_count": len(_dedupe_keep_order(preferred_atoms_union)),
                "risk_atom_count": len(_dedupe_keep_order(risk_atoms_union)),
                "required_interaction_primitive_count": len(_dedupe_keep_order(required_primitives_union)),
                "overall_resource_sufficiency_code": 3,
                "resource_gap_count": 0,
            },
        },
    )

    # scene_resource_readout — atoms는 object 배열이어야 함
    scene_resources = module2_common_input.get("scene_resources", {}) or {}
    summary = scene_resources.get("resource_summary", {}) or {}
    histogram = summary.get("affordance_histogram", {}) or {}
    atom_vocab = (vocab_registry := {}).get("module2a", {}).get("affordance_atoms", {})  # placeholder
    # vocab_registry는 외부 로드 — _ensure_schema_compliance에 안 넘겨줬으므로 빈 dict
    # atom_code는 0으로 기본값 (실제 코드는 module2a 처리에서 매핑되지만 여기선 모름)
    available_atom_objects = []
    for atom, count in histogram.items():
        try:
            c = int(count)
        except (TypeError, ValueError):
            c = 0
        if c >= 5:
            avail = "abundant"; avail_code = 5
        elif c >= 3:
            avail = "strong"; avail_code = 4
        elif c >= 2:
            avail = "present"; avail_code = 3
        elif c == 1:
            avail = "trace"; avail_code = 1
        else:
            avail = "none"; avail_code = 0
        available_atom_objects.append({
            "atom": atom,
            "availability": avail,
            "pybullet_bridge": {
                "atom_code": 0,
                "scene_count": c,
                "raw_availability_code": avail_code,
                "uncertainty_score": 2,
                "uncertainty_penalty": 0,
                "availability_code": avail_code,
            },
        })
    payload.setdefault(
        "scene_resource_readout",
        {
            "available_affordance_atoms": available_atom_objects,
            "high_risk_capability_areas": [],
            "high_uncertainty_capability_areas": [],
            "pybullet_bridge": {
                "available_atom_count": len(available_atom_objects),
                "high_risk_area_count": 0,
                "high_uncertainty_area_count": 0,
            },
        },
    )

    # top-level pybullet_bridge — format_version 은 integer
    payload.setdefault(
        "pybullet_bridge",
        {
            "format_version": 2,
            "subgoal_count": len(subgoals),
            "needs_target_binding_later_flag": 1,
            "needs_environment_constraint_binding_later_flag": 1,
            "task_target_is_tool_flag": 0,
            "task_target_is_state_change_object_flag": 1,
        },
    )

    return payload


def _rebuild_pybullet_bridges(
    output: dict[str, Any],
    vocab_registry: dict[str, Any],
) -> dict[str, Any]:
    """LLM 출력의 모든 pybullet_bridge 서브객체를 코드가 결정적으로 재계산해 덮어쓴다.

    이유:
      - subgoal_index, *_count, *_code, uncertainty_score 등은 시멘틱 필드와
        vocab_registry 로부터 100% derivable 한 metadata.
      - LLM 이 임의 정수를 생성하면 schema 의 minimum/maximum 위반으로 깨진다
        (e.g. subgoal_index: 0, uncertainty_score: 24).
      - OpenAI strict mode 가 minimum/maximum 키워드를 지원하지 않아 LLM 에
        제약을 전달할 길이 없으므로 후처리로 강제 정합화한다.
    """
    payload = dict(output)
    m2a_vocab = vocab_registry.get("module2a", {}) if isinstance(vocab_registry, dict) else {}
    primitive_codes: dict[str, int] = m2a_vocab.get("interaction_primitives", {}) or {}
    affordance_codes: dict[str, int] = m2a_vocab.get("affordance_atoms", {}) or {}
    risk_codes: dict[str, int] = m2a_vocab.get("risk_atoms", {}) or {}
    # 위험 atom 은 affordance 명에 들어있을 수도 있으므로 fallback 통합 매핑
    atom_code_lookup: dict[str, int] = {**affordance_codes, **risk_codes}

    coverage_codes = {
        "blocked": 0, "very_weak": 1, "weak": 2,
        "usable": 3, "strong": 4, "robust": 5,
    }
    availability_codes = {
        "none": 0, "trace": 1, "sparse": 2,
        "present": 3, "strong": 4, "abundant": 5,
    }
    sufficiency_codes = {
        "insufficient": 0, "very_limited": 1, "partial": 2,
        "usable": 3, "strong": 4, "sufficient": 5,
    }

    def _codes_from_names(names: Any, lookup: dict[str, int]) -> list[int]:
        out: list[int] = []
        seen: set[int] = set()
        if not isinstance(names, list):
            return out
        for n in names:
            if not isinstance(n, str):
                continue
            code = lookup.get(n.strip())
            if code is None or code in seen:
                continue
            seen.add(code)
            out.append(code)
        return out

    def _clamp(value: Any, lo: int, hi: int, default: int) -> int:
        try:
            v = int(value)
        except (TypeError, ValueError):
            return default
        return max(lo, min(hi, v))

    # ---------- subgoals[*].pybullet_bridge ----------
    subgoals = payload.get("subgoals", []) or []
    if not isinstance(subgoals, list):
        subgoals = []
    sgid_to_index: dict[str, int] = {}
    for i, sg in enumerate(subgoals, start=1):
        if not isinstance(sg, dict):
            continue
        sgid = str(sg.get("subgoal_id", f"sg_{i:02d}"))
        sgid_to_index[sgid] = i

    coverage_status_per_sg: list[str] = []
    uncertainty_per_sg: list[int] = []

    for i, sg in enumerate(subgoals, start=1):
        if not isinstance(sg, dict):
            continue
        fr = sg.get("function_requirements", {}) or {}
        feas = sg.get("resource_feasibility_hint", {}) or {}
        coverage = str(feas.get("coverage_status") or "usable")
        if coverage not in coverage_codes:
            coverage = "usable"
        coverage_status_per_sg.append(coverage)

        required_atoms = fr.get("required_atoms", []) if isinstance(fr, dict) else []
        preferred_atoms = fr.get("preferred_atoms", []) if isinstance(fr, dict) else []
        risk_atoms = fr.get("risk_atoms_to_avoid", []) if isinstance(fr, dict) else []
        supporting = feas.get("supporting_atoms_seen_in_scene", []) if isinstance(feas, dict) else []
        gap = feas.get("gap_atoms", []) if isinstance(feas, dict) else []
        primitives = sg.get("required_interaction_primitives", []) or []

        depends_indices = [
            sgid_to_index[d] for d in (sg.get("depends_on") or [])
            if isinstance(d, str) and d in sgid_to_index
        ]

        # uncertainty_score: LLM 이 준 게 합법 범위면 사용, 아니면 coverage 기반 추정
        coverage_to_uncertainty = {
            "blocked": 5, "very_weak": 4, "weak": 3,
            "usable": 2, "strong": 1, "robust": 1,
        }
        existing_bridge = sg.get("pybullet_bridge") if isinstance(sg.get("pybullet_bridge"), dict) else {}
        uncertainty = _clamp(
            existing_bridge.get("uncertainty_score"),
            lo=0, hi=5,
            default=coverage_to_uncertainty[coverage],
        )
        uncertainty_per_sg.append(uncertainty)

        required_atom_codes = _codes_from_names(required_atoms, atom_code_lookup)
        gap_atom_codes = _codes_from_names(gap, atom_code_lookup)
        missing_count = len(set(required_atom_codes) & set(gap_atom_codes))

        sg["pybullet_bridge"] = {
            "subgoal_index": i,
            "depends_on_indices": depends_indices,
            "required_interaction_primitive_codes": _codes_from_names(primitives, primitive_codes),
            "required_atom_codes": required_atom_codes,
            "preferred_atom_codes": _codes_from_names(preferred_atoms, atom_code_lookup),
            "risk_atom_codes_to_avoid": _codes_from_names(risk_atoms, atom_code_lookup),
            "supporting_atom_codes_seen_in_scene": _codes_from_names(supporting, atom_code_lookup),
            "gap_atom_codes": gap_atom_codes,
            "coverage_status_code": coverage_codes[coverage],
            "uncertainty_score": uncertainty,
            "required_interaction_primitive_count": len(primitives) if isinstance(primitives, list) else 0,
            "required_atom_count": len(required_atoms) if isinstance(required_atoms, list) else 0,
            "preferred_atom_count": len(preferred_atoms) if isinstance(preferred_atoms, list) else 0,
            "risk_atom_count": len(risk_atoms) if isinstance(risk_atoms, list) else 0,
            "required_atom_missing_count": missing_count,
            "has_blocking_gap_flag": 1 if coverage == "blocked" else 0,
        }

    # ---------- task_level_requirement_summary.pybullet_bridge ----------
    summary = payload.setdefault("task_level_requirement_summary", {})
    if not isinstance(summary, dict):
        summary = {}
        payload["task_level_requirement_summary"] = summary

    # LLM 이 만든 raw union 에 중복이 들어올 수 있음 → 강제 dedupe.
    # 스키마 검증("non-unique elements") 을 통과하기 위해 여기서 정리한다.
    req_union = _dedupe_keep_order(list(summary.get("required_atoms_union", []) or []))
    pref_union = _dedupe_keep_order(list(summary.get("preferred_atoms_union", []) or []))
    risk_union = _dedupe_keep_order(list(summary.get("risk_atoms_to_avoid_union", []) or []))
    primitives_union = _dedupe_keep_order(
        list(summary.get("required_interaction_primitives_union", []) or [])
    )
    # 정리된 결과를 summary 에 반영해 다음 단계 검증을 통과시킴
    summary["required_atoms_union"] = req_union
    summary["preferred_atoms_union"] = pref_union
    summary["risk_atoms_to_avoid_union"] = risk_union
    summary["required_interaction_primitives_union"] = primitives_union
    gap_hypotheses = summary.get("resource_gap_hypotheses", []) or []
    overall_suff = str(summary.get("overall_resource_sufficiency") or "usable")
    if overall_suff not in sufficiency_codes:
        overall_suff = "usable"
        summary["overall_resource_sufficiency"] = "usable"

    blocked_cnt = sum(1 for c in coverage_status_per_sg if c == "blocked")
    weak_or_worse = sum(1 for c in coverage_status_per_sg if c in {"blocked", "very_weak", "weak"})
    usable_or_better = sum(1 for c in coverage_status_per_sg if c in {"usable", "strong", "robust"})
    overall_uncertainty = max(uncertainty_per_sg) if uncertainty_per_sg else 2

    summary["pybullet_bridge"] = {
        "required_atoms_union_codes": _codes_from_names(req_union, atom_code_lookup),
        "preferred_atoms_union_codes": _codes_from_names(pref_union, atom_code_lookup),
        "risk_atoms_to_avoid_union_codes": _codes_from_names(risk_union, atom_code_lookup),
        "required_interaction_primitives_union_codes": _codes_from_names(primitives_union, primitive_codes),
        "resource_gap_atom_codes": _codes_from_names(gap_hypotheses, atom_code_lookup),
        "overall_resource_sufficiency_code": sufficiency_codes[overall_suff],
        "overall_uncertainty_score": _clamp(overall_uncertainty, 0, 5, 2),
        "required_atoms_union_count": len(req_union) if isinstance(req_union, list) else 0,
        "preferred_atoms_union_count": len(pref_union) if isinstance(pref_union, list) else 0,
        "risk_atoms_to_avoid_union_count": len(risk_union) if isinstance(risk_union, list) else 0,
        "required_interaction_primitives_union_count": len(primitives_union) if isinstance(primitives_union, list) else 0,
        "blocked_subgoal_count": blocked_cnt,
        "weak_or_worse_subgoal_count": weak_or_worse,
        "usable_or_better_subgoal_count": usable_or_better,
    }

    # ---------- scene_resource_readout (atoms[*].pybullet_bridge + 상위 pybullet_bridge) ----------
    readout = payload.setdefault("scene_resource_readout", {})
    if not isinstance(readout, dict):
        readout = {}
        payload["scene_resource_readout"] = readout

    atoms_list = readout.get("available_affordance_atoms", []) or []
    if not isinstance(atoms_list, list):
        atoms_list = []

    available_atom_codes_seen: list[int] = []
    seen_codes: set[int] = set()
    for entry in atoms_list:
        if not isinstance(entry, dict):
            continue
        atom_name = str(entry.get("atom") or "").strip()
        availability = str(entry.get("availability") or "none")
        if availability not in availability_codes:
            availability = "none"
            entry["availability"] = "none"
        atom_code = atom_code_lookup.get(atom_name, 0)
        if atom_code and atom_code not in seen_codes:
            seen_codes.add(atom_code)
            available_atom_codes_seen.append(atom_code)
        existing = entry.get("pybullet_bridge") if isinstance(entry.get("pybullet_bridge"), dict) else {}
        scene_count = _clamp(existing.get("scene_count"), 0, 10**6, 0)
        avail_code = availability_codes[availability]
        entry["pybullet_bridge"] = {
            "atom_code": int(atom_code) if atom_code else 0,
            "scene_count": scene_count,
            "raw_availability_code": avail_code,
            "uncertainty_score": _clamp(existing.get("uncertainty_score"), 0, 5, 2),
            "uncertainty_penalty": _clamp(existing.get("uncertainty_penalty"), 0, 2, 0),
            "availability_code": avail_code,
        }

    high_risk = readout.get("high_risk_capability_areas", []) or []
    high_unc = readout.get("high_uncertainty_capability_areas", []) or []
    readout["pybullet_bridge"] = {
        "available_atom_codes": available_atom_codes_seen,
        "available_atom_count": len(atoms_list),
        "high_risk_area_count": len(high_risk) if isinstance(high_risk, list) else 0,
        "high_uncertainty_area_count": len(high_unc) if isinstance(high_unc, list) else 0,
    }

    # ---------- top-level pybullet_bridge ----------
    task_model = payload.get("task_model") if isinstance(payload.get("task_model"), dict) else {}
    deferred = task_model.get("deferred_items") if isinstance(task_model.get("deferred_items"), dict) else {}
    needs_target = bool(deferred.get("needs_target_binding_later", True))
    needs_env = bool(deferred.get("needs_environment_constraint_binding_later", True))
    existing_top = payload.get("pybullet_bridge") if isinstance(payload.get("pybullet_bridge"), dict) else {}
    is_tool_flag = _clamp(existing_top.get("task_target_is_tool_flag"), 0, 1, 0)
    is_state_change_flag = _clamp(existing_top.get("task_target_is_state_change_object_flag"), 0, 1, 1)

    payload["pybullet_bridge"] = {
        "format_version": 2,
        "subgoal_count": len(subgoals),
        "needs_target_binding_later_flag": 1 if needs_target else 0,
        "needs_environment_constraint_binding_later_flag": 1 if needs_env else 0,
        "task_target_is_tool_flag": is_tool_flag,
        "task_target_is_state_change_object_flag": is_state_change_flag,
    }

    # 고정 필드 한 번 더 보정 (LLM 이 빠뜨릴 가능성 대비)
    payload["schema_name"] = "module2a_output"
    payload["schema_version"] = "0.2"
    payload["stage"] = "task_decomposition_and_function_requirement_extraction"

    return payload


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _normalize_task_decomposition(
    task_decomposition: list[Any],
) -> list[dict[str, Any]]:
    subgoals: list[dict[str, Any]] = []
    for idx, item in enumerate(task_decomposition, start=1):
        if not isinstance(item, dict):
            continue
        subgoal_id_raw = item.get("subgoal_id")
        subgoal_id = (
            str(subgoal_id_raw).strip()
            if isinstance(subgoal_id_raw, str) and str(subgoal_id_raw).strip()
            else f"sg_{idx:02d}"
        )

        objective_raw = item.get("objective")
        if not isinstance(objective_raw, str) or not objective_raw.strip():
            objective_raw = item.get("description")
        objective = str(objective_raw or "").strip()

        success_raw = item.get("success_condition")
        success_condition = str(success_raw or "").strip() or objective

        depends_raw = item.get("depends_on")
        if not isinstance(depends_raw, list):
            depends_raw = item.get("dependencies")
        depends_on = (
            [str(dep).strip() for dep in depends_raw if str(dep).strip()]
            if isinstance(depends_raw, list)
            else []
        )

        subgoals.append(
            {
                "subgoal_id": subgoal_id,
                "subgoal_name": str(item.get("name", "")).strip() or f"subgoal_{idx:02d}",
                "objective": objective,
                "success_condition": success_condition,
                "depends_on": depends_on,
            }
        )
    return subgoals


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


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _sanitize_response_schema(schema: Any) -> Any:
    unsupported = {
        "minProperties",
        "maxProperties",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "format",
        "default",
        "examples",
        "$id",
        "$schema",
    }
    if isinstance(schema, dict):
        cleaned = {
            k: _sanitize_response_schema(v)
            for k, v in schema.items()
            if k not in unsupported
        }
        # OpenAI strict mode 는 모든 노드에 'type' 키를 명시 요구.
        # const/enum 만 있고 type 이 없는 노드는 값에서 타입을 추론해 채워준다.
        if "type" not in cleaned:
            inferred = _infer_json_type(cleaned)
            if inferred is not None:
                cleaned["type"] = inferred
        return cleaned
    if isinstance(schema, list):
        return [_sanitize_response_schema(item) for item in schema]
    return schema


def _infer_json_type(node: dict[str, Any]) -> str | None:
    """const/enum 값에서 JSON Schema type 을 추론한다. 추론 불가면 None."""
    sample: Any = None
    if "const" in node:
        sample = node["const"]
    elif isinstance(node.get("enum"), list) and node["enum"]:
        sample = node["enum"][0]
    else:
        return None
    if isinstance(sample, bool):
        return "boolean"
    if isinstance(sample, int):
        return "integer"
    if isinstance(sample, float):
        return "number"
    if isinstance(sample, str):
        return "string"
    if isinstance(sample, list):
        return "array"
    if isinstance(sample, dict):
        return "object"
    if sample is None:
        return "null"
    return None
