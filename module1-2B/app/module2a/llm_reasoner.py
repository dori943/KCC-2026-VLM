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
        or "gpt-4.1-mini"
    )
    base = api_base or os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"
    api_url = base.rstrip("/") + "/chat/completions"
    schema = _sanitize_response_schema(
        load_json(root / "schemas" / "module2a_output.schema.json")
    )
    prompt_spec = _load_prompt_spec(
        path=prompt_spec_path or (root / "specs" / "module2a_prompt_spec.md")
    )

    system_prompt = (
        "You are Module 2-A task decomposition and function requirement extraction reasoner. "
        "Return only strict JSON that follows the provided JSON schema."
    )
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

    body = {
        "model": resolved_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
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
    except ValueError:
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
        return {
            k: _sanitize_response_schema(v)
            for k, v in schema.items()
            if k not in unsupported
        }
    if isinstance(schema, list):
        return [_sanitize_response_schema(item) for item in schema]
    return schema
