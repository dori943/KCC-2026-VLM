"""Module 2-B LLM-only pipeline orchestration and artifact export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.module2b.llm_reasoner import generate_module2b_output_with_llm
from app.module2b.normalizer import normalize_module2b_bundle
from app.module2b.providers import FileBundleProvider, Module2BBundleProvider
from app.module2b.validators import Module2BInputValidator, Module2BOutputValidator
from app.utils import (
    build_task_output_root,
    derive_task_name,
    dump_json,
    ensure_unique_run_dir,
    load_json,
    load_yaml,
    project_root,
    timestamp_id,
)


def run_module2b_pipeline(
    provider: Module2BBundleProvider | None = None,
    bundle_path: Path | None = None,
    module2_common_path: Path | None = None,
    module2a_output_path: Path | None = None,
    case_id: str | None = None,
    output_root: Path | None = None,
    variant: str | None = None,
    task_name: str | None = None,
    api_key: str | None = None,
    model: str = "gpt-4.1-mini",
    reasoner_mode: str = "llm",
) -> dict[str, Any]:
    """Run Module 2-B with LLM-only reasoning and export layered artifacts."""
    if reasoner_mode != "llm":
        raise ValueError(
            "Unsupported Module 2-B reasoner mode: "
            f"{reasoner_mode}. This build supports only: llm."
        )

    root = project_root()
    output_root = output_root or (root / "outputs")
    run_id = timestamp_id()

    bundle_provider = provider or FileBundleProvider()
    provider_result = bundle_provider.get_bundle(
        bundle_path=bundle_path,
        module2_common_path=module2_common_path,
        module2a_output_path=module2a_output_path,
        case_id=case_id,
    )
    raw_bundle = provider_result.bundle

    suffix = case_id or _infer_suffix(provider_result.metadata)
    resolved_task_name = derive_task_name(
        task_name=task_name,
        bundle_path=bundle_path or provider_result.metadata.get("bundle_path"),
        case_id=case_id,
    )
    task_root = build_task_output_root(output_root, resolved_task_name)
    run_dir = ensure_unique_run_dir(task_root, f"module2b_{run_id}_{suffix}")

    prompt_registry = load_yaml(root / "configs" / "prompt_registry.yaml")
    vocab_registry = load_json(root / "configs" / "vocab_registry.json")

    input_validator = Module2BInputValidator(root=root)
    input_validation = input_validator.validate(raw_bundle)
    normalized_context = normalize_module2b_bundle(raw_bundle)

    module2b_output, llm_trace = generate_module2b_output_with_llm(
        raw_bundle=raw_bundle,
        normalized_context=normalized_context,
        api_key=api_key,
        model=model,
    )

    output_validator = Module2BOutputValidator(root=root)
    output_validation = output_validator.validate(
        payload=module2b_output,
        inventory_ids=normalized_context.object_id_order,
        subgoal_ids=[subgoal.subgoal_id for subgoal in normalized_context.subgoals],
    )

    target_trace = _build_target_binding_trace(module2b_output)
    environment_trace = _build_environment_trace(module2b_output)
    numeric_trace = _build_numeric_trace(module2b_output)
    constraint_trace = _build_constraint_trace(module2b_output)
    handoff_preview = _build_handoff_preview(module2b_output)
    confidence_breakdown = _build_confidence_breakdown(module2b_output)

    validation_report = {
        "schema_name": "module2b_validation_report",
        "schema_version": "0.1",
        "input_validation": input_validation.to_dict(),
        "output_validation": output_validation.to_dict(),
    }

    diagnostics = {
        "schema_name": "module2b_diagnostics",
        "schema_version": "0.1",
        "run_id": run_id,
        "reasoner_mode": reasoner_mode,
        "llm_reasoner": {
            "mode": llm_trace.get("mode"),
            "model": llm_trace.get("model"),
            "api_url": llm_trace.get("api_url"),
            "api_usage": llm_trace.get("api_usage"),
        },
        "validation_report": validation_report,
        "trace_refs": {
            "target_binding_candidates": "target_binding_candidates.json",
            "environment_structure_candidates": "environment_structure_candidates.json",
            "numeric_estimates_trace": "numeric_estimates_trace.json",
            "derived_constraints_trace": "derived_constraints_trace.json",
        },
        "deferred_item_reasons": _collect_deferred_reasons(module2b_output),
        "confidence_component_breakdown": confidence_breakdown,
    }

    target_binding = module2b_output.get("target_binding", {})
    env_context = module2b_output.get("environment_context", {})
    derived = module2b_output.get("derived_constraints", {})
    module3_handoff = module2b_output.get("module3_handoff", {})
    usage = llm_trace.get("api_usage", {})

    summary = {
        "run_id": run_id,
        "task_id": normalized_context.task_id,
        "provider": provider_result.metadata.get("provider"),
        "case_id": provider_result.metadata.get("case_id"),
        "reasoner_mode": reasoner_mode,
        "target_mode": target_binding.get("target_mode"),
        "binding_status": target_binding.get("binding_status"),
        "environment_structure_count": len(env_context.get("relevant_structures", [])),
        "numeric_estimate_count": len(env_context.get("numeric_estimates", [])),
        "constraint_count": len(derived.get("constraint_catalog", [])),
        "handoff_status": module3_handoff.get("handoff_status"),
        "api_call_count": int(usage.get("api_call_count", 0) or 0),
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
        "input_valid": input_validation.valid,
        "output_valid": output_validation.valid,
    }

    selected_variant = variant or "llm_only_v1"
    manifest = {
        "schema_name": "module2b_run_manifest",
        "schema_version": "0.1",
        "run_id": run_id,
        "provider_metadata": provider_result.metadata,
        "prompt_variant": prompt_registry["defaults"]["active_module2b_variant"],
        "run_variant": selected_variant,
        "reasoner_mode": reasoner_mode,
        "module2b_reasoner": {
            "mode": llm_trace.get("mode"),
            "model": llm_trace.get("model"),
            "api_url": llm_trace.get("api_url"),
            "api_usage": llm_trace.get("api_usage"),
        },
        "versions": {
            "module2b_output_schema": "module2b_output_env_only@0.1",
            "module2_common_schema": "module2_common_input_for_module2b_derived_min@0.1",
            "module2a_output_schema": "module2a_output@0.2",
            "vocab_registry_version": vocab_registry["registry_version"],
        },
        "validation": {
            "input_valid": input_validation.valid,
            "output_valid": output_validation.valid,
            "input_error_count": len(input_validation.errors),
            "output_error_count": len(output_validation.errors),
        },
        "artifacts": {
            "run_manifest": "run_manifest.json",
            "raw_input_bundle": "raw_input_bundle.json",
            "normalized_context": "normalized_context.json",
            "validation_report": "validation_report.json",
            "target_binding_candidates": "target_binding_candidates.json",
            "environment_structure_candidates": "environment_structure_candidates.json",
            "numeric_estimates_trace": "numeric_estimates_trace.json",
            "derived_constraints_trace": "derived_constraints_trace.json",
            "module2b_output": "module2b_output.json",
            "module3_handoff_preview": "module3_handoff_preview.json",
            "module2b_diagnostics": "module2b_diagnostics.json",
            "summary": "summary.json",
            "module2b_llm_trace": "module2b_llm_trace.json",
        },
    }

    dump_json(raw_bundle, run_dir / "raw_input_bundle.json")
    dump_json(normalized_context.to_dict(), run_dir / "normalized_context.json")
    dump_json(validation_report, run_dir / "validation_report.json")
    dump_json(target_trace, run_dir / "target_binding_candidates.json")
    dump_json(environment_trace, run_dir / "environment_structure_candidates.json")
    dump_json(numeric_trace, run_dir / "numeric_estimates_trace.json")
    dump_json(constraint_trace, run_dir / "derived_constraints_trace.json")
    dump_json(module2b_output, run_dir / "module2b_output.json")
    dump_json(handoff_preview, run_dir / "module3_handoff_preview.json")
    dump_json(diagnostics, run_dir / "module2b_diagnostics.json")
    dump_json(summary, run_dir / "summary.json")
    dump_json(llm_trace, run_dir / "module2b_llm_trace.json")
    dump_json(manifest, run_dir / "run_manifest.json")

    if not input_validation.valid:
        raise ValueError(
            "Module 2-B input validation failed: "
            + " | ".join(input_validation.errors[:3])
            + (" ..." if len(input_validation.errors) > 3 else "")
        )
    if not output_validation.valid:
        raise ValueError(
            "Module 2-B output validation failed: "
            + " | ".join(output_validation.errors[:3])
            + (" ..." if len(output_validation.errors) > 3 else "")
        )

    return {
        "run_dir": str(run_dir),
        "summary": summary,
        "manifest": manifest,
    }


def export_module2b_normalized_context(
    bundle_path: Path | None = None,
    module2_common_path: Path | None = None,
    module2a_output_path: Path | None = None,
    case_id: str | None = None,
    provider: Module2BBundleProvider | None = None,
    output_root: Path | None = None,
    task_name: str | None = None,
) -> dict[str, Any]:
    """Export only Layer 2 normalized context from Module 2-B input."""
    root = project_root()
    output_root = output_root or (root / "outputs")
    run_id = timestamp_id()
    bundle_provider = provider or FileBundleProvider()
    provider_result = bundle_provider.get_bundle(
        bundle_path=bundle_path,
        module2_common_path=module2_common_path,
        module2a_output_path=module2a_output_path,
        case_id=case_id,
    )
    normalized_context = normalize_module2b_bundle(provider_result.bundle)

    suffix = case_id or _infer_suffix(provider_result.metadata)
    resolved_task_name = derive_task_name(
        task_name=task_name,
        bundle_path=bundle_path or provider_result.metadata.get("bundle_path"),
        case_id=case_id,
    )
    task_root = build_task_output_root(output_root, resolved_task_name)
    run_dir = ensure_unique_run_dir(task_root, f"module2b_normalized_{run_id}_{suffix}")
    dump_json(provider_result.bundle, run_dir / "raw_input_bundle.json")
    dump_json(normalized_context.to_dict(), run_dir / "normalized_context.json")
    dump_json(
        {
            "schema_name": "module2b_normalized_export_manifest",
            "schema_version": "0.1",
            "run_id": run_id,
            "provider_metadata": provider_result.metadata,
            "artifacts": {
                "raw_input_bundle": "raw_input_bundle.json",
                "normalized_context": "normalized_context.json",
            },
        },
        run_dir / "run_manifest.json",
    )
    return {
        "run_dir": str(run_dir),
        "normalized_path": str(run_dir / "normalized_context.json"),
    }


def compare_module2b_outputs(
    run_a: Path,
    run_b: Path,
) -> dict[str, Any]:
    """Compare two Module 2-B runs and return structural and value diffs."""
    payload_a = _load_module2b_output(run_a)
    payload_b = _load_module2b_output(run_b)

    structural_diff = _structural_diff(payload_a, payload_b)
    value_diff = _value_diff(payload_a, payload_b)

    return {
        "schema_name": "module2b_comparison",
        "schema_version": "0.1",
        "run_a": str(run_a),
        "run_b": str(run_b),
        "same_structure": len(structural_diff) == 0,
        "same_values": len(value_diff) == 0,
        "structural_diff_count": len(structural_diff),
        "value_diff_count": len(value_diff),
        "structural_diff": structural_diff,
        "value_diff": value_diff,
    }


def _build_target_binding_trace(module2b_output: dict[str, Any]) -> dict[str, Any]:
    target_binding = module2b_output.get("target_binding", {})
    candidate_ids = target_binding.get("candidate_ids_ranked", [])
    if not isinstance(candidate_ids, list):
        candidate_ids = []
    candidate_scoring = [
        {
            "object_id": str(object_id),
            "final_score": round(max(0.0, 1.0 - 0.1 * idx), 4),
            "reason": "llm_ranked_candidate",
        }
        for idx, object_id in enumerate(candidate_ids)
    ]
    selected_object_id = None
    primary_targets = target_binding.get("primary_targets", [])
    if isinstance(primary_targets, list) and primary_targets:
        head = primary_targets[0]
        if isinstance(head, dict):
            selected_object_id = head.get("object_id")
    if selected_object_id is None and candidate_ids:
        selected_object_id = candidate_ids[0]

    top_score = candidate_scoring[0]["final_score"] if candidate_scoring else 0.0
    second_score = candidate_scoring[1]["final_score"] if len(candidate_scoring) > 1 else 0.0
    return {
        "mode": "llm_only",
        "candidate_scoring": candidate_scoring,
        "resolution": {
            "selected_object_id": selected_object_id,
            "top_score": top_score,
            "second_score": second_score,
            "thresholds": {"strong_margin_min": 0.15},
        },
        "binding_status": target_binding.get("binding_status"),
    }


def _build_environment_trace(module2b_output: dict[str, Any]) -> dict[str, Any]:
    env_context = module2b_output.get("environment_context", {})
    return {
        "mode": "llm_only",
        "topology_tags": env_context.get("topology_tags", []),
        "relevant_structures": env_context.get("relevant_structures", []),
        "access_path_profile": env_context.get("access_path_profile", {}),
    }


def _build_numeric_trace(module2b_output: dict[str, Any]) -> dict[str, Any]:
    env_context = module2b_output.get("environment_context", {})
    deferred = module2b_output.get("deferred_items", {})
    omissions = deferred.get("not_numericized_items", [])
    return {
        "mode": "llm_only",
        "measurements": env_context.get("numeric_estimates", []),
        "omissions": omissions if isinstance(omissions, list) else [],
    }


def _build_constraint_trace(module2b_output: dict[str, Any]) -> dict[str, Any]:
    derived = module2b_output.get("derived_constraints", {})
    return {
        "mode": "llm_only",
        "constraint_catalog": derived.get("constraint_catalog", []),
        "global_constraint_ids": derived.get("global_constraint_ids", []),
        "subgoal_bindings": derived.get("subgoal_bindings", []),
    }


def _build_handoff_preview(module2b_output: dict[str, Any]) -> dict[str, Any]:
    handoff = module2b_output.get("module3_handoff", {})
    handoff_ids = handoff.get("handoff_constraint_ids", [])
    if not isinstance(handoff_ids, list):
        handoff_ids = []
    return {
        "schema_name": "module3_handoff_preview",
        "schema_version": "0.1",
        "handoff_status": handoff.get("handoff_status"),
        "target_binding_id": handoff.get("target_binding_id"),
        "handoff_constraint_ids": handoff_ids,
        "handoff_constraint_count": len(handoff_ids),
        "constraint_units_policy": handoff.get("constraint_units_policy"),
        "pending_merge_sources": handoff.get("pending_merge_sources", []),
        "omitted_constraint_families": [],
    }


def _build_confidence_breakdown(module2b_output: dict[str, Any]) -> dict[str, Any]:
    target_binding = module2b_output.get("target_binding", {})
    env_context = module2b_output.get("environment_context", {})
    derived = module2b_output.get("derived_constraints", {})

    target_conf = _safe_float(target_binding.get("confidence"), 0.0)
    env_components = [
        _safe_float(item.get("confidence"), 0.0)
        for item in env_context.get("relevant_structures", [])
        if isinstance(item, dict)
    ] + [
        _safe_float(item.get("confidence"), 0.0)
        for item in env_context.get("topology_tags", [])
        if isinstance(item, dict)
    ]
    constraint_components = [
        _safe_float(item.get("confidence"), 0.0)
        for item in derived.get("constraint_catalog", [])
        if isinstance(item, dict)
    ]

    env_mean = (sum(env_components) / len(env_components)) if env_components else 0.0
    constraint_mean = (
        (sum(constraint_components) / len(constraint_components))
        if constraint_components
        else 0.0
    )
    return {
        "target_binding": {
            "final": round(target_conf, 4),
            "source": "target_binding.confidence",
        },
        "environment_binding": {
            "components": [round(value, 4) for value in env_components],
            "mean": round(env_mean, 4),
        },
        "constraint_set": {
            "components": [round(value, 4) for value in constraint_components],
            "mean": round(constraint_mean, 4),
        },
    }


def _collect_deferred_reasons(module2b_output: dict[str, Any]) -> list[str]:
    deferred = module2b_output.get("deferred_items", {})
    if not isinstance(deferred, dict):
        return []
    reasons: list[str] = []
    for value in deferred.values():
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, dict):
                continue
            reason = item.get("reason")
            if isinstance(reason, str) and reason:
                reasons.append(reason)
    return _unique_keep_order(reasons)


def _load_module2b_output(path: Path) -> dict[str, Any]:
    if path.is_dir():
        return load_json(path / "module2b_output.json")
    return load_json(path)


def _structural_diff(a: Any, b: Any, path: str = "$") -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []
    if isinstance(a, dict) and isinstance(b, dict):
        keys = sorted(set(a.keys()) | set(b.keys()))
        for key in keys:
            if key not in a:
                diffs.append({"path": f"{path}.{key}", "type": "missing_in_a"})
                continue
            if key not in b:
                diffs.append({"path": f"{path}.{key}", "type": "missing_in_b"})
                continue
            diffs.extend(_structural_diff(a[key], b[key], f"{path}.{key}"))
        return diffs

    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(
                {
                    "path": path,
                    "type": "length_mismatch",
                    "len_a": len(a),
                    "len_b": len(b),
                }
            )
        for idx, (item_a, item_b) in enumerate(zip(a, b)):
            diffs.extend(_structural_diff(item_a, item_b, f"{path}[{idx}]"))
        return diffs

    if type(a) is not type(b):
        diffs.append(
            {
                "path": path,
                "type": "type_mismatch",
                "type_a": type(a).__name__,
                "type_b": type(b).__name__,
            }
        )
    return diffs


def _value_diff(a: Any, b: Any, path: str = "$") -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a.keys()) & set(b.keys())):
            diffs.extend(_value_diff(a[key], b[key], f"{path}.{key}"))
        return diffs

    if isinstance(a, list) and isinstance(b, list):
        for idx, (item_a, item_b) in enumerate(zip(a, b)):
            diffs.extend(_value_diff(item_a, item_b, f"{path}[{idx}]"))
        return diffs

    if a != b:
        diffs.append({"path": path, "value_a": a, "value_b": b})
    return diffs


def _infer_suffix(metadata: dict[str, Any]) -> str:
    for key in ("case_id", "bundle_path", "module2_common_path"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return Path(value).stem
    return "ad_hoc"


def _unique_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
