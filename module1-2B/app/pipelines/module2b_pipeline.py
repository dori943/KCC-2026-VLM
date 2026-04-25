"""Module 2-B env-only pipeline orchestration and artifact export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.module2b.normalizer import normalize_module2b_bundle
from app.module2b.providers import FileBundleProvider, Module2BBundleProvider
from app.module2b.reasoners.constraint_generator import generate_constraints
from app.module2b.reasoners.environment_binding import run_environment_binding
from app.module2b.reasoners.handoff_builder import build_module3_handoff
from app.module2b.reasoners.numeric_estimator import derive_numeric_estimates
from app.module2b.reasoners.target_binding import run_target_binding
from app.module2b.utils import dedupe_keep_order, stable_round
from app.module2b.validators import (
    Module2BInputValidator,
    Module2BOutputValidator,
)
from app.utils import dump_json, ensure_dir, load_json, load_yaml, project_root, timestamp_id


def run_module2b_pipeline(
    provider: Module2BBundleProvider | None = None,
    bundle_path: Path | None = None,
    module2_common_path: Path | None = None,
    module2a_output_path: Path | None = None,
    case_id: str | None = None,
    output_root: Path | None = None,
    variant: str | None = None,
) -> dict[str, Any]:
    """Run deterministic Module 2-B env-only reasoning and export layered artifacts."""
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
    run_dir = _ensure_unique_run_dir(output_root=output_root, stem=f"module2b_{run_id}_{suffix}")

    prompt_registry = load_yaml(root / "configs" / "prompt_registry.yaml")
    run_variants = load_yaml(root / "configs" / "module2b_run_variants.yaml")
    alias_registry = load_yaml(root / "configs" / "target_alias_registry.yaml")
    vocab_registry = load_json(root / "configs" / "vocab_registry.json")

    selected_variant = variant or run_variants["default_variant"]
    variant_cfg = run_variants["variants"][selected_variant]

    target_rules = load_yaml(root / variant_cfg["target_binding_rules"])
    environment_rules = load_yaml(root / variant_cfg["environment_rules"])
    numeric_rules = load_yaml(root / variant_cfg["numericization_rules"])
    constraint_rules = load_yaml(root / variant_cfg["constraint_rules"])

    input_validator = Module2BInputValidator(root=root)
    input_validation = input_validator.validate(raw_bundle)

    normalized_context = normalize_module2b_bundle(raw_bundle)

    target_binding, target_trace = run_target_binding(
        context=normalized_context,
        rules_cfg=target_rules,
        alias_registry=alias_registry,
    )

    environment_result, environment_trace = run_environment_binding(
        context=normalized_context,
        target_binding=target_binding,
        rules_cfg=environment_rules,
    )

    _link_target_context_refs(target_binding=target_binding, environment_result=environment_result)

    measurements, numeric_trace, numeric_omissions = derive_numeric_estimates(
        context=normalized_context,
        target_binding=target_binding,
        environment_result=environment_result,
        rules_cfg=numeric_rules,
    )

    constraints, constraint_trace = generate_constraints(
        context=normalized_context,
        target_binding=target_binding,
        environment_result=environment_result,
        measurements=measurements,
        rules_cfg=constraint_rules,
    )

    deferred_items = _build_deferred_items(
        target_binding=target_binding,
        environment_result=environment_result,
        numeric_omissions=numeric_omissions,
        measurements=measurements,
    )

    confidence_summary, confidence_breakdown = _build_confidence_summary(
        target_binding=target_binding,
        environment_result=environment_result,
        constraints=constraints,
        deferred_items=deferred_items,
    )

    module3_handoff, handoff_preview = build_module3_handoff(
        target_binding=target_binding,
        measurements=measurements,
        derived_constraints=constraints,
        pending_merge_sources=["material_reasoner"],
    )

    module2b_output = {
        "schema_name": "module2b_output_env_only",
        "schema_version": "0.1",
        "stage": "target_object_and_environment_constraints_env_only",
        "task_id": normalized_context.task_id,
        "target_binding": target_binding,
        "environment_context": {
            "topology_tags": environment_result["topology_tags"],
            "relevant_structures": environment_result["relevant_structures"],
            "access_path_profile": environment_result["access_path_profile"],
            "numeric_estimates": measurements,
        },
        "derived_constraints": constraints,
        "module3_handoff": module3_handoff,
        "deferred_items": deferred_items,
        "confidence_summary": confidence_summary,
    }

    output_validator = Module2BOutputValidator(root=root)
    output_validation = output_validator.validate(
        payload=module2b_output,
        inventory_ids=normalized_context.object_id_order,
        subgoal_ids=[subgoal.subgoal_id for subgoal in normalized_context.subgoals],
    )

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
        "rule_versions": {
            "target_binding": target_rules.get("rule_version", "unknown"),
            "environment": environment_rules.get("rule_version", "unknown"),
            "numericization": numeric_rules.get("rule_version", "unknown"),
            "constraint": constraint_rules.get("rule_version", "unknown"),
        },
        "validation_report": validation_report,
        "trace_refs": {
            "target_binding_candidates": "target_binding_candidates.json",
            "environment_structure_candidates": "environment_structure_candidates.json",
            "numeric_estimates_trace": "numeric_estimates_trace.json",
            "derived_constraints_trace": "derived_constraints_trace.json",
        },
        "deferred_item_reasons": dedupe_keep_order(
            [
                item["reason"]
                for key in deferred_items.keys()
                for item in deferred_items[key]
            ]
        ),
        "confidence_component_breakdown": confidence_breakdown,
        "dedup_rules": [
            "target candidates: score desc + inventory order tie-break",
            "environment structures: first evidence order + structure role priority",
            "measurements: (environment_structure_id, parameter_name, bound_type)",
            "constraints: (subgoal_order, priority, category, parameter_name, applies_to)",
        ],
    }

    summary = {
        "run_id": run_id,
        "task_id": normalized_context.task_id,
        "provider": provider_result.metadata.get("provider"),
        "case_id": provider_result.metadata.get("case_id"),
        "target_mode": target_binding["target_mode"],
        "binding_status": target_binding["binding_status"],
        "environment_structure_count": len(environment_result["relevant_structures"]),
        "numeric_estimate_count": len(measurements),
        "constraint_count": len(constraints["constraint_catalog"]),
        "handoff_status": module3_handoff["handoff_status"],
        "input_valid": input_validation.valid,
        "output_valid": output_validation.valid,
    }

    manifest = {
        "schema_name": "module2b_run_manifest",
        "schema_version": "0.1",
        "run_id": run_id,
        "provider_metadata": provider_result.metadata,
        "prompt_variant": prompt_registry["defaults"]["active_module2b_variant"],
        "run_variant": selected_variant,
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
    run_dir = _ensure_unique_run_dir(
        output_root=output_root,
        stem=f"module2b_normalized_{run_id}_{suffix}",
    )
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


def _link_target_context_refs(target_binding: dict[str, Any], environment_result: dict[str, Any]) -> None:
    target_ids = {
        item.get("object_id")
        for item in target_binding.get("primary_targets", [])
        if isinstance(item, dict)
    }
    context_refs: list[str] = []

    for structure in environment_result.get("relevant_structures", []):
        related_ids = set(structure.get("related_object_ids", []))
        if target_ids & related_ids:
            context_refs.append(structure["environment_structure_id"])

    context_refs = dedupe_keep_order(context_refs)
    target_binding["context_refs"] = context_refs
    for target in target_binding.get("primary_targets", []):
        target["context_refs"] = list(context_refs)


def _build_deferred_items(
    target_binding: dict[str, Any],
    environment_result: dict[str, Any],
    numeric_omissions: list[dict[str, Any]],
    measurements: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    unresolved_target_ambiguities: list[dict[str, Any]] = []
    unresolved_environment_ambiguities: list[dict[str, Any]] = []
    needs_user_or_scale_anchor: list[dict[str, Any]] = []

    for reason in target_binding.get("deferred_reasons", []):
        unresolved_target_ambiguities.append(
            {
                "item": "target_binding",
                "reason": reason,
                "source_refs": [target_binding["binding_id"]],
            }
        )

    if not environment_result.get("relevant_structures"):
        unresolved_environment_ambiguities.append(
            {
                "item": "environment_structure",
                "reason": "no_environment_structure_synthesized",
                "source_refs": [target_binding["binding_id"]],
            }
        )

    low_conf_structures = [
        item
        for item in environment_result.get("relevant_structures", [])
        if float(item.get("confidence", 0.0)) < 0.45
    ]
    if low_conf_structures:
        unresolved_environment_ambiguities.append(
            {
                "item": "environment_structure",
                "reason": "low_confidence_environment_structure",
                "source_refs": dedupe_keep_order(
                    [item["environment_structure_id"] for item in low_conf_structures]
                ),
            }
        )

    for measurement in measurements:
        unit = measurement.get("unit")
        basis = measurement.get("estimate_basis")
        if unit == "level_1_to_5" or basis == "task_text_prior":
            needs_user_or_scale_anchor.append(
                {
                    "item": measurement["parameter_name"],
                    "reason": "weak_scale_anchor",
                    "source_refs": [measurement["measurement_id"]],
                }
            )

    return {
        "unresolved_target_ambiguities": unresolved_target_ambiguities,
        "unresolved_environment_ambiguities": unresolved_environment_ambiguities,
        "not_numericized_items": numeric_omissions,
        "needs_user_or_scale_anchor": needs_user_or_scale_anchor,
    }


def _build_confidence_summary(
    target_binding: dict[str, Any],
    environment_result: dict[str, Any],
    constraints: dict[str, Any],
    deferred_items: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    target_conf = float(target_binding.get("confidence", 0.0))

    env_conf_components = [
        float(item.get("confidence", 0.0))
        for item in environment_result.get("relevant_structures", [])
    ] + [
        float(item.get("confidence", 0.0))
        for item in environment_result.get("topology_tags", [])
    ]
    env_conf = (
        sum(env_conf_components) / float(len(env_conf_components))
        if env_conf_components
        else 0.0
    )

    constraint_conf_components = [
        float(item.get("confidence", 0.0))
        for item in constraints.get("constraint_catalog", [])
    ]
    constraint_conf = (
        sum(constraint_conf_components) / float(len(constraint_conf_components))
        if constraint_conf_components
        else 0.0
    )

    high_impact_uncertainties = dedupe_keep_order(
        [
            item["reason"]
            for key in deferred_items.keys()
            for item in deferred_items[key]
        ]
    )

    summary = {
        "target_binding_confidence": stable_round(target_conf, 4),
        "environment_binding_confidence": stable_round(env_conf, 4),
        "constraint_set_confidence": stable_round(constraint_conf, 4),
        "high_impact_uncertainties": high_impact_uncertainties,
    }

    breakdown = {
        "target_binding": {
            "final": stable_round(target_conf, 4),
            "source": "target_binding.confidence",
        },
        "environment_binding": {
            "components": [stable_round(item, 4) for item in env_conf_components],
            "mean": stable_round(env_conf, 4),
        },
        "constraint_set": {
            "components": [stable_round(item, 4) for item in constraint_conf_components],
            "mean": stable_round(constraint_conf, 4),
        },
    }
    return summary, breakdown


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


def _ensure_unique_run_dir(output_root: Path, stem: str) -> Path:
    candidate = output_root / stem
    if not candidate.exists():
        return ensure_dir(candidate)
    index = 1
    while True:
        fallback = output_root / f"{stem}_{index:02d}"
        if not fallback.exists():
            return ensure_dir(fallback)
        index += 1
