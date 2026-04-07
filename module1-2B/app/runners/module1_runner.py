"""End-to-end Module 1 experiment runner and artifact exporter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.bridges.module1_to_module2a import build_module2_bridge_package
from app.bridges.module1_to_pybullet import map_module1_to_pybullet
from app.models.module1_normalizer import normalize_module1_raw
from app.providers.base import Module1Provider
from app.pybullet.proxy_generator import generate_proxy_specs
from app.pybullet.runner import run_pybullet_experiments
from app.utils import (
    dump_json,
    ensure_dir,
    load_json,
    load_yaml,
    project_root,
    timestamp_id,
    write_csv,
)
from app.validators.module1_validator import Module1Validator
from app.validators.schema_validator import validate_with_schema


def run_module1_pipeline(
    provider: Module1Provider,
    image_path: Path | None = None,
    case_id: str | None = None,
    module1_output_path: Path | None = None,
    scenarios: list[str] | None = None,
    output_root: Path | None = None,
    prompt_variant: str | None = None,
) -> dict[str, Any]:
    """Run full Module 1-only pipeline and export all artifacts."""
    root = project_root()
    output_root = output_root or (root / "outputs")
    run_id = timestamp_id()
    suffix = case_id or (image_path.stem if image_path else "ad_hoc")
    run_dir = _ensure_unique_run_dir(output_root=output_root, stem=f"run_{run_id}_{suffix}")

    mapping_cfg = load_yaml(root / "configs" / "module1_to_pybullet_map.yaml")
    atom_cfg = load_yaml(root / "configs" / "module1_to_module2a_atom_rules.yaml")
    prompt_registry = load_yaml(root / "configs" / "prompt_registry.yaml")
    vocab_registry = load_json(root / "configs" / "vocab_registry.json")

    provider_result = provider.get_module1_output(
        image_path=image_path,
        case_id=case_id,
        module1_output_path=module1_output_path,
    )
    raw_output = provider_result.raw_output

    validator = Module1Validator()
    validation = validator.validate(raw_output)
    if not validation.valid:
        raise ValueError(
            "Module 1 validation failed. "
            + " | ".join(validation.errors[:3])
            + (" ..." if len(validation.errors) > 3 else "")
        )

    normalized = normalize_module1_raw(raw_output)
    surrogate_spec = map_module1_to_pybullet(normalized=normalized, mapping_cfg=mapping_cfg)
    proxy_spec = generate_proxy_specs(normalized=normalized, mapping_cfg=mapping_cfg)

    if normalized.objects:
        pybullet_result = run_pybullet_experiments(
            proxy_spec=proxy_spec,
            surrogate_spec=surrogate_spec,
            mapping_cfg=mapping_cfg,
            scenarios=scenarios,
        )
    else:
        pybullet_result = {
            "seed": mapping_cfg["scenario_defaults"]["seed"],
            "fixed_timestep_s": mapping_cfg["scenario_defaults"]["fixed_timestep_s"],
            "scenarios": scenarios or ["drop_test", "slide_test", "force_response_test"],
            "metrics": {},
            "trajectory_rows": [],
            "applied_dynamics": {
                "schema_name": "pybullet_applied_dynamics",
                "schema_version": "0.1",
                "rows": [],
            },
        }

    bridge_artifacts = build_module2_bridge_package(
        normalized=normalized,
        rule_cfg=atom_cfg,
        vocab_registry=vocab_registry,
    )

    module2_template_errors = validate_with_schema(
        payload=bridge_artifacts["module2_common_input_template"],
        schema_path=root / "schemas" / "module2_common_input_template_derived_min.schema.json",
    )
    bridge_diag_errors = validate_with_schema(
        payload=bridge_artifacts["module2_bridge_diagnostics"],
        schema_path=root / "schemas" / "module2_bridge_diagnostics.schema.json",
    )

    metrics = {
        "schema_name": "module1_experiment_metrics",
        "schema_version": "0.1",
        "per_object": pybullet_result["metrics"],
    }
    summary = {
        "run_id": run_id,
        "provider": provider_result.metadata.get("provider"),
        "case_id": provider_result.metadata.get("case_id"),
        "prompt_variant": prompt_variant or prompt_registry["defaults"]["active_module1_variant"],
        "object_count": len(normalized.objects),
        "scenario_count": len(pybullet_result["scenarios"]),
        "validation_warning_count": len(validation.warnings),
        "module2_template_schema_error_count": len(module2_template_errors),
        "bridge_diagnostics_schema_error_count": len(bridge_diag_errors),
        "affordance_atom_count": len(
            bridge_artifacts["scene_resources_from_module1"]["resource_summary"][
                "affordance_histogram"
            ]
        ),
        "capability_unit_count": len(
            bridge_artifacts["scene_resources_from_module1"]["resource_inventory"]
        ),
        "metric_rollup": _summarize_metrics(pybullet_result["metrics"]),
    }

    manifest = {
        "schema_name": "module1_run_manifest",
        "schema_version": "0.1",
        "run_id": run_id,
        "provider_metadata": provider_result.metadata,
        "prompt_variant": prompt_variant or prompt_registry["defaults"]["active_module1_variant"],
        "module2a_variant": prompt_registry["defaults"]["active_module2a_variant"],
        "versions": {
            "module1_raw_schema": f"{raw_output.get('schema_name')}@{raw_output.get('schema_version')}",
            "module1_normalized_schema": "module1_normalized_internal@0.1",
            "pybullet_mapping_version": mapping_cfg["map_version"],
            "module2_bridge_rule_version": atom_cfg["bridge_rule_version"],
            "vocab_registry_version": vocab_registry["registry_version"],
        },
        "validation": {
            "valid": validation.valid,
            "errors": validation.errors,
            "warnings": validation.warnings + normalized.warnings,
        },
        "schema_checks": {
            "module2_common_input_template_errors": module2_template_errors,
            "module2_bridge_diagnostics_errors": bridge_diag_errors,
        },
        "artifacts": {
            "raw_module1_output": "raw_module1_output.json",
            "normalized_module1_output": "normalized_module1_output.json",
            "pybullet_surrogate_params": "pybullet_surrogate_params.json",
            "pybullet_proxy_spec": "pybullet_proxy_spec.json",
            "applied_dynamics": "applied_dynamics.json",
            "scene_resources_from_module1": "scene_resources_from_module1.json",
            "module2_common_input_template": "module2_common_input_template.json",
            "module2_bridge_diagnostics": "module2_bridge_diagnostics.json",
            "trajectory": "trajectory.csv",
            "metrics": "metrics.json",
            "summary": "summary.json",
        },
    }

    dump_json(manifest, run_dir / "run_manifest.json")
    dump_json(raw_output, run_dir / "raw_module1_output.json")
    dump_json(normalized.to_dict(), run_dir / "normalized_module1_output.json")
    dump_json(surrogate_spec, run_dir / "pybullet_surrogate_params.json")
    dump_json(proxy_spec, run_dir / "pybullet_proxy_spec.json")
    dump_json(pybullet_result["applied_dynamics"], run_dir / "applied_dynamics.json")
    dump_json(
        bridge_artifacts["scene_resources_from_module1"],
        run_dir / "scene_resources_from_module1.json",
    )
    dump_json(
        bridge_artifacts["module2_common_input_template"],
        run_dir / "module2_common_input_template.json",
    )
    dump_json(
        bridge_artifacts["module2_bridge_diagnostics"],
        run_dir / "module2_bridge_diagnostics.json",
    )
    write_csv(pybullet_result["trajectory_rows"], run_dir / "trajectory.csv")
    dump_json(metrics, run_dir / "metrics.json")
    dump_json(summary, run_dir / "summary.json")

    return {
        "run_dir": str(run_dir),
        "summary": summary,
        "manifest": manifest,
    }


def export_module2_bridge_only(
    provider: Module1Provider,
    image_path: Path | None = None,
    case_id: str | None = None,
    module1_output_path: Path | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Export only bridge-related artifacts without simulation runs."""
    root = project_root()
    output_root = output_root or (root / "outputs")
    run_id = timestamp_id()
    suffix = case_id or (image_path.stem if image_path else "ad_hoc")
    run_dir = _ensure_unique_run_dir(output_root=output_root, stem=f"bridge_{run_id}_{suffix}")

    atom_cfg = load_yaml(root / "configs" / "module1_to_module2a_atom_rules.yaml")
    vocab_registry = load_json(root / "configs" / "vocab_registry.json")
    provider_result = provider.get_module1_output(
        image_path=image_path,
        case_id=case_id,
        module1_output_path=module1_output_path,
    )
    raw_output = provider_result.raw_output
    validation = Module1Validator().validate(raw_output)
    if not validation.valid:
        raise ValueError("Module 1 validation failed for bridge export.")
    normalized = normalize_module1_raw(raw_output)
    bridge_artifacts = build_module2_bridge_package(
        normalized=normalized,
        rule_cfg=atom_cfg,
        vocab_registry=vocab_registry,
    )

    dump_json(
        bridge_artifacts["scene_resources_from_module1"],
        run_dir / "scene_resources_from_module1.json",
    )
    dump_json(
        bridge_artifacts["module2_common_input_template"],
        run_dir / "module2_common_input_template.json",
    )
    dump_json(
        bridge_artifacts["module2_bridge_diagnostics"],
        run_dir / "module2_bridge_diagnostics.json",
    )
    dump_json(
        {
            "schema_name": "module2_bridge_export_manifest",
            "schema_version": "0.1",
            "run_id": run_id,
            "provider_metadata": provider_result.metadata,
            "warnings": validation.warnings + normalized.warnings,
        },
        run_dir / "run_manifest.json",
    )
    return {"run_dir": str(run_dir)}


def _summarize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    if not metrics:
        return {}
    scenario_rollups: dict[str, list[float]] = {}
    for object_metrics in metrics.values():
        for scenario_name, scenario_metrics in object_metrics.items():
            for key, value in scenario_metrics.items():
                scenario_rollups.setdefault(f"{scenario_name}.{key}", []).append(float(value))
    return {
        key: round(sum(values) / len(values), 6) if values else 0.0
        for key, values in sorted(scenario_rollups.items())
    }


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
