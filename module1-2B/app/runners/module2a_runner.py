"""Module 2-A runner for subgoal decomposition and requirement extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.bridges.module1_to_module2a import build_module2_bridge_package
from app.models.module1_normalizer import normalize_module1_raw
from app.module2a.llm_reasoner import generate_module2a_output_with_llm
from app.providers.base import Module1Provider
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
from app.validators.module1_validator import Module1Validator
from app.validators.schema_validator import validate_with_schema


def run_module2a_pipeline(
    module2_input_path: Path | None = None,
    provider: Module1Provider | None = None,
    image_path: Path | None = None,
    case_id: str | None = None,
    module1_output_path: Path | None = None,
    user_goal: str | None = None,
    success_criteria: list[str] | None = None,
    task_notes: list[str] | None = None,
    reasoner_mode: str = "llm",
    output_root: Path | None = None,
    task_name: str | None = None,
) -> dict[str, Any]:
    """Generate Module 2-A output from module2_common_input or Module 1 bridge.

    Output은 outputs/<task_name>/module2a_<timestamp>_<suffix>/ 구조로 저장된다.
    """
    root = project_root()
    output_root = output_root or (root / "outputs")
    run_id = timestamp_id()

    suffix = (
        case_id
        or (image_path.stem if image_path else None)
        or (module2_input_path.stem if module2_input_path else None)
        or "ad_hoc"
    )
    resolved_task_name = derive_task_name(
        task_name=task_name,
        bundle_path=module2_input_path or module1_output_path,
        image_path=image_path,
        case_id=case_id,
    )
    task_root = build_task_output_root(output_root, resolved_task_name)
    run_dir = ensure_unique_run_dir(task_root, f"module2a_{run_id}_{suffix}")

    vocab_registry = load_json(root / "configs" / "vocab_registry.json")
    prompt_registry = load_yaml(root / "configs" / "prompt_registry.yaml")

    provider_metadata: dict[str, Any] = {}
    bridge_artifacts: dict[str, Any] | None = None
    raw_module1_output: dict[str, Any] | None = None
    normalized_module1: dict[str, Any] | None = None
    validation_warnings: list[str] = []

    if module2_input_path is not None:
        module2_common_input = load_json(module2_input_path)
        provider_metadata = {
            "input_mode": "module2_common_input_file",
            "module2_input_path": str(module2_input_path),
        }
    else:
        if provider is None:
            raise ValueError("provider is required when --module2-input is not supplied.")
        atom_cfg = load_yaml(root / "configs" / "module1_to_module2a_atom_rules.yaml")
        provider_result = provider.get_module1_output(
            image_path=image_path,
            case_id=case_id,
            module1_output_path=module1_output_path,
        )
        raw_module1_output = provider_result.raw_output
        provider_metadata = dict(provider_result.metadata)
        validation = Module1Validator().validate(raw_module1_output)
        if not validation.valid:
            raise ValueError(
                "Module 1 validation failed before Module 2-A generation: "
                + " | ".join(validation.errors[:3])
                + (" ..." if len(validation.errors) > 3 else "")
            )
        normalized = normalize_module1_raw(raw_module1_output)
        normalized_module1 = normalized.to_dict()
        validation_warnings = validation.warnings + normalized.warnings
        bridge_artifacts = build_module2_bridge_package(
            normalized=normalized,
            rule_cfg=atom_cfg,
            vocab_registry=vocab_registry,
        )
        module2_common_input = bridge_artifacts["module2_common_input_template"]

    _inject_task_overrides(
        module2_common_input=module2_common_input,
        user_goal=user_goal,
        success_criteria=success_criteria,
        task_notes=task_notes,
    )
    module2_output, module2a_reasoner_metadata = _generate_module2a_output(
        module2_common_input=module2_common_input,
        vocab_registry=vocab_registry,
        reasoner_mode=reasoner_mode,
    )
    provider_metadata["module2a_reasoner"] = module2a_reasoner_metadata

    module2_output_errors = validate_with_schema(
        payload=module2_output,
        schema_path=root / "schemas" / "module2a_output.schema.json",
    )

    manifest = {
        "schema_name": "module2a_run_manifest",
        "schema_version": "0.1",
        "run_id": run_id,
        "provider_metadata": provider_metadata,
        "prompt_variant": prompt_registry["defaults"]["active_module2a_variant"],
        "versions": {
            "module2a_output_schema": (
                f"{module2_output.get('schema_name', 'module2a_output')}"
                f"@{module2_output.get('schema_version', 'unknown')}"
            ),
            "module2a_prompt_variant": prompt_registry["defaults"]["active_module2a_variant"],
            "vocab_registry_version": vocab_registry["registry_version"],
        },
        "validation": {
            "module1_warning_count": len(validation_warnings),
            "module2a_schema_error_count": len(module2_output_errors),
            "module2a_schema_errors": module2_output_errors,
        },
        "artifacts": {
            "module2_common_input": "module2_common_input.json",
            "module2a_output": "module2a_output.json",
            "run_manifest": "run_manifest.json",
        },
    }

    if raw_module1_output is not None:
        manifest["artifacts"]["raw_module1_output"] = "raw_module1_output.json"
        dump_json(raw_module1_output, run_dir / "raw_module1_output.json")
    if normalized_module1 is not None:
        manifest["artifacts"]["normalized_module1_output"] = "normalized_module1_output.json"
        dump_json(normalized_module1, run_dir / "normalized_module1_output.json")
    if bridge_artifacts is not None:
        manifest["artifacts"]["scene_resources_from_module1"] = "scene_resources_from_module1.json"
        manifest["artifacts"]["module2_bridge_diagnostics"] = "module2_bridge_diagnostics.json"
        dump_json(
            bridge_artifacts["scene_resources_from_module1"],
            run_dir / "scene_resources_from_module1.json",
        )
        dump_json(
            bridge_artifacts["module2_bridge_diagnostics"],
            run_dir / "module2_bridge_diagnostics.json",
        )

    dump_json(module2_common_input, run_dir / "module2_common_input.json")
    dump_json(module2_output, run_dir / "module2a_output.json")
    dump_json(manifest, run_dir / "run_manifest.json")

    return {
        "run_dir": str(run_dir),
        "summary": {
            "subgoal_count": len(module2_output.get("subgoals", [])),
            "overall_resource_sufficiency": module2_output.get(
                "task_level_requirement_summary", {}
            ).get("overall_resource_sufficiency"),
            "module2a_schema_error_count": len(module2_output_errors),
        },
        "manifest": manifest,
    }


def _generate_module2a_output(
    module2_common_input: dict[str, Any],
    vocab_registry: dict[str, Any],
    reasoner_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if reasoner_mode != "llm":
        raise ValueError(
            "Unsupported Module 2-A reasoner mode: "
            f"{reasoner_mode}. This build supports only: llm."
        )
    return generate_module2a_output_with_llm(
        module2_common_input=module2_common_input,
        vocab_registry=vocab_registry,
    )


def _inject_task_overrides(
    module2_common_input: dict[str, Any],
    user_goal: str | None,
    success_criteria: list[str] | None,
    task_notes: list[str] | None,
) -> None:
    task_brief = module2_common_input.setdefault("task_brief", {})
    task_brief.setdefault("user_goal", None)
    task_brief.setdefault("success_criteria", [])
    task_brief.setdefault("task_notes", [])

    if user_goal is not None:
        task_brief["user_goal"] = user_goal
    if success_criteria is not None:
        task_brief["success_criteria"] = [item for item in success_criteria if item]
    if task_notes is not None:
        task_brief["task_notes"] = [item for item in task_notes if item]


