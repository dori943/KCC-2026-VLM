"""Generic JSON Schema validator helper with no-dependency fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.utils import load_json

try:  # pragma: no cover - optional dependency
    from jsonschema import Draft202012Validator  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Draft202012Validator = None


def validate_with_schema(payload: dict[str, Any], schema_path: Path) -> list[str]:
    """Return list of validation errors (empty on success)."""
    if Draft202012Validator is not None:
        schema = load_json(schema_path)
        validator = Draft202012Validator(schema)
        errors = sorted(validator.iter_errors(payload), key=lambda e: e.path)
        formatted: list[str] = []
        for err in errors:
            path = "$"
            if err.path:
                path += "." + ".".join(str(item) for item in err.path)
            formatted.append(f"{path}: {err.message}")
        return formatted

    # no jsonschema dependency installed; run strict manual checks for known schemas used here
    name = schema_path.name
    if name == "module2_common_input_template_derived_min.schema.json":
        return _validate_module2_template(payload)
    if name == "module2_common_input_for_module2b_derived_min.schema.json":
        return _validate_module2b_common_input(payload)
    if name == "module2b_input_bundle.schema.json":
        return _validate_module2b_input_bundle(payload)
    if name == "module2b_output_env_only.schema.json":
        return _validate_module2b_output(payload)
    if name == "module2b_diagnostics.schema.json":
        return _validate_module2b_diagnostics(payload)
    if name == "module2_bridge_diagnostics.schema.json":
        return _validate_bridge_diagnostics(payload)
    if name == "module2a_output.schema.json":
        return _validate_module2a_output(payload)
    return []


def _validate_module2_template(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = {"schema_name", "schema_version", "task_brief", "scene_resources"}
    _exact_keys(payload, expected, "$", errors)
    if payload.get("schema_name") != "module2_common_input_template_derived_min":
        errors.append("$.schema_name mismatch")
    if payload.get("schema_version") != "0.1":
        errors.append("$.schema_version mismatch")
    task = payload.get("task_brief")
    if isinstance(task, dict):
        _exact_keys(task, {"user_goal", "success_criteria", "task_notes"}, "$.task_brief", errors)
    else:
        errors.append("$.task_brief must be object")
    scene = payload.get("scene_resources")
    if isinstance(scene, dict):
        _exact_keys(scene, {"resource_summary", "resource_inventory"}, "$.scene_resources", errors)
        summary = scene.get("resource_summary")
        if isinstance(summary, dict):
            _exact_keys(
                summary,
                {"affordance_histogram", "risk_histogram", "primitive_histogram"},
                "$.scene_resources.resource_summary",
                errors,
            )
        else:
            errors.append("$.scene_resources.resource_summary must be object")
    else:
        errors.append("$.scene_resources must be object")
    return errors


def _validate_module2b_common_input(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = {"schema_name", "schema_version", "task_id", "task_brief", "scene_resources"}
    _exact_keys(payload, expected, "$", errors)
    if payload.get("schema_name") != "module2_common_input_for_module2b_derived_min":
        errors.append("$.schema_name mismatch")
    if payload.get("schema_version") != "0.1":
        errors.append("$.schema_version mismatch")
    if not isinstance(payload.get("task_id"), str):
        errors.append("$.task_id must be string")
    task = payload.get("task_brief")
    if isinstance(task, dict):
        _exact_keys(task, {"user_goal", "success_criteria", "task_notes"}, "$.task_brief", errors)
    else:
        errors.append("$.task_brief must be object")
    scene = payload.get("scene_resources")
    if isinstance(scene, dict):
        _exact_keys(scene, {"resource_summary", "resource_inventory"}, "$.scene_resources", errors)
    else:
        errors.append("$.scene_resources must be object")
    return errors


def _validate_module2b_input_bundle(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _exact_keys(payload, {"module2_common_input", "module2a_output"}, "$", errors)
    if not isinstance(payload.get("module2_common_input"), dict):
        errors.append("$.module2_common_input must be object")
    if not isinstance(payload.get("module2a_output"), dict):
        errors.append("$.module2a_output must be object")
    return errors


def _validate_module2b_output(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = {
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
    }
    _exact_keys(payload, expected, "$", errors)
    if payload.get("schema_name") != "module2b_output_env_only":
        errors.append("$.schema_name mismatch")
    if payload.get("schema_version") != "0.1":
        errors.append("$.schema_version mismatch")
    return errors


def _validate_module2b_diagnostics(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = {
        "schema_name",
        "schema_version",
        "run_id",
        "rule_versions",
        "validation_report",
        "trace_refs",
        "deferred_item_reasons",
        "confidence_component_breakdown",
        "dedup_rules",
    }
    _exact_keys(payload, expected, "$", errors)
    if payload.get("schema_name") != "module2b_diagnostics":
        errors.append("$.schema_name mismatch")
    if payload.get("schema_version") != "0.1":
        errors.append("$.schema_version mismatch")
    return errors


def _validate_bridge_diagnostics(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = {
        "schema_name",
        "schema_version",
        "bridge_rule_version",
        "counting_rule",
        "assumptions",
        "warnings",
        "object_count",
        "capability_unit_count",
        "atom_histogram",
        "risk_histogram",
        "primitive_histogram",
        "uncertainty_summary",
        "atom_activation",
        "risk_activation",
        "skipped_objects",
        "mapping_provenance",
        "vocab_registry_refs",
    }
    _exact_keys(payload, expected, "$", errors)
    if payload.get("schema_name") != "module2_bridge_diagnostics":
        errors.append("$.schema_name mismatch")
    if payload.get("schema_version") != "0.1":
        errors.append("$.schema_version mismatch")
    return errors


def _validate_module2a_output(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = {
        "schema_name",
        "schema_version",
        "stage",
        "task_model",
        "subgoals",
        "task_level_requirement_summary",
        "scene_resource_readout",
        "pybullet_bridge",
    }
    _exact_keys(payload, expected, "$", errors)
    if payload.get("schema_name") != "module2a_output":
        errors.append("$.schema_name mismatch")
    if payload.get("schema_version") != "0.2":
        errors.append("$.schema_version mismatch")
    if payload.get("stage") != "task_decomposition_and_function_requirement_extraction":
        errors.append("$.stage mismatch")

    task_model = payload.get("task_model")
    if isinstance(task_model, dict):
        _exact_keys(
            task_model,
            {
                "task_restatement",
                "primary_success_condition",
                "secondary_success_conditions",
                "decomposition_principle",
                "assumptions",
                "deferred_items",
            },
            "$.task_model",
            errors,
        )
    else:
        errors.append("$.task_model must be object")

    subgoals = payload.get("subgoals")
    if isinstance(subgoals, list):
        if not subgoals:
            errors.append("$.subgoals must not be empty")
        for idx, subgoal in enumerate(subgoals):
            path = f"$.subgoals[{idx}]"
            if not isinstance(subgoal, dict):
                errors.append(f"{path} must be object")
                continue
            _exact_keys(
                subgoal,
                {
                    "subgoal_id",
                    "subgoal_name",
                    "objective",
                    "success_condition",
                    "depends_on",
                    "required_interaction_primitives",
                    "function_requirements",
                    "physical_rationale",
                    "resource_feasibility_hint",
                    "failure_risks",
                    "pybullet_bridge",
                },
                path,
                errors,
            )
            feasibility = subgoal.get("resource_feasibility_hint")
            if isinstance(feasibility, dict):
                status = feasibility.get("coverage_status")
                if status not in {"blocked", "very_weak", "weak", "usable", "strong", "robust"}:
                    errors.append(f"{path}.resource_feasibility_hint.coverage_status invalid")
    else:
        errors.append("$.subgoals must be array")

    summary = payload.get("task_level_requirement_summary")
    if isinstance(summary, dict):
        overall = summary.get("overall_resource_sufficiency")
        if overall not in {
            "insufficient",
            "very_limited",
            "partial",
            "usable",
            "strong",
            "sufficient",
        }:
            errors.append("$.task_level_requirement_summary.overall_resource_sufficiency invalid")
    else:
        errors.append("$.task_level_requirement_summary must be object")

    readout = payload.get("scene_resource_readout")
    if isinstance(readout, dict):
        atoms = readout.get("available_affordance_atoms")
        if isinstance(atoms, list):
            for idx, item in enumerate(atoms):
                path = f"$.scene_resource_readout.available_affordance_atoms[{idx}]"
                if not isinstance(item, dict):
                    errors.append(f"{path} must be object")
                    continue
                availability = item.get("availability")
                if availability not in {"none", "trace", "sparse", "present", "strong", "abundant"}:
                    errors.append(f"{path}.availability invalid")
        else:
            errors.append("$.scene_resource_readout.available_affordance_atoms must be array")
    else:
        errors.append("$.scene_resource_readout must be object")

    return errors


def _exact_keys(obj: Any, expected: set[str], path: str, errors: list[str]) -> None:
    if not isinstance(obj, dict):
        errors.append(f"{path} must be object")
        return
    keys = set(obj.keys())
    for key in sorted(expected - keys):
        errors.append(f"{path}.{key} missing")
    for key in sorted(keys - expected):
        errors.append(f"{path}.{key} unexpected")
