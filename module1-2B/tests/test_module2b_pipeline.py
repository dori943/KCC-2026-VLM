from __future__ import annotations

from pathlib import Path

import pytest

from app.module2b.providers import MockBundleProvider
from app.module2b.validators import Module2BInputValidator, Module2BOutputValidator
from app.pipelines.module2b_pipeline import compare_module2b_outputs, run_module2b_pipeline
from app.utils import ensure_dir, load_json, project_root
from app.validators.schema_validator import validate_with_schema


CASES = [
    "coin_in_narrow_gap_case",
    "mug_under_overhang_case",
    "bottle_in_deep_recess_case",
]


@pytest.fixture
def module2b_provider() -> MockBundleProvider:
    return MockBundleProvider(fixtures_root=project_root() / "fixtures")


@pytest.fixture
def module2b_output_root() -> Path:
    return ensure_dir(project_root() / "outputs" / "test_module2b_pipeline")


@pytest.mark.parametrize("case_id", CASES)
def test_module2b_input_derived_min_schema_validation(case_id: str):
    case_dir = project_root() / "fixtures" / "module2b_cases" / case_id
    module2_common_input = load_json(case_dir / "module2_common_input.json")
    errors = validate_with_schema(
        payload=module2_common_input,
        schema_path=project_root()
        / "schemas"
        / "module2_common_input_for_module2b_derived_min.schema.json",
    )
    assert errors == []


@pytest.mark.parametrize("case_id", CASES)
def test_module2a_output_strict_validation_for_module2b_cases(case_id: str):
    case_dir = project_root() / "fixtures" / "module2b_cases" / case_id
    module2a_output = load_json(case_dir / "module2a_output.json")
    errors = validate_with_schema(
        payload=module2a_output,
        schema_path=project_root() / "schemas" / "module2a_output.schema.json",
    )
    assert errors == []


@pytest.mark.parametrize("case_id", CASES)
def test_module2b_input_bundle_validation(module2b_provider: MockBundleProvider, case_id: str):
    bundle = module2b_provider.get_bundle(case_id=case_id).bundle
    report = Module2BInputValidator(root=project_root()).validate(bundle)
    assert report.valid, report.errors


@pytest.mark.parametrize("case_id", CASES)
def test_module2b_output_schema_and_integrity(
    module2b_provider: MockBundleProvider, case_id: str, module2b_output_root: Path
):
    result = run_module2b_pipeline(
        provider=module2b_provider,
        case_id=case_id,
        output_root=module2b_output_root,
    )
    run_dir = Path(result["run_dir"])
    output_payload = load_json(run_dir / "module2b_output.json")
    normalized_context = load_json(run_dir / "normalized_context.json")

    schema_errors = validate_with_schema(
        payload=output_payload,
        schema_path=project_root() / "schemas" / "module2b_output_env_only.schema.json",
    )
    assert schema_errors == []

    output_validator = Module2BOutputValidator(root=project_root())
    report = output_validator.validate(
        payload=output_payload,
        inventory_ids=normalized_context["object_id_order"],
        subgoal_ids=[item["subgoal_id"] for item in normalized_context["subgoals"]],
    )
    assert report.valid, report.errors


def test_primary_target_object_id_exists_and_target_not_tool(
    module2b_provider: MockBundleProvider, module2b_output_root: Path
):
    result = run_module2b_pipeline(
        provider=module2b_provider,
        case_id="coin_in_narrow_gap_case",
        output_root=module2b_output_root,
    )
    output_payload = load_json(Path(result["run_dir"]) / "module2b_output.json")
    primary_targets = output_payload["target_binding"]["primary_targets"]
    assert primary_targets
    target_id = primary_targets[0]["object_id"]
    assert target_id == "obj_01"


def test_subgoal_bindings_preserve_module2a_order(
    module2b_provider: MockBundleProvider, module2b_output_root: Path
):
    result = run_module2b_pipeline(
        provider=module2b_provider,
        case_id="mug_under_overhang_case",
        output_root=module2b_output_root,
    )
    run_dir = Path(result["run_dir"])
    output_payload = load_json(run_dir / "module2b_output.json")
    bundle = load_json(run_dir / "raw_input_bundle.json")

    expected_order = [item["subgoal_id"] for item in bundle["module2a_output"]["subgoals"]]
    actual_order = [item["subgoal_id"] for item in output_payload["derived_constraints"]["subgoal_bindings"]]
    assert actual_order == expected_order


def test_source_refs_and_numeric_bounds_and_units(
    module2b_provider: MockBundleProvider, module2b_output_root: Path
):
    result = run_module2b_pipeline(
        provider=module2b_provider,
        case_id="bottle_in_deep_recess_case",
        output_root=module2b_output_root,
    )
    output_payload = load_json(Path(result["run_dir"]) / "module2b_output.json")

    known_units = {"m", "deg", "level_1_to_5"}
    for measure in output_payload["environment_context"]["numeric_estimates"]:
        assert measure["source_refs"]
        assert measure["unit"] in known_units
        lower = measure["lower_value"]
        upper = measure["upper_value"]
        if lower is not None and upper is not None:
            assert lower <= upper

    for constraint in output_payload["derived_constraints"]["constraint_catalog"]:
        assert constraint["source_refs"]


def test_target_binding_mode_status_expected(
    module2b_provider: MockBundleProvider, module2b_output_root: Path
):
    for case_id in CASES:
        expected = load_json(
            project_root() / "fixtures" / "module2b_cases" / case_id / "expected.json"
        )
        result = run_module2b_pipeline(
            provider=module2b_provider,
            case_id=case_id,
            output_root=module2b_output_root,
        )
        output_payload = load_json(Path(result["run_dir"]) / "module2b_output.json")
        assert output_payload["target_binding"]["target_mode"] == expected["expected_target_binding"]["target_mode"]
        assert (
            output_payload["target_binding"]["binding_status"]
            == expected["expected_target_binding"]["binding_status"]
        )


def test_environment_structure_id_determinism(
    module2b_provider: MockBundleProvider, module2b_output_root: Path
):
    result = run_module2b_pipeline(
        provider=module2b_provider,
        case_id="coin_in_narrow_gap_case",
        output_root=module2b_output_root,
    )
    output_payload = load_json(Path(result["run_dir"]) / "module2b_output.json")
    env_ids = [item["environment_structure_id"] for item in output_payload["environment_context"]["relevant_structures"]]
    assert env_ids == sorted(env_ids)
    assert env_ids[0] == "env_01"


def test_numeric_estimate_provenance_trace(
    module2b_provider: MockBundleProvider, module2b_output_root: Path
):
    result = run_module2b_pipeline(
        provider=module2b_provider,
        case_id="mug_under_overhang_case",
        output_root=module2b_output_root,
    )
    run_dir = Path(result["run_dir"])
    trace = load_json(run_dir / "numeric_estimates_trace.json")
    assert "deduped_candidates" in trace
    assert "omissions" in trace


def test_derived_constraint_family_and_handoff(
    module2b_provider: MockBundleProvider, module2b_output_root: Path
):
    case_id = "bottle_in_deep_recess_case"
    expected = load_json(project_root() / "fixtures" / "module2b_cases" / case_id / "expected.json")
    result = run_module2b_pipeline(
        provider=module2b_provider,
        case_id=case_id,
        output_root=module2b_output_root,
    )
    output_payload = load_json(Path(result["run_dir"]) / "module2b_output.json")
    parameter_names = {
        item["parameter_name"] for item in output_payload["derived_constraints"]["constraint_catalog"]
    }
    for parameter_name in expected["expected_derived_constraint_family"]:
        assert parameter_name in parameter_names

    handoff = output_payload["module3_handoff"]
    assert handoff["handoff_status"] == expected["expected_handoff_status"]
    assert "material_reasoner" in handoff["pending_merge_sources"]


def test_deterministic_repeat_run(module2b_provider: MockBundleProvider, module2b_output_root: Path):
    result_a = run_module2b_pipeline(
        provider=module2b_provider,
        case_id="coin_in_narrow_gap_case",
        output_root=module2b_output_root,
    )
    result_b = run_module2b_pipeline(
        provider=module2b_provider,
        case_id="coin_in_narrow_gap_case",
        output_root=module2b_output_root,
    )
    comparison = compare_module2b_outputs(
        run_a=Path(result_a["run_dir"]),
        run_b=Path(result_b["run_dir"]),
    )
    assert comparison["same_structure"]
    assert comparison["same_values"]


def test_module2b_end_to_end_smoke(
    module2b_provider: MockBundleProvider, module2b_output_root: Path
):
    result = run_module2b_pipeline(
        provider=module2b_provider,
        case_id="mug_under_overhang_case",
        output_root=module2b_output_root,
    )
    run_dir = Path(result["run_dir"])
    required_files = [
        "run_manifest.json",
        "raw_input_bundle.json",
        "normalized_context.json",
        "validation_report.json",
        "target_binding_candidates.json",
        "environment_structure_candidates.json",
        "numeric_estimates_trace.json",
        "derived_constraints_trace.json",
        "module2b_output.json",
        "module3_handoff_preview.json",
        "summary.json",
    ]
    for file_name in required_files:
        assert (run_dir / file_name).exists(), file_name
