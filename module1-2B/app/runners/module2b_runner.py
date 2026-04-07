"""Runners for Module 2-B single run, batch run, and comparison artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.module2b.providers import (
    FileBundleProvider,
    MockBundleProvider,
    list_module2b_case_ids,
)
from app.module2b.validators import Module2BInputValidator
from app.pipelines.module2b_pipeline import compare_module2b_outputs, run_module2b_pipeline
from app.utils import dump_json, ensure_dir, project_root, timestamp_id, write_csv


def validate_module2b_input(
    bundle_path: Path | None = None,
    module2_common_path: Path | None = None,
    module2a_output_path: Path | None = None,
) -> dict[str, Any]:
    """Validate Module 2-B input payload from bundle or split files."""
    provider = FileBundleProvider()
    result = provider.get_bundle(
        bundle_path=bundle_path,
        module2_common_path=module2_common_path,
        module2a_output_path=module2a_output_path,
        case_id=None,
    )
    validator = Module2BInputValidator(root=project_root())
    report = validator.validate(result.bundle)
    return report.to_dict()


def run_module2b_batch(
    cases: list[str] | str,
    provider_name: str = "mock",
    repeats: int = 1,
    output_root: Path | None = None,
    variant: str | None = None,
) -> dict[str, Any]:
    """Run batch Module 2-B experiments and export summary CSV/JSON."""
    root = project_root()
    output_root = output_root or (root / "outputs")
    batch_id = timestamp_id()
    batch_dir = ensure_dir(output_root / f"module2b_batch_{batch_id}")

    case_ids = _resolve_case_ids(cases)
    provider = _build_provider(provider_name)

    rows: list[dict[str, Any]] = []
    case_runs: dict[str, list[Path]] = {case_id: [] for case_id in case_ids}

    for case_id in case_ids:
        for repeat_idx in range(repeats):
            result = run_module2b_pipeline(
                provider=provider,
                case_id=case_id,
                output_root=batch_dir,
                variant=variant,
            )
            run_dir = Path(result["run_dir"])
            case_runs[case_id].append(run_dir)
            summary = result["summary"]
            rows.append(
                {
                    "case_id": case_id,
                    "repeat_idx": repeat_idx,
                    "run_dir": str(run_dir),
                    "target_mode": summary["target_mode"],
                    "binding_status": summary["binding_status"],
                    "environment_structure_count": summary["environment_structure_count"],
                    "numeric_estimate_count": summary["numeric_estimate_count"],
                    "constraint_count": summary["constraint_count"],
                    "handoff_status": summary["handoff_status"],
                    "input_valid": summary["input_valid"],
                    "output_valid": summary["output_valid"],
                }
            )

    repeatability: list[dict[str, Any]] = []
    for case_id, runs in case_runs.items():
        if len(runs) <= 1:
            repeatability.append(
                {
                    "case_id": case_id,
                    "reference_run": str(runs[0]) if runs else None,
                    "comparison_count": 0,
                    "all_equal": True,
                    "mismatches": [],
                }
            )
            continue

        reference = runs[0]
        mismatches: list[dict[str, Any]] = []
        for candidate in runs[1:]:
            comparison = compare_module2b_outputs(reference, candidate)
            if not (comparison["same_structure"] and comparison["same_values"]):
                mismatches.append(
                    {
                        "run": str(candidate),
                        "same_structure": comparison["same_structure"],
                        "same_values": comparison["same_values"],
                        "structural_diff_count": comparison["structural_diff_count"],
                        "value_diff_count": comparison["value_diff_count"],
                    }
                )
        repeatability.append(
            {
                "case_id": case_id,
                "reference_run": str(reference),
                "comparison_count": len(runs) - 1,
                "all_equal": len(mismatches) == 0,
                "mismatches": mismatches,
            }
        )

    write_csv(rows, batch_dir / "batch_summary.csv")
    summary_json = {
        "schema_name": "module2b_batch_summary",
        "schema_version": "0.1",
        "batch_id": batch_id,
        "provider": provider_name,
        "repeats": repeats,
        "case_ids": case_ids,
        "row_count": len(rows),
        "rows": rows,
        "repeatability": repeatability,
    }
    dump_json(summary_json, batch_dir / "batch_summary.json")

    return {
        "batch_dir": str(batch_dir),
        "summary": summary_json,
    }


def run_module2b_comparison(
    run_a: Path,
    run_b: Path,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Compare two Module 2-B runs and export diff artifacts."""
    root = project_root()
    output_root = output_root or (root / "outputs")
    compare_id = timestamp_id()
    compare_dir = ensure_dir(output_root / f"module2b_compare_{compare_id}")

    comparison = compare_module2b_outputs(run_a=run_a, run_b=run_b)
    structural_diff = comparison["structural_diff"]
    value_diff = comparison["value_diff"]

    summary = {
        "schema_name": "module2b_comparison_summary",
        "schema_version": "0.1",
        "run_a": str(run_a),
        "run_b": str(run_b),
        "same_structure": comparison["same_structure"],
        "same_values": comparison["same_values"],
        "structural_diff_count": len(structural_diff),
        "value_diff_count": len(value_diff),
    }

    dump_json(summary, compare_dir / "comparison_summary.json")
    dump_json(structural_diff, compare_dir / "structural_diff.json")
    dump_json(value_diff, compare_dir / "value_diff.json")

    return {
        "comparison_dir": str(compare_dir),
        "summary": summary,
    }


def _resolve_case_ids(cases: list[str] | str) -> list[str]:
    if isinstance(cases, list):
        return cases
    if cases != "all":
        return [cases]
    return list_module2b_case_ids(fixtures_root=project_root() / "fixtures")


def _build_provider(provider_name: str):
    if provider_name == "mock":
        return MockBundleProvider(fixtures_root=project_root() / "fixtures")
    if provider_name == "file":
        return FileBundleProvider()
    raise ValueError(f"Unsupported Module 2-B provider: {provider_name}")
