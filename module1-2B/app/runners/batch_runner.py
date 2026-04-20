"""Batch runner with prompt variant comparison scaffolding."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.providers.mock_provider import MockProvider
from app.runners.module1_runner import run_module1_pipeline
from app.utils import dump_json, ensure_dir, load_json, load_yaml, project_root, timestamp_id, write_csv


def run_batch(
    cases: list[str] | str,
    provider_name: str = "mock",
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Run a batch of cases and export comparison summary artifacts."""
    root = project_root()
    output_root = output_root or (root / "outputs")
    batch_id = timestamp_id()
    batch_dir = ensure_dir(output_root / f"batch_{batch_id}")

    case_ids = _resolve_case_ids(cases=cases)
    prompt_registry = load_yaml(root / "configs" / "prompt_registry.yaml")
    module1_variants = list(prompt_registry["module1"]["variants"].keys())

    if provider_name != "mock":
        raise ValueError("Batch mode currently supports provider=mock only.")
    provider = MockProvider()

    rows: list[dict[str, Any]] = []
    case_results: list[dict[str, Any]] = []

    for case_id in case_ids:
        for variant in module1_variants:
            result = run_module1_pipeline(
                provider=provider,
                case_id=case_id,
                image_path=None,
                module1_output_path=None,
                scenarios=["all"],
                output_root=batch_dir,
                prompt_variant=variant,
            )
            summary = result["summary"]
            row = {
                "case_id": case_id,
                "prompt_variant": variant,
                "run_dir": result["run_dir"],
                "object_count": summary["object_count"],
                "scenario_count": summary["scenario_count"],
                "capability_unit_count": summary["capability_unit_count"],
                "affordance_atom_count": summary["affordance_atom_count"],
                "validation_warning_count": summary["validation_warning_count"],
            }
            row.update(summary.get("metric_rollup", {}))
            rows.append(row)
            case_results.append(
                {
                    "case_id": case_id,
                    "prompt_variant": variant,
                    "run_dir": result["run_dir"],
                    "summary": summary,
                }
            )

    write_csv(rows, batch_dir / "batch_summary.csv")
    batch_summary = {
        "schema_name": "batch_summary",
        "schema_version": "0.1",
        "batch_id": batch_id,
        "provider": provider_name,
        "case_count": len(case_ids),
        "variant_count": len(module1_variants),
        "rows": case_results,
    }
    dump_json(batch_summary, batch_dir / "batch_summary.json")
    return {"batch_dir": str(batch_dir), "summary": batch_summary}


def _resolve_case_ids(cases: list[str] | str) -> list[str]:
    if isinstance(cases, list):
        return cases
    if cases != "all":
        return [cases]
    root = project_root()
    index = load_json(root / "fixtures" / "cases" / "index.json")
    return list(index["cases"])
