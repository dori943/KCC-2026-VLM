from __future__ import annotations

from pathlib import Path

from app.runners.module2b_reasoner_eval_runner import (
    evaluate_existing_module2b_runs,
    run_module2b_reasoner_evaluation,
)
from app.utils import ensure_dir, project_root


def test_module2b_reasoner_eval_runner_smoke():
    output_root = ensure_dir(project_root() / "outputs" / "test_module2b_reasoner_eval")
    result = run_module2b_reasoner_evaluation(
        cases=["coin_in_narrow_gap_case"],
        repeats=2,
        output_root=output_root,
    )

    report_path = Path(result["report_path"])
    assert report_path.exists()

    report = result["report"]
    assert report["schema_name"] == "module2b_reasoner_evaluation"
    assert report["repeats_per_case"] == 2
    assert set(report["metrics"].keys()) == {"TBDS", "NCR", "MHRR"}
    assert 0.0 <= report["metrics"]["TBDS"]["score"] <= 1.0
    assert 0.0 <= report["metrics"]["NCR"]["score"] <= 1.0
    assert 0.0 <= report["metrics"]["MHRR"]["score"] <= 1.0
    tbds_case = report["metrics"]["TBDS"]["by_case"]["coin_in_narrow_gap_case"]
    ncr_case = report["metrics"]["NCR"]["by_case"]["coin_in_narrow_gap_case"]
    mhrr_case = report["metrics"]["MHRR"]["by_case"]["coin_in_narrow_gap_case"]
    assert tbds_case["score"] > 0.0
    assert ncr_case["run_details"][0]["numeric_estimate_count"] > 0
    assert mhrr_case["run_success_rate"] == 1.0


def test_evaluate_existing_module2b_runs_smoke():
    output_root = ensure_dir(project_root() / "outputs" / "test_module2b_reasoner_eval_existing")
    generated = run_module2b_reasoner_evaluation(
        cases=["coin_in_narrow_gap_case"],
        repeats=1,
        output_root=output_root,
    )
    run_dir = Path(generated["report"]["runs"][0]["run_dir"])

    result = evaluate_existing_module2b_runs(
        run_dirs=[run_dir],
        output_root=output_root,
    )

    report = result["report"]
    assert report["schema_name"] == "module2b_reasoner_evaluation"
    assert report["repeats_per_case"] is None
    assert report["run_count"] == 1
    assert set(report["metrics"].keys()) == {"TBDS", "NCR", "MHRR"}
