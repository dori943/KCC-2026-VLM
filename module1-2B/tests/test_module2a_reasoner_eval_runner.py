from __future__ import annotations

from pathlib import Path

from app.runners.module2a_reasoner_eval_runner import run_module2a_reasoner_evaluation
from app.utils import ensure_dir, project_root


def test_module2a_reasoner_eval_runner_smoke():
    output_root = ensure_dir(project_root() / "outputs" / "test_module2a_reasoner_eval")
    result = run_module2a_reasoner_evaluation(
        cases=["coin_in_narrow_gap_case"],
        repeats=2,
        output_root=output_root,
    )

    report_path = Path(result["report_path"])
    assert report_path.exists()

    report = result["report"]
    assert report["schema_name"] == "module2a_reasoner_evaluation"
    assert report["repeats_per_case"] == 2
    assert set(report["metrics"].keys()) == {"TCRS", "RCR", "SUR", "SSR"}
    assert 0.0 <= report["metrics"]["RCR"]["score"] <= 1.0
    assert 0.0 <= report["metrics"]["SUR"]["score"] <= 1.0
    assert 0.0 <= report["metrics"]["SSR"]["score"] <= 1.0
    assert report["metrics"]["SSR"]["by_case"]["coin_in_narrow_gap_case"]["score"] == 1.0
