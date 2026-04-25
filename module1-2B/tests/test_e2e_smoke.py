from __future__ import annotations

from pathlib import Path

import pytest

from app.providers.mock_provider import MockProvider
from app.runners.module1_runner import run_module1_pipeline
from app.utils import ensure_dir, project_root


pytest.importorskip("pybullet")


def test_end_to_end_smoke():
    provider = MockProvider(fixtures_root=project_root() / "fixtures")
    output_root = ensure_dir(project_root() / "outputs" / "test_smoke")
    result = run_module1_pipeline(
        provider=provider,
        case_id="wooden_block_like_object",
        scenarios=["all"],
        output_root=output_root,
    )
    run_dir = Path(result["run_dir"])
    required = [
        "run_manifest.json",
        "raw_module1_output.json",
        "normalized_module1_output.json",
        "pybullet_surrogate_params.json",
        "pybullet_proxy_spec.json",
        "applied_dynamics.json",
        "scene_resources_from_module1.json",
        "module2_common_input_template.json",
        "module2_bridge_diagnostics.json",
        "trajectory.csv",
        "metrics.json",
        "summary.json",
    ]
    for filename in required:
        assert (run_dir / filename).exists(), filename
