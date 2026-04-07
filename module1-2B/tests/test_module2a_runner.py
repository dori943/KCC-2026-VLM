from __future__ import annotations

from pathlib import Path

from app.providers.mock_provider import MockProvider
from app.runners.module2a_runner import run_module2a_pipeline
from app.utils import ensure_dir, project_root


def test_module2a_runner_from_module1_bridge():
    provider = MockProvider(fixtures_root=project_root() / "fixtures")
    output_root = ensure_dir(project_root() / "outputs" / "test_module2a")
    result = run_module2a_pipeline(
        provider=provider,
        case_id="Gemini_Generated_Image_gvc8a5gvc8a5gvc8",
        user_goal="Arrange target objects into a stable placement state.",
        success_criteria=["Objects remain stable after release."],
        task_notes=["Prefer low-risk contact paths."],
        output_root=output_root,
    )
    run_dir = Path(result["run_dir"])
    required_files = [
        "run_manifest.json",
        "module2_common_input.json",
        "module2a_output.json",
        "raw_module1_output.json",
        "normalized_module1_output.json",
        "scene_resources_from_module1.json",
        "module2_bridge_diagnostics.json",
    ]
    for filename in required_files:
        assert (run_dir / filename).exists(), filename
    assert result["summary"]["subgoal_count"] >= 1
    assert result["summary"]["module2a_schema_error_count"] == 0
