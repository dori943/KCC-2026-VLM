"""Module 3 pipeline orchestration and artifact export."""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any

from app.module3.models import Module3Input
from app.module3.providers import FileInputProvider, Module3InputProvider
from app.module3.reasoners.pose_calculator import calculate_pose
from app.module3.validators import Module3InputValidator, Module3OutputValidator
from app.utils import dump_json, ensure_dir, timestamp_id


def run_module3_pipeline(
    provider: Module3InputProvider | None = None,
    bundle_path: Path | None = None,
    case_id: str | None = None,
    output_root: Path | None = None,
    api_key: str | None = None,
    model: str = "gpt-4o",
    temperature: float = 0.1,
) -> dict[str, Any]:
    """Run Module 3 pose calculation and export artifacts."""
    output_root = output_root or (Path(__file__).resolve().parents[2] / "outputs")
    run_id = timestamp_id()
    suffix = case_id or (Path(str(bundle_path)).stem if bundle_path else "ad_hoc")
    run_dir = _ensure_unique_run_dir(output_root, f"module3_{run_id}_{suffix}")

    bundle_provider = provider or FileInputProvider()
    provider_result = bundle_provider.get_bundle(bundle_path=bundle_path, case_id=case_id)
    raw_bundle = provider_result.bundle

    input_validator = Module3InputValidator()
    input_validation = input_validator.validate(raw_bundle)
    if not input_validation.valid:
        raise ValueError("Module 3 입력 검증 실패: " + " | ".join(input_validation.errors[:3]))

    input_data = Module3Input.from_dict(raw_bundle)

    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_api_key:
        raise ValueError("OPENAI_API_KEY 환경변수 또는 --api-key 옵션이 필요합니다.")

    output_dict, pose_trace = calculate_pose(
        input_data=input_data,
        api_key=resolved_api_key,
        model=model,
        temperature=temperature,
    )

    output_validator = Module3OutputValidator()
    output_validation = output_validator.validate(output_dict)

    summary = {
        "run_id": run_id,
        "case_id": case_id,
        "provider": provider_result.metadata.get("provider"),
        "task": input_data.task[:80],
        "selected_candidate_id": input_data.selected_candidate.get("candidate_id"),
        "assembly_step_count": pose_trace.get("assembly_step_count"),
        "is_valid": pose_trace.get("is_valid"),
        "need_feedback_to_module2c": pose_trace.get("need_feedback"),
        "input_valid": input_validation.valid,
        "output_valid": output_validation.valid,
        "input_warnings": input_validation.warnings,
        "output_warnings": output_validation.warnings,
        "model": model,
        "prompt_tokens": pose_trace.get("prompt_tokens"),
        "completion_tokens": pose_trace.get("completion_tokens"),
    }

    validation_report = {
        "schema_name": "module3_validation_report",
        "schema_version": "0.1",
        "input_validation": input_validation.to_dict(),
        "output_validation": output_validation.to_dict(),
    }

    manifest = {
        "schema_name": "module3_run_manifest",
        "schema_version": "0.1",
        "run_id": run_id,
        "provider_metadata": provider_result.metadata,
        "model": model,
        "temperature": temperature,
        "artifacts": {
            "run_manifest": "run_manifest.json",
            "raw_input_bundle": "raw_input_bundle.json",
            "module3_output": "module3_output.json",
            "pose_trace": "pose_trace.json",
            "validation_report": "validation_report.json",
            "summary": "summary.json",
        },
    }

    dump_json(raw_bundle,         run_dir / "raw_input_bundle.json")
    dump_json(output_dict,        run_dir / "module3_output.json")
    dump_json(pose_trace,         run_dir / "pose_trace.json")
    dump_json(validation_report,  run_dir / "validation_report.json")
    dump_json(summary,            run_dir / "summary.json")
    dump_json(manifest,           run_dir / "run_manifest.json")

    if not output_validation.valid:
        raise ValueError("Module 3 출력 검증 실패: " + " | ".join(output_validation.errors[:3]))

    return {"run_dir": str(run_dir), "summary": summary, "manifest": manifest}


def _ensure_unique_run_dir(output_root: Path, stem: str) -> Path:
    from app.utils import ensure_dir
    candidate = output_root / stem
    if not candidate.exists():
        return ensure_dir(candidate)
    index = 1
    while True:
        fallback = output_root / f"{stem}_{index:02d}"
        if not fallback.exists():
            return ensure_dir(fallback)
        index += 1
