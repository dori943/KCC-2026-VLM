"""Module 2-D pipeline orchestration and artifact export."""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any

from app.module2d.models import Module2DInput, Module2DOutput
from app.module2d.providers import FileInputProvider, Module2DInputProvider
from app.module2d.reasoners.candidate_filter import filter_candidates
from app.module2d.validators import Module2DInputValidator, Module2DOutputValidator
from app.utils import (
    build_task_output_root,
    derive_task_name,
    dump_json,
    ensure_unique_run_dir,
    timestamp_id,
)


def run_module2d_pipeline(
    provider: Module2DInputProvider | None = None,
    bundle_path: Path | None = None,
    case_id: str | None = None,
    output_root: Path | None = None,
    api_key: str | None = None,
    model: str = "gpt-4o",
    temperature: float = 0.2,
    task_name: str | None = None,
) -> dict[str, Any]:
    """Run Module 2-D filtering and export artifacts.

    Output은 outputs/<task_name>/module2d_<timestamp>_<suffix>/ 구조로 저장된다.
    """
    output_root = output_root or (Path(__file__).resolve().parents[2] / "outputs")
    resolved_task_name = derive_task_name(
        task_name=task_name, bundle_path=bundle_path, case_id=case_id,
    )
    task_root = build_task_output_root(output_root, resolved_task_name)

    run_id = timestamp_id()
    suffix = case_id or (Path(str(bundle_path)).stem if bundle_path else "ad_hoc")
    run_dir = ensure_unique_run_dir(task_root, f"module2d_{run_id}_{suffix}")

    bundle_provider = provider or FileInputProvider()
    provider_result = bundle_provider.get_bundle(bundle_path=bundle_path, case_id=case_id)
    raw_bundle = provider_result.bundle

    input_validator = Module2DInputValidator()
    input_validation = input_validator.validate(raw_bundle)
    if not input_validation.valid:
        raise ValueError("Module 2-D 입력 검증 실패: " + " | ".join(input_validation.errors[:3]))

    input_data = Module2DInput.from_dict(raw_bundle)

    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_api_key:
        raise ValueError("OPENAI_API_KEY 환경변수 또는 --api-key 옵션이 필요합니다.")

    evaluated, selected_id, feedback, filter_trace = filter_candidates(
        input_data=input_data,
        api_key=resolved_api_key,
        model=model,
        temperature=temperature,
    )

    output = Module2DOutput(
        evaluated_candidates=evaluated,
        selected_candidate_id=selected_id,
        feedback_decision=feedback,
        filter_counts=filter_trace.get("filter_counts", {}),
    )
    output_dict = output.to_dict()

    output_validator = Module2DOutputValidator()
    output_validation = output_validator.validate(output_dict)

    summary = {
        "run_id": run_id,
        "task_name": resolved_task_name,
        "case_id": case_id,
        "provider": provider_result.metadata.get("provider"),
        "task": input_data.task[:80],
        "evaluated_count": len(evaluated),
        "pass_count": sum(1 for c in evaluated if c.passed),
        "selected_candidate_id": selected_id,
        "need_feedback": feedback.need_feedback,
        "feedback_target": feedback.feedback_target,
        "input_valid": input_validation.valid,
        "output_valid": output_validation.valid,
        "model": model,
        "prompt_tokens": filter_trace.get("prompt_tokens"),
        "completion_tokens": filter_trace.get("completion_tokens"),
    }

    validation_report = {
        "schema_name": "module2d_validation_report",
        "schema_version": "0.1",
        "input_validation": input_validation.to_dict(),
        "output_validation": output_validation.to_dict(),
    }

    manifest = {
        "schema_name": "module2d_run_manifest",
        "schema_version": "0.1",
        "run_id": run_id,
        "task_name": resolved_task_name,
        "provider_metadata": provider_result.metadata,
        "model": model,
        "temperature": temperature,
        "artifacts": {
            "run_manifest": "run_manifest.json",
            "raw_input_bundle": "raw_input_bundle.json",
            "module2d_output": "module2d_output.json",
            "filter_trace": "filter_trace.json",
            "validation_report": "validation_report.json",
            "summary": "summary.json",
        },
    }

    dump_json(raw_bundle,         run_dir / "raw_input_bundle.json")
    dump_json(output_dict,        run_dir / "module2d_output.json")
    dump_json(filter_trace,       run_dir / "filter_trace.json")
    dump_json(validation_report,  run_dir / "validation_report.json")
    dump_json(summary,            run_dir / "summary.json")
    dump_json(manifest,           run_dir / "run_manifest.json")

    if not output_validation.valid:
        raise ValueError("Module 2-D 출력 검증 실패: " + " | ".join(output_validation.errors[:3]))

    return {"run_dir": str(run_dir), "summary": summary, "manifest": manifest}
