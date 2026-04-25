"""Core metric evaluator for Module 2-B reasoner quality.

Metric definitions follow the user-specified formulas:

- TBDS: Target Binding Decisiveness Score
  status_weight * min(1, (top_score - second_score) / strong_margin_min)

- NCR: Numericization Coverage Ratio
  numeric_estimate_count / (numeric_estimate_count + not_numericized_item_count)

- MHRR: Module3 Handoff Readiness Rate
  For each run, 1 if all of the following hold, else 0:
  1) module3_handoff_preview.json parse succeeds
  2) handoff_constraint_count == len(handoff_constraint_ids)
  3) every handoff_constraint_id exists in constraint_catalog
  4) downstream Module 3 reader succeeds
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.module2b.providers import MockBundleProvider, list_module2b_case_ids
from app.module3.handoff_reader import read_module3_handoff_payload
from app.pipelines.module2b_pipeline import run_module2b_pipeline
from app.utils import dump_json, ensure_dir, load_json, project_root, timestamp_id


_TBDS_STATUS_WEIGHTS = {
    "resolved": 1.0,
    "partial": 0.75,
    "partially_resolved": 0.75,
    "ambiguous": 0.40,
}


def run_module2b_reasoner_evaluation(
    cases: list[str] | str,
    repeats: int = 5,
    output_root: Path | None = None,
    variant: str | None = None,
) -> dict[str, Any]:
    """Run repeated fixture-backed Module 2-B experiments and evaluate metrics."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    root = project_root()
    output_root = output_root or (root / "outputs")
    eval_id = timestamp_id()
    eval_dir = ensure_dir(output_root / f"module2b_reasoner_eval_{eval_id}")

    case_ids = _resolve_case_ids(cases=cases)
    provider = MockBundleProvider(fixtures_root=root / "fixtures")
    run_records: list[dict[str, Any]] = []

    for case_id in case_ids:
        for repeat_idx in range(repeats):
            result = run_module2b_pipeline(
                provider=provider,
                case_id=case_id,
                output_root=eval_dir,
                variant=variant,
            )
            run_dir = Path(result["run_dir"])
            run_records.append(
                _load_run_record(
                    run_dir=run_dir,
                    case_key=case_id,
                    repeat_idx=repeat_idx,
                )
            )

    report = _build_report(
        evaluation_id=eval_id,
        case_keys=case_ids,
        repeats=repeats,
        run_records=run_records,
    )
    dump_json(report, eval_dir / "module2b_reasoner_evaluation.json")
    return {
        "evaluation_dir": str(eval_dir),
        "report_path": str(eval_dir / "module2b_reasoner_evaluation.json"),
        "report": report,
    }


def evaluate_existing_module2b_runs(
    run_dirs: list[Path | str],
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Evaluate existing Module 2-B run directories without re-running the pipeline."""
    if not run_dirs:
        raise ValueError("run_dirs must not be empty")

    root = project_root()
    output_root = output_root or (root / "outputs")
    eval_id = timestamp_id()
    eval_dir = ensure_dir(output_root / f"module2b_run_eval_{eval_id}")

    normalized_run_dirs = [Path(item) for item in run_dirs]
    run_records = [
        _load_run_record(
            run_dir=run_dir,
            case_key=None,
            repeat_idx=idx,
        )
        for idx, run_dir in enumerate(normalized_run_dirs)
    ]

    case_keys = sorted({rec["case_key"] for rec in run_records})
    report = _build_report(
        evaluation_id=eval_id,
        case_keys=case_keys,
        repeats=None,
        run_records=run_records,
    )
    dump_json(report, eval_dir / "module2b_reasoner_evaluation.json")
    return {
        "evaluation_dir": str(eval_dir),
        "report_path": str(eval_dir / "module2b_reasoner_evaluation.json"),
        "report": report,
    }


def _build_report(
    evaluation_id: str,
    case_keys: list[str],
    repeats: int | None,
    run_records: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = _compute_reasoner_metrics(run_records=run_records)
    report = {
        "schema_name": "module2b_reasoner_evaluation",
        "schema_version": "0.2",
        "evaluation_id": evaluation_id,
        "cases": case_keys,
        "repeats_per_case": repeats,
        "run_count": len(run_records),
        "metrics": metrics,
        "runs": [
            {
                "case_key": rec["case_key"],
                "repeat_idx": rec["repeat_idx"],
                "run_dir": rec["run_dir"],
            }
            for rec in run_records
        ],
    }
    return report


def _load_run_record(
    run_dir: Path,
    case_key: str | None,
    repeat_idx: int,
) -> dict[str, Any]:
    summary = load_json(run_dir / "summary.json")
    module2b_output = load_json(run_dir / "module2b_output.json")
    target_binding_candidates = load_json(run_dir / "target_binding_candidates.json")
    numeric_estimates_trace = load_json(run_dir / "numeric_estimates_trace.json")
    preview_payload, preview_parse_success, preview_error = _try_load_json(
        run_dir / "module3_handoff_preview.json"
    )

    resolved_case_key = (
        case_key
        or str(summary.get("case_id") or "").strip()
        or str(summary.get("task_id") or "").strip()
        or run_dir.name
    )

    return {
        "case_key": resolved_case_key,
        "repeat_idx": repeat_idx,
        "run_dir": str(run_dir),
        "summary": summary,
        "module2b_output": module2b_output,
        "target_binding_candidates": target_binding_candidates,
        "numeric_estimates_trace": numeric_estimates_trace,
        "module3_handoff_preview": preview_payload,
        "module3_handoff_preview_parse_success": preview_parse_success,
        "module3_handoff_preview_parse_error": preview_error,
    }


def _try_load_json(path: Path) -> tuple[dict[str, Any] | None, bool, str | None]:
    try:
        return load_json(path), True, None
    except Exception as exc:  # pragma: no cover - defensive path
        return None, False, str(exc)


def _compute_reasoner_metrics(run_records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "TBDS": _compute_tbds(run_records=run_records),
        "NCR": _compute_ncr(run_records=run_records),
        "MHRR": _compute_mhrr(run_records=run_records),
    }


def _compute_tbds(run_records: list[dict[str, Any]]) -> dict[str, Any]:
    run_scores: list[float] = []
    by_case: dict[str, list[dict[str, Any]]] = {}

    for rec in run_records:
        binding = rec["module2b_output"].get("target_binding", {})
        trace = rec["target_binding_candidates"]
        resolution = trace.get("resolution", {}) if isinstance(trace, dict) else {}
        candidate_scoring = trace.get("candidate_scoring", []) if isinstance(trace, dict) else []

        top_score = _to_float(
            resolution.get("top_score"),
            _candidate_score(candidate_scoring, 0),
        )
        second_score = _to_float(
            resolution.get("second_score"),
            _candidate_score(candidate_scoring, 1),
        )
        thresholds = resolution.get("thresholds", {}) if isinstance(resolution, dict) else {}
        strong_margin_min = _to_float(thresholds.get("strong_margin_min"), 0.0)
        margin = max(0.0, top_score - second_score)
        status = str(binding.get("binding_status", "")).strip()
        status_weight = _TBDS_STATUS_WEIGHTS.get(status, 0.0)
        margin_ratio = min(1.0, (margin / strong_margin_min)) if strong_margin_min > 0 else 0.0
        score = status_weight * margin_ratio

        detail = {
            "repeat_idx": rec["repeat_idx"],
            "score": round(score, 6),
            "binding_status": status,
            "status_weight": round(status_weight, 6),
            "top_score": round(top_score, 6),
            "second_score": round(second_score, 6),
            "margin": round(margin, 6),
            "strong_margin_min": round(strong_margin_min, 6),
            "margin_ratio": round(margin_ratio, 6),
        }
        run_scores.append(score)
        by_case.setdefault(rec["case_key"], []).append(detail)

    return {
        "score": round(_mean(run_scores), 6),
        "by_case": {
            case_key: {
                "score": round(_mean([item["score"] for item in rows]), 6),
                "run_details": rows,
            }
            for case_key, rows in sorted(by_case.items())
        },
    }


def _compute_ncr(run_records: list[dict[str, Any]]) -> dict[str, Any]:
    run_scores: list[float] = []
    by_case: dict[str, list[dict[str, Any]]] = {}

    for rec in run_records:
        output = rec["module2b_output"]
        trace = rec["numeric_estimates_trace"]
        numeric_estimates = output.get("environment_context", {}).get("numeric_estimates", [])
        omissions = trace.get("omissions", []) if isinstance(trace, dict) else []

        numeric_count = len(numeric_estimates) if isinstance(numeric_estimates, list) else 0
        omitted_count = len(omissions) if isinstance(omissions, list) else 0
        denominator = numeric_count + omitted_count
        score = (numeric_count / denominator) if denominator > 0 else 0.0

        detail = {
            "repeat_idx": rec["repeat_idx"],
            "score": round(score, 6),
            "numeric_estimate_count": numeric_count,
            "not_numericized_item_count": omitted_count,
            "not_numericized_items": [
                str(item.get("item"))
                for item in omissions
                if isinstance(item, dict) and item.get("item") is not None
            ],
        }
        run_scores.append(score)
        by_case.setdefault(rec["case_key"], []).append(detail)

    return {
        "score": round(_mean(run_scores), 6),
        "by_case": {
            case_key: {
                "score": round(_mean([item["score"] for item in rows]), 6),
                "run_details": rows,
            }
            for case_key, rows in sorted(by_case.items())
        },
    }


def _compute_mhrr(run_records: list[dict[str, Any]]) -> dict[str, Any]:
    run_scores: list[float] = []
    by_case: dict[str, list[dict[str, Any]]] = {}

    for rec in run_records:
        output = rec["module2b_output"]
        preview = rec["module3_handoff_preview"]
        preview_parse_success = bool(rec["module3_handoff_preview_parse_success"])
        preview_parse_error = rec["module3_handoff_preview_parse_error"]

        handoff = output.get("module3_handoff", {})
        handoff_ids = handoff.get("handoff_constraint_ids", []) if isinstance(handoff, dict) else []
        handoff_ids = [str(item) for item in handoff_ids if isinstance(item, str)]

        preview_count = None
        if preview_parse_success and isinstance(preview, dict):
            preview_count = preview.get("handoff_constraint_count")
        count_match = preview_parse_success and preview_count == len(handoff_ids)

        constraint_catalog = output.get("derived_constraints", {}).get("constraint_catalog", [])
        constraint_ids = {
            str(item.get("constraint_id"))
            for item in constraint_catalog
            if isinstance(item, dict) and item.get("constraint_id") is not None
        }
        ids_exist = all(item in constraint_ids for item in handoff_ids)

        reader_success = False
        reader_error = None
        if preview_parse_success and isinstance(preview, dict):
            try:
                read_module3_handoff_payload(
                    preview_payload=preview,
                    module2b_output=output,
                )
                reader_success = True
            except Exception as exc:  # pragma: no cover - defensive path
                reader_error = str(exc)

        run_success = (
            preview_parse_success
            and count_match
            and ids_exist
            and reader_success
        )
        score = 1.0 if run_success else 0.0

        detail = {
            "repeat_idx": rec["repeat_idx"],
            "score": round(score, 6),
            "preview_parse_success": preview_parse_success,
            "preview_parse_error": preview_parse_error,
            "handoff_constraint_count": len(handoff_ids),
            "preview_handoff_constraint_count": preview_count,
            "count_match": count_match,
            "all_handoff_ids_exist_in_constraint_catalog": ids_exist,
            "reader_success": reader_success,
            "reader_error": reader_error,
            "run_success": run_success,
        }
        run_scores.append(score)
        by_case.setdefault(rec["case_key"], []).append(detail)

    return {
        "score": round(_mean(run_scores), 6),
        "by_case": {
            case_key: {
                "score": round(_mean([item["score"] for item in rows]), 6),
                "run_success_rate": round(_mean([1.0 if item["run_success"] else 0.0 for item in rows]), 6),
                "run_details": rows,
            }
            for case_key, rows in sorted(by_case.items())
        },
    }


def _candidate_score(candidate_scoring: Any, index: int) -> float:
    if not isinstance(candidate_scoring, list):
        return 0.0
    if index >= len(candidate_scoring):
        return 0.0
    item = candidate_scoring[index]
    if not isinstance(item, dict):
        return 0.0
    return _to_float(item.get("final_score"), 0.0)


def _mean(values: list[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolve_case_ids(cases: list[str] | str) -> list[str]:
    if isinstance(cases, list):
        return cases
    if cases != "all":
        return [cases]
    return list_module2b_case_ids(fixtures_root=project_root() / "fixtures")
