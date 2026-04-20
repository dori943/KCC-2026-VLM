"""Core metric evaluator for Module 1 using SESR/APRE/FSC/RSMS."""

from __future__ import annotations

import csv
import itertools
from pathlib import Path
from typing import Any

from app.providers.mock_provider import MockProvider
from app.runners.module1_runner import run_module1_pipeline
from app.utils import dump_json, ensure_dir, load_json, load_yaml, project_root, timestamp_id, to_float


DEFAULT_SCENARIOS = ["drop_test", "slide_test", "force_response_test"]


def run_module1_core_evaluation(
    cases: list[str] | str,
    provider_name: str = "mock",
    repeats: int = 5,
    output_root: Path | None = None,
    prompt_variant: str | None = None,
) -> dict[str, Any]:
    """Run repeated experiments and evaluate SESR/APRE/FSC/RSMS."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    root = project_root()
    output_root = output_root or (root / "outputs")
    eval_id = timestamp_id()
    eval_dir = ensure_dir(output_root / f"module1_core_eval_{eval_id}")

    case_ids = _resolve_case_ids(cases=cases)
    provider = _build_provider(provider_name=provider_name)
    mapping_cfg = load_yaml(root / "configs" / "module1_to_pybullet_map.yaml")

    run_records: list[dict[str, Any]] = []
    for case_id in case_ids:
        for repeat_idx in range(repeats):
            result = run_module1_pipeline(
                provider=provider,
                case_id=case_id,
                scenarios=["all"],
                output_root=eval_dir,
                prompt_variant=prompt_variant,
            )
            run_dir = Path(result["run_dir"])
            run_records.append(
                {
                    "case_id": case_id,
                    "repeat_idx": repeat_idx,
                    "run_dir": str(run_dir),
                    "summary": load_json(run_dir / "summary.json"),
                    "metrics": load_json(run_dir / "metrics.json"),
                    "surrogate": load_json(run_dir / "pybullet_surrogate_params.json"),
                    "applied_dynamics": load_json(run_dir / "applied_dynamics.json"),
                    "trajectory_rows": _load_trajectory_rows(run_dir / "trajectory.csv"),
                }
            )

    metric_results = _compute_core_metrics(
        run_records=run_records,
        mapping_cfg=mapping_cfg,
        repeats=repeats,
    )

    report = {
        "schema_name": "module1_core_evaluation",
        "schema_version": "0.1",
        "evaluation_id": eval_id,
        "provider": provider_name,
        "cases": case_ids,
        "repeats_per_case": repeats,
        "run_count": len(run_records),
        "scenarios": list(DEFAULT_SCENARIOS),
        "metrics": metric_results,
        "runs": [
            {
                "case_id": rec["case_id"],
                "repeat_idx": rec["repeat_idx"],
                "run_dir": rec["run_dir"],
            }
            for rec in run_records
        ],
    }
    dump_json(report, eval_dir / "module1_core_evaluation.json")
    return {
        "evaluation_dir": str(eval_dir),
        "report_path": str(eval_dir / "module1_core_evaluation.json"),
        "report": report,
    }


def _compute_core_metrics(
    run_records: list[dict[str, Any]],
    mapping_cfg: dict[str, Any],
    repeats: int,
) -> dict[str, Any]:
    sesr = _compute_sesr(run_records)
    apre = _compute_apre(run_records, mapping_cfg)
    fsc = _compute_fsc(run_records)
    rsms = _compute_rsms(run_records, repeats=repeats)
    return {
        "SESR": sesr,
        "APRE": apre,
        "FSC": fsc,
        "RSMS": rsms,
    }


def _compute_sesr(run_records: list[dict[str, Any]]) -> dict[str, Any]:
    successful_pairs = 0
    total_pairs = 0
    failed_pairs: list[dict[str, Any]] = []

    for rec in run_records:
        per_object = rec["metrics"].get("per_object", {})
        object_ids = [obj["object_id"] for obj in rec["surrogate"].get("objects", [])]
        trajectory_counts: dict[tuple[str, str], int] = {}
        for row in rec["trajectory_rows"]:
            key = (str(row.get("object_id")), str(row.get("scenario")))
            trajectory_counts[key] = trajectory_counts.get(key, 0) + 1

        for object_id in object_ids:
            for scenario in DEFAULT_SCENARIOS:
                total_pairs += 1
                has_metric = scenario in per_object.get(object_id, {})
                has_trajectory = trajectory_counts.get((object_id, scenario), 0) > 0
                if has_metric and has_trajectory:
                    successful_pairs += 1
                else:
                    failed_pairs.append(
                        {
                            "case_id": rec["case_id"],
                            "repeat_idx": rec["repeat_idx"],
                            "object_id": object_id,
                            "scenario": scenario,
                            "has_metric": has_metric,
                            "has_trajectory": has_trajectory,
                        }
                    )

    score = (successful_pairs / total_pairs) if total_pairs else 0.0
    return {
        "score": round(score, 6),
        "successful_pairs": successful_pairs,
        "total_pairs": total_pairs,
        "failed_pair_count": len(failed_pairs),
        "failed_pairs": failed_pairs[:20],
    }


def _compute_apre(run_records: list[dict[str, Any]], mapping_cfg: dict[str, Any]) -> dict[str, Any]:
    clamp_ranges = mapping_cfg["clamp_ranges"]
    mass_span = float(clamp_ranges["mass_kg"][1]) - float(clamp_ranges["mass_kg"][0])
    fric_span = float(clamp_ranges["lateral_friction"][1]) - float(
        clamp_ranges["lateral_friction"][0]
    )
    rest_span = float(clamp_ranges["restitution"][1]) - float(clamp_ranges["restitution"][0])

    pair_errors: list[float] = []
    per_field_errors = {
        "mass_norm_abs_error": [],
        "lateral_friction_norm_abs_error": [],
        "restitution_norm_abs_error": [],
    }

    for rec in run_records:
        for row in rec["applied_dynamics"].get("rows", []):
            requested = row.get("requested_dynamics", {})
            actual = row.get("actual_dynamics", {})

            req_mass = _value_or_fallback(requested.get("mass_kg"), row.get("mass_kg"))
            req_fric = _value_or_fallback(
                requested.get("lateral_friction"), row.get("lateral_friction")
            )
            req_rest = _value_or_fallback(requested.get("restitution"), row.get("restitution"))
            act_mass = _value_or_fallback(actual.get("mass_kg"), row.get("mass_kg"))
            act_fric = _value_or_fallback(
                actual.get("lateral_friction"), row.get("lateral_friction")
            )
            act_rest = _value_or_fallback(actual.get("restitution"), row.get("restitution"))

            if None in {req_mass, req_fric, req_rest, act_mass, act_fric, act_rest}:
                continue

            e_mass = abs(req_mass - act_mass) / mass_span if mass_span > 0 else 0.0
            e_fric = abs(req_fric - act_fric) / fric_span if fric_span > 0 else 0.0
            e_rest = abs(req_rest - act_rest) / rest_span if rest_span > 0 else 0.0
            pair_errors.append((e_mass + e_fric + e_rest) / 3.0)
            per_field_errors["mass_norm_abs_error"].append(e_mass)
            per_field_errors["lateral_friction_norm_abs_error"].append(e_fric)
            per_field_errors["restitution_norm_abs_error"].append(e_rest)

    return {
        "score": round(_mean(pair_errors), 6),
        "sample_count": len(pair_errors),
        "per_field_mean": {
            key: round(_mean(values), 6) for key, values in per_field_errors.items()
        },
    }


def _compute_fsc(run_records: list[dict[str, Any]]) -> dict[str, Any]:
    # One observation per (run, object) using slide_test stopping distance.
    observations: list[dict[str, Any]] = []
    for rec in run_records:
        metrics_per_object = rec["metrics"].get("per_object", {})
        friction_by_object = _slide_friction_by_object(rec["applied_dynamics"].get("rows", []))
        for object_id, obj_metrics in metrics_per_object.items():
            slide_metrics = obj_metrics.get("slide_test")
            if not isinstance(slide_metrics, dict):
                continue
            stopping_distance = to_float(slide_metrics.get("stopping_distance_m"), default=None)
            friction = friction_by_object.get(object_id)
            if stopping_distance is None or friction is None:
                continue
            observations.append(
                {
                    "case_id": rec["case_id"],
                    "repeat_idx": rec["repeat_idx"],
                    "object_id": object_id,
                    "friction": friction,
                    "stopping_distance_m": stopping_distance,
                }
            )

    eps = 1e-9
    consistent = 0
    comparable = 0
    for a, b in itertools.combinations(observations, 2):
        fric_diff = a["friction"] - b["friction"]
        dist_diff = a["stopping_distance_m"] - b["stopping_distance_m"]
        if abs(fric_diff) <= eps or abs(dist_diff) <= eps:
            continue
        comparable += 1
        if fric_diff * dist_diff < 0:
            consistent += 1

    score = (consistent / comparable) if comparable else 0.0
    return {
        "score": round(score, 6),
        "observation_count": len(observations),
        "comparable_pair_count": comparable,
        "consistent_pair_count": consistent,
    }


def _compute_rsms(run_records: list[dict[str, Any]], repeats: int) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for rec in run_records:
        by_case.setdefault(rec["case_id"], []).append(rec)

    all_spreads: list[float] = []
    case_reports: dict[str, Any] = {}
    for case_id, records in sorted(by_case.items()):
        if len(records) < 2:
            case_reports[case_id] = {
                "repeat_count": len(records),
                "metric_path_count": 0,
                "mean_spread": 0.0,
                "max_spread": 0.0,
            }
            continue

        path_values: dict[str, list[float]] = {}
        for rec in records:
            per_object = rec["metrics"].get("per_object", {})
            for object_id, scenario_map in per_object.items():
                for scenario, metric_map in scenario_map.items():
                    for key, value in metric_map.items():
                        path = f"{object_id}.{scenario}.{key}"
                        path_values.setdefault(path, []).append(float(value))

        spreads: list[float] = []
        for values in path_values.values():
            if len(values) != len(records):
                continue
            spreads.append(max(values) - min(values))

        all_spreads.extend(spreads)
        case_reports[case_id] = {
            "repeat_count": len(records),
            "expected_repeats": repeats,
            "metric_path_count": len(spreads),
            "mean_spread": round(_mean(spreads), 6),
            "max_spread": round(max(spreads), 6) if spreads else 0.0,
        }

    return {
        "mean_spread": round(_mean(all_spreads), 6),
        "max_spread": round(max(all_spreads), 6) if all_spreads else 0.0,
        "path_count": len(all_spreads),
        "by_case": case_reports,
    }


def _slide_friction_by_object(rows: list[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, float] = {}
    for row in rows:
        if row.get("scenario") != "slide_test":
            continue
        requested = row.get("requested_dynamics", {})
        actual = row.get("actual_dynamics", {})
        friction = _value_or_fallback(actual.get("lateral_friction"), requested.get("lateral_friction"))
        if friction is None:
            friction = to_float(row.get("lateral_friction"), default=None)
        object_id = str(row.get("object_id"))
        if friction is not None and object_id:
            values[object_id] = friction
    return values


def _value_or_fallback(primary: Any, fallback: Any) -> float | None:
    if primary is not None:
        return to_float(primary, default=None)
    if fallback is not None:
        return to_float(fallback, default=None)
    return None


def _mean(values: list[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def _build_provider(provider_name: str):
    if provider_name == "mock":
        return MockProvider(fixtures_root=project_root() / "fixtures")
    raise ValueError("Core evaluation currently supports provider=mock only.")


def _resolve_case_ids(cases: list[str] | str) -> list[str]:
    if isinstance(cases, list):
        return cases
    if cases != "all":
        return [cases]
    root = project_root()
    index = load_json(root / "fixtures" / "cases" / "index.json")
    return list(index["cases"])


def _load_trajectory_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows
