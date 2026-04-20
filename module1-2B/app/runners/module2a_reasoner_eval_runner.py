"""Core metric evaluator for Module 2-A reasoner quality."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.runners.module2a_runner import run_module2a_pipeline
from app.utils import dump_json, ensure_dir, load_json, project_root, timestamp_id
from app.validators.schema_validator import validate_with_schema


def run_module2a_reasoner_evaluation(
    cases: list[str] | str,
    repeats: int = 5,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Run repeated Module 2-A reasoning and evaluate TCRS/RCR/SUR/SSR."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    root = project_root()
    output_root = output_root or (root / "outputs")
    eval_id = timestamp_id()
    eval_dir = ensure_dir(output_root / f"module2a_reasoner_eval_{eval_id}")

    case_ids = _resolve_case_ids(cases=cases)
    run_records: list[dict[str, Any]] = []

    for case_id in case_ids:
        module2_input_path = root / "fixtures" / "module2b_cases" / case_id / "module2_common_input.json"
        for repeat_idx in range(repeats):
            result = run_module2a_pipeline(
                module2_input_path=module2_input_path,
                output_root=eval_dir,
            )
            run_dir = Path(result["run_dir"])
            module2_input = load_json(run_dir / "module2_common_input.json")
            module2a_output = load_json(run_dir / "module2a_output.json")
            schema_errors = validate_with_schema(
                payload=module2a_output,
                schema_path=root / "schemas" / "module2a_output.schema.json",
            )
            run_records.append(
                {
                    "case_id": case_id,
                    "repeat_idx": repeat_idx,
                    "run_dir": str(run_dir),
                    "module2_input": module2_input,
                    "module2a_output": module2a_output,
                    "module2a_schema_error_count": len(schema_errors),
                    "module2a_schema_errors": schema_errors,
                }
            )

    metrics = _compute_reasoner_metrics(run_records=run_records)
    report = {
        "schema_name": "module2a_reasoner_evaluation",
        "schema_version": "0.1",
        "evaluation_id": eval_id,
        "cases": case_ids,
        "repeats_per_case": repeats,
        "run_count": len(run_records),
        "metrics": metrics,
        "runs": [
            {
                "case_id": rec["case_id"],
                "repeat_idx": rec["repeat_idx"],
                "run_dir": rec["run_dir"],
            }
            for rec in run_records
        ],
    }
    dump_json(report, eval_dir / "module2a_reasoner_evaluation.json")
    return {
        "evaluation_dir": str(eval_dir),
        "report_path": str(eval_dir / "module2a_reasoner_evaluation.json"),
        "report": report,
    }


def _compute_reasoner_metrics(run_records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "TCRS": _compute_tcrs(run_records=run_records),
        "RCR": _compute_rcr(run_records=run_records),
        "SUR": _compute_sur(run_records=run_records),
        "SSR": _compute_ssr(run_records=run_records),
    }


def _compute_tcrs(run_records: list[dict[str, Any]]) -> dict[str, Any]:
    active_constraints_total = 0
    reflected_constraints_total = 0
    by_case: dict[str, dict[str, Any]] = {}

    for rec in run_records:
        case_id = rec["case_id"]
        module2_input = rec["module2_input"]
        module2a_output = rec["module2a_output"]
        constraints = _detect_task_constraints(module2_input=module2_input)
        reflected_ids = []
        missing_ids = []
        for item in constraints:
            if not item["active"]:
                continue
            active_constraints_total += 1
            if _is_constraint_reflected(constraint_id=item["id"], module2a_output=module2a_output):
                reflected_constraints_total += 1
                reflected_ids.append(item["id"])
            else:
                missing_ids.append(item["id"])

        case_stats = by_case.setdefault(
            case_id,
            {
                "active_constraints": 0,
                "reflected_constraints": 0,
                "evaluated_runs": 0,
                "run_details": [],
            },
        )
        case_stats["active_constraints"] += len(reflected_ids) + len(missing_ids)
        case_stats["reflected_constraints"] += len(reflected_ids)
        case_stats["evaluated_runs"] += 1
        case_stats["run_details"].append(
            {
                "repeat_idx": rec["repeat_idx"],
                "active_constraint_ids": [item["id"] for item in constraints if item["active"]],
                "reflected_constraint_ids": reflected_ids,
                "missing_constraint_ids": missing_ids,
            }
        )

    for stats in by_case.values():
        if stats["active_constraints"] > 0:
            stats["score"] = round(
                stats["reflected_constraints"] / stats["active_constraints"], 6
            )
        else:
            stats["score"] = None

    score = (
        reflected_constraints_total / active_constraints_total
        if active_constraints_total > 0
        else 0.0
    )
    return {
        "score": round(score, 6),
        "reflected_constraints": reflected_constraints_total,
        "active_constraints": active_constraints_total,
        "by_case": dict(sorted(by_case.items())),
    }


def _compute_rcr(run_records: list[dict[str, Any]]) -> dict[str, Any]:
    run_scores: list[float] = []
    by_case: dict[str, list[float]] = {}
    details: dict[str, list[dict[str, Any]]] = {}

    for rec in run_records:
        subgoal_scores: list[float] = []
        subgoal_details: list[dict[str, Any]] = []
        for subgoal in rec["module2a_output"].get("subgoals", []):
            required_atoms = list(subgoal["function_requirements"]["required_atoms"])
            supporting_atoms = list(subgoal["resource_feasibility_hint"]["supporting_atoms_seen_in_scene"])
            required_set = set(required_atoms)
            covered = len(required_set.intersection(supporting_atoms))
            score = (covered / len(required_set)) if required_set else 0.0
            subgoal_scores.append(score)
            subgoal_details.append(
                {
                    "subgoal_id": subgoal["subgoal_id"],
                    "required_atom_count": len(required_set),
                    "covered_required_atom_count": covered,
                    "score": round(score, 6),
                }
            )

        run_score = (sum(subgoal_scores) / len(subgoal_scores)) if subgoal_scores else 0.0
        run_scores.append(run_score)
        by_case.setdefault(rec["case_id"], []).append(run_score)
        details.setdefault(rec["case_id"], []).append(
            {
                "repeat_idx": rec["repeat_idx"],
                "subgoals": subgoal_details,
                "score": round(run_score, 6),
            }
        )

    return {
        "score": round(sum(run_scores) / len(run_scores), 6) if run_scores else 0.0,
        "by_case": {
            case_id: {
                "score": round(sum(scores) / len(scores), 6),
                "run_scores": [round(score, 6) for score in scores],
                "run_details": details[case_id],
            }
            for case_id, scores in sorted(by_case.items())
        },
    }


def _compute_sur(run_records: list[dict[str, Any]]) -> dict[str, Any]:
    run_scores: list[float] = []
    by_case: dict[str, list[float]] = {}
    details: dict[str, list[dict[str, Any]]] = {}

    for rec in run_records:
        subgoals = rec["module2a_output"].get("subgoals", [])
        usable_count = 0
        subgoal_details = []
        for subgoal in subgoals:
            coverage_code = int(subgoal["pybullet_bridge"]["coverage_status_code"])
            usable = coverage_code >= 3
            if usable:
                usable_count += 1
            subgoal_details.append(
                {
                    "subgoal_id": subgoal["subgoal_id"],
                    "coverage_status": subgoal["resource_feasibility_hint"]["coverage_status"],
                    "coverage_status_code": coverage_code,
                    "usable_or_better": usable,
                }
            )

        run_score = (usable_count / len(subgoals)) if subgoals else 0.0
        run_scores.append(run_score)
        by_case.setdefault(rec["case_id"], []).append(run_score)
        details.setdefault(rec["case_id"], []).append(
            {
                "repeat_idx": rec["repeat_idx"],
                "usable_subgoal_count": usable_count,
                "total_subgoals": len(subgoals),
                "score": round(run_score, 6),
                "subgoals": subgoal_details,
            }
        )

    return {
        "score": round(sum(run_scores) / len(run_scores), 6) if run_scores else 0.0,
        "by_case": {
            case_id: {
                "score": round(sum(scores) / len(scores), 6),
                "run_scores": [round(score, 6) for score in scores],
                "run_details": details[case_id],
            }
            for case_id, scores in sorted(by_case.items())
        },
    }


def _compute_ssr(run_records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[tuple[str, ...]]] = {}
    for rec in run_records:
        grouped.setdefault(rec["case_id"], []).append(_subgoal_sequence(rec["module2a_output"]))

    by_case: dict[str, Any] = {}
    case_scores: list[float] = []
    for case_id, sequences in sorted(grouped.items()):
        if len(sequences) < 2:
            score = 1.0
            unique_sequences = [list(sequences[0])] if sequences else []
        else:
            total_pairs = 0
            matched_pairs = 0
            for idx in range(len(sequences)):
                for jdx in range(idx + 1, len(sequences)):
                    total_pairs += 1
                    if sequences[idx] == sequences[jdx]:
                        matched_pairs += 1
            score = (matched_pairs / total_pairs) if total_pairs else 1.0
            unique_sequences = [list(item) for item in sorted(set(sequences))]
        case_scores.append(score)
        by_case[case_id] = {
            "score": round(score, 6),
            "repeat_count": len(sequences),
            "unique_sequences": unique_sequences,
        }

    return {
        "score": round(sum(case_scores) / len(case_scores), 6) if case_scores else 0.0,
        "by_case": by_case,
    }


def _detect_task_constraints(module2_input: dict[str, Any]) -> list[dict[str, Any]]:
    task_brief = module2_input.get("task_brief", {})
    merged = " ".join(
        [str(task_brief.get("user_goal") or "")]
        + [str(item) for item in task_brief.get("success_criteria", [])]
        + [str(item) for item in task_brief.get("task_notes", [])]
    ).lower()
    return [
        {
            "id": "constrained_entry",
            "active": _contains_any(
                merged,
                [
                    "narrow gap",
                    "slot",
                    "partially occluded",
                    "occluded",
                    "overhang",
                    "recess",
                    "limited rotation",
                    "side access",
                    "top entry",
                    "lip",
                ],
            ),
        },
        {
            "id": "deep_reach",
            "active": _contains_any(
                merged,
                [
                    "deep recess",
                    "reach depth",
                    "deep in recess",
                    "deep inside",
                    "deep in the",
                    "depth is likely limiting",
                ],
            ),
        },
        {
            "id": "damage_avoidance",
            "active": _contains_any(
                merged,
                [
                    "avoid damage",
                    "surface is not damaged",
                    "surrounding surface",
                    "avoid scratching",
                    "avoid gouging",
                    "without bending or tearing",
                    "avoid bending",
                    "avoid tearing",
                    "avoid sharp",
                ],
            ),
        },
        {
            "id": "post_release_stability",
            "active": _contains_any(
                merged,
                [
                    "stable",
                    "upright",
                    "remains upright",
                    "remains stable",
                ],
            ),
        },
        {
            "id": "contact_control",
            "active": _contains_any(
                merged,
                [
                    "controllable",
                    "controlled",
                    "without pushing it deeper",
                    "limited rotation",
                ],
            ),
        },
    ]


def _is_constraint_reflected(constraint_id: str, module2a_output: dict[str, Any]) -> bool:
    summary = module2a_output.get("task_level_requirement_summary", {})
    required_atoms_union = set(summary.get("required_atoms_union", []))
    preferred_atoms_union = set(summary.get("preferred_atoms_union", []))
    risk_atoms_union = set(summary.get("risk_atoms_to_avoid_union", []))
    text = _output_text(module2a_output)

    if constraint_id == "constrained_entry":
        return _contains_any(
            text,
            [
                "narrow-gap",
                "low-profile",
                "side-entry",
                "collision-free",
                "limited rotation",
                "recess opening",
                "deep-reach",
                "deep reach",
                "constrained path",
                "constrained region",
            ],
        )
    if constraint_id == "deep_reach":
        return "elongated_reach" in required_atoms_union or _contains_any(
            text,
            ["deep-reach", "deep reach", "recess opening", "recess depth", "deep recess"],
        )
    if constraint_id == "damage_avoidance":
        return (
            "sharp_contact_risk" in risk_atoms_union
            or "deform_prone" in risk_atoms_union
            or _contains_any(
                text,
                ["damage", "sharp", "bend", "tear", "crease", "scratch", "gouge"],
            )
        )
    if constraint_id == "post_release_stability":
        return "stable_support_face" in required_atoms_union or _contains_any(
            text,
            ["stable", "upright", "placement", "stabilize"],
        )
    if constraint_id == "contact_control":
        return (
            "frictional_contact" in required_atoms_union
            or "frictional_contact" in preferred_atoms_union
            or _contains_any(
                text,
                ["controlled", "controllable", "stable contact", "bounded force"],
            )
        )
    return False


def _output_text(module2a_output: dict[str, Any]) -> str:
    chunks: list[str] = []
    task_model = module2a_output.get("task_model", {})
    summary = module2a_output.get("task_level_requirement_summary", {})
    chunks.append(str(task_model.get("decomposition_principle", "")))
    chunks.extend(str(item) for item in task_model.get("assumptions", []))
    chunks.extend(str(item) for item in summary.get("resource_gap_hypotheses", []))
    chunks.extend(str(item) for item in summary.get("risk_atoms_to_avoid_union", []))
    chunks.extend(str(item) for item in summary.get("required_atoms_union", []))
    chunks.extend(str(item) for item in summary.get("preferred_atoms_union", []))

    for subgoal in module2a_output.get("subgoals", []):
        chunks.append(str(subgoal.get("subgoal_name", "")))
        chunks.append(str(subgoal.get("objective", "")))
        chunks.append(str(subgoal.get("success_condition", "")))
        chunks.append(str(subgoal.get("physical_rationale", "")))
        chunks.extend(str(item) for item in subgoal.get("failure_risks", []))
        reqs = subgoal.get("function_requirements", {})
        chunks.extend(str(item) for item in reqs.get("required_atoms", []))
        chunks.extend(str(item) for item in reqs.get("preferred_atoms", []))
        chunks.extend(str(item) for item in reqs.get("risk_atoms_to_avoid", []))
    return " ".join(chunks).lower()


def _subgoal_sequence(module2a_output: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(item.get("subgoal_name", "")) for item in module2a_output.get("subgoals", []))


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _resolve_case_ids(cases: list[str] | str) -> list[str]:
    if isinstance(cases, list):
        return cases
    if cases != "all":
        return [cases]
    root = project_root()
    index = load_json(root / "fixtures" / "module2b_cases" / "index.json")
    return list(index["cases"])
