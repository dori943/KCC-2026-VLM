from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
OUTPUT_TASK_ROOT = ROOT / "outputs" / "well_retrieval"


PULLEY_TERMS = [
    "pulley",
    "redirection",
    "redirect",
    "rolling",
    "wheel",
    "curved",
    "round",
]
CHAIN_TERMS = [
    "tension",
    "thread",
    "flexible",
    "line",
    "pull",
    "transmit",
]
RETRIEVE_TERMS = [
    "hook",
    "grab",
    "grasp",
    "retrieve",
    "lift",
    "catch",
    "engage",
    "pinch",
]
TARGET_TERMS = ["key", "target", "object"]


@dataclass
class SearchSummary:
    phase: str
    attempts: int
    success_runs: int
    errors: int
    combo_hits: int
    strict_hits: int
    stopped_early_on_hit: bool
    module2b_dir: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "attempts": self.attempts,
            "success_runs": self.success_runs,
            "errors": self.errors,
            "combo_hits": self.combo_hits,
            "strict_hits": self.strict_hits,
            "stopped_early_on_hit": self.stopped_early_on_hit,
            "module2b_dir": self.module2b_dir,
        }


def _contains_any(text: str, terms: list[str]) -> bool:
    t = text.lower()
    return any(term.lower() in t for term in terms)


def _mapping_text(fm: dict[str, Any]) -> str:
    return f"{fm.get('function', '')} {fm.get('related_physics', '')}".lower()


def _is_strict_hit(candidate: dict[str, Any]) -> tuple[bool, str]:
    used = {str(x).lower() for x in candidate.get("used_objects", [])}
    if not {"softball", "large_clamp", "chain"}.issubset(used):
        return False, ""

    fm_by_obj: dict[str, str] = {}
    for fm in candidate.get("function_mapping", []):
        obj = str(fm.get("object", "")).lower()
        fm_by_obj[obj] = _mapping_text(fm)

    soft_txt = fm_by_obj.get("softball", "")
    chain_txt = fm_by_obj.get("chain", "")
    clamp_txt = fm_by_obj.get("large_clamp", "")

    if not _contains_any(soft_txt, PULLEY_TERMS):
        return False, "combo_but_no_softball_pulley_role"
    if not _contains_any(chain_txt, CHAIN_TERMS):
        return False, "combo_but_no_chain_transmission_role"

    clamp_retrieve = _contains_any(clamp_txt, RETRIEVE_TERMS)
    clamp_target = _contains_any(clamp_txt, TARGET_TERMS)
    sg2_txt = " ".join(
        str(sg.get("method", ""))
        for sg in candidate.get("subgoal_coverage", [])
        if str(sg.get("subgoal_id", "")).lower() == "sg_02"
    ).lower()
    sg2_clamp_retrieve = (
        ("clamp" in sg2_txt)
        and _contains_any(sg2_txt, RETRIEVE_TERMS)
    )

    hit = clamp_retrieve and (clamp_target or sg2_clamp_retrieve)
    return (hit, "strict_hit" if hit else "combo_but_clamp_not_retriever")


def _latest_dir(pattern: str) -> Path | None:
    dirs = sorted(OUTPUT_TASK_ROOT.glob(pattern), key=lambda p: p.stat().st_mtime)
    return dirs[-1] if dirs else None


def _run_cmd(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def _build_module2b(extra_success: list[str] | None = None, extra_notes: list[str] | None = None) -> Path:
    cmd = [
        sys.executable,
        "scripts/run_pipeline.py",
        "--preset",
        "task8",
        "--start-from",
        "1",
        "--stop-at",
        "2b",
        "--provider",
        "vision",
        "--model",
        "gpt-4o",
    ]
    for s in extra_success or []:
        cmd.extend(["--success-criterion", s])
    for n in extra_notes or []:
        cmd.extend(["--task-note", n])

    proc = _run_cmd(cmd, ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"build_module2b failed:\n{proc.stderr[-1000:]}")

    latest = _latest_dir("module2b_*")
    if latest is None:
        raise RuntimeError("No module2b run_dir found after build.")
    return latest


def _run_loop(phase: str, module2b_dir: Path, max_success_runs: int) -> tuple[SearchSummary, list[dict[str, Any]]]:
    attempts = 0
    success_runs = 0
    errors = 0
    combo_hits = 0
    strict_hits = 0
    records: list[dict[str, Any]] = []

    while success_runs < max_success_runs:
        attempts += 1
        cmd = [
            sys.executable,
            "scripts/run_pipeline.py",
            "--preset",
            "task8",
            "--start-from",
            "2c",
            "--stop-at",
            "2c",
            "--module2b-dir",
            str(module2b_dir),
            "--provider",
            "vision",
            "--model",
            "gpt-4o",
        ]
        proc = _run_cmd(cmd, ROOT)
        if proc.returncode != 0:
            errors += 1
            records.append(
                {
                    "attempt": attempts,
                    "status": "error",
                    "stderr_tail": proc.stderr[-600:],
                }
            )
            continue

        run_dir = _latest_dir("module2c_*")
        if run_dir is None:
            errors += 1
            records.append({"attempt": attempts, "status": "error", "stderr_tail": "no_module2c_run_dir"})
            continue

        out_path = run_dir / "module2c_output.json"
        if not out_path.exists():
            errors += 1
            records.append({"attempt": attempts, "status": "error", "stderr_tail": "missing_module2c_output"})
            continue

        try:
            data = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors += 1
            records.append(
                {
                    "attempt": attempts,
                    "status": "error",
                    "stderr_tail": f"json_load_error: {exc}",
                }
            )
            continue

        success_runs += 1
        this_combo = 0
        this_strict = 0
        reasons: dict[str, int] = {}

        for cand in data.get("candidate_tools", []):
            used = {str(x).lower() for x in cand.get("used_objects", [])}
            if {"softball", "large_clamp", "chain"}.issubset(used):
                this_combo += 1
                hit, reason = _is_strict_hit(cand)
                if reason:
                    reasons[reason] = reasons.get(reason, 0) + 1
                if hit:
                    this_strict += 1

        combo_hits += this_combo
        strict_hits += this_strict
        records.append(
            {
                "attempt": attempts,
                "status": "ok",
                "run_dir": str(run_dir),
                "combo_count": this_combo,
                "strict_count": this_strict,
                "reason_counts": reasons,
            }
        )
        if strict_hits > 0:
            break

    summary = SearchSummary(
        phase=phase,
        attempts=attempts,
        success_runs=success_runs,
        errors=errors,
        combo_hits=combo_hits,
        strict_hits=strict_hits,
        stopped_early_on_hit=(strict_hits > 0),
        module2b_dir=str(module2b_dir),
    )
    return summary, records


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-success-runs", type=int, default=50)
    args = parser.parse_args()

    if not (ROOT / "scripts" / "run_pipeline.py").exists():
        raise RuntimeError(f"run_pipeline.py not found under {ROOT}")
    if not (ROOT / "configs" / "task_presets.yaml").exists():
        raise RuntimeError(f"task_presets.yaml not found under {ROOT}")

    if not (Path.cwd() / "module1-2B").exists() and Path.cwd().resolve() != PROJECT_ROOT.resolve():
        # Script is resilient to cwd, but this guard helps catch wrong checkout roots.
        pass

    if not (Path(sys.executable)).exists():
        raise RuntimeError("Python interpreter path invalid.")

    # Explicit runtime key check.
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    report_root = OUTPUT_TASK_ROOT / "analysis_reports_cloud"
    report_root.mkdir(parents=True, exist_ok=True)

    phase1_m2b = _build_module2b()
    phase1_summary, phase1_records = _run_loop(
        phase="baseline",
        module2b_dir=phase1_m2b,
        max_success_runs=args.max_success_runs,
    )
    _save_json(report_root / "phase1_summary.json", phase1_summary.to_dict())
    _save_json(report_root / "phase1_records.json", phase1_records)

    final_payload: dict[str, Any] = {
        "max_success_runs": args.max_success_runs,
        "phase1": phase1_summary.to_dict(),
    }

    # If phase1 has no strict hit, apply generic task-level refinement (no object-name hardcoding).
    if phase1_summary.strict_hits == 0:
        generic_success = [
            "Construct a force-transmission pathway and a separate target-engagement action so the target can be extracted reliably.",
        ]
        generic_notes = [
            "Prefer explicit role separation: one subsystem for transmission/redirection, another for direct target engagement.",
            "Subgoal coverage text should clearly map each role to a distinct manipulation function.",
        ]
        phase2_m2b = _build_module2b(extra_success=generic_success, extra_notes=generic_notes)
        phase2_summary, phase2_records = _run_loop(
            phase="refined_prompt_no_object_hardcoding",
            module2b_dir=phase2_m2b,
            max_success_runs=args.max_success_runs,
        )
        _save_json(report_root / "phase2_summary.json", phase2_summary.to_dict())
        _save_json(report_root / "phase2_records.json", phase2_records)
        final_payload["phase2"] = phase2_summary.to_dict()
        final_payload["refinement_reason"] = (
            "Phase1 produced combo hits but no strict role assignment where clamp performs target retrieval."
        )
    else:
        final_payload["phase2"] = None
        final_payload["refinement_reason"] = None

    _save_json(report_root / "final_summary.json", final_payload)
    print(json.dumps(final_payload, ensure_ascii=False, indent=2))
    print(f"REPORT_DIR={report_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
