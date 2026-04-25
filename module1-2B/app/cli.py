"""CLI entry point for Module 1 + Module 2-A + Module 2-B experiment harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.module2b.providers import FileBundleProvider as Module2BFileBundleProvider
from app.module2b.providers import MockBundleProvider as Module2BMockBundleProvider
from app.pipelines.module2b_pipeline import export_module2b_normalized_context, run_module2b_pipeline
from app.providers.file_provider import FileProvider
from app.providers.mock_provider import MockProvider
from app.providers.vision_provider import VisionProvider
from app.runners.batch_runner import run_batch
from app.runners.module1_core_eval_runner import run_module1_core_evaluation
from app.runners.module2a_reasoner_eval_runner import run_module2a_reasoner_evaluation
from app.runners.module2b_reasoner_eval_runner import run_module2b_reasoner_evaluation
from app.runners.module1_runner import export_module2_bridge_only, run_module1_pipeline
from app.runners.module2a_runner import run_module2a_pipeline
from app.runners.module2b_runner import (
    run_module2b_batch,
    run_module2b_comparison,
    validate_module2b_input,
)
from app.utils import load_json, project_root
from app.validators.module1_validator import Module1Validator


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate-module1":
        return _cmd_validate_module1(args)
    if args.command == "run-experiments":
        return _cmd_run_experiments(args)
    if args.command == "export-module2-bridge":
        return _cmd_export_module2_bridge(args)
    if args.command == "run-module2a":
        return _cmd_run_module2a(args)
    if args.command == "batch":
        return _cmd_batch(args)
    if args.command == "evaluate-module1-core":
        return _cmd_evaluate_module1_core(args)
    if args.command == "evaluate-module2a-reasoner":
        return _cmd_evaluate_module2a_reasoner(args)
    if args.command == "evaluate-module2b-reasoner":
        return _cmd_evaluate_module2b_reasoner(args)
    if args.command == "validate-module2b-input":
        return _cmd_validate_module2b_input(args)
    if args.command == "run-module2b":
        return _cmd_run_module2b(args)
    if args.command == "batch-module2b":
        return _cmd_batch_module2b(args)
    if args.command == "compare-module2b-runs":
        return _cmd_compare_module2b_runs(args)
    if args.command == "export-module2b-normalized":
        return _cmd_export_module2b_normalized(args)
    parser.print_help()
    return 1


def _cmd_validate_module1(args: argparse.Namespace) -> int:
    payload = load_json(Path(args.input))
    report = Module1Validator().validate(payload)
    print(
        json.dumps(
            {
                "valid": report.valid,
                "errors": report.errors,
                "warnings": report.warnings,
            },
            indent=2,
        )
    )
    return 0 if report.valid else 2


def _cmd_run_experiments(args: argparse.Namespace) -> int:
    provider = _build_provider(args.provider)
    scenarios = _parse_scenarios(args.scenarios)
    result = run_module1_pipeline(
        provider=provider,
        image_path=Path(args.image) if args.image else None,
        case_id=args.case_id,
        module1_output_path=Path(args.module1_output) if args.module1_output else None,
        scenarios=scenarios,
        output_root=Path(args.output_root) if args.output_root else None,
        prompt_variant=args.prompt_variant,
<<<<<<< HEAD
=======
        task_name=getattr(args, "task_name", None),
>>>>>>> origin/subin/module2c-3-pipeline
    )
    print(json.dumps(result, indent=2))
    return 0


def _cmd_export_module2_bridge(args: argparse.Namespace) -> int:
    provider = _build_provider(args.provider)
    result = export_module2_bridge_only(
        provider=provider,
        image_path=Path(args.image) if args.image else None,
        case_id=args.case_id,
        module1_output_path=Path(args.module1_output) if args.module1_output else None,
        output_root=Path(args.output_root) if args.output_root else None,
<<<<<<< HEAD
=======
        task_name=getattr(args, "task_name", None),
>>>>>>> origin/subin/module2c-3-pipeline
    )
    print(json.dumps(result, indent=2))
    return 0


def _cmd_batch(args: argparse.Namespace) -> int:
    if args.cases == "all":
        cases: Any = "all"
    else:
        cases = [case.strip() for case in args.cases.split(",") if case.strip()]
    result = run_batch(
        cases=cases,
        provider_name=args.provider,
        output_root=Path(args.output_root) if args.output_root else None,
    )
    print(json.dumps(result, indent=2))
    return 0


def _cmd_run_module2a(args: argparse.Namespace) -> int:
    provider = None if args.module2_input else _build_provider(args.provider)
    result = run_module2a_pipeline(
        module2_input_path=Path(args.module2_input) if args.module2_input else None,
        provider=provider,
        image_path=Path(args.image) if args.image else None,
        case_id=args.case_id,
        module1_output_path=Path(args.module1_output) if args.module1_output else None,
        user_goal=args.user_goal,
        success_criteria=args.success_criteria,
        task_notes=args.task_notes,
        output_root=Path(args.output_root) if args.output_root else None,
<<<<<<< HEAD
=======
        task_name=getattr(args, "task_name", None),
>>>>>>> origin/subin/module2c-3-pipeline
    )
    print(json.dumps(result, indent=2))
    return 0


def _cmd_evaluate_module1_core(args: argparse.Namespace) -> int:
    if args.cases == "all":
        cases: Any = "all"
    else:
        cases = [case.strip() for case in args.cases.split(",") if case.strip()]
    result = run_module1_core_evaluation(
        cases=cases,
        provider_name=args.provider,
        repeats=int(args.repeats),
        output_root=Path(args.output_root) if args.output_root else None,
        prompt_variant=args.prompt_variant,
    )
    print(json.dumps(result, indent=2))
    return 0


def _cmd_evaluate_module2a_reasoner(args: argparse.Namespace) -> int:
    if args.cases == "all":
        cases: Any = "all"
    else:
        cases = [case.strip() for case in args.cases.split(",") if case.strip()]
    result = run_module2a_reasoner_evaluation(
        cases=cases,
        repeats=int(args.repeats),
        output_root=Path(args.output_root) if args.output_root else None,
    )
    print(json.dumps(result, indent=2))
    return 0


def _cmd_evaluate_module2b_reasoner(args: argparse.Namespace) -> int:
    if args.cases == "all":
        cases: Any = "all"
    else:
        cases = [case.strip() for case in args.cases.split(",") if case.strip()]
    result = run_module2b_reasoner_evaluation(
        cases=cases,
        repeats=int(args.repeats),
        output_root=Path(args.output_root) if args.output_root else None,
        variant=args.variant,
    )
    print(json.dumps(result, indent=2))
    return 0


def _cmd_validate_module2b_input(args: argparse.Namespace) -> int:
    report = validate_module2b_input(
        bundle_path=Path(args.bundle) if args.bundle else None,
        module2_common_path=Path(args.module2_common) if args.module2_common else None,
        module2a_output_path=Path(args.module2a_output) if args.module2a_output else None,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["valid"] else 2


def _cmd_run_module2b(args: argparse.Namespace) -> int:
    provider = _build_module2b_provider(args.provider)
    result = run_module2b_pipeline(
        provider=provider,
        bundle_path=Path(args.bundle) if args.bundle else None,
        module2_common_path=Path(args.module2_common) if args.module2_common else None,
        module2a_output_path=Path(args.module2a_output) if args.module2a_output else None,
        case_id=args.case_id,
        output_root=Path(args.output_root) if args.output_root else None,
        variant=args.variant,
<<<<<<< HEAD
=======
        task_name=getattr(args, "task_name", None),
>>>>>>> origin/subin/module2c-3-pipeline
    )
    print(json.dumps(result, indent=2))
    return 0


def _cmd_batch_module2b(args: argparse.Namespace) -> int:
    if args.cases == "all":
        cases: Any = "all"
    else:
        cases = [case.strip() for case in args.cases.split(",") if case.strip()]
    result = run_module2b_batch(
        cases=cases,
        provider_name=args.provider,
        repeats=int(args.repeats),
        output_root=Path(args.output_root) if args.output_root else None,
        variant=args.variant,
    )
    print(json.dumps(result, indent=2))
    return 0


def _cmd_compare_module2b_runs(args: argparse.Namespace) -> int:
    result = run_module2b_comparison(
        run_a=Path(args.run_a),
        run_b=Path(args.run_b),
        output_root=Path(args.output_root) if args.output_root else None,
    )
    print(json.dumps(result, indent=2))
    return 0


def _cmd_export_module2b_normalized(args: argparse.Namespace) -> int:
    provider = _build_module2b_provider(args.provider)
    result = export_module2b_normalized_context(
        bundle_path=Path(args.bundle) if args.bundle else None,
        module2_common_path=Path(args.module2_common) if args.module2_common else None,
        module2a_output_path=Path(args.module2a_output) if args.module2a_output else None,
        case_id=args.case_id,
        provider=provider,
        output_root=Path(args.output_root) if args.output_root else None,
<<<<<<< HEAD
=======
        task_name=getattr(args, "task_name", None),
>>>>>>> origin/subin/module2c-3-pipeline
    )
    print(json.dumps(result, indent=2))
    return 0


def _build_provider(name: str):
    if name == "mock":
        return MockProvider(fixtures_root=project_root() / "fixtures")
    if name == "file":
        return FileProvider()
    if name == "vision":
        return VisionProvider()
    raise ValueError(f"Unknown provider: {name}")


def _build_module2b_provider(name: str):
    if name == "mock":
        return Module2BMockBundleProvider(fixtures_root=project_root() / "fixtures")
    if name == "file":
        return Module2BFileBundleProvider()
    raise ValueError(f"Unknown Module 2-B provider: {name}")


def _parse_scenarios(raw: str) -> list[str]:
    if raw == "all":
        return ["all"]
    return [item.strip() for item in raw.split(",") if item.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Module 1 + Module 2-A + Module 2-B experiment harness CLI")
    subparsers = parser.add_subparsers(dest="command")

    validate_parser = subparsers.add_parser(
        "validate-module1", help="Validate Module 1 raw output JSON"
    )
    validate_parser.add_argument("--input", required=True, help="Path to Module 1 raw JSON")

    run_parser = subparsers.add_parser(
        "run-experiments",
        help="Run validation, normalization, bridge export, and PyBullet scenarios",
    )
    run_parser.add_argument("--provider", default="file", choices=["file", "mock", "vision"])
    run_parser.add_argument("--image", default=None)
    run_parser.add_argument("--case-id", default=None)
    run_parser.add_argument("--module1-output", default=None)
    run_parser.add_argument("--scenarios", default="all", help="all or comma-separated list")
    run_parser.add_argument("--prompt-variant", default=None)
    run_parser.add_argument("--output-root", default=None)
<<<<<<< HEAD
=======
    run_parser.add_argument("--task-name", default=None,
                            help="Task 폴더 이름 (outputs/<task-name>/). 미지정 시 image 파일명으로 자동.")
>>>>>>> origin/subin/module2c-3-pipeline

    bridge_parser = subparsers.add_parser(
        "export-module2-bridge",
        help="Export scene_resources and module2_common_input_template only",
    )
    bridge_parser.add_argument("--provider", default="file", choices=["file", "mock", "vision"])
    bridge_parser.add_argument("--image", default=None)
    bridge_parser.add_argument("--case-id", default=None)
    bridge_parser.add_argument("--module1-output", default=None)
    bridge_parser.add_argument("--output-root", default=None)
<<<<<<< HEAD
=======
    bridge_parser.add_argument("--task-name", default=None,
                               help="Task 폴더 이름 (outputs/<task-name>/).")
>>>>>>> origin/subin/module2c-3-pipeline

    module2a_parser = subparsers.add_parser(
        "run-module2a",
        help="Generate Module 2-A output from module2_common_input or Module 1 bridge",
    )
    module2a_parser.add_argument("--module2-input", default=None)
    module2a_parser.add_argument("--provider", default="file", choices=["file", "mock", "vision"])
    module2a_parser.add_argument("--image", default=None)
    module2a_parser.add_argument("--case-id", default=None)
    module2a_parser.add_argument("--module1-output", default=None)
    module2a_parser.add_argument("--user-goal", default=None)
    module2a_parser.add_argument("--success-criteria", action="append", default=None)
    module2a_parser.add_argument("--task-notes", action="append", default=None)
    module2a_parser.add_argument("--output-root", default=None)
<<<<<<< HEAD
=======
    module2a_parser.add_argument("--task-name", default=None,
                                 help="Task 폴더 이름 (outputs/<task-name>/).")
>>>>>>> origin/subin/module2c-3-pipeline

    batch_parser = subparsers.add_parser("batch", help="Run batch fixture experiments")
    batch_parser.add_argument("--cases", default="all", help="all or comma-separated case ids")
    batch_parser.add_argument("--provider", default="mock", choices=["mock"])
    batch_parser.add_argument("--output-root", default=None)

    eval_parser = subparsers.add_parser(
        "evaluate-module1-core",
        help="Run repeated Module 1 experiments and compute SESR/APRE/FSC/RSMS",
    )
    eval_parser.add_argument("--cases", default="all", help="all or comma-separated case ids")
    eval_parser.add_argument("--provider", default="mock", choices=["mock"])
    eval_parser.add_argument("--repeats", default=5, type=int)
    eval_parser.add_argument("--prompt-variant", default=None)
    eval_parser.add_argument("--output-root", default=None)

    eval_m2a_parser = subparsers.add_parser(
        "evaluate-module2a-reasoner",
        help="Run repeated Module 2-A reasoning and compute TCRS/RCR/SUR/SSR",
    )
    eval_m2a_parser.add_argument("--cases", default="all", help="all or comma-separated case ids")
    eval_m2a_parser.add_argument("--repeats", default=5, type=int)
    eval_m2a_parser.add_argument("--output-root", default=None)

    eval_m2b_parser = subparsers.add_parser(
        "evaluate-module2b-reasoner",
        help="Run repeated Module 2-B reasoning and compute TBDS/NCR/MHRR",
    )
    eval_m2b_parser.add_argument("--cases", default="all", help="all or comma-separated case ids")
    eval_m2b_parser.add_argument("--repeats", default=5, type=int)
    eval_m2b_parser.add_argument("--variant", default=None)
    eval_m2b_parser.add_argument("--output-root", default=None)

    validate_m2b_parser = subparsers.add_parser(
        "validate-module2b-input",
        help="Validate Module 2-B input bundle (bundle file or split files)",
    )
    validate_m2b_parser.add_argument("--bundle", default=None)
    validate_m2b_parser.add_argument("--module2-common", default=None)
    validate_m2b_parser.add_argument("--module2a-output", default=None)

    run_m2b_parser = subparsers.add_parser(
        "run-module2b",
        help="Run deterministic Module 2-B env-only reasoning",
    )
    run_m2b_parser.add_argument("--provider", default="file", choices=["file", "mock"])
    run_m2b_parser.add_argument("--case-id", default=None)
    run_m2b_parser.add_argument("--bundle", default=None)
    run_m2b_parser.add_argument("--module2-common", default=None)
    run_m2b_parser.add_argument("--module2a-output", default=None)
    run_m2b_parser.add_argument("--variant", default=None)
    run_m2b_parser.add_argument("--output-root", default=None)
<<<<<<< HEAD
=======
    run_m2b_parser.add_argument("--task-name", default=None,
                                help="Task 폴더 이름 (outputs/<task-name>/).")
>>>>>>> origin/subin/module2c-3-pipeline

    batch_m2b_parser = subparsers.add_parser(
        "batch-module2b",
        help="Run Module 2-B over fixture cases with repeatability summary",
    )
    batch_m2b_parser.add_argument("--cases", default="all", help="all or comma-separated case ids")
    batch_m2b_parser.add_argument("--provider", default="mock", choices=["mock"])
    batch_m2b_parser.add_argument("--repeats", default=1, type=int)
    batch_m2b_parser.add_argument("--variant", default=None)
    batch_m2b_parser.add_argument("--output-root", default=None)

    compare_m2b_parser = subparsers.add_parser(
        "compare-module2b-runs",
        help="Compare two Module 2-B run outputs",
    )
    compare_m2b_parser.add_argument("--run-a", required=True)
    compare_m2b_parser.add_argument("--run-b", required=True)
    compare_m2b_parser.add_argument("--output-root", default=None)

    normalized_m2b_parser = subparsers.add_parser(
        "export-module2b-normalized",
        help="Export Layer 2 normalized context only",
    )
    normalized_m2b_parser.add_argument("--provider", default="file", choices=["file", "mock"])
    normalized_m2b_parser.add_argument("--case-id", default=None)
    normalized_m2b_parser.add_argument("--bundle", default=None)
    normalized_m2b_parser.add_argument("--module2-common", default=None)
    normalized_m2b_parser.add_argument("--module2a-output", default=None)
    normalized_m2b_parser.add_argument("--output-root", default=None)
<<<<<<< HEAD
=======
    normalized_m2b_parser.add_argument("--task-name", default=None,
                                        help="Task 폴더 이름 (outputs/<task-name>/).")
>>>>>>> origin/subin/module2c-3-pipeline

    return parser


if __name__ == "__main__":
    raise SystemExit(main())
