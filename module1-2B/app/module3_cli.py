"""CLI commands for Module 3 pipeline."""

from __future__ import annotations
import os
from pathlib import Path
import click
from app.pipelines.module3_pipeline import run_module3_pipeline


@click.group()
def cli() -> None:
    """Module 3: 도구 조립 위치 및 pose 계산 CLI"""


@cli.command("run-module3")
@click.option("--bundle", "bundle_path", type=click.Path(path_type=Path), default=None)
@click.option("--case-id", default=None)
@click.option("--task-name", "task_name", default=None,
              help="Task 폴더 이름 (outputs/<task-name>/ 밑에 저장). 미지정 시 bundle path에서 자동 추출 (module2b_<task>).")
@click.option("--provider", "provider_name", default="file", type=click.Choice(["file", "mock", "module2d"]))
@click.option("--output-root", type=click.Path(path_type=Path), default=None)
@click.option("--api-key", default=None)
@click.option("--model", default="gpt-4o", show_default=True)
@click.option("--temperature", default=0.1, show_default=True, type=float)
def run_module3(bundle_path, case_id, task_name, provider_name, output_root, api_key, model, temperature):
    """Module 3 조립 pose 계산 실행."""
    from app.module3.providers import FileInputProvider, MockInputProvider, Module2DOutputProvider

    if provider_name == "mock":
        provider = MockInputProvider()
    elif provider_name == "module2d":
        provider = Module2DOutputProvider()
    else:
        provider = FileInputProvider()

    result = run_module3_pipeline(
        provider=provider,
        bundle_path=bundle_path,
        case_id=case_id,
        output_root=output_root,
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        model=model,
        temperature=temperature,
        task_name=task_name,
    )

    click.echo(f"✅ Module 3 완료")
    click.echo(f"   task_name              : {result['summary']['task_name']}")
    click.echo(f"   run_dir                : {result['run_dir']}")
    click.echo(f"   selected_candidate_id  : {result['summary']['selected_candidate_id']}")
    click.echo(f"   assembly_step_count    : {result['summary']['assembly_step_count']}")
    click.echo(f"   is_valid               : {result['summary']['is_valid']}")
    click.echo(f"   need_feedback_to_2a    : {result['summary']['need_feedback_to_module2a']}")
    click.echo(f"   feedback_iteration     : {result['summary']['feedback_iteration']}")
    click.echo(f"   task_abandoned         : {result['summary']['task_abandoned']}")
    click.echo(f"   tokens                 : {result['summary']['prompt_tokens']} + {result['summary']['completion_tokens']}")


if __name__ == "__main__":
    cli()
