"""CLI commands for Module 2-D pipeline."""

from __future__ import annotations
import os
from pathlib import Path
import click
from app.pipelines.module2d_pipeline import run_module2d_pipeline


@click.group()
def cli() -> None:
    """Module 2-D: 도구 조합 후보 사전 필터링 CLI"""


@cli.command("run-module2d")
@click.option("--bundle", "bundle_path", type=click.Path(path_type=Path), default=None)
@click.option("--case-id", default=None)
@click.option("--provider", "provider_name", default="file", type=click.Choice(["file", "mock", "module2c"]))
@click.option("--output-root", type=click.Path(path_type=Path), default=None)
@click.option("--api-key", default=None)
@click.option("--model", default="gpt-4o", show_default=True)
@click.option("--temperature", default=0.2, show_default=True, type=float)
def run_module2d(bundle_path, case_id, provider_name, output_root, api_key, model, temperature):
    """Module 2-D 후보 필터링 실행."""
    from app.module2d.providers import FileInputProvider, MockInputProvider, Module2COutputProvider

    if provider_name == "mock":
        provider = MockInputProvider()
    elif provider_name == "module2c":
        provider = Module2COutputProvider()
    else:
        provider = FileInputProvider()

    result = run_module2d_pipeline(
        provider=provider,
        bundle_path=bundle_path,
        case_id=case_id,
        output_root=output_root,
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        model=model,
        temperature=temperature,
    )

    click.echo(f"✅ Module 2-D 완료")
    click.echo(f"   run_dir              : {result['run_dir']}")
    click.echo(f"   evaluated_count      : {result['summary']['evaluated_count']}")
    click.echo(f"   pass_count           : {result['summary']['pass_count']}")
    click.echo(f"   selected_candidate_id: {result['summary']['selected_candidate_id']}")
    click.echo(f"   need_feedback        : {result['summary']['need_feedback_to_module2a']}")
    click.echo(f"   tokens               : {result['summary']['prompt_tokens']} + {result['summary']['completion_tokens']}")


if __name__ == "__main__":
    cli()
