"""CLI commands for Module 2-C pipeline."""

from __future__ import annotations
import os
from pathlib import Path
import click
from app.pipelines.module2c_pipeline import run_module2c_pipeline


@click.group()
def cli() -> None:
    """Module 2-C: 도구 조합 후보 생성 CLI"""


@cli.command("run-module2c")
@click.option("--bundle", "bundle_path", type=click.Path(path_type=Path), default=None)
@click.option("--case-id", default=None)
@click.option("--provider", "provider_name", default="file", type=click.Choice(["file", "mock", "module2b"]))
@click.option("--output-root", type=click.Path(path_type=Path), default=None)
@click.option("--api-key", default=None)
@click.option("--model", default="gpt-4o", show_default=True)
@click.option("--temperature", default=0.3, show_default=True, type=float)
# ↓ Material Reasoner용 옵션 추가
@click.option("--image", "image_path", type=click.Path(path_type=Path), default=None, help="타겟 물체 이미지 경로 (Material Reasoner용)")
@click.option("--target-name", default="business card", show_default=True, help="타겟 물체 이름")
@click.option("--task", "task_text", default=None, help="task 설명 (없으면 2-B output에서 자동 추출)")
def run_module2c(bundle_path, case_id, provider_name, output_root, api_key, model, temperature,
                 image_path, target_name, task_text):
    """Module 2-C 후보 생성 실행."""
    from app.module2c.providers import FileInputProvider, MockInputProvider, Module2BOutputProvider

    if provider_name == "mock":
        provider = MockInputProvider()
    elif provider_name == "module2b":
        provider = Module2BOutputProvider(
            image_path=image_path,
            target_name=target_name,
            task=task_text,
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        )
    else:
        provider = FileInputProvider()

    result = run_module2c_pipeline(
        provider=provider,
        bundle_path=bundle_path,
        case_id=case_id,
        output_root=output_root,
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        model=model,
        temperature=temperature,
    )

    click.echo(f"✅ Module 2-C 완료")
    click.echo(f"   run_dir        : {result['run_dir']}")
    click.echo(f"   candidate_count: {result['summary']['candidate_count']}")
    click.echo(f"   input_valid    : {result['summary']['input_valid']}")
    click.echo(f"   output_valid   : {result['summary']['output_valid']}")
    click.echo(f"   tokens         : {result['summary']['prompt_tokens']} + {result['summary']['completion_tokens']}")


if __name__ == "__main__":
    cli()