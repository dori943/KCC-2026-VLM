"""End-to-end orchestrator: Module 1 → 2a → 2b → 2c → 2d → 3.

사용 예:
  # preset 방식 (configs/task_presets.yaml 기반)
  python scripts/run_pipeline.py --preset task2

  # 개별 옵션 방식
  python scripts/run_pipeline.py \
      --image cases/images/deep_hole_reach.jpg \
      --target-name "target object" \
      --task-description "깊은 구멍 바닥에서 꺼내기"

  # 중간부터 (이미 module2b까지 돈 상태면)
  python scripts/run_pipeline.py --preset task2 \
      --start-from 2c \
      --module2b-dir outputs/deep_hole_reach/module2b_.../

옵션:
  --preset           task_presets.yaml의 task id (task1~task5)
  --image            scene 이미지 경로 (preset 미사용 시 필수)
  --task-name        output 폴더 이름 (미지정 시 image stem)
  --target-name      타겟 물체 이름 (module2c)
  --task-description task 설명 (module2c)
  --provider         module1 provider (vision | mock | file)
  --model            LLM 모델 (default: gpt-4o)
  --start-from       시작 단계 (1 | 2a | 2b | 2c | 2d | 3)
  --stop-at          마지막 단계 (1 | 2a | 2b | 2c | 2d | 3)
  --module1-dir / --module2a-dir / --module2b-dir / --module2c-dir / --module2d-dir
                     --start-from이 2a 이후면 이전 단계 결과 dir 전달
  --api-key          OpenAI API key (env OPENAI_API_KEY도 가능)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.pipelines.module2b_pipeline import run_module2b_pipeline
from app.pipelines.module2c_pipeline import run_module2c_pipeline
from app.pipelines.module2d_pipeline import run_module2d_pipeline
from app.pipelines.module3_pipeline import run_module3_pipeline
from app.runners.module1_runner import run_module1_pipeline
from app.runners.module2a_runner import run_module2a_pipeline
from app.utils import load_yaml


STAGES = ["1", "2a", "2b", "2c", "2d", "3"]


def _build_provider(name: str):
    """Module 1 provider 빌드."""
    if name == "vision":
        from app.providers.vision_provider import VisionProvider
        return VisionProvider()
    if name == "mock":
        from app.providers.mock_provider import MockProvider
        return MockProvider()
    from app.providers.file_provider import FileProvider
    return FileProvider()


def _load_preset(preset_id: str) -> dict[str, Any]:
    cfg = load_yaml(PROJECT_ROOT / "configs" / "task_presets.yaml")
    tasks = cfg.get("tasks", {})
    if preset_id not in tasks:
        raise ValueError(
            f"preset '{preset_id}' 없음. 사용 가능: {list(tasks.keys())}"
        )
    return tasks[preset_id]


def _stage_index(stage: str) -> int:
    try:
        return STAGES.index(stage)
    except ValueError:
        raise ValueError(f"stage '{stage}'는 유효하지 않음. 허용: {STAGES}")


def _banner(step: int, total: int, label: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  [{step}/{total}] {label}")
    print(f"{'=' * 70}")


def _elapsed(start: float) -> str:
    sec = time.time() - start
    return f"{sec:.1f}s"


def run_full_pipeline(
    image_path: Path | None,
    task_name: str,
    target_name: str,
    task_description: str,
    provider_name: str,
    model: str,
    api_key: str,
    start_from: str,
    stop_at: str,
    preloaded_dirs: dict[str, Path],
) -> dict[str, Path]:
    """모듈 1~3 순차 실행. preloaded_dirs는 --start-from이 2a 이상일 때 필요."""
    start_idx = _stage_index(start_from)
    stop_idx  = _stage_index(stop_at)
    if start_idx > stop_idx:
        raise ValueError(f"start-from({start_from}) > stop-at({stop_at}) 안 됨.")

    total = stop_idx - start_idx + 1
    step = 0
    dirs: dict[str, Path] = dict(preloaded_dirs)

    # ── Module 1 ──
    if start_idx <= 0 <= stop_idx:
        step += 1
        _banner(step, total, "Module 1 — Scene Resource Parser")
        if image_path is None:
            raise ValueError("Module 1 실행에는 --image 필요.")
        t0 = time.time()
        m1 = run_module1_pipeline(
            provider=_build_provider(provider_name),
            image_path=image_path,
            case_id=None,
            task_name=task_name,
        )
        dirs["1"] = Path(m1["run_dir"])
        print(f"  → run_dir: {dirs['1']}  ({_elapsed(t0)})")

    # ── Module 2a ──
    if start_idx <= 1 <= stop_idx:
        step += 1
        _banner(step, total, "Module 2a — Task-to-Function Reasoner")
        m1_dir = dirs.get("1") or preloaded_dirs.get("1")
        if m1_dir is None:
            raise ValueError("Module 2a는 --module1-dir 필요 (또는 module 1부터 실행).")
        common_input = m1_dir / "module2_common_input_template.json"
        if not common_input.exists():
            raise FileNotFoundError(f"{common_input} 없음.")
        t0 = time.time()
        # task_description을 user_goal로 주입 — rule-based intent 추론에 필수
        m2a = run_module2a_pipeline(
            module2_input_path=common_input,
            task_name=task_name,
            user_goal=task_description or None,
        )
        dirs["2a"] = Path(m2a["run_dir"])
        print(f"  → run_dir: {dirs['2a']}  ({_elapsed(t0)})")

    # ── Module 2b ──
    if start_idx <= 2 <= stop_idx:
        step += 1
        _banner(step, total, "Module 2b — Environment Constraint Generator")
        m2a_dir = dirs.get("2a") or preloaded_dirs.get("2a")
        if m2a_dir is None:
            raise ValueError("Module 2b는 --module2a-dir 필요.")
        m2a_output = m2a_dir / "module2a_output.json"
        if not m2a_output.exists():
            raise FileNotFoundError(f"{m2a_output} 없음.")
        # module2_common은 module2a 결과에서 (있으면) 또는 module1 결과에서 가져옴
        m2_common = m2a_dir / "module2_common_input.json"
        if not m2_common.exists():
            m1_dir = dirs.get("1") or preloaded_dirs.get("1")
            if m1_dir is None:
                raise FileNotFoundError(
                    f"module2_common_input.json 없음 — {m2a_dir} 또는 module1 run_dir 필요."
                )
            m2_common = m1_dir / "module2_common_input_template.json"
            if not m2_common.exists():
                raise FileNotFoundError(f"{m2_common} 없음.")
        t0 = time.time()
        m2b = run_module2b_pipeline(
            module2_common_path=m2_common,
            module2a_output_path=m2a_output,
            task_name=task_name,
        )
        dirs["2b"] = Path(m2b["run_dir"])
        print(f"  → run_dir: {dirs['2b']}  ({_elapsed(t0)})")

    # ── Module 2c ──
    if start_idx <= 3 <= stop_idx:
        step += 1
        _banner(step, total, "Module 2c — Tool Candidate Generator")
        m2b_dir = dirs.get("2b") or preloaded_dirs.get("2b")
        if m2b_dir is None:
            raise ValueError("Module 2c는 --module2b-dir 필요.")
        from app.module2c.providers import Module2BOutputProvider
        provider_2c = Module2BOutputProvider(
            image_path=image_path,
            target_name=target_name,
            task=task_description,
            api_key=api_key,
        )
        t0 = time.time()
        m2c = run_module2c_pipeline(
            provider=provider_2c,
            bundle_path=m2b_dir,
            api_key=api_key,
            model=model,
            task_name=task_name,
        )
        dirs["2c"] = Path(m2c["run_dir"])
        print(f"  → run_dir: {dirs['2c']}  (candidates={m2c['summary']['candidate_count']}, {_elapsed(t0)})")

    # ── Module 2d ──
    if start_idx <= 4 <= stop_idx:
        step += 1
        _banner(step, total, "Module 2d — Candidate Filter")
        m2c_dir = dirs.get("2c") or preloaded_dirs.get("2c")
        if m2c_dir is None:
            raise ValueError("Module 2d는 --module2c-dir 필요.")
        from app.module2d.providers import Module2COutputProvider
        t0 = time.time()
        m2d = run_module2d_pipeline(
            provider=Module2COutputProvider(),
            bundle_path=m2c_dir,
            api_key=api_key,
            model=model,
            task_name=task_name,
        )
        dirs["2d"] = Path(m2d["run_dir"])
        s = m2d["summary"]
        print(f"  → run_dir: {dirs['2d']}  (selected={s['selected_candidate_id']}, {_elapsed(t0)})")

    # ── Module 3 ──
    if start_idx <= 5 <= stop_idx:
        step += 1
        _banner(step, total, "Module 3 — Tool Assembly Generator")
        m2d_dir = dirs.get("2d") or preloaded_dirs.get("2d")
        if m2d_dir is None:
            raise ValueError("Module 3는 --module2d-dir 필요.")
        from app.module3.providers import Module2DOutputProvider
        t0 = time.time()
        m3 = run_module3_pipeline(
            provider=Module2DOutputProvider(),
            bundle_path=m2d_dir,
            api_key=api_key,
            model=model,
            task_name=task_name,
        )
        dirs["3"] = Path(m3["run_dir"])
        s = m3["summary"]
        print(f"  → run_dir: {dirs['3']}")
        print(f"     is_valid={s['is_valid']}, "
              f"need_feedback_to_2a={s['need_feedback_to_module2a']}, "
              f"iteration={s['feedback_iteration']}, "
              f"abandoned={s['task_abandoned']}  ({_elapsed(t0)})")

    return dirs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default=None, help="configs/task_presets.yaml의 task id")
    parser.add_argument("--image", default=None, help="scene 이미지 경로")
    parser.add_argument("--task-name", default=None, help="output 폴더 이름")
    parser.add_argument("--target-name", default=None, help="타겟 물체 이름 (module2c)")
    parser.add_argument("--task-description", default=None, help="task 설명")
    parser.add_argument("--provider", default="vision", choices=["vision", "mock", "file"])
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--start-from", default="1", choices=STAGES)
    parser.add_argument("--stop-at", default="3", choices=STAGES)
    parser.add_argument("--module1-dir", default=None)
    parser.add_argument("--module2a-dir", default=None)
    parser.add_argument("--module2b-dir", default=None)
    parser.add_argument("--module2c-dir", default=None)
    parser.add_argument("--module2d-dir", default=None)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    # preset 병합 (명시 옵션이 우선)
    preset: dict[str, Any] = {}
    if args.preset:
        preset = _load_preset(args.preset)

    image_path       = Path(args.image) if args.image else (Path(preset["image"]) if preset.get("image") else None)
    task_name        = args.task_name       or preset.get("name")
    target_name      = args.target_name     or preset.get("target_name", "target object")
    task_description = args.task_description or preset.get("description", "")

    if image_path and not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path

    if task_name is None and image_path is not None:
        task_name = image_path.stem
    if task_name is None:
        parser.error("--task-name 또는 --preset 또는 --image 중 하나 필요.")

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        parser.error("OPENAI_API_KEY 환경변수 또는 --api-key 필요.")

    preloaded_dirs: dict[str, Path] = {}
    for stage, arg in [("1", args.module1_dir), ("2a", args.module2a_dir),
                       ("2b", args.module2b_dir), ("2c", args.module2c_dir),
                       ("2d", args.module2d_dir)]:
        if arg:
            preloaded_dirs[stage] = Path(arg)

    if image_path is not None and not image_path.exists():
        print(f"⚠  이미지 없음: {image_path} — start-from=1이면 실패합니다.", file=sys.stderr)

    print(f"Task:         {task_name}")
    print(f"Image:        {image_path}")
    print(f"Target:       {target_name}")
    print(f"Description:  {task_description[:60]}{'...' if len(task_description) > 60 else ''}")
    print(f"Stages:       {args.start_from} → {args.stop_at}")
    print(f"Model:        {args.model}")

    total_t0 = time.time()
    try:
        dirs = run_full_pipeline(
            image_path=image_path,
            task_name=task_name,
            target_name=target_name,
            task_description=task_description,
            provider_name=args.provider,
            model=args.model,
            api_key=api_key,
            start_from=args.start_from,
            stop_at=args.stop_at,
            preloaded_dirs=preloaded_dirs,
        )
    except Exception as e:
        print(f"\n❌ 파이프라인 실패: {e}", file=sys.stderr)
        return 1

    print(f"\n{'=' * 70}")
    print(f"  ✅ 파이프라인 완료 (총 {_elapsed(total_t0)})")
    print(f"{'=' * 70}")
    for stage, path in sorted(dirs.items(), key=lambda x: STAGES.index(x[0])):
        print(f"  [{stage}] {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
