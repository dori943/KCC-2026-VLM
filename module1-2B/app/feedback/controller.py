"""FeedbackController — 논문 F1·F2 피드백 경로의 연결(구현) 계층.

배경(README_inference_pipeline.md 참조):
- 논문은 두 피드백 경로를 서술하지만, 기존 오케스트레이터(scripts/run_pipeline.py)는
  1→2a→2b→2c→2d→3 완전 선형이라 각 모듈이 피드백 '결정'을 JSON에 기록만 하고
  아무도 소비해 상위 모듈을 재호출하지 않았다. 이 컨트롤러가 그 결정을 읽어
  (1) 논문대로 분기를 판정하고 (2) 재호출 대상·행동을 정하고 (3) 공통 로그(§4.5)를
  남기며 (4) 재시도 상한을 강제한다.

F1 — Tool Composition Generator (논문 §3.3):
    "통과 후보가 없을 경우, 환경 제약 일괄 탈락 시 상위 제약 생성 단계에 완화를
     요청하고, 그 외 실패는 후보 재생성 및 직전 추론 단계로 피드백되어 재시도된다."
    ┌ 분기 판정은 module2d의 filter_counts(3단 필터 in/out)에 의존한다.
    ├ 환경 제약 필터(1단)에서 후보가 전부 제거 → env_constraint_wipeout
    │   → action=relax_request_to_module2, target=module2b(수치 제약 완화 재생성)
    └ 그 외 통과 후보 없음 → other
        → action=regenerate_candidates, target=module2c(후보 재생성) + 직전 단계

F2 — Tool Assembly Generator (논문 §3.4):
    "8개 항목으로 검증되며, 실패 시 직전 단계로 피드백이 전달되어 재수행한다."
    → 직전 단계 = Tool Composition Generator(레포 module2c). target=module2c.
    ※ 주의: module3 코드 내부는 역사적으로 feedback_target="module2a"로 적어 왔다
      (코드 주석의 '논문 3.1.4' 번호 기준). 본 작업의 전달 문서(§3.4)는 직전 단계를
      Tool Composition Generator로 규정하므로, 연결 계층에서는 module2c로 라우팅하고
      그 사실을 로그·README에 남긴다. module3 출력 스키마 자체는 건드리지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.feedback.log_record import FailureLogRecord, append_record
from app.feedback.verification_items import violated_items

# module2d failed_stage 문자열 → 논문 3단 필터 단계 매핑.
# (1) 환경 제약 필터 / (2) 접합 구조 필터 / (3) 다차원 점수 평가
_ENV_STAGE = {"environment"}
_JOINT_STAGE = {"assembly", "handle_feasibility", "hook_feasibility"}
_SCORE_STAGE = {
    "geometry", "physics", "commonsense", "task_fit",
    "emergence", "necessity", "role_contribution", "total",
    "subgoal_coverage_critical",
}

_DEFAULT_RETRY_CAPS = {"F1": 2, "F2": 2}


@dataclass(slots=True)
class FeedbackDirective:
    """컨트롤러 판정 결과. 오케스트레이터가 이걸 보고 재호출 여부/대상을 결정한다."""

    triggered: bool                 # 피드백이 필요한 실패인가
    failed_at: str                  # "F1" | "F2"
    should_retry: bool              # 재시도 상한 내인가 (triggered && attempt < cap)
    exhausted: bool                 # 상한 초과 → task_abandoned
    action: str | None              # relax_request_to_module2 | regenerate_candidates | reassemble
    target_module: str | None       # module2b | module2c
    branch: str | None              # F1: env_constraint_wipeout | other
    attempt: int
    record: FailureLogRecord | None  # 공통 로그 레코드(아직 미기록, result=pending)
    detail: dict[str, Any] | None = None  # 디버깅용 부가 정보

    @property
    def needs_feedback(self) -> bool:
        return self.triggered


def compute_filter_counts(
    evaluated: list[dict[str, Any]],
    rejected_for_unknown_names: list[Any] | None = None,
) -> dict[str, Any]:
    """module2d 평가 결과에서 논문 3단 필터의 순차 in/out 카운트를 산출한다.

    순차 필터이므로 앞 단계에서 탈락한 후보는 뒷 단계 in에 포함되지 않는다.
    env_constraint_wipeout 판정(F1 분기)이 이 카운트에 의존한다.
    """
    rejected_for_unknown_names = rejected_for_unknown_names or []
    invalid_out = len(rejected_for_unknown_names)

    env_out = joint_out = score_out = passed = 0
    for c in evaluated:
        stage = c.get("failed_stage")
        if c.get("pass") is True or stage is None:
            passed += 1
        elif stage in _ENV_STAGE:
            env_out += 1
        elif stage in _JOINT_STAGE:
            joint_out += 1
        elif stage in _SCORE_STAGE:
            score_out += 1
        else:
            # 미분류 실패는 보수적으로 점수 단계 탈락으로 본다.
            score_out += 1

    env_in = len(evaluated)                 # scene-name 유효 후보 전부가 1단에 진입
    joint_in = env_in - env_out
    score_in = joint_in - joint_out
    passed_final = score_in - score_out

    total_in = invalid_out + env_in
    return {
        "invalid_scene_name": {"in": total_in, "out": invalid_out},
        "env_constraint": {"in": env_in, "out": env_out},
        "joint_structure": {"in": joint_in, "out": joint_out},
        "score_eval": {"in": score_in, "out": score_out},
        "passed": passed_final,
    }


class FeedbackController:
    """F1·F2 피드백 판정 + 공통 로그 기록 + 재시도 상한 강제."""

    def __init__(
        self,
        retry_caps: dict[str, int] | None = None,
        log_path: Path | None = None,
        policy_path: Path | None = None,
    ) -> None:
        if retry_caps is None and policy_path is not None:
            retry_caps = _load_caps_from_yaml(policy_path)
        self.retry_caps = {**_DEFAULT_RETRY_CAPS, **(retry_caps or {})}
        self.log_path = Path(log_path) if log_path else None

    # ── F1: Tool Composition Generator ────────────────────────────────
    def evaluate_f1(
        self,
        module2d_output: dict[str, Any],
        scenario: str,
        attempt: int = 0,
    ) -> FeedbackDirective:
        evaluated = module2d_output.get("evaluated_candidates", []) or []
        passed_any = any(c.get("pass") is True for c in evaluated)

        filter_counts = module2d_output.get("filter_counts")
        if filter_counts is None:
            filter_counts = compute_filter_counts(evaluated)

        if passed_any:
            return FeedbackDirective(
                triggered=False, failed_at="F1", should_retry=False,
                exhausted=False, action=None, target_module=None,
                branch=None, attempt=attempt, record=None,
                detail={"filter_counts": filter_counts},
            )

        # 통과 후보 없음 → 두 갈래 판정
        env = filter_counts.get("env_constraint", {}) or {}
        env_in = int(env.get("in", 0) or 0)
        env_out = int(env.get("out", 0) or 0)
        env_wipeout = env_in > 0 and env_out == env_in

        if env_wipeout:
            branch = "env_constraint_wipeout"
            action = "relax_request_to_module2"
            target = "module2b"   # 상위 수치 제약 생성 단계
        else:
            branch = "other"
            action = "regenerate_candidates"
            target = "module2c"   # 후보 재생성 + 직전 추론 단계 피드백

        return self._finalize(
            failed_at="F1", scenario=scenario, module="ToolCompositionGenerator",
            action=action, target=target, branch=branch, attempt=attempt,
            filter_counts=filter_counts, violated_checks=None,
        )

    # ── F2: Tool Assembly Generator ───────────────────────────────────
    def evaluate_f2(
        self,
        module3_output: dict[str, Any],
        scenario: str,
        attempt: int = 0,
    ) -> FeedbackDirective:
        verification = module3_output.get("verification", {}) or {}
        checks = verification.get("checks", []) or []
        violated = violated_items(checks)
        is_valid = verification.get("is_valid")
        # is_valid가 명시 안 됐으면 checks로 판정.
        if is_valid is None:
            is_valid = len(violated) == 0

        if is_valid and not violated:
            return FeedbackDirective(
                triggered=False, failed_at="F2", should_retry=False,
                exhausted=False, action=None, target_module=None,
                branch=None, attempt=attempt, record=None,
                detail={"violated_checks": []},
            )

        return self._finalize(
            failed_at="F2", scenario=scenario, module="ToolAssemblyGenerator",
            action="reassemble", target="module2c", branch=None, attempt=attempt,
            filter_counts=None, violated_checks=violated,
        )

    # ── 공통 마무리: 상한 판정 + 레코드 생성 ──────────────────────────
    def _finalize(
        self, *, failed_at: str, scenario: str, module: str, action: str,
        target: str, branch: str | None, attempt: int,
        filter_counts: dict[str, Any] | None, violated_checks: list[str] | None,
    ) -> FeedbackDirective:
        cap = int(self.retry_caps.get(failed_at, _DEFAULT_RETRY_CAPS[failed_at]))
        exhausted = attempt >= cap
        should_retry = not exhausted
        record = FailureLogRecord(
            scenario=scenario, failed_at=failed_at, module=module,
            action=action, attempt=attempt, result="pending",
            filter_counts=filter_counts, branch=branch,
            violated_checks=violated_checks, target_module=target,
        )
        return FeedbackDirective(
            triggered=True, failed_at=failed_at, should_retry=should_retry,
            exhausted=exhausted, action=action, target_module=target,
            branch=branch, attempt=attempt, record=record,
        )

    # ── 로그 확정 기록 ────────────────────────────────────────────────
    def finalize_record(
        self,
        directive: FeedbackDirective,
        result: str,
        log_path: Path | None = None,
    ) -> FailureLogRecord | None:
        """재호출 결과(success/failed/abandoned)를 레코드에 반영하고 JSONL로 남긴다."""
        if directive.record is None:
            return None
        directive.record.result = result  # type: ignore[assignment]
        target = log_path or self.log_path
        if target is not None:
            append_record(directive.record, Path(target))
        return directive.record


def _load_caps_from_yaml(path: Path) -> dict[str, int]:
    from app.utils import load_yaml

    cfg = load_yaml(Path(path))
    caps = (cfg or {}).get("retry_caps", {}) or {}
    return {k: int(v) for k, v in caps.items()}
