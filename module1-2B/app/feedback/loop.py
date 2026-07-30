"""FeedbackRunner — 논문 F1·F2 피드백을 실제로 재호출하는 오케스트레이션 루프.

FeedbackController가 '판정'을 담당한다면, 이 루프는 그 판정을 소비해 상위 모듈을
실제로 재호출한다. 엔진/네트워크에 독립적이도록 스테이지를 콜러블로 주입받는다
(그래서 GPT 호출 없이 mock 스테이지로 단위 테스트 가능하다).

- F1: Module 2-D 결과에 통과 후보가 없으면 컨트롤러가 두 갈래로 분기.
    env_constraint_wipeout → relax_module2(2b 수치 제약 완화) 후 2c→2d 재생성
    other                 → regenerate_2c(후보 재생성) 후 2d 재필터
  재시도 상한(F1) 내에서 반복하고, 각 실패를 §4.5 로그로 남긴다.
- F2: Module 3 조립 검증(8항목)이 실패하면 직전 단계(2c)로 피드백 후 2d→3 재수행.
  재시도 상한(F2) 내 반복.

각 스테이지 콜러블 계약:
    read_module2d_output()          -> dict   (현재 2d 출력)
    regenerate_2c(directive)        -> None    (2c 후보 재생성 + 2d 재필터 실행; 부수효과)
    relax_module2(directive)        -> None    (2b 제약 완화 + 2c→2d 재실행; 부수효과)
    read_module3_output()           -> dict    (현재 3 출력)
    reassemble_via_2c(directive)    -> None    (2c부터 2d→3 재수행; 부수효과)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.feedback.controller import FeedbackController, FeedbackDirective
from app.feedback.log_record import FailureLogRecord

ReadFn = Callable[[], dict[str, Any]]
ReinvokeFn = Callable[[FeedbackDirective], None]


@dataclass(slots=True)
class FeedbackLoopResult:
    stage: str                       # "F1" | "F2"
    resolved: bool                   # 재시도로 해소됐는가
    attempts: int                    # 재호출 횟수
    abandoned: bool                  # 상한 초과로 포기했는가
    records: list[FailureLogRecord] = field(default_factory=list)


def _run_loop(
    *,
    stage: str,
    controller: FeedbackController,
    scenario: str,
    read_output: ReadFn,
    evaluate: Callable[[dict[str, Any], str, int], FeedbackDirective],
    reinvoke: ReinvokeFn,
    log_path: Path | None,
) -> FeedbackLoopResult:
    """단일 피드백 경로(F1 또는 F2)의 판정→재호출→로그 루프."""
    attempt = 0
    pending: list[FailureLogRecord] = []
    reinvokes = 0

    while True:
        output = read_output()
        directive = evaluate(output, scenario, attempt)

        if not directive.triggered:
            # 통과(또는 재시도로 해소). 직전 실패 레코드를 success로 확정.
            if pending:
                pending[-1].result = "success"  # type: ignore[assignment]
            _flush(controller, pending, log_path)
            return FeedbackLoopResult(
                stage=stage, resolved=True, attempts=reinvokes,
                abandoned=False, records=pending,
            )

        if not directive.should_retry:
            # 상한 초과 → 포기.
            if directive.record is not None:
                directive.record.result = "abandoned"  # type: ignore[assignment]
                pending.append(directive.record)
            _flush(controller, pending, log_path)
            return FeedbackLoopResult(
                stage=stage, resolved=False, attempts=reinvokes,
                abandoned=True, records=pending,
            )

        # 재시도 예정: 실패 레코드 적재(결과는 다음 순회에서 확정).
        if directive.record is not None:
            directive.record.result = "failed"  # 갱신 안 되면 실패로 남음
            pending.append(directive.record)
        attempt += 1
        reinvokes += 1
        reinvoke(directive)


def _flush(
    controller: FeedbackController,
    records: list[FailureLogRecord],
    log_path: Path | None,
) -> None:
    from app.feedback.log_record import append_record

    target = log_path or controller.log_path
    if target is None:
        return
    for rec in records:
        if rec.result == "pending":
            rec.result = "failed"  # type: ignore[assignment]
        append_record(rec, Path(target))


class FeedbackRunner:
    """F1(2d 이후)·F2(3 이후) 피드백 루프를 순차 실행."""

    def __init__(
        self,
        controller: FeedbackController,
        scenario: str,
        log_path: Path | None = None,
    ) -> None:
        self.controller = controller
        self.scenario = scenario
        self.log_path = Path(log_path) if log_path else controller.log_path

    def resolve_f1(
        self,
        read_module2d_output: ReadFn,
        reinvoke: ReinvokeFn,
    ) -> FeedbackLoopResult:
        return _run_loop(
            stage="F1", controller=self.controller, scenario=self.scenario,
            read_output=read_module2d_output,
            evaluate=self.controller.evaluate_f1,
            reinvoke=reinvoke, log_path=self.log_path,
        )

    def resolve_f2(
        self,
        read_module3_output: ReadFn,
        reinvoke: ReinvokeFn,
    ) -> FeedbackLoopResult:
        return _run_loop(
            stage="F2", controller=self.controller, scenario=self.scenario,
            read_output=read_module3_output,
            evaluate=self.controller.evaluate_f2,
            reinvoke=reinvoke, log_path=self.log_path,
        )
