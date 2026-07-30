"""FeedbackRunner 재호출 루프 테스트 — GPT 없이 mock 스테이지로 검증.

'컨트롤러+재호출 배선'이 실제로 상위 모듈을 재호출하고(부수효과 카운트),
재시도 상한을 지키며, §4.5 로그를 남기는지 확인한다.
"""

from __future__ import annotations

from pathlib import Path

from app.feedback import FeedbackController, FeedbackRunner, read_records

_FAIL_2D = {
    "evaluated_candidates": [
        {"candidate_id": "t1", "pass": False, "failed_stage": "environment"},
        {"candidate_id": "t2", "pass": False, "failed_stage": "environment"},
    ],
    "filter_counts": {
        "invalid_scene_name": {"in": 2, "out": 0},
        "env_constraint": {"in": 2, "out": 2},
        "joint_structure": {"in": 0, "out": 0},
        "score_eval": {"in": 0, "out": 0},
        "passed": 0,
    },
}
_PASS_2D = {
    "evaluated_candidates": [
        {"candidate_id": "t1", "pass": True, "failed_stage": None},
    ],
    "filter_counts": {"env_constraint": {"in": 1, "out": 0}, "passed": 1},
}


def test_f1_reinvokes_then_resolves(tmp_path: Path) -> None:
    """2d가 fail→fail→pass 순서로 바뀌면, 2회 재호출 후 해소되고 로그 2건."""
    log = tmp_path / "fb.jsonl"
    controller = FeedbackController(retry_caps={"F1": 3}, log_path=log)
    runner = FeedbackRunner(controller, scenario="chain", log_path=log)

    outputs = [_FAIL_2D, _FAIL_2D, _PASS_2D]
    state = {"i": 0, "reinvokes": 0}

    def read():
        return outputs[state["i"]]

    def reinvoke(directive):
        state["reinvokes"] += 1
        state["i"] += 1  # 재호출로 다음 상태 반영

    result = runner.resolve_f1(read, reinvoke)

    assert result.resolved is True
    assert result.abandoned is False
    assert state["reinvokes"] == 2          # 실제로 2회 재호출됨
    records = read_records(log)
    assert len(records) == 2
    assert records[-1]["result"] == "success"
    assert records[0]["branch"] == "env_constraint_wipeout"
    assert records[0]["action"] == "relax_request_to_module2"


def test_f1_abandons_at_cap(tmp_path: Path) -> None:
    """계속 실패하면 상한(2)에서 포기하고 abandoned 로그를 남긴다."""
    log = tmp_path / "fb.jsonl"
    controller = FeedbackController(retry_caps={"F1": 2}, log_path=log)
    runner = FeedbackRunner(controller, scenario="chain", log_path=log)

    state = {"reinvokes": 0}

    def read():
        return _FAIL_2D  # 항상 실패

    def reinvoke(directive):
        state["reinvokes"] += 1

    result = runner.resolve_f1(read, reinvoke)

    assert result.resolved is False
    assert result.abandoned is True
    assert state["reinvokes"] == 2          # cap==2 → 2회만 재호출
    records = read_records(log)
    assert records[-1]["result"] == "abandoned"


def test_f1_no_feedback_when_passing(tmp_path: Path) -> None:
    log = tmp_path / "fb.jsonl"
    controller = FeedbackController(retry_caps={"F1": 2}, log_path=log)
    runner = FeedbackRunner(controller, scenario="chain", log_path=log)

    result = runner.resolve_f1(lambda: _PASS_2D, lambda d: None)
    assert result.resolved is True
    assert result.attempts == 0
    assert read_records(log) == []
