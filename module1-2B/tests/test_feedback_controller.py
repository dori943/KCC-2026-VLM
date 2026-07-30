"""F1·F2 피드백 연결 계층 테스트.

종료 조건 ① 검증: mock 실패 입력을 넣었을 때 논문 서술대로 분기하고 로그를 남긴다.
- F1: env_constraint_wipeout(→module2b 완화 요청) vs other(→module2c 후보 재생성)
- F2: 8항목 검증 실패 → module2c 재수행, violated_checks 기록
- 재시도 상한(구현 기본값 2) 강제 → 초과 시 exhausted/abandoned
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.feedback import (
    ALL_ITEMS,
    PAPER_NAMED_ITEMS,
    FeedbackController,
    compute_filter_counts,
    read_records,
)
from app.utils import load_json

CASES = Path(__file__).resolve().parents[1] / "fixtures" / "feedback_cases"


@pytest.fixture
def controller() -> FeedbackController:
    return FeedbackController(retry_caps={"F1": 2, "F2": 2})


# ── F1 ────────────────────────────────────────────────────────────────
def test_f1_env_constraint_wipeout(controller: FeedbackController) -> None:
    out = load_json(CASES / "f1_env_wipeout" / "module2d_output.json")
    d = controller.evaluate_f1(out, scenario="chain", attempt=0)

    assert d.triggered is True
    assert d.branch == "env_constraint_wipeout"
    assert d.action == "relax_request_to_module2"
    assert d.target_module == "module2b"
    assert d.should_retry is True
    assert d.exhausted is False
    # 로그 레코드(§4.5) 형태 확인
    rec = d.record.to_dict()
    assert rec["failed_at"] == "F1"
    assert rec["scenario"] == "chain"
    assert rec["module"] == "ToolCompositionGenerator"
    assert rec["branch"] == "env_constraint_wipeout"
    assert rec["filter_counts"]["env_constraint"]["in"] == rec["filter_counts"]["env_constraint"]["out"]


def test_f1_other_branch(controller: FeedbackController) -> None:
    out = load_json(CASES / "f1_other" / "module2d_output.json")
    d = controller.evaluate_f1(out, scenario="balloon", attempt=0)

    assert d.triggered is True
    assert d.branch == "other"
    assert d.action == "regenerate_candidates"
    assert d.target_module == "module2c"


def test_f1_no_feedback_when_a_candidate_passes(controller: FeedbackController) -> None:
    out = load_json(CASES / "f1_other" / "module2d_output.json")
    out["evaluated_candidates"][1]["pass"] = True  # 통과 후보 1개 주입
    d = controller.evaluate_f1(out, scenario="balloon", attempt=0)
    assert d.triggered is False
    assert d.record is None


def test_f1_retry_cap_exhausted(controller: FeedbackController) -> None:
    out = load_json(CASES / "f1_env_wipeout" / "module2d_output.json")
    d = controller.evaluate_f1(out, scenario="chain", attempt=2)  # cap == 2
    assert d.triggered is True
    assert d.should_retry is False
    assert d.exhausted is True


# ── filter_counts 유도 정합성 ───────────────────────────────────────────
def test_compute_filter_counts_matches_declared() -> None:
    out = load_json(CASES / "f1_other" / "module2d_output.json")
    derived = compute_filter_counts(out["evaluated_candidates"])
    declared = out["filter_counts"]
    for stage in ("env_constraint", "joint_structure", "score_eval"):
        assert derived[stage] == declared[stage], stage
    assert derived["passed"] == declared["passed"]


# ── F2 ────────────────────────────────────────────────────────────────
def test_f2_check_fail(controller: FeedbackController) -> None:
    out = load_json(CASES / "f2_check_fail" / "module3_output.json")
    d = controller.evaluate_f2(out, scenario="pet", attempt=0)

    assert d.triggered is True
    assert d.failed_at == "F2"
    assert d.action == "reassemble"
    assert d.target_module == "module2c"  # 직전 단계 = Tool Composition Generator
    rec = d.record.to_dict()
    assert rec["module"] == "ToolAssemblyGenerator"
    assert "violated_checks" in rec
    assert "force_transfer" in rec["violated_checks"]
    assert "functional_end_exposed" in rec["violated_checks"]
    # 순서: ORDERED_ITEMS 기준 functional_end_exposed < force_transfer
    assert rec["violated_checks"] == ["functional_end_exposed", "force_transfer"]


def test_f2_valid_no_feedback(controller: FeedbackController) -> None:
    out = load_json(CASES / "f2_check_fail" / "module3_output.json")
    for c in out["verification"]["checks"]:
        c["result"] = "pass"
    out["verification"]["is_valid"] = True
    d = controller.evaluate_f2(out, scenario="pet", attempt=0)
    assert d.triggered is False


def test_f2_retry_cap_exhausted(controller: FeedbackController) -> None:
    out = load_json(CASES / "f2_check_fail" / "module3_output.json")
    d = controller.evaluate_f2(out, scenario="pet", attempt=2)
    assert d.exhausted is True
    assert d.should_retry is False


# ── 로그 기록(JSONL) ────────────────────────────────────────────────────
def test_finalize_record_writes_jsonl(tmp_path: Path, controller: FeedbackController) -> None:
    log_path = tmp_path / "feedback_failures.jsonl"
    out = load_json(CASES / "f1_env_wipeout" / "module2d_output.json")
    d = controller.evaluate_f1(out, scenario="chain", attempt=0)
    controller.finalize_record(d, result="success", log_path=log_path)

    records = read_records(log_path)
    assert len(records) == 1
    assert records[0]["result"] == "success"
    assert records[0]["failed_at"] == "F1"


# ── 8항목 정의 단일 소스 ────────────────────────────────────────────────
def test_verification_items_single_source() -> None:
    assert len(ALL_ITEMS) == 8
    # 논문 본문 명시 3항목이 포함돼야 한다.
    assert PAPER_NAMED_ITEMS <= ALL_ITEMS
    assert {"collision", "functional_end_exposed", "force_transfer"} == set(PAPER_NAMED_ITEMS)


def test_module3_validator_uses_single_source() -> None:
    from app.module3.validators import Module3OutputValidator

    assert Module3OutputValidator.REQUIRED_CHECK_ITEMS == set(ALL_ITEMS)
