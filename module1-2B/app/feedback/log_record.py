"""공통 실패 로그 레코드 — 논문 §4.5 형식.

F1(§3.3)·F2(§3.4) 실패 1건당 한 레코드를 남긴다. 나중에 실패 원인 집계 표로
바로 쓰이도록 스키마를 통일한다. 시나리오는 필드로 구분하며, 시나리오별로 파일을
가르지 않는다(README의 "출력 스키마 통일" 원칙).

레코드 예시(§4.5):
    {
      "scenario": "chain",
      "failed_at": "F1",
      "module": "ToolCompositionGenerator",
      "filter_counts": {"env_constraint": {"in": 12, "out": 0}, ...},
      "branch": "env_constraint_wipeout",
      "action": "relax_request_to_module2",
      "attempt": 1,
      "result": "success"
    }

F2 레코드는 filter_counts/branch 대신 violated_checks 필드를 쓴다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

FailedAt = Literal["F1", "F2"]
RecordResult = Literal["pending", "success", "failed", "abandoned"]


@dataclass(slots=True)
class FailureLogRecord:
    """단일 피드백 실패 이벤트. F1/F2 공통 스키마, 미해당 필드는 None으로 생략."""

    scenario: str
    failed_at: FailedAt
    module: str
    action: str
    attempt: int
    result: RecordResult = "pending"
    # F1 전용
    filter_counts: dict[str, Any] | None = None
    branch: str | None = None
    # F2 전용
    violated_checks: list[str] | None = None
    # 재호출 대상 상위 모듈(예: module2b, module2c). 집계·디버깅 보조.
    target_module: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "scenario": self.scenario,
            "failed_at": self.failed_at,
            "module": self.module,
            "action": self.action,
            "attempt": self.attempt,
            "result": self.result,
        }
        if self.target_module is not None:
            d["target_module"] = self.target_module
        if self.failed_at == "F1":
            d["filter_counts"] = self.filter_counts or {}
            d["branch"] = self.branch
        else:  # F2
            d["violated_checks"] = self.violated_checks or []
        return d


def append_record(record: FailureLogRecord, log_path: Path) -> Path:
    """레코드 한 건을 JSONL 파일에 append. 상위 디렉토리는 자동 생성."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return log_path


def read_records(log_path: Path) -> list[dict[str, Any]]:
    """JSONL 로그를 읽어 dict 리스트로 반환(없으면 빈 리스트)."""
    log_path = Path(log_path)
    if not log_path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out
