"""피드백 연결 계층 — 논문 F1(§3.3)·F2(§3.4) 경로를 실제로 소비/재호출한다.

공개 API:
    FeedbackController, FeedbackDirective, compute_filter_counts
    FailureLogRecord, append_record, read_records
    VerificationItem, ALL_ITEMS, PAPER_NAMED_ITEMS, violated_items
"""

from app.feedback.controller import (
    FeedbackController,
    FeedbackDirective,
    compute_filter_counts,
)
from app.feedback.loop import (
    FeedbackLoopResult,
    FeedbackRunner,
)
from app.feedback.log_record import (
    FailureLogRecord,
    append_record,
    read_records,
)
from app.feedback.verification_items import (
    ALL_ITEMS,
    PAPER_NAMED_ITEMS,
    VerificationItem,
    violated_items,
)

__all__ = [
    "FeedbackController",
    "FeedbackDirective",
    "FeedbackRunner",
    "FeedbackLoopResult",
    "compute_filter_counts",
    "FailureLogRecord",
    "append_record",
    "read_records",
    "VerificationItem",
    "ALL_ITEMS",
    "PAPER_NAMED_ITEMS",
    "violated_items",
]
