"""Module 4 (Tool Assembly Generator) 8항목 조립 검증 — 단일 정의 소스.

논문 §3.4:
    "조립 결과는 충돌, 기능부 노출, 힘 전달 등 8개 항목으로 검증되며,
     실패 시 직전 단계로 피드백이 전달되어 재수행한다."

논문이 본문에 명시한 항목은 3개(충돌 / 기능부 노출 / 힘 전달)이고,
나머지 5개는 코드(FINAL_SYSTEM_PROMPT, Module3OutputValidator)에서 확인해 여기에
문서화한다. 이 파일이 8항목의 유일한 정의처이며, validator·controller·README가
모두 이 목록을 참조한다.
"""

from __future__ import annotations

from enum import Enum


class VerificationItem(str, Enum):
    """조립 검증 8항목. value는 module3 verification.checks[].item 문자열과 일치한다."""

    ALIGNMENT = "alignment"                          # 물체가 task 방향으로 정렬됐는가
    COLLISION = "collision"                          # ★논문 명시: 비의도적 충돌이 없는가
    FUNCTIONAL_END_EXPOSED = "functional_end_exposed"  # ★논문 명시: 기능단(tip/edge) 노출
    HANDLE_REGION_FREE = "handle_region_free"        # 파지 영역 확보
    FORCE_TRANSFER = "force_transfer"                # ★논문 명시: 힘이 도구 끝까지 전달되는가
    WEAK_POINT_MITIGATION = "weak_point_mitigation"  # 이전 단계 취약 항목(약점)의 완화 반영
    SUBGOAL_SUPPORT = "subgoal_support"              # subgoal의 required_atoms 지원
    CONTACT_FEASIBILITY = "contact_feasibility"      # 도구가 타겟과 실제 접촉 가능


# 논문 본문이 직접 이름을 밝힌 3개 항목.
PAPER_NAMED_ITEMS: frozenset[str] = frozenset({
    VerificationItem.COLLISION.value,
    VerificationItem.FUNCTIONAL_END_EXPOSED.value,
    VerificationItem.FORCE_TRANSFER.value,
})

# 8항목 전체(검증 순서와 무관한 집합). validator가 이 집합으로 누락을 점검한다.
ALL_ITEMS: frozenset[str] = frozenset(item.value for item in VerificationItem)

# 사람이 읽기 위한 순서 있는 목록(README/로그 표기용).
ORDERED_ITEMS: tuple[str, ...] = tuple(item.value for item in VerificationItem)

# 각 항목의 한 줄 설명(README 자동화·디버깅용).
ITEM_DESCRIPTIONS: dict[str, str] = {
    VerificationItem.ALIGNMENT.value: "물체가 task 방향으로 올바르게 정렬됐는가",
    VerificationItem.COLLISION.value: "물체 간 비의도적 충돌이 없는가 (논문 명시)",
    VerificationItem.FUNCTIONAL_END_EXPOSED.value: "기능단(tip/edge)이 충분히 노출됐는가 (논문 명시)",
    VerificationItem.HANDLE_REGION_FREE.value: "파지 영역이 확보됐는가",
    VerificationItem.FORCE_TRANSFER.value: "힘이 도구 끝까지 전달 가능한가 (논문 명시)",
    VerificationItem.WEAK_POINT_MITIGATION.value: "이전 단계에서 기록된 취약 항목이 조립에 반영됐는가",
    VerificationItem.SUBGOAL_SUPPORT.value: "각 subgoal의 required_atoms가 지원되는가",
    VerificationItem.CONTACT_FEASIBILITY.value: "도구가 타겟과 실제 접촉 가능한가",
}


def violated_items(checks: list[dict]) -> list[str]:
    """verification.checks에서 result=='fail'인 항목 이름을 순서대로 반환."""
    failed = {
        c.get("item")
        for c in (checks or [])
        if isinstance(c, dict) and c.get("result") == "fail"
    }
    # ORDERED_ITEMS 순서를 유지하되, 8항목에 없는 임의 항목도 뒤에 붙인다.
    ordered = [i for i in ORDERED_ITEMS if i in failed]
    extra = [i for i in failed if i not in ALL_ITEMS and i is not None]
    return ordered + sorted(extra)
