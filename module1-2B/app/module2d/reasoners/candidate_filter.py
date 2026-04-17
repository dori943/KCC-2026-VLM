"""Candidate filter for Module 2-D using GPT-4o."""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from app.module2d.models import (
    EvaluatedCandidate, FeedbackDecision, Module2DInput,
    RepairAnalysis, StageScores, WeakPoint,
)

SYSTEM_PROMPT = """너는 Module 2-D: 도구 조합 후보 필터링 및 정량 평가기이다.

목표:
입력으로 받은 도구 조합 후보들을 3단계로 평가하여
최종 후보를 선정한다.

# 입력 파일 및 활용 방법

[task]
수행해야 할 작업 설명이다.

[tool_constraints] ← module2b_output.json

환경 구조 분석 결과이다.
derived_constraints의 각 항목을 1단계 하드 필터로 사용한다.

- global_constraints: 모든 후보가 만족해야 할 전역 제약
- derived_constraints: 환경 기반 수치 제약 (하드 필터 기준)
- numeric_estimates: 환경 수치 정보
- subgoal_constraints: subgoal별 기능 요건
  - required_atoms: 반드시 충족해야 할 기능 원자
  - required_interaction_primitives: subgoal 수행에 필요한 동작 목록
    → 도구 조합이 해당 primitive를 수행 가능한지 상식/맥락 평가에서 활용하라

interaction primitive 목록:
reach, approach, orient, probe, contact, push, pull, drag, slide, lift, lower,
hold, release, place, align, guide, press, tap, squeeze, grasp, pinch, clamp,
hook, wedge, pry, insert, thread, twist, rotate, separate, scrape, sweep, scoop,
contain, pour, funnel, cover, block, support, brace, stabilize, stack, hang,
roll, spin, fasten, clip, anchor, bridge

[candidate_tools] ← module2c_output.json

평가 대상 도구 조합 후보 목록이다.

[scene_objects] ← scene_info.json

각 물체의 기하 정보이다.
- name, aabb_min, aabb_max, center_world

[object_physical_properties] ← raw_module1_output.json

각 물체의 물리속성 정보이다.
- name, shape_category, surface_friction, rigidity, inferred_functions
- connection_modes: 물체가 다른 물체와 결합 가능한 방식 및 점수
  예: insert_into_gap, clip_onto_edge, mate_flat_face, wrap_around 등
  → 접합 가능성 하드 필터에서 결합 방식 판단에 활용하라

# 평가 절차

[1단계] 환경 제약 하드 필터 ← module2b_output.json derived_constraints

derived_constraints의 각 항목에 대해 위반 여부를 판단하라.
하나라도 위반하면 즉시 탈락시키고 이후 평가를 수행하지 마라.

판단 기준:
- max_tip_thickness: 삽입 물체 두께 > 제약값 → 탈락
- min_effective_reach: 물체 길이 < 제약값 → 탈락
- max_required_entry_angle_deg: 진입 각도 > 제약값 → 탈락
- 그 외 derived_constraints 항목도 동일하게 적용

[2단계] 접합 가능성 하드 필터

아래 조건 중 하나라도 해당하면 즉시 탈락시키고 이후 평가를 수행하지 마라.

- 두 물체의 connection_modes 중 호환 가능한 결합 방식이 존재하지 않는 경우
- 두 물체의 크기/형태가 물리적으로 결합 불가능한 경우
- 유연한 물체가 강성 구조 역할을 수행해야 하는 경우
  예: 천, 고무줄 → 지렛대/삽입 역할
- 결합 후 로봇이 파지할 수 있는 영역이 사라지는 경우
- 결합 후 기능 수행 부위가 막히는 경우

[3단계] 정량 평가 (1~2단계 통과 후보만)

각 항목은 Yes=1점, No=0점으로 평가하라.
score = Yes 개수 / 전체 항목 수 × 5

[3-1] 기하 평가

1. 두 물체 사이에 물리적 접촉 가능한 면이 존재하는가?
2. 두 물체의 크기가 결합 가능한 수준인가?
3. 결합 후 기능 수행 부위(tip/edge)가 task 수행을 위해 충분히 노출되는가?
4. 두 물체의 주축이 task 수행 방향에 맞게 정렬 가능한가?
5. 결합했을 때 전체 도구의 길이가 task 수행에 충분한가?
6. 두 물체의 접촉 면적이 task 수행 중 안정적인 결합을 유지하기에 충분한가?

geometry_score = Yes 개수 / 6 × 5

[3-2] 물리 평가

1. 무게 중심이 task를 수행할 수 있을 만큼 안정적인가?
2. 회전 모멘트가 task 수행을 방해하지 않을 수준인가?
3. 구조가 task 수행 중 휘거나 불안정해지지 않는가?
4. task 수행에 필요한 힘이 도구 끝까지 전달 가능한가?
5. 접합부가 task 수행 중 분리되지 않을 만큼 안정적인가?
6. task 수행에 필요한 마찰력이 충분한가?

physics_score = Yes 개수 / 6 × 5

[3-3] 상식/맥락 평가

1. 공간 제약 내에서 도구 조합을 사용할 수 있는가?
2. 로봇이 실제로 조작 가능한 구조인가?
3. 각 subgoal의 required_atoms 및 required_interaction_primitives를 도구 조합 내 물체가 실제로 수행 가능한가?
4. 도구 조합이 타겟 물체를 손상시키지 않고 task를 수행할 수 있는가?
5. 도구 조합이 주변 환경을 손상시키지 않는가?

commonsense_score = Yes 개수 / 5 × 5

[3-4] 총점 계산

total_score = (geometry_score + physics_score + commonsense_score) / 3

[3-5] pass 판정

아래 조건을 모두 만족해야 pass=true:
- total_score >= 3.5
- geometry_score >= 1.5
- physics_score >= 1.5
- commonsense_score >= 1.5

하나라도 미달 시 pass=false

[4단계] 보완 가능성 분석 (pass=false 후보만)

geometry/physics/commonsense 점수를 기반으로 분류하라:

- 1개 항목만 < 1.5 → local_fix
  해당 항목과 보완 방법을 제시하라

- 2개 이상 항목이 < 1.5 → global_redesign
  구조 전체 재설계 방향을 제시하라

# 최종 선정

pass=true 후보 중 total_score가 가장 높은 후보를 선정하라.
total_score가 동점인 경우 used_objects 수가 많은 후보를 선정하라.
pass=true 후보가 없으면 null을 반환하라.

# 피드백 판단

모든 후보를 종합하여 아래 조건을 판단하라:

[피드백 필요 조건]
- pass 후보가 0개인 경우
- 동일한 constraint에서 3개 이상 후보가 반복적으로 탈락하는 경우
- 환경 하드 필터에서 모든 후보가 탈락하는 경우
- 특정 required_atoms가 어떤 후보에서도 충족되지 않는 경우

[피드백 내용]
- feedback_target: 피드백을 보낼 단계 ("module2a" | "module2c" | null)
  - required_atoms 관련 실패 → "module2a"
  - 후보 조합 품질 문제 → "module2c"
  - 피드백 불필요 → null
- dominant_failure_pattern: 반복적으로 실패한 제약/항목
- suggested_relaxations: 완화 가능한 제약 제안
  예: required_atoms 중 일부를 preferred_atoms로 변경
  예: 환경 수치 제약의 범위 완화

# 출력 규칙

1. JSON만 출력하라.
2. JSON 바깥의 설명문은 출력하지 마라.
3. 1~2단계에서 탈락한 후보는 stage_scores를 null로 출력하라.
4. 각 평가 항목의 Yes/No 결과를 checklist에 기록하라.

# 출력 형식

{
  "evaluated_candidates": [
    {
      "candidate_id": "...",
      "environment_filter": {
        "pass": true,
        "violated_constraints": []
      },
      "assembly_filter": {
        "pass": true,
        "violated_reasons": []
      },
      "checklist": {
        "geometry": [
          {"item": "...", "result": true}
        ],
        "physics": [
          {"item": "...", "result": true}
        ],
        "commonsense": [
          {"item": "...", "result": true}
        ]
      },
      "stage_scores": {
        "geometry": 0.0,
        "physics": 0.0,
        "commonsense": 0.0
      },
      "total_score": 0.0,
      "pass": true,
      "failed_stage": "environment | assembly | geometry | physics | commonsense | null",
      "weak_points": [
        {
          "stage": "...",
          "item": "...",
          "score": 0.0,
          "reason": "..."
        }
      ],
      "repair_analysis": {
        "repair_type": "local_fix | global_redesign",
        "target_issue": "...",
        "reason": "...",
        "repair_strategy": ["..."]
      }
    }
  ],
  "selected_candidate_id": "string | null",
  "feedback_decision": {
    "need_feedback": false,
    "feedback_target": "module2a | module2c | null",
    "reason": "...",
    "dominant_failure_pattern": ["..."],
    "suggested_relaxations": ["..."]
  }
}"""


def filter_candidates(
    input_data: Module2DInput,
    api_key: str | None = None,
    model: str = "gpt-4o",
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> tuple[list[EvaluatedCandidate], str | None, FeedbackDecision, dict[str, Any]]:
    """Call GPT-4o to evaluate and filter candidates.

    Returns:
        (evaluated_candidates, selected_candidate_id, feedback_decision, trace)
    """
    client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    user_payload = {
        "task": input_data.task,
        "tool_constraints": input_data.tool_constraints,
        "candidate_tools": input_data.candidate_tools,
        "scene_objects": input_data.scene_objects,
        "object_physical_properties": input_data.object_physical_properties,
    }

    user_message = (
        "다음 입력을 분석하여 도구 조합 후보를 평가하고 필터링하라.\n\n"
        + json.dumps(user_payload, ensure_ascii=False, indent=2)
    )

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
    )

    raw_text = response.choices[0].message.content or ""
    usage = response.usage

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Module 2-D: GPT 응답 JSON 파싱 실패: {e}\n응답: {raw_text[:300]}")

    evaluated = _parse_evaluated_candidates(parsed.get("evaluated_candidates", []))
    selected_id = parsed.get("selected_candidate_id")
    feedback = _parse_feedback(parsed.get("feedback_decision", {}))

    trace = {
        "model": model,
        "temperature": temperature,
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "evaluated_count": len(evaluated),
        "selected_candidate_id": selected_id,
        "raw_response_preview": raw_text[:500],
    }

    return evaluated, selected_id, feedback, trace


def _parse_evaluated_candidates(raw: list[dict[str, Any]]) -> list[EvaluatedCandidate]:
    result = []
    for item in raw:
        scores = item.get("stage_scores", {})
        stage_scores = StageScores(
            geometry=float(scores.get("geometry", 0.0)),
            physics=float(scores.get("physics", 0.0)),
            commonsense=float(scores.get("commonsense", 0.0)),
        )
        weak_points = [
            WeakPoint(
                stage=wp["stage"],
                item=wp["item"],
                score=float(wp["score"]),
                reason=wp["reason"],
            )
            for wp in item.get("weak_points", [])
        ]
        ra = item.get("repair_analysis", {})
        repair_analysis = RepairAnalysis(
            repair_type=ra.get("repair_type", "local_fix"),
            target_issue=ra.get("target_issue", ""),
            reason=ra.get("reason", ""),
            repair_strategy=ra.get("repair_strategy", []),
        )
        result.append(EvaluatedCandidate(
            candidate_id=item["candidate_id"],
            stage_scores=stage_scores,
            total_score=float(item.get("total_score", stage_scores.total())),
            passed=bool(item.get("pass", False)),
            failed_stage=item.get("failed_stage"),
            weak_points=weak_points,
            repair_analysis=repair_analysis,
        ))
    return result


def _parse_feedback(raw: dict[str, Any]) -> FeedbackDecision:
    return FeedbackDecision(
        need_feedback_to_module2a=bool(raw.get("need_feedback_to_module2a", False)),
        reason=raw.get("reason", ""),
        dominant_failure_pattern=raw.get("dominant_failure_pattern", []),
        suggested_relaxations=raw.get("suggested_relaxations", []),
    )
