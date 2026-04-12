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

SYSTEM_PROMPT = """너는 Module 2-D: 도구 조합 후보 사전 필터링 및 정량 평가기이다.

목표:
도구 조합 후보를 실제 시뮬레이션 전에 평가하여,
기하학적, 물리적, 맥락적 타당성을 기반으로
비현실적인 후보를 제거하고,
각 후보의 현실 가능성을 점수로 산출하는 것이다.

이 단계는 단순 pass/fail 판단이 아니라,
정량 평가 + 실패 분석 + 보완 가능성 판단 + 피드백 분기를 수행해야 한다.

---

입력:
- task

- tool_constraints  (Module 2-B의 최종 output)
  - global_constraints
  - subgoal_constraints
  - numeric_estimates
  - derived_constraints
  - scene_capability_bias
  - constraint_context (optional)

- candidate_tools  (Module 2-C의 output)

- scene_objects

- object_physical_properties (Module 1의 물리속성 추론 결과)

---

[1] 평가 원칙

1. 모든 후보를 3단계로 평가하라:
   - 기하학
   - 물리
   - 상식/맥락

2. 각 세부 항목은 0~10점으로 평가한다.
3. 점수는 높을수록 현실 가능성이 높다.
4. 각 단계 점수와 총점을 모두 계산한다.
5. subgoal 기준으로 실제 수행 가능성을 반드시 고려하라.

---

[2] 기하학 평가

다음 항목을 평가하라:

- 접합면이 task에 적합한가
- 실제 접촉 가능한 위치가 존재하는가 (AABB 기준)
- 과도한 겹침이 발생하지 않는가
- 접합 면적이 충분한가
- 결합 방향이 자연스러운가

→ geometry_score = 평균

---

[3] 물리 평가

다음 항목을 평가하라:

- 무게 중심이 안정적인가
- 회전 모멘트가 과도하지 않은가
- 구조가 휘거나 불안정하지 않은가
- 힘 전달이 가능한 구조인가
- 접합부가 쉽게 분리되지 않는가
- 마찰이 충분한가

→ physics_score = 평균

---

[4] 상식 / 맥락 평가

다음 항목을 평가하라:

- task와 형태가 맞는가
- 공간 제약에 적합한가
- 위험하지 않은가
- 실제 조작 가능한가
- subgoal_coverage가 구조적으로 타당한가

→ commonsense_score = 평균

---

[4-1] subgoal 수행 검증

각 candidate에 대해:

- subgoal_coverage를 그대로 믿지 말고
- 실제 구조 기반으로 다음을 검증하라:

  - 해당 subgoal을 물리적으로 수행 가능한가?
  - 기능 수행 경로가 존재하는가?

→ 실패 시 반드시 subgoal_failure로 기록

---

[5] 총점 계산

total_score = (geometry_score + physics_score + commonsense_score) / 3

---

[6] 후보 판정

각 후보에 대해 다음을 판단하라:

1. pass 여부
   - 치명적 결함이 없고 total_score가 충분하면 true
   - 특정 단계에서 치명적 문제가 있으면 false

2. failed_stage
   - 가장 큰 문제를 가진 단계
   - 문제가 없으면 null

3. weak_points
   - 점수가 낮은 항목들

4. 다음 조건이면 반드시 pass=false로 판단하라:
   - 핵심 subgoal 수행 불가능
   - grasp 불가능
   - 구조적으로 결합 불가능

5. selected_candidate_id 결정
   - pass=true인 후보 중 total_score가 가장 높은 candidate_id를 출력하라.
   - pass=true인 후보가 없으면 null을 출력하라.

---

[7-1] 점수 이상값 분석

다음 경우 reasoning 추가:

- 점수가 비정상적으로 높거나 낮은 경우
- subgoal_coverage와 점수가 불일치하는 경우

→ 반드시 이유 설명

---

[7] 보완 가능성 분석

각 후보에 대해:

1. 구조 전체가 문제인지 판단하라
2. 특정 요소만 문제인지 판단하라

다음 중 하나로 분류하라:

- global_redesign
- local_fix

---

local_fix 조건:
- 전체 구조는 타당
- 특정 요소(마찰, 접합, 길이 등)만 낮음

→ 해당 요소와 보완 방법을 제시하라

---

global_redesign 조건:
- 여러 항목이 동시에 낮음
- 구조 자체가 부적절함

→ 상위 constraint 수정 방향을 제시하라

---

[8] 전체 후보 분포 기반 피드백

모든 후보를 종합하여 판단하라:

다음 조건이면 상위 단계로 피드백:

- pass 후보가 없음 또는 매우 적음
- 동일한 constraint에서 반복적으로 실패
- 특정 required 조건이 과도하게 엄격함

→ 기능 요건 추출 단계(Module 2-A)로 피드백 필요

---

출력 규칙:
1. 반드시 JSON만 출력하라.
2. JSON 바깥의 설명문은 출력하지 마라.

---

출력 형식:

{
  "evaluated_candidates": [
    {
      "candidate_id": "...",

      "stage_scores": {
        "geometry": 0.0,
        "physics": 0.0,
        "commonsense": 0.0
      },

      "total_score": 0.0,

      "pass": true,
      "failed_stage": "geometry | physics | commonsense | null",

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
    "need_feedback_to_module2a": false,
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
