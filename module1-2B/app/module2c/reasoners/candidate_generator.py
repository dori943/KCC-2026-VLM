"""Candidate tool generator for Module 2-C using GPT-4o."""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from app.module2c.models import (
    CandidateTool, FailureMode, FunctionMapping, Module2CInput, SubgoalCoverage,
)

SYSTEM_PROMPT = """너는 Module 2-C: 도구 조합 후보 생성기이다.

목표:
각 물체의 물리속성 추론 output과
task 분해 및 기능 요건 추출 모듈의 최종 output(tool_constraints)을 함께 참고하여,
subgoal 기반 constraint를 만족하려는 물체 조합 도구 후보를 생성하는 것이다.

이 단계는 정답을 찾는 것이 아니라,
가능한 후보 공간을 탐색하는 단계이며,
각 후보의 실패 가능성을 반드시 분석해야 한다.

반드시 3개 이상의 후보를 생성하라.
입력 데이터가 부족하더라도 주어진 정보를 최대한 활용하여 후보를 생성해야 한다.

---

입력:
- task

- tool_constraints  
  - global_constraints
  - subgoal_constraints
  - numeric_estimates
  - derived_constraints
  - scene_capability_bias
  - constraint_context (optional)
  - target_material_constraints (optional)
    - physical_properties.fragility: 타겟 물체 손상 위험도
    - physical_properties.slip_tendency: 타겟 물체 미끄럼 경향
    - manipulation_analysis.constraints: 타겟 물체 조작 제약 목록

- scene_objects

- object_physical_properties (Module 1의 물리속성 추론 결과)

---

지시사항:

[1] 역할 이해 (중요)

1. tool_constraints는 후보 생성의 1차 기준이다.
2. object_physical_properties는 각 물체가 어떤 기능을 수행할 수 있는지 판단하는 근거이다.
3. 이 모듈의 핵심은 "필요한 기능"과 "가능한 물체"를 매칭하는 것이다.
4. target_material_constraints가 있으면 타겟 물체의 fragility와 slip_tendency를 반드시 반영하라.

---

[2] 입력 데이터 부족 시 처리 규칙

1. scene_objects가 비어있거나 AABB가 [0,0,0]인 경우:
   → tool_constraints의 scene_capability_bias와 task 텍스트를 기반으로
     물체의 기능을 추론하여 후보를 생성하라.

2. subgoal_constraints가 비어있는 경우:
   → task 텍스트에서 직접 필요한 기능을 추론하라.
   예: "좁은 틈에서 명함 꺼내기" → thin_insertable, frictional_contact, elongated_reach 필요

---

[3] constraint 해석

1. global_constraints는 도구 전체가 만족해야 할 기능이다.
2. subgoal_constraints는 각 단계별 필수 기능이다.
3. 각 subgoal의 required_atoms는 반드시 충족하려고 시도해야 한다.
4. preferred_atoms는 가능하면 반영하라.
5. risk_atoms_to_avoid는 가능한 피하라.

---

[4] 물체 기능 해석

각 scene_object에 대해:

- object_physical_properties를 기반으로 해당 물체가 수행 가능한 기능을 추론하라

예:
- elongated shape → reach 가능
- thin edge → 삽입 가능
- high friction → 미끄럼 방지
- rigid structure → 힘 전달 가능

---

[5] 기능-물체 매칭 (핵심)

각 subgoal에 대해:

1. required_atoms를 충족할 수 있는 물체를 찾는다
2. 하나의 물체로 부족하면 여러 물체를 조합한다
3. 각 물체가 어떤 역할을 수행하는지 명확히 할당한다
4. 각 subgoal에 대해, 해당 조합이 "어떤 물리적 방식으로 기능을 만족시키는지" 반드시 설명하라

---

[6] 도구 조합 후보 생성

다음을 만족하도록 후보를 생성하라:

1. 반드시 3개 이상 생성하라. 다음 유형을 포함하라:
   - 유형 A: 단일 물체 최적 후보 (가장 많은 subgoal 커버)
   - 유형 B: 2개 물체 조합 후보 (서로 다른 기능 보완)
   - 유형 C: 대안적 접근 후보 (다른 방식으로 task 해결)
2. 각 subgoal을 어떤 물체 조합으로 해결하는지 명시하라
3. 단일 물체 또는 다중 물체 조합 모두 허용한다
4. 구조적으로 현실적인 결합을 가정하라
5. 실제로 로봇이 사용할 수 있는 구조인지 고려하라
   - 파지 가능한 위치 존재 여부
   - 기능 수행 부위와 파지 부위의 충돌 여부

---

[7] 구조 표현

각 후보는 반드시 다음을 포함해야 한다:

- 물체 간 결합 구조 설명
- 각 물체의 역할 정의
- subgoal별 대응 관계

---

[8] Failure 분석 (필수)

각 후보에 대해 반드시 분석하라:

1. 어떤 subgoal에서 실패할 수 있는가
2. 실패의 물리적 원인은 무엇인가
3. 단순 실패가 아니라 "왜 구조적으로 실패하는지" 설명하라

예:
- friction 부족 → slip 발생
- CoM 불안정 → 회전 발생
- thickness 초과 → 삽입 실패
- fragility 높음 → 타겟 물체 손상

---

다음 조건이면 후보로 생성하지 말 것:
- 핵심 subgoal 수행 불가능
- grasp 가능한 위치 없음
- 구조적으로 결합 불가능

단, 위 조건에 해당하더라도 최소 1개 이상의 후보는 반드시 생성하라.

---

출력 규칙:
1. 반드시 JSON만 출력하라.
2. JSON 바깥의 설명문은 출력하지 마라.
3. candidate_tools는 반드시 3개 이상 포함하라.
4. 반드시 모든 출력 텍스트는 한국어로 작성하라.

---

출력 형식:

{
  "candidate_tools": [
    {
      "candidate_id": "...",

      "used_objects": ["...", "..."],

      "structure_description": "...",

      "function_mapping": [
        {
          "object": "...",
          "function": "...",
          "related_physics": "..."
        }
      ],

      "subgoal_coverage": [
        {
          "subgoal_id": "sg_01",
          "covered": true,
          "method": "..."
        }
      ],

      "failure_modes": [
        {
          "subgoal_id": "sg_02",
          "failure_type": "...",
          "cause": "...",
          "related_constraint": "..."
        }
      ]
    }
  ]
}"""


def generate_candidates(
    input_data: Module2CInput,
    api_key: str | None = None,
    model: str = "gpt-4o",
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> tuple[list[CandidateTool], dict[str, Any]]:
    client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    user_payload = {
        "task": input_data.task,
        "tool_constraints": input_data.tool_constraints,
        "scene_objects": input_data.scene_objects,
        "object_physical_properties": input_data.object_physical_properties,
    }

    user_message = (
        "다음 입력을 분석하여 도구 조합 후보를 생성하라.\n\n"
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
        raise ValueError(f"Module 2-C: GPT 응답 JSON 파싱 실패: {e}\n응답: {raw_text[:300]}")

    candidates = _parse_candidates(parsed.get("candidate_tools", []))

    trace = {
        "model": model,
        "temperature": temperature,
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "candidate_count": len(candidates),
        "raw_response_preview": raw_text[:500],
    }

    return candidates, trace


def _parse_candidates(raw: list[dict[str, Any]]) -> list[CandidateTool]:
    candidates = []
    for item in raw:
        candidates.append(CandidateTool(
            candidate_id=item["candidate_id"],
            used_objects=item["used_objects"],
            structure_description=item["structure_description"],
            function_mapping=[
                FunctionMapping(object=fm["object"], function=fm["function"],
                                related_physics=fm["related_physics"])
                for fm in item.get("function_mapping", [])
            ],
            subgoal_coverage=[
                SubgoalCoverage(subgoal_id=sc["subgoal_id"], covered=bool(sc["covered"]),
                                method=sc["method"])
                for sc in item.get("subgoal_coverage", [])
            ],
            failure_modes=[
                FailureMode(subgoal_id=fm["subgoal_id"], failure_type=fm["failure_type"],
                            cause=fm["cause"], related_constraint=fm.get("related_constraint", ""))
                for fm in item.get("failure_modes", [])
            ],
        ))
    return candidates
