"""Candidate tool generator for Module 2-C using GPT-4o."""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from app.module2c.models import (
    CandidateTool, FailureMode, FunctionMapping, Module2CInput, SubgoalCoverage,
)

SYSTEM_PROMPT = """너는 Module 2-C: 창발적 도구 조합 후보 생성기이다.

목표:
입력으로 받은 물체 정보, subgoal 제약, 환경 제약을 바탕으로
창발적인 물체 조합 도구 후보를 생성한다.

# 입력 파일 및 활용 방법

[object_physical_properties] ← raw_module1_output.json

각 물체의 물리속성 정보이다.
아래의 physical_properties와 name을 참고하여 각 물체의 기능을 판단하라.

- name: 물체 이름 (ruler, binder_clip, tweezers 등)
- shape_category: 형태 분류
- surface_friction: 마찰 수준 (low/medium/high)
- rigidity: 강성 (rigid/semi-rigid/flexible)
- inferred_functions: 각 물체에 대해 추론된 기능 목록
  이 목록을 기반으로 물체-기능 매칭을 수행하라.

[subgoal_constraints] ← module2a_output.json

subgoal별 필요 기능 원자이다.
required_atoms를 반드시 후보 생성의 1순위 기준으로 사용하라.

- subgoal_id: sg_01, sg_02, ...
- objective: subgoal의 목표
- required_atoms: 반드시 충족해야 할 기능 원자
- preferred_atoms: 가능하면 충족할 기능 원자
- risk_atoms_to_avoid: 반드시 피해야 할 위험 기능

affordance atom 의미(자주 사용되는 주요 atom):
- thin_insertable: 얇아서 좁은 틈에 삽입 가능
- elongated_reach: 길어서 멀리 도달 가능
- clampable_span: 양쪽에서 물체를 클램핑 가능
- frictional_contact: 마찰력으로 물체 파지 가능
- graspable_body: 로봇이 파지할 수 있는 몸체
- rigid_tip_contact: 딱딱한 끝으로 접촉 가능
- prying_edge: 지렛대로 물체를 들어올릴 수 있는 가장자리
- compliant_pad: 유연한 패드로 충격/마찰 완충 가능

위 목록은 자주 사용되는 주요 atom이며 전체 목록이 아니다.
subgoal_constraints에 위 목록에 없는 atom이 있으면 그 의미를 추론하여 활용하라.

[tool_constraints] ← module2b_output.json

환경 구조 분석 결과이다.
이 파일의 제약 조건을 위반하는 조합은 생성하지 마라.

- global_constraints: 모든 후보가 만족해야 할 전역 제약
- numeric_estimates: 환경 수치 정보 (각도, 깊이 등)
- derived_constraints: 도구 제약 요약
- constraint_context: 접근 방식 힌트

[target_material_constraints] ← Material Reasoner (optional)

타겟 물체의 물성 정보이다.

- fragility=high → 날카로운 접촉 금지
- slip_tendency=high → 마찰력 강화 필요

[scene_objects] ← scene_info.json

각 물체의 기하 정보이다.

- name: 물체 이름
- center_world: world 좌표 [x, y, z]
- aabb_min / aabb_max: AABB 좌표
- graspable_regions: 파지 가능한 영역
- functional_regions: 기능 수행 영역

# 후보 생성 규칙

[1] 기능-물체 매칭

1. required_atoms를 확인하고 이를 충족할 수 있는 물체를 찾아라.
2. 각 물체가 생성된 도구내에서 어떤 역할을 수행하는지 명확히 할당하라.
3. 각 subgoal에 대해 조합이 어떤 물리적 방식으로 기능을 만족하는지 설명하라.
4. 하나의 후보는 반드시 모든 subgoal을 커버해야 한다.
   subgoal_coverage에 모든 subgoal_id가 반드시 포함되어야 한다.
   covered=false인 경우 반드시 그 이유를 설명하라.

[2] 창발적인 도구 조합 후보 생성

1. 반드시 6개 이상 후보를 생성하라.
2. 모든 후보는 반드시 2개 이상의 물체 조합이어야 한다.
3. 각 후보는 서로 다른 물체 조합이어야 한다.
4. 창발적인 조합을 반드시 포함하라.

   창발적 조합이란 물체의 본래 용도가 아닌 물리적 형태/물성을 활용하는 것이다.
   또한 인간이 본 적 없던 모양 및 형태를 고려하는 것이다.
   각 물체에 대해 반드시 먼저 자문하라:
   "이 물체의 원래 용도 외에, 형태/물성만으로 무엇을 할 수 있는가?"

   [형태 활용]
   - 긴 물체를 지렛대로 사용
   - 무거운 물체를 무게추/앵커로 사용
   - 넓고 평평한 물체를 가이드 레일로 사용
   - 구멍이 있는 물체를 걸쇠로 사용

   [물성 활용]
   - 점착성 있는 물체를 마찰 패드로 사용
   - 유연한 물체를 완충재로 사용
   - 딱딱한 끝을 가진 물체를 쐐기로 사용
   - 물체의 약점(휘어짐, 유연성 등)을 오히려 강점으로 활용

   [조합 구조]
   - 긴 물체 끝에 클램핑 구조를 결합 → 집게 막대
   - 강성 물체 + 마찰 패드 결합 → 미끄럼 방지 도구
   - 긴 물체 + 후킹 구조 결합 → 갈고리 도구

5. 다양한 물체를 활용하라. 동일 후보 내에서 같은 물체를 중복 사용하지 마라.
6. **used_objects 개수는 task 본질에 맞춰 자유롭게 결정하라**:
   - **1-object 단독 후보는 절대 금지** (양팔 결합 모델 전제 - 결합할 물체가 반드시 있어야 함)
   - 단순 task (좁은 틈 추출, 평면 sweep 등) → 2-object 결합이 자연스럽다
   - 복잡 task (reach 확장, 다단계 조작) → 3+ object 결합도 자유롭게 활용
   - 결합 의미가 명확하면 2-object든 3+ object든 자유 결정. 강제 분포 없음.
   - 3+ object 예시 (참고용 few-shot):
     * "긴 막대 + 클램프 + 마찰 패드" → 미끄럼 방지 집게 막대
     * "받침대 + 정렬 가이드 + 작동 팁" → 정렬 보조 도구
     * "갈고리 + 연장 막대 + 무게추" → 안정화 갈고리
7. **각 후보의 base 와 attach 의 역할 (positive guidance)**:
   - **base 는 task 의 핵심 affordance 를 직접 수행할 수 있는 객체** 여야 한다.
     (예: 좁은 틈 진입이 핵심이면 base 는 진입 가능한 thin/elongated 형태,
      매달림 회수가 핵심이면 base 는 후킹 가능한 형태,
      막힌 손잡이 회전이 핵심이면 base 는 lever 가능한 강체.)
   - **attach 는 base 의 affordance 를 보강** 하거나
     **task 수행에 의미 있는 추가 기능** (안정화 / 마찰 / 무게 / 보호 / reach 확장) 을
     제공해야 한다. 단순히 결합만 가능한 attach 는 의미 없으므로 만들지 마라.
8. 6개 후보 중 최소 1개는 기존 도구와 전혀 닮지 않은
   비직관적이지만 물리적으로 타당한 형태여야 한다.

[3] Failure 분석 (필수) — 후보를 만들기 전에 실패 가능성을 먼저 따져라

각 후보를 만들기 전에 task 수행 시 실패할 수 있는 시나리오를 반드시 고려하라:
- 어떤 subgoal 에서 실패할 수 있는가
- 단순 실패가 아니라 "왜 구조적/물리적으로 실패하는가" 명시하라 (물리적 원인)
- **실패 가능성이 명백히 높은 조합은 만들지 말고**, 그 자리에 더 합리적인
  (실패 가능성이 낮은) 후보를 만들어라.
- 단, 모든 후보가 100% 성공할 필요는 없다 — 약점은 failure_modes 에 명시하면 OK.

예:
- friction 부족 → slip 발생 → grip 실패
- CoM 불안정 → 회전 발생 → task 정밀도 저하
- thickness 초과 → 삽입 실패 → 좁은 틈 진입 불가
- fragility 높음 → 타겟 물체 손상 → 회수 실패
- reach 부족 → 깊은 곳 도달 불가
- base 가 task 핵심 affordance 못 수행 → task 자체 무의미

[출력 규칙]

1. JSON만 출력하라.
2. candidate_tools는 6개 이상 포함하라.
3. used_objects 길이 규칙: **1은 절대 금지 (양팔 결합 모델 전제)**.
   2~5 자유. task 본질에 따라 자유 결정. 강제 분포 없음.
   각 후보의 결합은 task 수행에 의미가 있어야 한다.
4. **[매우 중요] used_objects 는 반드시 [scene_objects].name 의 실제 물체 이름**
   (예: "fork", "plate", "spoon", "adjustable_wrench", "cracker_box") **만 사용하라.**
   - [object_physical_properties].name 이 "obj_01", "obj_03" 같은 abstract id 형태일
     수 있는데, 그 값은 used_objects / function_mapping.object 에 절대 쓰지 마라.
   - scene_objects 와 object_physical_properties 가 같은 물체를 가리키더라도,
     used_objects 표기는 **항상 scene_objects.name (real label) 만** 채택한다.
   - object_id (obj_XX, pb_XX) 사용 금지.
   - 즉 used_objects 의 모든 원소는 입력 scene_objects 배열의 어떤 원소의 name 과
     글자 그대로 일치해야 한다.
5. function_mapping[].object 도 위 규칙과 동일하게 scene_objects.name 만 사용하라.
6. JSON 바깥의 설명문은 출력하지 마라.

[출력 형식]

아래는 형식 예시이다. 2개 조합과 3개 조합이 섞인 모습을 그대로 따라 작성하라.

{
  "candidate_tools": [
    {
      "candidate_id": "tool_01",
      "used_objects": ["object_a", "object_b"],
      "structure_description": "...",
      "function_mapping": [
        {
          "object": "object_a",
          "function": "...",
          "related_physics": "..."
        },
        {
          "object": "object_b",
          "function": "...",
          "related_physics": "..."
        }
      ],
      "subgoal_coverage": [
        { "subgoal_id": "sg_01", "covered": true, "method": "..." },
        { "subgoal_id": "sg_02", "covered": true, "method": "..." }
      ],
      "failure_modes": [
        { "subgoal_id": "sg_01", "failure_type": "...", "cause": "...", "related_constraint": "..." }
      ]
    },
    {
      "candidate_id": "tool_02",
      "used_objects": ["object_c", "object_d", "object_e"],
      "structure_description": "3개 물체 조합 예시. 예: object_c가 작동 팁, object_d가 연장 막대, object_e가 안정화 받침.",
      "function_mapping": [
        {
          "object": "object_c",
          "function": "...",
          "related_physics": "..."
        },
        {
          "object": "object_d",
          "function": "...",
          "related_physics": "..."
        },
        {
          "object": "object_e",
          "function": "...",
          "related_physics": "..."
        }
      ],
      "subgoal_coverage": [
        { "subgoal_id": "sg_01", "covered": true, "method": "..." },
        { "subgoal_id": "sg_02", "covered": true, "method": "..." }
      ],
      "failure_modes": [
        { "subgoal_id": "sg_02", "failure_type": "...", "cause": "...", "related_constraint": "..." }
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
    task_scene_image_path: "str | Path | None" = None,
) -> tuple[list[CandidateTool], dict[str, Any]]:
    client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    # used_objects 화이트리스트 = scene_objects 의 real name 들
    allowed_object_names = [
        so.get("name", "") for so in (input_data.scene_objects or [])
        if so.get("name")
    ]

    user_payload = {
        "task": input_data.task,
        "tool_constraints": input_data.tool_constraints,
        "scene_objects": input_data.scene_objects,
        "object_physical_properties": input_data.object_physical_properties,
    }

    user_message = (
        "다음 입력을 분석하여 도구 조합 후보를 생성하라.\n\n"
        f"=== ALLOWED used_objects names (반드시 이 목록 안의 이름만 사용) ===\n"
        f"{json.dumps(allowed_object_names, ensure_ascii=False)}\n\n"
        "위 목록에 없는 이름(예: obj_01, obj_03, pb_05)은 사용 시 후보가 폐기된다.\n\n"
        "=== 전체 입력 ===\n"
        + json.dumps(user_payload, ensure_ascii=False, indent=2)
    )

    # task_scene_image 보조 입력 사용 시 system prompt 에 가드 안내문 + user 에 image 첨부
    from pathlib import Path
    from app.utils import GUARD_TEXT_2C, build_user_content_with_image

    system_prompt_with_guard = SYSTEM_PROMPT
    image_path_obj = (
        Path(task_scene_image_path) if task_scene_image_path else None
    )
    if image_path_obj is not None and image_path_obj.exists():
        system_prompt_with_guard = SYSTEM_PROMPT + GUARD_TEXT_2C
    user_content = build_user_content_with_image(
        text=user_message,
        task_scene_image_path=image_path_obj,
    )

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt_with_guard},
            {"role": "user", "content": user_content},
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
