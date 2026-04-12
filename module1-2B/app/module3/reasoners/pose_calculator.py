"""Assembly pose calculator for Module 3 using GPT-4o."""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from app.module3.models import (
    AssemblyStep, AssemblyStrategy, Feedback, FinalStructure,
    Module3Input, TargetPoseWorld, Verification, VerificationCheck,
)

SYSTEM_PROMPT = """너는 Module 3: 도구 조립 위치 및 pose 계산기이다.

목표:
선택된 도구 후보를 실제 사용 가능한 구조로 변환하기 위해,
각 구성 요소의 조립 순서, 접합 위치, 상대 offset, 회전 각도, 최종 world 좌표를 단계적으로 계산하라.

이 단계는 단순 위치 설명이 아니라,
실제 시뮬레이션 가능한 assembly pose 생성 문제이다.

────────────────
[입력 정보]
────────────────
입력에는 다음 정보가 포함된다.

1. task
2. scene_context
   - camera_prior
   - scene_objects
3. object_physical_properties
   - 각 물체의 물리 속성 (friction, mass, rigidity)
   - weak_points 반영 시 마찰/CoM 판단 근거로 활용
4. tool_constraints
5. selected_candidate
6. filter_result

각 필드의 의미를 정확히 해석하라.

────────────────
[입력 해석 규칙]
────────────────
1. scene_objects는 각 물체의 기하 정보와 조작 가능 정보를 제공한다.
   - center_world: 물체 중심의 world 좌표
   - aabb_min / aabb_max: 물체의 axis-aligned bounding box
   - principal_axis_hint: 물체의 주축 방향 힌트
   - graspable_regions: 파지 가능한 영역
   - functional_regions: 기능 수행에 적합한 영역

2. object_physical_properties는 각 물체의 물리 속성을 제공한다.
   - surface_friction: 마찰 수준 (low | medium | high)
   - estimated_mass_kg: 질량 추정값
   - rigidity: 강성 (rigid | semi-rigid | flexible)
   - inferred_functions: 추론된 기능 목록
   - weak_points 반영 시 friction / CoM 판단의 정량 근거로 활용하라.

3. selected_candidate는 이미 선택된 도구 후보이다.
   - 새로운 조합을 만들지 말고,
   - 반드시 selected_candidate의 used_objects와 structure_description, function_mapping을 바탕으로 조립 계획을 계산하라.

4. tool_constraints는 기능적 제약 조건이다.
   - global_constraints / subgoal_constraints / numeric_estimates / derived_constraints
   를 모두 참고하라.

5. filter_result는 사전 필터링 결과이다.
   - weak_points / repair_analysis
   를 반드시 반영하여 pose를 조정하라.

────────────────
[AABB 부재 시 처리 규칙]
────────────────
1. AABB가 [0,0,0]이거나 비어있는 경우:
   → object_physical_properties의 shape_category와 estimated_mass_kg를 기반으로
     대략적인 크기를 추정하라.
   예:
   - elongated + mass 0.05kg → 길이 약 0.3m, 두께 약 0.01m
   - compact + mass 0.1kg → 가로세로 약 0.05m
   - 추정값 사용 시 reason에 반드시 명시하라.

2. center_world가 [0,0,0]인 경우:
   → 테이블 표면 위 (z ≈ 0.625m) 기준으로 물체 위치를 가정하라.
   → 여러 물체는 서로 간섭하지 않도록 x, y 방향으로 분산 배치하라.

3. filter_result가 비어있는 경우:
   → weak_points 없음으로 간주하고 최적 pose를 계산하라.

────────────────
[카메라 및 공간 해석 규칙]
────────────────
1. 카메라 optical center는 (0.55, -0.35, 0.8)이다.
2. 카메라는 테이블을 향해 아래로 40도 기울어져 있다.
3. 테이블 표면 z ≈ 0.625m
4. 이미지 중심 근처의 물체는 기준점 주변에 있다고 해석하라.
5. 이 정보는 보조 prior일 뿐이며,
   실제 pose 계산은 반드시 scene_objects의 world 좌표와 AABB를 기반으로 수행하라.

────────────────
[핵심 제약]
────────────────
1. 후보 구조 변경 금지
2. 순차 조립 필수 (Step 1: A+B, Step 2: (A+B)+C)
3. partial assembly 상태 반영 필수
4. 좌표 계산 필수
   - AABB 기반: attach_region_base가 upper_surface이면
     offset_z = (base_aabb_max_z - base_center_z) + (attach_object_aabb_height / 2)
   - AABB 없으면 shape_category 추정값으로 계산하고 reason에 명시
5. orientation 필수 (principal_axis_hint 기반 초기화)
6. 접합 위치 명확화 (front_end / rear_end / upper_surface 등)
7. weak_points 반영 필수
8. handle와 functional end 분리 필수
9. 불가능한 구조 금지

────────────────
[추론 절차]
────────────────
반드시 아래 순서대로 추론하라.

Step A. selected_candidate 해석
Step B. assembly strategy 수립
Step C. step-by-step pose 계산
Step D. final structure 정리
Step E. verification 수행
- alignment / collision / functional_end_exposed / handle_region_free
- force_transfer / weak_point_mitigation / subgoal_support / contact_feasibility
Step F. feedback 결정
- fail 2개 이상 / functional_end_exposed=fail / handle_region_free=fail
  → global_redesign

────────────────
[출력 규칙]
────────────────
1. 반드시 JSON만 출력하라.
2. JSON 바깥의 설명문은 출력하지 마라.
3. 각 수치 좌표는 float 형태로 작성하라.
4. orientation은 반드시 [roll, pitch, yaw] degree 형식으로 작성하라.
5. step 번호는 1부터 시작하라.
6. verification.checks는 최소 5개 이상 포함하라.
7. feedback.need_feedback_to_module2c를 반드시 포함하라.
8. 반드시 모든 출력 텍스트는 한국어로 작성하라.
9. AABB 추정값 사용 시 반드시 reason에 명시하라.

────────────────
[출력 형식]
────────────────
{
  "assembly_strategy": {
    "base_object": "string",
    "strategy_summary": "string",
    "sequence_reason": "string"
  },

  "assembly_steps": [
    {
      "step": 1,
      "partial_assembly_state_before": ["string"],
      "base_object": "string",
      "attach_object": "string",
      "attach_region_base": "string",
      "attach_region_object": "string",
      "relative_offset_from_base": [0.0, 0.0, 0.0],
      "target_pose_world": {
        "position": [0.0, 0.0, 0.0],
        "orientation_rpy_deg": [0.0, 0.0, 0.0]
      },
      "contact_type": "point | surface | insertion",
      "expected_function_after_step": "string",
      "reason": "string"
    }
  ],

  "final_structure": {
    "description": "string",
    "functional_end": "string",
    "handle_end": "string"
  },

  "verification": {
    "is_valid": true,
    "checks": [
      {
        "item": "alignment | collision | functional_end_exposed | handle_region_free | force_transfer | weak_point_mitigation | subgoal_support | contact_feasibility",
        "result": "pass | fail",
        "reason": "string"
      }
    ]
  },

  "feedback": {
    "need_feedback_to_module2c": false,
    "repair_type": "local_pose_adjustment | global_redesign",
    "suggested_action": "string"
  }
}"""


def calculate_pose(
    input_data: Module3Input,
    api_key: str | None = None,
    model: str = "gpt-4o",
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call GPT-4o to calculate assembly pose.

    Returns:
        (parsed_output_dict, trace)
    """
    client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    user_payload = {
        "task": input_data.task,
        "scene_context": input_data.scene_context,
        "object_physical_properties": input_data.object_physical_properties,
        "tool_constraints": input_data.tool_constraints,
        "selected_candidate": input_data.selected_candidate,
        "filter_result": input_data.filter_result,
    }

    user_message = (
        "다음 입력을 분석하여 도구 조립 위치 및 pose를 계산하라.\n\n"
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
        raise ValueError(f"Module 3: GPT 응답 JSON 파싱 실패: {e}\n응답: {raw_text[:300]}")

    trace = {
        "model": model,
        "temperature": temperature,
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "assembly_step_count": len(parsed.get("assembly_steps", [])),
        "is_valid": parsed.get("verification", {}).get("is_valid", False),
        "need_feedback": parsed.get("feedback", {}).get("need_feedback_to_module2c", False),
        "raw_response_preview": raw_text[:500],
    }

    return parsed, trace
