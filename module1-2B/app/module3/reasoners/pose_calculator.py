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

# 입력 파일 및 활용 방법

[task]
수행해야 할 작업 설명이다.

[scene_objects] ← scene_info.json

각 물체의 기하 정보이다.
- center_world: 물체 중심의 world 좌표
- aabb_min / aabb_max: axis-aligned bounding box
- principal_axis_hint: 물체의 주축 방향 힌트 (x_axis | z_axis)
- graspable_regions: 파지 가능한 영역
- functional_regions: 기능 수행에 적합한 영역

[object_physical_properties] ← raw_module1_output.json

각 물체의 물리속성 정보이다.
- surface_friction: 마찰 수준 (low/medium/high)
- estimated_mass_kg: 질량 추정값
- rigidity: 강성 (rigid/semi-rigid/flexible)
- inferred_functions: 추론된 기능 목록

[tool_constraints] ← module2b_output.json

기능적 제약 조건이다. 각 필드를 다음과 같이 활용하라:
- numeric_estimates → 도달 거리/진입 각도 제약으로 pose 범위 결정
- derived_constraints → 삽입 두께/깊이 제약으로 attach_region 및 offset 결정
- global_constraints → 모든 step에 적용되는 전역 제약
- subgoal_constraints → 각 step의 목표 기능 달성 여부 확인

[selected_candidate] ← module2d_output.json

이미 선택된 도구 후보이다.
새로운 조합을 만들지 말고,
반드시 used_objects와 structure_description, function_mapping을 바탕으로 조립 계획을 계산하라.

[filter_result] ← module2d_output.json

2-D 평가에서 pass된 후보의 결과이다. pose 계산 시 반드시 반영하라.

- checklist: 기하/물리/상식 항목별 Yes/No 결과
  → No 항목이 있으면 해당 부분을 보완하는 방향으로 pose 조정
- stage_scores: geometry/physics/commonsense 점수 (0~5)
  → geometry_score 낮음: 접합 위치/면적을 더 신중하게 계산
  → physics_score 낮음: 무게 중심/힘 전달 경로를 더 신중하게 계산
  → commonsense_score 낮음: task 수행 맥락을 재검토
- weak_points: 점수가 낮은 항목
  → stage별로 다음과 같이 pose 조정에 반영하라:
  - weak_points.stage=geometry → 접합 위치/면적/방향 재조정
  - weak_points.stage=physics → 무게 중심 보정, 접합부 강화 방향으로 offset 조정
  - weak_points.stage=commonsense → functional_end 노출 재확인, task 수행 방향 재검토

# 로봇 파지 제약

Panda gripper는 rigid 물체만 파지 가능하다.
rigidity=flexible 또는 semi-rigid인 물체는 파지 대상으로 사용할 수 없다.
flexible/semi-rigid 물체는 보조 역할(마찰 패드, 완충재 등)로만 사용 가능하다.

# AABB 부재 시 처리 규칙

AABB가 [0,0,0]이거나 비어있는 경우:
반드시 shape_category와 estimated_mass_kg 기반으로 크기를 추정하고 실제 수치로 계산하라.
[0,0,0] 그대로 사용 금지.

추정 기준:
- elongated + mass 0.05kg → length=0.30m, width=0.03m, height=0.01m
- elongated + mass 0.1kg  → length=0.20m, width=0.01m, height=0.01m
- compact + mass 0.05kg   → length=0.05m, width=0.03m, height=0.03m
- center_world가 [0,0,0]이면 테이블 위 (z=0.625m) 기준으로 배치
- 추정값 사용 시 reason에 반드시 "AABB 추정값 사용" 명시

# 카메라 및 공간 해석 규칙

- 카메라 optical center: (0.55, -0.35, 0.8)
- 카메라는 테이블을 향해 아래로 40도 기울어짐
- 테이블 표면 z ≈ 0.625m
- 실제 pose 계산은 반드시 scene_objects의 world 좌표와 AABB를 기반으로 수행하라.
- 카메라 정보는 보조 prior로만 활용하라.

# 핵심 제약

1. 후보 구조 변경 금지
2. 순차 조립 필수
   - used_objects가 N개이면 반드시 N단계 이상으로 분리하라.
   - Step 1: base_object 단독 배치 (attach_object=null)
   - Step 2~N: 이전 단계 결과에 다음 물체를 순서대로 부착
     예: 3개 물체 (A, B, C)
     Step 1: A 단독 배치
     Step 2: A + B 부착
     Step 3: (A+B) + C 부착
3. partial assembly 상태 반영 필수
4. 좌표 계산 필수
   - AABB 기반: attach_region_base가 upper_surface이면
     offset_z = (base_aabb_max_z - base_center_z) + (attach_object_aabb_height / 2)
   - AABB 없으면 shape_category 추정값으로 계산하고 reason에 명시
5. orientation 계산 기준
   - principal_axis_hint=x_axis → 물체가 x축 방향으로 놓임 → yaw=0
   - principal_axis_hint=z_axis → 물체가 z축 방향으로 세워짐 → yaw=90
   - filter_result.checklist "주축이 task 수행 방향에 맞게 정렬 가능한가?" No → orientation 재조정
   - filter_result.checklist "결합 후 기능 수행 부위가 충분히 노출되는가?" No → attach_region 재조정
6. contact_type 선택 기준
   - point: 끝점이 물체에 접촉하는 경우 (예: 드라이버 끝이 물체에 닿을 때)
   - surface: 면이 면에 접촉하는 경우 (예: 자 위에 클립을 올릴 때)
   - insertion: 물체가 틈/구멍에 삽입되는 경우 (예: 핀셋을 좁은 틈에 넣을 때)
7. attach_region 선택 기준
   - front_end: 물체의 기능 수행 끝부분 (예: 드라이버 팁, 핀셋 끝)
   - rear_end: 물체의 파지 끝부분 (예: 드라이버 손잡이 끝)
   - upper_surface: 물체의 윗면 (예: 자 위에 클립 올릴 때)
   - lower_surface: 물체의 아랫면
   - mid_body: 물체의 중간 몸통 부분 (예: 자 중간에 클립 고정)
   - side_face: 물체의 옆면
8. weak_points 반영 필수
9. handle과 functional end 분리 필수
10. 불가능한 구조 금지

# 추론 절차

반드시 아래 순서대로 추론하라.

Step A. selected_candidate 해석 (function_mapping 기반 역할 확인)
Step B. filter_result 반영 (stage_scores, weak_points, checklist 확인)
Step C. assembly strategy 수립
Step D. step-by-step pose 계산
Step E. final structure 정리
Step F. verification 수행
아래 항목을 각각 pass/fail로 판정하라:
- alignment: 물체들이 task 수행 방향으로 올바르게 정렬됐는가
- collision: 물체 간 비의도적 충돌이 없는가
- functional_end_exposed: 기능 수행 부위(tip/edge)가 충분히 노출됐는가
- handle_region_free: 로봇이 파지할 수 있는 영역이 확보됐는가
- force_transfer: task 수행에 필요한 힘이 도구 끝까지 전달 가능한가
- weak_point_mitigation: filter_result의 weak_points가 pose 조정에 반영됐는가
- subgoal_support: 각 subgoal의 required_atoms를 도구 구조가 지원하는가
- contact_feasibility: 도구가 타겟 물체와 실제로 접촉 가능한 위치에 있는가
Step G. feedback 결정
- fail 2개 이상 / functional_end_exposed=fail / handle_region_free=fail
  → need_feedback_to_module2c=true, feedback_target="module2c"

# 출력 규칙

1. JSON만 출력하라.
2. JSON 바깥의 설명문은 출력하지 마라.
3. 각 수치 좌표는 float 형태로 작성하라.
4. orientation은 반드시 [roll, pitch, yaw] degree 형식으로 작성하라.
5. step 번호는 1부터 시작하라.
6. verification.checks는 최소 5개 이상 포함하라.
7. AABB 추정값 사용 시 반드시 reason에 명시하라.

# 출력 형식

{
  "assembly_strategy": {
    "base_object": "string",
    "strategy_summary": "string",
    "sequence_reason": "string",
    "filter_result_reflection": "string"
  },

  "assembly_steps": [
    {
      "step": 1,
      "partial_assembly_state_before": ["string"],
      "base_object": "string",
      "attach_object": "string | null",
      "attach_region_base": "string",
      "attach_region_object": "string | null",
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
    "feedback_target": "module2c | null",
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
        "feedback_target": parsed.get("feedback", {}).get("feedback_target"),
        "raw_response_preview": raw_text[:500],
    }

    return parsed, trace
