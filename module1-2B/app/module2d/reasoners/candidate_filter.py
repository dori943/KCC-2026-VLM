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

CANDIDATE_SYSTEM_PROMPT = """너는 Module 2-D: 로봇 task를 위한 도구 조합 후보를 평가하는 전문가다.
하나의 후보만 평가한다. 아래 절차를 순서대로 따르라.

# 입력

- task: 로봇이 수행할 작업
- tool_constraints: 환경 제약 (derived_constraints, subgoal_constraints 포함)
- candidate: 평가할 단일 후보 (used_objects, structure_description, function_mapping)
- object_profiles: 각 물체의 기하·물리 프로파일 요약 (Python이 계산한 값)
- object_physical_properties: 각 물체의 shape_category, rigidity, surface_friction, connection_modes, inferred_functions, geometry_profile, numeric_profile
- scene_objects: 각 물체의 AABB 기하 정보

# 1단계 전 필수 분석 (pre_analysis — JSON 출력에 포함)

used_objects 각각에 대해 아래 4가지를 명시하라. 이 분석이 이후 G3·G4 판단의 근거가 된다.
1. functional_end: task를 직접 수행하는 부위 (예: "jaw opening", "thin tip", "flat edge")
2. mechanism: 기능 수행에 필요한 물리적 동작 (예: "jaw opens/closes", "insert into gap", "lever rotation")
3. attach_zone: 상대 물체와 결합하는 위치
4. blockable: 결합 후 functional_end가 차단될 가능성 ("high" | "medium" | "low"), 이유 포함

# 1단계: 환경 제약 하드 필터

derived_constraints 항목을 위반하면 즉시 탈락. 이후 평가 없음.
- max_tip_thickness 등 수치 제약 직접 비교

# 2단계: 접합 구조 하드 필터 (구조적 불가능성만)

아래 조건 중 하나라도 해당하면 탈락. 슬립·손상·힘 비효율은 여기서 탈락시키지 않는다.
- 호환 가능한 connection_modes가 전혀 없음
- 크기 차이가 극단적으로 커서 물리적 접촉 불가
- 비강성 물체가 강성 역할(지렛대, 삽입 쐐기) 필수 수행
- 결합 후 파지 영역이 기하학적으로 완전 소멸
- 결합 후 기능 수행부가 물리적으로 완전히 차단됨

# 3단계: 체크리스트 평가 (하드 필터 통과 시)

각 항목에 대해 Yes/No와 No인 경우 그 이유를 판단하라.
물체의 실제 물리 속성(rigidity, surface_friction, shape_category, connection_modes)을 직접 참조하라.

[기하 G1~G7]
G1. 두 물체 사이에 물리적 접촉 가능한 면이 존재하는가?
G2. 두 물체의 크기가 결합 가능한 수준인가? (AABB 참조)
G3. 결합 후 기능 수행부(tip/edge/jaw)가 task 수행에 충분히 노출되는가?
    ← geometry_profile.has_open_cavity=True인 물체(clip/tweezers 등)는 jaw 입구가 상대 물체에 의해 막히면 반드시 No.
    ← mechanism_type=spring_jaw인 경우: 결합 위치가 jaw 입구 쪽이면 기능 수행 불가.
G4. 결합이 각 물체의 기계적 동작(벌어짐, 삽입, 회전 등)을 방해하지 않는가?
    ← has_open_cavity=True인 물체: 상대 물체가 jaw 열림을 물리적으로 제한하면 No.
G5. 두 물체의 주축이 task 수행 방향에 맞게 정렬 가능한가?
G6. 결합했을 때 전체 도구의 길이가 task 수행에 충분한가?
G7. 두 물체의 접촉 면적이 안정적 결합을 유지하기에 충분한가?

[물리 P1~P6]
P1. 무게 중심이 안정적인가?
P2. 회전 모멘트가 task를 방해하지 않는가?
P3. 구조가 task 중 휘거나 불안정해지지 않는가? (rigidity 참조)
P4. task에 필요한 힘이 도구 끝까지 전달 가능한가?
P5. 접합부가 task 중 분리되지 않는가?
P6. task에 필요한 마찰력이 충분한가? (surface_friction 참조)

[상식 C1~C5]
C1. 공간 제약 내에서 사용 가능한가?
C2. 로봇이 실제로 조작 가능한 구조인가?
C3. required_atoms 및 required_interaction_primitives를 이 도구 조합이 실제로 수행 가능한가?
    참고 가능한 interaction primitive 목록:
    reach, approach, orient, probe, contact, push, pull, drag, slide, lift, lower,
    hold, release, place, align, guide, press, tap, squeeze, grasp, pinch, clamp,
    hook, wedge, pry, insert, thread, twist, rotate, separate, scrape, sweep, scoop,
    contain, pour, funnel, cover, block, support, brace, stabilize, stack, hang,
    roll, spin, fasten, clip, anchor, bridge
C4. 타겟 물체를 손상시키지 않고 task 수행 가능한가?
C5. 주변 환경을 손상시키지 않는가?

# 4단계: 보완 분석

false 항목이 있으면 repair_analysis를 채워라.
- false 항목이 1개 그룹에만 있으면 → local_fix
- 2개 이상 그룹에 걸치면 → global_redesign
- false 항목 없으면 → repair_type: "none"

# 출력 규칙

1. JSON만 출력하라.
2. stage_scores/total_score/pass/failed_stage/weak_points는 출력하지 마라. 시스템이 계산한다.
3. checklist는 반드시 G1~G7(7개) + P1~P6(6개) + C1~C5(5개) = 18개 전부 출력하라.
4. result=false인 항목만 false_reason을 채워라. true면 null.

# 출력 형식

{
  "candidate_id": "...",
  "pre_analysis": {
    "object_profiles": [
      {
        "name": "물체명",
        "functional_end": "...",
        "mechanism": "...",
        "attach_zone": "...",
        "blockable": "high | medium | low — 이유"
      }
    ]
  },
  "environment_filter": {
    "pass": true,
    "violated_constraints": [],
    "reason": "derived_constraints 검토 결과 한 문장"
  },
  "assembly_filter": {
    "pass": true,
    "violated_reasons": [],
    "reason": "connection_modes 및 구조적 결합 가능성 한 문장"
  },
  "checklist": {
    "geometry": [
      {"item": "G1", "result": true, "false_reason": null},
      {"item": "G2", "result": true, "false_reason": null},
      {"item": "G3", "result": true, "false_reason": null},
      {"item": "G4", "result": true, "false_reason": null},
      {"item": "G5", "result": true, "false_reason": null},
      {"item": "G6", "result": true, "false_reason": null},
      {"item": "G7", "result": true, "false_reason": null}
    ],
    "physics": [
      {"item": "P1", "result": true, "false_reason": null},
      {"item": "P2", "result": true, "false_reason": null},
      {"item": "P3", "result": true, "false_reason": null},
      {"item": "P4", "result": true, "false_reason": null},
      {"item": "P5", "result": true, "false_reason": null},
      {"item": "P6", "result": false, "false_reason": "이유"}
    ],
    "commonsense": [
      {"item": "C1", "result": true, "false_reason": null},
      {"item": "C2", "result": true, "false_reason": null},
      {"item": "C3", "result": true, "false_reason": null},
      {"item": "C4", "result": false, "false_reason": "이유"},
      {"item": "C5", "result": true, "false_reason": null}
    ]
  },
  "repair_analysis": {
    "repair_type": "local_fix | global_redesign | none",
    "target_issue": "...",
    "reason": "...",
    "repair_strategy": ["..."]
  }
}"""


def filter_candidates(
    input_data: Module2DInput,
    api_key: str | None = None,
    model: str = "gpt-4o",
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> tuple[list[EvaluatedCandidate], str | None, FeedbackDecision, dict[str, Any]]:
    """Evaluate each candidate with an independent GPT call, then aggregate.

    Returns:
        (evaluated_candidates, selected_candidate_id, feedback_decision, trace)
    """
    client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    # Build lookup maps shared across all candidates
    prop_map = {p["name"]: p for p in input_data.object_physical_properties}
    geo_map = {o["name"]: o for o in input_data.scene_objects}

    evaluated: list[EvaluatedCandidate] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    raw_previews: dict[str, str] = {}

    for candidate in input_data.candidate_tools:
        cid = candidate["candidate_id"]
        used_names = candidate.get("used_objects", [])

        # Idea 3: Python-built object profile summary (geometry_profile + numeric_profile)
        profile_lines: list[str] = []
        for name in used_names:
            prop = prop_map.get(name)
            if not prop:
                continue
            line_parts = [
                f"[{name}]",
                f"  shape={prop.get('shape_category','?')}",
                f"  rigidity={prop.get('rigidity','?')}",
                f"  friction={prop.get('surface_friction','?')}",
                f"  functions={prop.get('inferred_functions',[])}",
                f"  connection_modes={prop.get('connection_modes',[])}",
            ]
            gp = prop.get("geometry_profile", {})
            if gp:
                gp_str = ", ".join(f"{k}={v}" for k, v in gp.items())
                line_parts.append(f"  geometry_profile: {gp_str}")
            np_ = prop.get("numeric_profile", {})
            if np_:
                np_str = ", ".join(f"{k}={v}" for k, v in np_.items())
                line_parts.append(f"  numeric_profile: {np_str}")
            geo = geo_map.get(name, {})
            aabb = geo.get("aabb_dimensions_m") or geo.get("aabb_size_m")
            if aabb:
                line_parts.append(f"  aabb_m: {aabb}")
            profile_lines.append("\n".join(line_parts))
        object_profiles_text = "\n".join(profile_lines) if profile_lines else "(없음)"

        user_payload = {
            "task": input_data.task,
            "tool_constraints": input_data.tool_constraints,
            "candidate": candidate,
            "object_physical_properties": [prop_map[n] for n in used_names if n in prop_map],
            "scene_objects": [geo_map[n] for n in used_names if n in geo_map],
        }
        user_message = (
            f"다음 단일 후보를 평가하라. candidate_id: {cid}\n\n"
            f"=== 물체 프로파일 요약 (Python 계산값) ===\n{object_profiles_text}\n\n"
            "=== 전체 입력 데이터 ===\n"
            + json.dumps(user_payload, ensure_ascii=False, indent=2)
        )

        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": CANDIDATE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
        )

        raw_text = response.choices[0].message.content or ""
        usage = response.usage
        total_prompt_tokens += usage.prompt_tokens if usage else 0
        total_completion_tokens += usage.completion_tokens if usage else 0
        raw_previews[cid] = raw_text[:300]

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Module 2-D [{cid}]: GPT 응답 JSON 파싱 실패: {e}\n응답: {raw_text[:200]}")

        evaluated.extend(_parse_evaluated_candidates([parsed]))

    selected_id = _select_best_candidate(evaluated, input_data.candidate_tools)
    feedback = _compute_feedback_decision(evaluated, input_data)

    trace = {
        "model": model,
        "temperature": temperature,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "evaluated_count": len(evaluated),
        "selected_candidate_id": selected_id,
        "raw_response_previews": raw_previews,
    }

    return evaluated, selected_id, feedback, trace


def _select_best_candidate(
    evaluated: list[EvaluatedCandidate],
    candidate_tools: list[dict[str, Any]],
) -> str | None:
    """Select best candidate: highest total_score among passed; tiebreak by used_objects count."""
    passed = [c for c in evaluated if c.passed]
    if not passed:
        return None

    obj_count = {c["candidate_id"]: len(c.get("used_objects", [])) for c in candidate_tools}
    max_score = max(c.total_score for c in passed)
    top = [c for c in passed if c.total_score == max_score]

    return max(top, key=lambda c: obj_count.get(c.candidate_id, 0)).candidate_id


_PASS_THRESHOLDS = {
    "geometry": 2.0,
    "physics": 1.5,
    "commonsense": 1.5,
    "total": 3.5,
}


def _score_from_checklist(items: list[dict[str, Any]]) -> float:
    if not items:
        return 0.0
    yes = sum(1 for i in items if i.get("result") is True)
    return round(yes / len(items) * 5, 4)


def _parse_evaluated_candidates(raw: list[dict[str, Any]]) -> list[EvaluatedCandidate]:
    result = []
    for item in raw:
        env_filter = item.get("environment_filter") or {}
        asm_filter = item.get("assembly_filter") or {}

        # Hard filter failures bypass quantitative scoring
        env_failed = not bool(env_filter.get("pass", True))
        asm_failed = not bool(asm_filter.get("pass", True))

        checklist = item.get("checklist") or {}
        geo_items = checklist.get("geometry", [])
        phys_items = checklist.get("physics", [])
        comm_items = checklist.get("commonsense", [])

        if env_failed or asm_failed:
            stage_scores = StageScores(geometry=0.0, physics=0.0, commonsense=0.0)
            passed = False
            failed_stage = "environment" if env_failed else "assembly"
        else:
            # Compute scores from checklist (Python-controlled, not GPT)
            g = _score_from_checklist(geo_items)
            p = _score_from_checklist(phys_items)
            c = _score_from_checklist(comm_items)
            stage_scores = StageScores(geometry=g, physics=p, commonsense=c)
            total = stage_scores.total()

            failed_stage = None
            if g < _PASS_THRESHOLDS["geometry"]:
                failed_stage = "geometry"
            elif p < _PASS_THRESHOLDS["physics"]:
                failed_stage = "physics"
            elif c < _PASS_THRESHOLDS["commonsense"]:
                failed_stage = "commonsense"
            elif total < _PASS_THRESHOLDS["total"]:
                failed_stage = "total"
            passed = failed_stage is None

        # Extract weak_points from checklist false items (score always 0.0)
        weak_points: list[WeakPoint] = []
        for stage_name, stage_items in [("geometry", geo_items), ("physics", phys_items), ("commonsense", comm_items)]:
            for ci in stage_items:
                if ci.get("result") is False:
                    weak_points.append(WeakPoint(
                        stage=stage_name,
                        item=ci.get("item", ""),
                        score=0.0,
                        reason=ci.get("false_reason") or "",
                    ))

        ra = item.get("repair_analysis") or {}
        repair_analysis = RepairAnalysis(
            repair_type=ra.get("repair_type", "none"),
            target_issue=ra.get("target_issue", ""),
            reason=ra.get("reason", ""),
            repair_strategy=ra.get("repair_strategy", []),
        )
        result.append(EvaluatedCandidate(
            candidate_id=item["candidate_id"],
            stage_scores=stage_scores,
            total_score=stage_scores.total(),
            passed=passed,
            failed_stage=failed_stage,
            weak_points=weak_points,
            repair_analysis=repair_analysis,
            environment_filter=item.get("environment_filter"),
            assembly_filter=item.get("assembly_filter"),
            checklist=checklist,
            pre_analysis=item.get("pre_analysis"),
        ))
    return result


def _compute_feedback_decision(
    evaluated: list[EvaluatedCandidate],
    input_data: Module2DInput,
) -> FeedbackDecision:
    """Determine feedback target using rule-based logic across all evaluated candidates."""
    from collections import Counter

    passed = [c for c in evaluated if c.passed]
    if passed:
        return FeedbackDecision(
            need_feedback=False,
            feedback_target=None,
            reason=f"{len(passed)}/{len(evaluated)}개 후보 통과.",
            dominant_failure_pattern=[],
            suggested_relaxations=[],
        )

    # Tally failure stages
    stage_counter: Counter[str] = Counter()
    for c in evaluated:
        if c.failed_stage:
            stage_counter[c.failed_stage] += 1

    # Tally weak point items across all candidates
    item_counter: Counter[str] = Counter()
    for c in evaluated:
        for wp in c.weak_points:
            item_counter[wp.item] += 1

    dominant = [item for item, cnt in item_counter.most_common(3) if cnt >= 2]
    dominant_stages = [s for s, cnt in stage_counter.most_common(3)]

    # Determine feedback target
    all_env_failed = all(
        not (c.environment_filter or {}).get("pass", True) for c in evaluated
    )
    c3_failures = sum(
        1 for c in evaluated
        for wp in c.weak_points if wp.item == "C3"
    )

    if all_env_failed:
        feedback_target = None
        reason = "모든 후보가 환경 제약 하드 필터에서 탈락. 제약 완화 필요."
        relaxations = ["derived_constraints 수치 범위 재검토"]
    elif c3_failures >= len(evaluated) // 2:
        feedback_target = "module2a"
        reason = "다수 후보에서 required_atoms/interaction_primitives 미충족."
        relaxations = ["required_atoms 일부를 preferred_atoms로 완화"]
    else:
        feedback_target = "module2c"
        reason = f"주요 실패 단계: {', '.join(dominant_stages)}. 후보 조합 품질 개선 필요."
        relaxations = ["더 다양한 connection_modes를 활용한 후보 재생성"]

    return FeedbackDecision(
        need_feedback=True,
        feedback_target=feedback_target,
        reason=reason,
        dominant_failure_pattern=dominant or dominant_stages,
        suggested_relaxations=relaxations,
    )
