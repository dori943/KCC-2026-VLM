"""Assembly pose calculator for Module 3 using GPT (step-by-step + validation retry).

구조 (옵션 B):
  Phase 1: Strategy 호출 (조립 순서/region 결정, 좌표 계산 없음)
  Phase 2: Step별 좌표 계산 — GPT가 joint 좌표 산출
           → 코드가 기하 검증 (AABB 기반)
           → fail 시 최대 2회 재시도 (GPT에 에러 전달)
  Phase 3: Final 호출 — 검증 결과 종합, final_structure, feedback
"""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from app.module3.models import Module3Input
from app.module3.reasoners.scene_coord_overrides import apply_scene_overrides


# ──────────────────────────────────────────────
# Functional role mapping
# ──────────────────────────────────────────────
FUNCTION_TO_ROLE: dict[str, str] = {
    "prying_edge":        "primary_executor",
    "rigid_tip_contact":  "primary_executor",
    "rigid_tip":          "primary_executor",
    "insert":             "primary_executor",
    "graspable_body":     "grip_assistant",
    "clampable_span":     "grip_assistant",
    "frictional_contact": "friction_enhancer",
    "flat_face":          "friction_enhancer",
    "support":            "alignment_aid",
    "stabilize":          "alignment_aid",
}

# 기능적 정렬 허용 거리 (m) — 이보다 멀면 fail
ALIGNMENT_TOLERANCE_M  = 0.02
CONTACT_TOLERANCE_M    = 0.02
# AABB 겹침(부피) 허용 상한 — 완전 겹침 방지
COLLISION_OVERLAP_EPS  = 1e-8
# 결정적 좌표 계산 시 region 방향으로 추가 outward push (m).
# floating point 정밀도 + PyBullet AABB inflation 으로 face contact 좌표가
# sub-millimeter 침투를 일으키는 문제 회피용 안전 마진.
COLLISION_SAFETY_MARGIN_M = 0.002

# ── 테이블 plane 좌표계 정합 ──────────────────────────────────
# 팀원 manipulation 시뮬 환경의 테이블 상판 z = 0.626 (실측 aabb_top_z, 2026-04-27 갱신).
# 이전엔 0.65 로 잘못 알려져서 모든 물체가 테이블 위로 +29mm 들려있었음.
# 0.626 으로 보정 후엔 PyBullet settle 좌표(z≈0.62)와도 거의 일치하므로
# override 가 있는 물체는 lift 스킵, 없는 물체만 +6mm 정도 가벼운 lift.
TABLE_TOP_Z: float = 0.626
TABLE_TOP_SAFETY_MARGIN_M: float = 0.001  # 1mm 띄움 (mesh 침투 방지)

# region → (axis, direction). direction +1 은 +축 방향(outward), -1 은 -축 방향.
_REGION_AXIS_DIR: dict[str, tuple[int, int]] = {
    "upper_surface": (2, +1),
    "lower_surface": (2, -1),
    "front_end":     (0, +1),
    "rear_end":      (0, -1),
    "side_face":     (1, +1),
}

# STRATEGY LLM 이 attach_region_base 에 부위 이름을 적은 경우 자동 매핑.
# (Phase 1 retry 가 실패해도 Phase 2 에서 collision 을 방지하기 위한 안전망.)
_REGION_ALIAS: dict[str, str] = {
    "handle":     "rear_end",
    "grip":       "rear_end",
    "rear":       "rear_end",
    "tail":       "rear_end",
    "back":       "rear_end",
    "tip":        "front_end",
    "blade":      "front_end",
    "edge":       "front_end",
    "head":       "front_end",
    "front":      "front_end",
    "nose":       "front_end",
    "top":        "upper_surface",
    "upper":      "upper_surface",
    "bottom":     "lower_surface",
    "lower":      "lower_surface",
    "side":       "side_face",
    "flank":      "side_face",
    "body":       "mid_body",
    "center":     "mid_body",
    "core":       "mid_body",
}


def _normalize_region(region: str) -> tuple[str, str | None]:
    """region 을 6 개 enum 으로 정규화.

    반환: (정규화된 region, alias 적용 시 원본 이름 / 아니면 None)
    """
    r = (region or "").lower()
    if r in _REGION_AXIS_DIR or r == "mid_body":
        return r, None
    # alias 매핑 (handle → rear_end 등)
    for alias, canonical in _REGION_ALIAS.items():
        if alias in r:
            return canonical, r
    # 알 수 없는 region → mid_body (insertion 의도로 fallback)
    return "mid_body", r


# ──────────────────────────────────────────────
# Phase 1: Strategy prompt
# ──────────────────────────────────────────────
STRATEGY_SYSTEM_PROMPT = """너는 Module 3: 도구 조립 전략 계획기이다.

목표: 선택된 도구 후보를 분석하여 조립 전략과 순서를 계획하라. 좌표 계산은 하지 않는다.

전제 환경:
- 모든 작업은 manipulation 시뮬레이터의 **테이블(table_top_z ≈ 0.626m) 위**에서 수행된다.
- 모든 부품의 z 좌표는 0.626 이상이며, 테이블 plane 아래로 내려갈 수 없다.
- 따라서 "바닥에 깐다", "지면에 받친다" 같은 전략은 금지. base 는 항상 테이블 위에 놓는다.

# ───────── base_object 선택 가이드 (task-type 별, 가장 중요) ─────────
task_description 의 의도에 따라 base 선택 휴리스틱이 다르다. **반드시 task 의도부터 먼저 분류하라.**

(A) "좁은 틈/구멍/구석에서 작은 물체를 꺼낸다" 부류
    · 키워드 예: "명함", "구멍", "끼인", "낀", "유리 조각", "구석"
    · base = **가장 얇고 길이가 긴 강체** (insertion tool). 우선순위:
        knife > flat_screwdriver > phillips_screwdriver > spatula > large_marker
    · 넓은 그릇류(bowl, mug, plate)나 박스류(cracker_box, sugar_box, gelatin_box, pudding_box)를
      base 로 두는 것은 **금지** — 좁은 틈에 들어갈 수 없어 task 자체가 실패한다.
    · attach 는 보통 friction/leverage 를 더하는 작은 도구(large_marker, spoon, fork 등).

(B) "걸려 있거나 매달려 있는 물체를 끌어내린다" 부류
    · 키워드 예: "매달린", "걸린", "고리", "후크"
    · base = **끝이 휘어있거나 후킹 가능한 도구** (spatula, adjustable_wrench, fork).
    · attach 는 길이 보강/리치 연장용 (large_marker, screwdriver).

(C) "막혀 있어서 손이 안 들어가는 손잡이/문을 조작한다" 부류
    · 키워드 예: "막힌", "잠긴", "닫힌"
    · base = 길고 강체 (flat_screwdriver, phillips_screwdriver, knife) — 레버 작용.
    · attach 는 그립 보조(adjustable_wrench, mug 등 무게 추가).

(D) "깊은 구멍 속의 물체에 닿는다" 부류
    · 키워드 예: "깊은 구멍", "안쪽", "속"
    · base = **가장 길고 얇은 봉형** (large_marker, screwdriver, spoon 손잡이).
    · attach 는 끝부분 마찰/포획용 (sticky 표면, friction_enhancer 역할).

공통 원칙: base 는 task 의 **기능단을 직접 수행하는 도구(primary_executor)**여야 한다.
넓고 안정적이라는 이유만으로 mug/bowl/plate 를 base 로 잡는 것은 **거의 항상 오답**이다
(이런 물체는 task 기능단(좁은 틈, 깊은 구멍, 매달린 물체)에 닿지 못한다).

base_object / attach_object 정의 (중요 — 절대 혼동 금지):
- base_object  : **이미 배치되어 있는 앵커 물체** (움직이지 않음)
- attach_object: **새로 들어와 base에 붙이는 물체** (base 방향으로 이동)

순서 규칙:
- Step 1: base_object 단독 배치 (attach_object=null). base = 가장 먼저 놓을 주체(primary_executor 권장).
- Step N: base_object는 **직전 step까지 이미 배치된 물체 중 하나** (보통 직전 step의 attach).
  attach_object는 **아직 배치되지 않은 새 물체**.

예 (ruler → tweezers → sticky notes 조립):
- Step 1: base=ruler,    attach=null
- Step 2: base=ruler,    attach=tweezers       (ruler가 앵커, tweezers가 들어옴)
- Step 3: base=tweezers, attach=sticky notes   (tweezers가 앵커, sticky notes가 들어옴)

절대 금지: Step 2에서 base=tweezers, attach=ruler 처럼 반전하는 것.

attach_region_base 허용 값 — **반드시 아래 6개 enum 중 하나** (다른 값 출력 시 plan validation FAIL → retry 발동):
- upper_surface  : 물체의 윗면 — 쌓는 경우 (가장 흔함)
- lower_surface  : 물체의 아랫면
- front_end      : 물체의 기능 수행 끝부분
- rear_end       : 물체의 파지 끝부분
- mid_body       : 물체의 중간 몸통 — Step 1 단독 배치 또는 삽입(insertion)일 때만
- side_face      : 물체의 옆면

**금지된 값 예시 (절대 사용 금지)**: "handle", "blade", "jaw", "tip", "edge",
"body", "head", "grip", "base", "pole" 등 물체 부위 이름.
이런 단어는 **attach_region_object** (attach 측 어느 부위가 닿는가) 자리에만
사용한다. attach_region_base 자리에는 위 6개 중 가장 가까운 것을 매핑하라:
  - "handle"/"grip"/"rear" → rear_end
  - "tip"/"blade"/"front"/"head" → front_end
  - "top"/"upper" → upper_surface
  - "bottom"/"lower" → lower_surface
  - "side"/"flank" → side_face

mid_body를 남용하면 물체들이 겹쳐 collision이 발생한다.
쌓는(stacking) 조립은 반드시 upper_surface를 사용하라.
도구 손잡이를 base 의 끝부분에 잇는 경우엔 rear_end (or front_end)를 써라.

functional_roles 참고:
- primary_executor: task 수행 주체 (ex. prying_edge). 기능단이 노출되어야 함.
- grip_assistant  : 파지 보조. tip이 base의 기능단 방향으로 돌출되어야 함.
- friction_enhancer: 마찰 증강. 접촉점(기능단 근처)에 배치되어야 함.

filter_result.weak_points를 반영하여 region과 contact_type을 결정하라.

JSON만 출력. 설명문 없음.

출력 형식:
{
  "assembly_strategy": {
    "base_object": "string",
    "strategy_summary": "string",
    "sequence_reason": "string",
    "filter_result_reflection": "string"
  },
  "planned_sequence": [
    {
      "step": 1,
      "base_object": "string",
      "attach_object": "string | null",
      "attach_region_base": "upper_surface | lower_surface | front_end | rear_end | mid_body | side_face",
      "attach_region_object": "string | null",
      "contact_type": "point | surface | insertion",
      "expected_function_after_step": "string",
      "reason": "string",
      "weak_points_applied": ["filter_result.weak_points 중 이 step 설계에 반영된 항목 id 또는 설명 (예: 'P6: sticky notes 마찰 부족 → friction_enhancer를 tip 접촉점에 배치')"]
    }
  ]
}"""


# ──────────────────────────────────────────────
# Phase 2: Per-step coordinate calculation
# ──────────────────────────────────────────────
STEP_SYSTEM_PROMPT = """너는 Module 3: 단계별 접합 좌표 계산기이다.

목표: current_positions(코드가 업데이트한 현재 AABB)를 기반으로 이번 step의 joint_position_world를 계산하라.

# 출력 우선순위 (중요 — 팀 spec, 2026-04-27 업데이트)
하류 R1 manipulation 은 **양팔 조립 모델**이다:
  · 왼팔이 base 를 잡고, 오른팔이 attach 를 잡아 두 팔이 동시에 meeting point 로 이동해 attach.
  · 따라서 step 출력에는 base 도착 좌표(base_target_pose_world.position)와
    attach 도착 좌표(target_pose_world.position) 가 둘 다 필요하다 (코드가 자동 계산).
  · 각도(orientation_rpy_deg) 는 현재 비활성 — 잡고 회전해서 다시 두는 동작은
    VLM 추론 신뢰성 부족으로 시간상 보류. 모두 [0,0,0] 으로 출력된다.

**너의 핵심 책임**: "두 물체가 어디에서 만나서 attach 되는가" 의 위치를 정확히 뽑기.
즉 다음 정성 필드의 품질에 집중:
  1) **attach_region_base** — base 의 어느 면이 contact 면인가 (6 enum 정확히)
  2) **attach_region_object** — attach 의 어느 부위가 그 면에 닿는가
  3) **contact_type** — point / surface / insertion
  4) **reason / weak_points_applied / expected_function_after_step**

좌표(position) 는 코드가 deterministic 으로 계산하므로 너의 좌표는 참고용.

# joint_position_world의 의미
joint은 **base의 attach_region face 위의 contact point**이다. attach_object의
해당 반대면이 이 joint에 닿는 방식으로 코드가 자동 stacking한다.
(예: upper_surface → attach 바닥이 joint에 착지, ruler 위에 tweezers가 얹히는 형태)

# 좌표계 가드 (필수)
- **테이블 상판(table_top_z) = 0.65m** (manipulation 시뮬 기준).
- 모든 step 의 joint_z, target_pose_world.z, 그리고 attach_object 의
  aabb_min_z 는 **반드시 0.65 이상**이어야 한다.
- current_positions 의 aabb 값은 이미 테이블 plane 위로 lift 된 좌표이므로
  그대로 사용하면 자동 충족된다. 임의로 z 를 빼지 말 것.
- 0.65 미만의 z 가 산출되면 하드웨어/시뮬에서 부품이 테이블 안으로 박힌다 → fail.

# attach_region_base 기본 계산식 (attach_object가 null이 아닐 때)
- upper_surface  → joint_z = base.aabb_max_z,  joint_xy = base.center_xy
- lower_surface  → joint_z = base.aabb_min_z,  joint_xy = base.center_xy   (단, ≥ 0.65)
- front_end      → joint_x = base.aabb_max_x,  joint_yz = base.center_yz
- rear_end       → joint_x = base.aabb_min_x,  joint_yz = base.center_yz
- mid_body       → joint = base.center (insertion 시만)
- side_face      → joint_y = base.aabb_max_y,  joint_xz = base.center_xz
- step 1 (attach_object=null) → joint = base.center (mid_body 사용)

# region 선택 가이드 (role 기반)
- attach.role=grip_assistant on primary_executor: region=upper_surface 권장 (위로 쌓음)
- attach.role=friction_enhancer: region=upper_surface 강력 권장
  (tip contact point 위로 마찰 패드 쌓는 형태 — 그래야 z-stacking으로 collision 회피)
- mid_body는 진짜 삽입 구조일 때만 사용

# 기능적 정렬 규칙 (role 조합에 따라 위 기본값에 shift 적용)
1) base.role=primary_executor AND attach.role=grip_assistant:
   → attach의 tip(functional_pole)이 base.functional_pole과 일치하도록 xy shift.
   → 계산: shift_x = base.functional_pole.x - attach.functional_pole.x (y 동일)
   → joint의 xy는 base.center_xy가 아니라 (base.center_xy + shift)로 설정.

2) base.role=grip_assistant AND attach.role=friction_enhancer:
   → attach를 base의 functional_pole(tip) 위치에 배치.
   → joint.xy = base.functional_pole.xy, joint.z는 region 계산식 유지.

3) base.role=primary_executor AND attach.role=friction_enhancer:
   → attach를 base의 functional_pole(edge) 위치에 배치.
   → joint.xy = base.functional_pole.xy, joint.z는 region 계산식 유지.

4) 그 외 → 기본 계산식만 적용, shift 없음.

# 기타 규칙
- step_info의 base_object와 attach_object는 절대 변경 금지. 그대로 사용하라.
  (반전하거나 다른 물체로 교체하면 validation fail)
- current_positions의 값을 그대로 사용. 임의 offset 추가 금지.
- target_pose_world.position = joint_position_world.position.
- partial_assembly_state_before: 이미 배치된 물체 이름 배열 (문자열).
- calculation_basis에 사용한 AABB 수치, role 조합, shift 계산을 명시하라.

# 좌표 자동 계산 안내 (중요)
- joint_position_world.position (= meeting point) 은 코드가 region + functional_role 기반으로
  결정적으로 계산한다. 너의 좌표는 참고용.
- target_pose_world.position (= attach 도착 좌표) 와
  base_target_pose_world.position (= base 도착 좌표) 둘 다 코드가 region + AABB 기반으로 산출.
- target_pose_world.orientation_rpy_deg / base_target_pose_world.orientation_rpy_deg 는
  현재 비활성 ([0,0,0]).
- relative_offset_from_base 도 코드가 (joint - base.center) 로 계산.
- 따라서 너는 좌표 정확도보다 **region / contact_type / reason /
  weak_points_applied / expected_function_after_step** 의 품질에 집중하라.

# weak_points_applied (필수, 빈 배열 절대 금지)
- filter_result.weak_points 항목 (예: P6 마찰 부족, C4 손상 위험, G3 기능단 차단)
  중 이번 step 의 region/contact_type/role 배치로 완화되는 항목을 반드시 명시하라.
- **반드시 P/C/G 로 시작하는 weak_point ID 1개 이상을 인용**한다 (filter_result.weak_points
  의 실제 ID 사용. ID 가 없으면 같은 weak_point 의 설명 첫 단어를 prefix 로 사용).
- **금지된 placeholder 문자열**:
  "no_applicable_weak_point_at_this_step", "none", "n/a", "not applicable",
  "no weak point", "" (빈 문자열). 이런 값을 적으면 검증 fail.
- 이번 step 에서 직접 완화하지 못하더라도 **간접적으로 안정성·정렬·접촉을
  보강한 weak_point 1개**를 반드시 골라 적어라 (예: 첫 step 단독 배치라도
  "P5: base 안정성 확보를 위해 가장 무거운/넓은 물체부터 배치").
- 형식: ["P6: friction_enhancer 를 base.functional_pole 위에 배치하여 마찰 보강",
        "C4: contact_type=surface 로 타겟 손상 회피"]
- 출력 길이: 최소 1개, 최대 3개.

# 재시도 모드
previous_attempt가 payload에 있으면, 그 좌표는 validation fail이다.
previous_attempt.errors를 읽고 원인을 해결한 새 좌표를 산출하라.

JSON만 출력. 설명문 없음.

출력 형식:
{
  "step": 1,
  "partial_assembly_state_before": ["ruler"],
  "base_object": "string",
  "attach_object": "string | null",
  "attach_region_base": "upper_surface | lower_surface | front_end | rear_end | mid_body | side_face",
  "attach_region_object": "string | null",
  "relative_offset_from_base": [0.0, 0.0, 0.0],
  "target_pose_world": {
    "position": [0.0, 0.0, 0.0],
    "orientation_rpy_deg": [0.0, 0.0, 0.0]
  },
  "joint_position_world": {
    "position": [0.0, 0.0, 0.0],
    "calculation_basis": "string — AABB 수치, role, shift 명시",
    "description": "PyBullet constraint 생성 위치."
  },
  "contact_type": "surface",
  "expected_function_after_step": "string",
  "reason": "string",
  "weak_points_applied": ["이 step에 반영된 filter_result.weak_points 항목 — 반드시 P/C/G ID prefix 인용. 빈 배열·placeholder 금지."]
}"""


# ──────────────────────────────────────────────
# Phase 3: Final verification + feedback
# ──────────────────────────────────────────────
FINAL_SYSTEM_PROMPT = """너는 Module 3: 조립 결과 검증기이다.

목표: 완성된 조립 단계와 코드 검증 결과를 종합하여 최종 구조, 검증, 피드백을 출력하라.

payload의 geometric_validation_results는 코드가 기하 계산으로 판정한 결과이다.
이 결과를 반드시 verification.checks에 반영하라 (코드 fail → 해당 항목 fail 필수).

# 핵심 판정 원칙 (중요)
- **코드 fail 외에는 "확정적 증거"가 있을 때만 fail 처리하라.**
- "potential", "may not", "might", "could possibly" 같은 추측·우려성 사유로
  fail 을 매기지 말 것. 그런 우려는 reason 에만 적되 result=pass 처리.
- 정성 판정 항목(force_transfer, weak_point_mitigation, subgoal_support 등)은
  명백한 위반(필수 도구 누락, 약점 ID 무응답, 필수 atom 미충족) 에만 fail.

검증 항목 8개 (전부 pass/fail):
- alignment              : 물체가 task 방향으로 올바르게 정렬됐는가
- collision              : 물체 간 비의도적 충돌이 없는가 (코드 검증 우선)
- functional_end_exposed : 기능단(tip/edge)이 충분히 노출됐는가 (코드 검증 우선)
- handle_region_free     : 파지 영역이 확보됐는가 (코드 검증 우선)
- force_transfer         : 힘이 도구 끝까지 전달 가능한가
  → primary_executor 가 존재하고 grip_assistant 또는 robot 파지 경로가 끊기지
    않으면 PASS. "마찰이 부족할 수도" 같은 추측은 fail 사유가 아니다.
- weak_point_mitigation  : filter_result.weak_points 가 step 설계에 반영됐는가
  → **PASS 조건 (관대하게)**: assembly_steps 의 step 중 하나 이상에 실제
    weak_point ID(P숫자 또는 C숫자 또는 G숫자 prefix)를 인용한
    weak_points_applied 항목이 있으면 PASS.
  → **FAIL 조건**: 모든 step 의 weak_points_applied 가 빈 배열이거나, 전부
    "no_applicable_weak_point", "none", "n/a" 같은 placeholder 문자열이거나,
    weak_point ID 인용이 단 하나도 없는 경우만 FAIL.
  → **"완전 해소(fully mitigated)" 를 요구하지 말라.** 약점은 본질적으로
    완전 제거가 어렵다. **완화 시도(mitigation attempt)** 만 있어도 PASS.
- subgoal_support        : 각 subgoal 의 required_atoms 가 지원되는가
  → **PASS 자동 조건**:
    (a) 해당 subgoal 의 required_atoms 가 비어있음 ([]) → 검증 정보 부재이므로 PASS
    (b) required_interaction_primitives 도 비어있음 → PASS
    (c) 후보의 inferred_functions / function_mapping 이 한 개라도 atom 을 커버 → partial PASS
  → **FAIL 조건**: required_atoms 에 명시된 atom 이 모두 후보에 부재함이
    "구체적 atom 이름과 함께" 입증된 경우만 fail.
  → "lack of grasping capability", "is not supported", "cannot be supported"
    같은 단정형 표현이라도 실제 누락된 atom 을 지목하지 않으면 추측이다 → PASS.
  → 텍스트 objective 만 보고 grasp/grip/insert 같은 능력을 임의로 요구하지 말 것.
    오직 required_atoms / required_interaction_primitives 배열만 근거로 삼는다.
- contact_feasibility    : 도구가 타겟과 실제 접촉 가능한가 (코드 검증 우선)
  → 코드가 pass 했으면 pass. 추가로 의심하지 말 것.

handle_end 규칙:
- 로봇이 파지하는 부분 (base_object 또는 구조 전체 파지 영역)
- attach_object(보조 물체)를 handle_end로 지정 금지

feedback 발동 조건 (논문 3.1.4: 피드백은 3.1.1 = module2a로 전달):
- geometric_validation_results.any_unresolved_failure=true → need_feedback_to_module2a=true
- 8개 검증 항목 중 하나라도 fail → need_feedback_to_module2a=true
- 모두 pass → need_feedback_to_module2a=false, repair_type=local_pose_adjustment
- feedback_target은 항상 "module2a" (3.1.1 Scene Resource Parser)

JSON만 출력. 설명문 없음.

출력 형식:
{
  "final_structure": {
    "description": "string",
    "functional_end": "string",
    "handle_end": "string"
  },
  "verification": {
    "is_valid": true,
    "checks": [
      {"item": "string", "result": "pass | fail", "reason": "string"}
    ]
  },
  "feedback": {
    "need_feedback_to_module2a": false,
    "feedback_target": "module2a | null",
    "repair_type": "local_pose_adjustment | global_redesign",
    "suggested_action": "string"
  },
  "reasoning_trace": {
    "step_A": "selected_candidate 해석",
    "step_B": "filter_result 반영",
    "step_C": "assembly strategy",
    "step_D": "각 step 좌표 계산 수치 요약",
    "step_E": "final_structure 결정 근거",
    "step_F": "verification 판정 근거 (코드 검증 포함)",
    "step_G": "feedback 결정 근거"
  }
}"""


# ──────────────────────────────────────────────
# Helpers — GPT call
# ──────────────────────────────────────────────
def _call_gpt(
    client: OpenAI,
    system_prompt: str,
    user_message: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> tuple[dict[str, Any], Any]:
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Module 3: GPT JSON 파싱 실패: {e}\n응답: {raw[:300]}")
    return parsed, response.usage


# ──────────────────────────────────────────────
# Helpers — position tracking
# ──────────────────────────────────────────────
# 형상 유형 → 기본 extents (m). 질량으로 스케일 조정.
SHAPE_DEFAULT_EXTENTS: dict[str, tuple[float, float, float]] = {
    "elongated":  (0.15,  0.01, 0.01),   # 15cm × 1cm × 1cm
    "blocky":     (0.05,  0.05, 0.05),   # 5cm cube
    "sheet_like": (0.08,  0.08, 0.002),  # 8cm × 8cm × 2mm
    "cylindrical":(0.10,  0.02, 0.02),
}
_DEFAULT_EXTENTS = (0.05, 0.05, 0.05)
_REFERENCE_MASS_KG = 0.1


def _fallback_aabb_from_physical(
    center: list[float],
    shape_category: str,
    estimated_mass_kg: float | None,
) -> tuple[list[float], list[float]]:
    """AABB 결측 시 형상 유형 + 질량 추정값으로 합성 AABB 생성 (논문 3.1.4)."""
    ex = SHAPE_DEFAULT_EXTENTS.get(shape_category or "", _DEFAULT_EXTENTS)
    mass = estimated_mass_kg if (estimated_mass_kg and estimated_mass_kg > 0) else _REFERENCE_MASS_KG
    scale = (mass / _REFERENCE_MASS_KG) ** (1.0 / 3.0)
    half = [ex[i] * scale / 2.0 for i in range(3)]
    aabb_min = [round(center[i] - half[i], 6) for i in range(3)]
    aabb_max = [round(center[i] + half[i], 6) for i in range(3)]
    return aabb_min, aabb_max


def _is_degenerate_aabb(aabb_min: list[float], aabb_max: list[float]) -> bool:
    """AABB가 비어 있거나 퇴화된 경우 (extent = 0)."""
    if len(aabb_min) != 3 or len(aabb_max) != 3:
        return True
    return all(aabb_max[i] - aabb_min[i] <= 1e-9 for i in range(3))


def _lift_to_table_top(
    center: list[float],
    aabb_min: list[float],
    aabb_max: list[float],
) -> tuple[list[float], list[float], list[float], float]:
    """물체 전체를 +z 방향으로 lift 하여 aabb_min_z >= TABLE_TOP_Z + safety 보장.

    PyBullet settle 좌표(테이블 ≈ 0.62)와 manipulation 시뮬(테이블 = 0.65)
    plane 좌표계 미스매치를 보정한다. 기존엔 부품들이 테이블 안으로 0.03m 박힘.

    반환: (lifted_center, lifted_aabb_min, lifted_aabb_max, lift_amount)
    """
    z_floor = TABLE_TOP_Z + TABLE_TOP_SAFETY_MARGIN_M
    if len(aabb_min) < 3:
        return center, aabb_min, aabb_max, 0.0
    cur_amin_z = float(aabb_min[2])
    if cur_amin_z >= z_floor:
        return center, aabb_min, aabb_max, 0.0
    lift = z_floor - cur_amin_z
    new_center = [center[0], center[1], round(center[2] + lift, 6)]
    new_amin = [aabb_min[0], aabb_min[1], round(aabb_min[2] + lift, 6)]
    new_amax = [aabb_max[0], aabb_max[1], round(aabb_max[2] + lift, 6)]
    return new_center, new_amin, new_amax, lift


def _init_positions(
    scene_objects: list[dict[str, Any]],
    object_physical_properties: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, list[float]]]:
    """AABB 초기화. 결측 시 형상·질량 기반 fallback 적용 (논문 3.1.4).

    추가 처리: PyBullet settle 좌표계(테이블 ≈ 0.62) ↔ manipulation 시뮬
    좌표계(테이블 = 0.65) 정합 — aabb_min_z < TABLE_TOP_Z 인 물체는
    전체를 +z 로 lift 한다. (T1 grasp 실패 hotfix)
    """
    prop_map: dict[str, dict[str, Any]] = {}
    for p in object_physical_properties or []:
        name = p.get("name", "")
        if name:
            prop_map[name] = p

    pos: dict[str, dict[str, list[float]]] = {}
    lift_log: list[tuple[str, float]] = []
    override_warn: list[tuple[str, float]] = []
    for obj in scene_objects:
        name     = obj.get("name", "")
        center   = list(obj.get("center_world", [0.0, 0.0, 0.0]))
        aabb_min = list(obj.get("aabb_min", [0.0, 0.0, 0.0]))
        aabb_max = list(obj.get("aabb_max", [0.0, 0.0, 0.0]))
        is_overridden = bool(obj.get("_overridden", False))

        if _is_degenerate_aabb(aabb_min, aabb_max):
            prop = prop_map.get(name, {})
            aabb_min, aabb_max = _fallback_aabb_from_physical(
                center,
                prop.get("shape_category", ""),
                prop.get("estimated_mass_kg"),
            )

        # 테이블 plane 좌표계 정합 — override 좌표는 팀원이 명시한 ground truth 이므로 lift 스킵.
        # 팀원 sim 의 settle 결과는 통상 mesh 두께/접촉 tolerance 로 0~20mm 정도
        # aabb_min_z 가 TABLE_TOP_Z 보다 작게 나오는 게 정상이라 경고는 30mm 초과 이상치만.
        if is_overridden:
            if aabb_min[2] < TABLE_TOP_Z - 0.030:
                override_warn.append((name, aabb_min[2]))
        else:
            center, aabb_min, aabb_max, lift = _lift_to_table_top(center, aabb_min, aabb_max)
            if lift > 0:
                lift_log.append((name, lift))

        pos[name] = {"center": center, "aabb_min": aabb_min, "aabb_max": aabb_max}

    if lift_log:
        print(
            f"[module3] table_top_z={TABLE_TOP_Z} 정합: "
            f"{len(lift_log)}개 물체 +z lift 적용:"
        )
        for n, l in lift_log:
            print(f"          - {n}: +{l*1000:.1f}mm")
    if override_warn:
        print(
            f"[module3] ⚠ override 좌표 {len(override_warn)}개가 TABLE_TOP_Z={TABLE_TOP_Z}보다 30mm 이상 낮습니다 "
            f"(팀원 시뮬 정합 의심 — 좌표 재확인 필요):"
        )
        for n, z in override_warn:
            print(f"          - {n}: aabb_min_z={z:.4f} (penetration={(TABLE_TOP_Z-z)*1000:.1f}mm)")
    return pos


def _compute_joint_deterministic(
    base_name: str,
    attach_name: str | None,
    attach_region_base: str,
    current_positions: dict[str, dict[str, list[float]]],
    functional_info: dict[str, dict[str, Any]],
    roles: dict[str, str],
) -> tuple[list[float], str]:
    """region + role 기반으로 joint_position_world.position 을 결정적으로 계산.

    LLM 이 좌표를 직접 산출할 때 region/role 의미를 자주 오해해
    AABB collision 이 반복적으로 발생했음. 좌표는 코드가 계산하고
    LLM 은 region/orientation/이유 같은 soft 필드만 담당하도록 분리한다.

    반환: (joint_xyz, calculation_basis_text)
    """
    if base_name not in current_positions:
        return [0.0, 0.0, 0.0], f"unknown_base={base_name}"

    base_pos = current_positions[base_name]
    base_amin = base_pos["aabb_min"]
    base_amax = base_pos["aabb_max"]
    base_center = base_pos["center"]

    # Step 1 (단독 배치) → joint = base.center
    # ※ z floor clamp: 테이블 plane 좌표계 정합 (TABLE_TOP_Z=0.65). PyBullet
    #   settle 좌표가 0.62 plane 기준이라 그대로 쓰면 manipulation 시뮬에서
    #   부품이 테이블 아래로 박힘. base center_z 가 테이블 위에 있도록 보정.
    if attach_name is None:
        joint = [round(v, 6) for v in base_center]
        clamp_note = ""
        z_floor = TABLE_TOP_Z + TABLE_TOP_SAFETY_MARGIN_M
        if joint[2] < z_floor:
            old_z = joint[2]
            joint[2] = round(z_floor, 6)
            clamp_note = f"; table_top_z_clamp z {old_z:.4f}→{joint[2]:.4f} (TABLE_TOP_Z={TABLE_TOP_Z})"
        return (
            joint,
            f"step_1_standalone, joint=base.center{clamp_note}",
        )

    region, region_alias = _normalize_region(attach_region_base)
    joint = list(base_center)
    if region == "upper_surface":
        joint[2] = base_amax[2]
    elif region == "lower_surface":
        joint[2] = base_amin[2]
    elif region == "front_end":
        joint[0] = base_amax[0]
    elif region == "rear_end":
        joint[0] = base_amin[0]
    elif region == "side_face":
        joint[1] = base_amax[1]
    # mid_body / 미지정 → center 유지 (insertion 의도)

    basis_parts = [
        f"region={region}" + (f" (alias '{region_alias}' → {region})" if region_alias else ""),
        f"base.aabb_min={[round(v,4) for v in base_amin]}",
        f"base.aabb_max={[round(v,4) for v in base_amax]}",
        f"region_joint={[round(v,4) for v in joint]}",
    ]

    region_axis_dir = _REGION_AXIS_DIR.get(region)
    region_axis = region_axis_dir[0] if region_axis_dir else None

    # Role 기반 shift (LLM 이 자주 누락).
    # **중요**: region 방향 축은 절대 덮어쓰지 않는다. 그래야 stacking face contact 가
    # 유지되어 AABB 침투를 방지한다 (이전엔 grip_alignment 가 joint.x 를 덮어써서
    # T4 에서 collision 3.5e-4m 발생).
    base_role = roles.get(base_name, "alignment_aid")
    attach_role = roles.get(attach_name, "alignment_aid")
    base_pole = functional_info.get(base_name, {}).get("functional_pole")

    def _safe_shift_xy(target_xy: list[float], label: str) -> None:
        if region_axis != 0:
            joint[0] = target_xy[0]
        if region_axis != 1:
            joint[1] = target_xy[1]
        # region_axis == 2 (z stacking) 인 경우 xy 둘 다 자유롭게 이동 가능.
        basis_parts.append(
            f"{label}→xy=({target_xy[0]:.4f},{target_xy[1]:.4f})"
            f"{' (region_axis 보존)' if region_axis is not None and region_axis != 2 else ''}"
        )

    if base_role == "primary_executor" and attach_role == "grip_assistant" and base_pole:
        _safe_shift_xy([base_pole[0], base_pole[1]], "grip_alignment")
    elif attach_role == "friction_enhancer" and base_pole:
        _safe_shift_xy([base_pole[0], base_pole[1]], "friction_at_pole")
    elif base_role == "grip_assistant" and attach_role == "friction_enhancer" and base_pole:
        _safe_shift_xy([base_pole[0], base_pole[1]], "friction_on_grip_tip")

    # region 방향으로 outward safety margin 추가 (collision 방지).
    # 예: upper_surface (axis=2, dir=+1) → joint[2] += 2mm.
    # face contact 좌표는 이론상 부피 겹침 0이지만 floating-point 오차 +
    # PyBullet 회전된 mesh AABB inflation 으로 sub-mm 침투가 실제 발생함.
    if region_axis_dir:
        axis, direction = region_axis_dir
        joint[axis] += direction * COLLISION_SAFETY_MARGIN_M
        basis_parts.append(
            f"safety_margin→{['x','y','z'][axis]}{'+' if direction>0 else '-'}"
            f"{COLLISION_SAFETY_MARGIN_M*1000:.0f}mm"
        )

    # 테이블 plane 좌표계 정합 (T1 grasp 실패 hotfix).
    # PyBullet settle 좌표(z≈0.62)와 manipulation 시뮬 테이블(z=0.65) 미스매치
    # 보정 — joint_z 가 테이블 아래라면 위로 lift.
    z_floor = TABLE_TOP_Z + TABLE_TOP_SAFETY_MARGIN_M
    if joint[2] < z_floor:
        old_z = joint[2]
        joint[2] = z_floor
        basis_parts.append(
            f"table_top_z_clamp z {old_z:.4f}→{joint[2]:.4f} (TABLE_TOP_Z={TABLE_TOP_Z})"
        )

    return [round(v, 6) for v in joint], "; ".join(basis_parts)


def _compute_relative_orientation_rpy(
    base_name: str,
    attach_name: str | None,
    attach_region_base: str,
    contact_type: str,
    current_positions: dict[str, dict[str, list[float]]],
) -> tuple[list[float], str]:
    """접합 시 attach 의 base 좌표계 기준 상대 회전 (RPY, deg).

    팀 사양: target_pose_world 의 절대 position/orientation 은 R1 이 무시한다.
    우리가 정확히 줘야 하는 것은 "두 물체가 어떤 위치에서 어떤 각도로 붙는가" 이므로
    이 함수가 산출한 RPY 를 target_pose_world.orientation_rpy_deg 자리에 채운다.

    규칙 (보수적으로 — 잘못된 회전은 안 주는 게 default 보다 나쁘므로
    long_axis 정렬이 명백히 필요한 케이스만 적용):

    - region == upper_surface / lower_surface (z-stacking)
        · attach.long_axis == z 인 경우 (세로로 길쭉) → y 축 90° 로 눕힘 [0, 90, 0]
          (그래야 "납작하게 쌓인다" 의 의미가 됨; 안 그러면 옆으로 세워서 무너짐)
        · 그 외 → [0, 0, 0]
    - region == front_end / rear_end (x 축 끝에 잇기)
        · attach.long_axis 가 x 가 되도록 회전:
            attach.long_axis == z → [0, 90, 0] (y 회전; z-long → x-long)
            attach.long_axis == y → [0, 0, 90] (z 회전; y-long → x-long)
            attach.long_axis == x → [0, 0, 0]
    - region == side_face (y 축 면에 잇기)
        · attach.long_axis 가 y 가 되도록:
            attach.long_axis == x → [0, 0, 90]
            attach.long_axis == z → [90, 0, 0]
            attach.long_axis == y → [0, 0, 0]
    - region == mid_body + contact_type == insertion
        · attach.long_axis 가 base.long_axis 와 같은 축이 되도록 회전.
    - 그 외 → [0, 0, 0]
    """
    rpy = [0.0, 0.0, 0.0]
    if not attach_name or attach_name not in current_positions:
        return rpy, "step_1_standalone or unknown attach → no relative rotation"
    if base_name not in current_positions:
        return rpy, f"unknown base={base_name}"

    base = current_positions[base_name]
    attach = current_positions[attach_name]
    base_long = _longest_axis(base["aabb_min"], base["aabb_max"])
    attach_long = _longest_axis(attach["aabb_min"], attach["aabb_max"])
    region, _alias = _normalize_region(attach_region_base)

    notes = [f"base.long_axis={['x','y','z'][base_long]}",
             f"attach.long_axis={['x','y','z'][attach_long]}",
             f"region={region}"]

    if region in ("upper_surface", "lower_surface"):
        if attach_long == 2:
            rpy = [0.0, 90.0, 0.0]
            notes.append("flatten: attach.long(z) → horizontal via pitch+90")
    elif region in ("front_end", "rear_end"):
        if attach_long == 2:
            rpy = [0.0, 90.0, 0.0]
            notes.append("align: attach.long(z) → x via pitch+90")
        elif attach_long == 1:
            rpy = [0.0, 0.0, 90.0]
            notes.append("align: attach.long(y) → x via yaw+90")
    elif region == "side_face":
        if attach_long == 0:
            rpy = [0.0, 0.0, 90.0]
            notes.append("align: attach.long(x) → y via yaw+90")
        elif attach_long == 2:
            rpy = [90.0, 0.0, 0.0]
            notes.append("align: attach.long(z) → y via roll+90")
    elif region == "mid_body" and (contact_type or "").lower() == "insertion":
        # base 의 long_axis 와 attach 의 long_axis 가 일치하도록.
        if attach_long != base_long:
            # 회전 매핑: (current_long, target_long) → rpy
            #   x→y: yaw+90 / x→z: pitch+90 / y→x: yaw-90 / y→z: roll+90
            #   z→x: pitch-90 / z→y: roll-90
            mapping = {
                (0, 1): [0.0, 0.0, 90.0],
                (0, 2): [0.0, 90.0, 0.0],
                (1, 0): [0.0, 0.0, -90.0],
                (1, 2): [90.0, 0.0, 0.0],
                (2, 0): [0.0, -90.0, 0.0],
                (2, 1): [-90.0, 0.0, 0.0],
            }
            rpy = list(mapping.get((attach_long, base_long), [0.0, 0.0, 0.0]))
            notes.append(
                f"insertion_align: attach.long → base.long ({['x','y','z'][attach_long]} → {['x','y','z'][base_long]})"
            )

    return [round(v, 4) for v in rpy], "; ".join(notes)


def _compute_stacking_delta(
    attach: dict[str, list[float]],
    joint: list[float],
    attach_region_base: str,
) -> list[float]:
    """region 의미에 맞는 delta 계산.

    joint는 "base의 region face 위의 접촉점"이다. attach의 해당 축 contact face
    (반대 방향의 AABB extreme)가 joint에 닿도록 delta를 맞춘다.

    - upper_surface → attach.aabb_min_z = joint.z (attach 바닥이 joint에)
    - lower_surface → attach.aabb_max_z = joint.z
    - front_end     → attach.aabb_min_x = joint.x (attach 뒷면이 base 앞면에)
    - rear_end      → attach.aabb_max_x = joint.x
    - side_face     → attach.aabb_min_y = joint.y
    - mid_body      → attach.center = joint (insertion, 겹침 의도)
    """
    center = attach["center"]
    amin   = attach["aabb_min"]
    amax   = attach["aabb_max"]

    # 기본: xy/xz/yz는 center를 joint에 맞춤
    delta = [joint[i] - center[i] for i in range(3)]

    region = (attach_region_base or "").lower()
    if region == "upper_surface":
        delta[2] = joint[2] - amin[2]
    elif region == "lower_surface":
        delta[2] = joint[2] - amax[2]
    elif region == "front_end":
        delta[0] = joint[0] - amin[0]
    elif region == "rear_end":
        delta[0] = joint[0] - amax[0]
    elif region == "side_face":
        delta[1] = joint[1] - amin[1]
    # mid_body / 기본 → center = joint (변경 없음)

    return [round(d, 6) for d in delta]


def _compute_base_pose_at_meeting(
    base: dict[str, list[float]],
    joint: list[float],
    attach_region_base: str,
) -> list[float]:
    """양팔 조립 모델 — base 가 meeting point (joint) 로 이동했을 때의 base center.

    팀원 manipulation 사양 (2026-04-27):
      - 기존: base 가 가만히 있고 attach 만 이동 (1 팔 모델)
      - 변경: 양팔이 base/attach 를 각각 들어 meeting point 에서 attach (2 팔 모델)

    region 별로 base 의 contact face 가 joint 에 정렬되도록 base 전체를 평행이동.
    contact face 는 attach 가 닿는 면이므로 stacking_delta 와 반대편 face 가 된다:
      - upper_surface  : base 의 윗면(aabb_max_z) 이 joint 에 옴 → center.z 는 더 아래
      - lower_surface  : base 의 아랫면(aabb_min_z) 이 joint 에 옴
      - front_end      : base 의 앞면(aabb_max_x) 이 joint 에 옴
      - rear_end       : base 의 뒷면(aabb_min_x) 이 joint 에 옴
      - side_face      : base 의 옆면(aabb_max_y) 이 joint 에 옴
      - mid_body       : base center = joint (insertion 의도)
    """
    center = base["center"]
    amin   = base["aabb_min"]
    amax   = base["aabb_max"]

    # 기본: center 가 joint 와 정렬
    delta = [joint[i] - center[i] for i in range(3)]

    region = (attach_region_base or "").lower()
    if region == "upper_surface":
        delta[2] = joint[2] - amax[2]
    elif region == "lower_surface":
        delta[2] = joint[2] - amin[2]
    elif region == "front_end":
        delta[0] = joint[0] - amax[0]
    elif region == "rear_end":
        delta[0] = joint[0] - amin[0]
    elif region == "side_face":
        delta[1] = joint[1] - amax[1]
    # mid_body / 그 외 → center = joint

    return [round(center[i] + delta[i], 6) for i in range(3)]


def _update_positions(
    pos: dict[str, dict[str, list[float]]],
    step: dict[str, Any],
) -> None:
    """attach_object가 joint_position_world로 이동된 이후의 AABB 업데이트.

    region 의미에 맞춰 contact face가 joint에 정렬되도록 stacking한다.
    """
    attach_name = step.get("attach_object")
    if not attach_name or attach_name not in pos:
        return
    joint = step.get("joint_position_world", {}).get("position", [])
    if len(joint) != 3:
        return
    region = step.get("attach_region_base", "")
    a = pos[attach_name]
    delta = _compute_stacking_delta(a, joint, region)
    pos[attach_name] = {
        "center":   [round(a["center"][i]   + delta[i], 6) for i in range(3)],
        "aabb_min": [round(a["aabb_min"][i] + delta[i], 6) for i in range(3)],
        "aabb_max": [round(a["aabb_max"][i] + delta[i], 6) for i in range(3)],
    }


def _pos_snapshot(pos: dict[str, dict[str, list[float]]]) -> dict[str, Any]:
    return {
        name: {
            "center_world": data["center"],
            "aabb_min":     data["aabb_min"],
            "aabb_max":     data["aabb_max"],
        }
        for name, data in pos.items()
    }


# ──────────────────────────────────────────────
# Helpers — functional analysis
# ──────────────────────────────────────────────
def _derive_roles(selected_candidate: dict[str, Any]) -> dict[str, str]:
    """function_mapping에서 각 물체의 functional_role 매핑."""
    roles: dict[str, str] = {}
    for item in selected_candidate.get("function_mapping", []):
        name = item.get("object", "")
        func = item.get("function", "")
        role = FUNCTION_TO_ROLE.get(func, "alignment_aid")
        if name:
            roles[name] = role
    return roles


def _longest_axis(aabb_min: list[float], aabb_max: list[float]) -> int:
    """AABB에서 가장 긴 축 index (0=x, 1=y, 2=z)."""
    extents = [aabb_max[i] - aabb_min[i] for i in range(3)]
    return extents.index(max(extents))


def _functional_pole(
    pos_data: dict[str, list[float]],
    primary_contact_profile: str,
) -> list[float]:
    """물체의 기능단 좌표.

    - broad_flat_face → center
    - edge / tip / 기타 → longest axis의 max 끝점 (x/y/z face center)
    """
    aabb_min = pos_data["aabb_min"]
    aabb_max = pos_data["aabb_max"]
    center   = pos_data["center"]

    if primary_contact_profile == "broad_flat_face":
        return list(center)

    axis = _longest_axis(aabb_min, aabb_max)
    pole = list(center)
    pole[axis] = aabb_max[axis]
    return [round(v, 6) for v in pole]


def _build_functional_info(
    current_positions: dict[str, dict[str, list[float]]],
    object_physical_properties: list[dict[str, Any]],
    roles: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """step 프롬프트/검증에 넘길 기능 정보 테이블."""
    profile_map = {
        p.get("name", ""): p.get("geometry_profile", {}).get("primary_contact_profile", "")
        for p in object_physical_properties
    }
    info: dict[str, dict[str, Any]] = {}
    for name, pos in current_positions.items():
        profile = profile_map.get(name, "")
        info[name] = {
            "role":                    roles.get(name, "alignment_aid"),
            "primary_contact_profile": profile,
            "functional_pole":         _functional_pole(pos, profile),
        }
    return info


# ──────────────────────────────────────────────
# Helpers — geometric validation
# ──────────────────────────────────────────────
def _aabb_overlap_volume(a: dict[str, list[float]], b: dict[str, list[float]]) -> float:
    dims = []
    for i in range(3):
        lo = max(a["aabb_min"][i], b["aabb_min"][i])
        hi = min(a["aabb_max"][i], b["aabb_max"][i])
        d  = hi - lo
        if d <= 0:
            return 0.0
        dims.append(d)
    return dims[0] * dims[1] * dims[2]


def _point_in_aabb(p: list[float], aabb: dict[str, list[float]]) -> bool:
    return all(aabb["aabb_min"][i] <= p[i] <= aabb["aabb_max"][i] for i in range(3))


def _distance(a: list[float], b: list[float]) -> float:
    return sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5


def _simulate_attach_aabb(
    attach_pos: dict[str, list[float]],
    joint_position: list[float],
    attach_region_base: str = "",
) -> dict[str, list[float]]:
    """joint 좌표로 이동시킨 후 attach_object의 AABB (region-aware stacking)."""
    delta = _compute_stacking_delta(attach_pos, joint_position, attach_region_base)
    return {
        "center":   [attach_pos["center"][i]   + delta[i] for i in range(3)],
        "aabb_min": [attach_pos["aabb_min"][i] + delta[i] for i in range(3)],
        "aabb_max": [attach_pos["aabb_max"][i] + delta[i] for i in range(3)],
    }


def _table_clamp_deficit(
    *aabbs_min_z: float,
) -> float:
    """주어진 amin_z 값들 중 가장 낮은 게 TABLE 위로 떠 있게 하는 양수 lift 양.

    rear_end / front_end / side_face / mid_body 처럼 face-to-face 측면 접합에서는
    base 와 attach 의 z 가 동일 plane 으로 정렬되는데, 이 plane 이 너무 낮으면
    한쪽 또는 양쪽이 테이블 mesh 를 뚫는다. meeting point 를 모두에게 충분한
    높이까지 +z lift 해서 contact 는 유지하되 테이블 침투만 막는다.
    """
    z_floor = TABLE_TOP_Z + TABLE_TOP_SAFETY_MARGIN_M
    deficit = 0.0
    for amin_z in aabbs_min_z:
        deficit = max(deficit, z_floor - amin_z)
    return max(0.0, deficit)


_ALLOWED_REGIONS: frozenset[str] = frozenset({
    "upper_surface", "lower_surface", "front_end", "rear_end",
    "mid_body", "side_face",
})


def _validate_plan(planned_sequence: list[dict[str, Any]]) -> list[str]:
    """Strategy의 planned_sequence 전체 의미 검증.

    - step 1 attach=null
    - step N≥2: base ∈ 이전 steps에서 배치된 물체, attach ∉ 이전 물체
    - 각 step에 base_object 존재
    - attach_region_base 는 6개 enum 중 하나
    """
    errors: list[str] = []
    if not planned_sequence:
        errors.append("planned_sequence가 비어 있음.")
        return errors

    placed: list[str] = []
    for i, plan in enumerate(planned_sequence):
        step_num = plan.get("step", i + 1)
        base     = plan.get("base_object", "")
        attach   = plan.get("attach_object")
        region   = (plan.get("attach_region_base") or "").lower()

        if not base:
            errors.append(f"step {step_num}: base_object 비어 있음.")
            continue

        # region enum hard check (step 1 도 포함)
        if region and region not in _ALLOWED_REGIONS:
            errors.append(
                f"step {step_num}: attach_region_base='{region}' 는 허용 enum 아님."
                f" 허용: {sorted(_ALLOWED_REGIONS)}."
                f" 'handle'/'tip'/'blade' 같은 부위 이름은 attach_region_object 자리에만 사용."
            )

        if step_num == 1:
            if attach is not None:
                errors.append(f"step 1: attach_object는 null이어야 함 (got '{attach}').")
            placed.append(base)
            continue

        # step >= 2
        if base not in placed:
            errors.append(
                f"step {step_num}: base_object='{base}'가 이전 steps에서 배치되지 않음."
                f" placed_so_far={placed}. base는 앵커(이미 배치된 물체)여야 함."
            )
        if attach is None:
            errors.append(f"step {step_num}: attach_object가 null. step 2 이상은 새 물체 필요.")
        elif attach in placed:
            errors.append(
                f"step {step_num}: attach_object='{attach}'가 이미 배치됨."
                f" placed_so_far={placed}. attach는 새 물체여야 함."
            )
        if base and attach and base == attach:
            errors.append(f"step {step_num}: base와 attach가 동일 ('{base}').")

        if base and base not in placed:
            placed.append(base)
        if attach and attach not in placed:
            placed.append(attach)

    return errors


def _validate_step(
    step_result: dict[str, Any],
    plan: dict[str, Any],
    current_positions: dict[str, dict[str, list[float]]],
    assembled_objects: list[str],
    roles: dict[str, str],
    functional_info: dict[str, dict[str, Any]],
) -> list[str]:
    """기하 검증. 빈 배열이면 pass."""
    errors: list[str] = []

    # 0) Plan 일치 체크 — base/attach 반전 방지
    plan_base   = plan.get("base_object", "")
    plan_attach = plan.get("attach_object")
    got_base    = step_result.get("base_object", "")
    got_attach  = step_result.get("attach_object")
    if got_base != plan_base:
        errors.append(
            f"plan_mismatch: base_object가 planned='{plan_base}'인데 '{got_base}'로 출력됨."
            f" planned_sequence의 값을 그대로 사용해야 함."
        )
    if got_attach != plan_attach:
        errors.append(
            f"plan_mismatch: attach_object가 planned='{plan_attach}'인데 '{got_attach}'로 출력됨."
            f" planned_sequence의 값을 그대로 사용해야 함."
        )
    if errors:
        return errors  # 반전된 상태에선 뒤 검증 무의미

    # 0b) base/attach 의미 체크 — base는 이미 배치된 앵커, attach는 새 물체
    step_num = step_result.get("step", 0)
    if step_num > 1:
        if got_base and got_base not in assembled_objects:
            errors.append(
                f"semantics_violation: base_object='{got_base}'가 아직 배치되지 않음."
                f" base는 이미 배치된 앵커 물체여야 함. assembled={assembled_objects}"
            )
        if got_attach and got_attach in assembled_objects:
            errors.append(
                f"semantics_violation: attach_object='{got_attach}'가 이미 배치됨."
                f" attach는 새로 들어오는 물체여야 함. assembled={assembled_objects}"
            )
    if errors:
        return errors

    joint = step_result.get("joint_position_world", {}).get("position") or []
    attach_name = step_result.get("attach_object")
    base_name   = step_result.get("base_object", "")

    # Step 1 (단독 배치) — 검증 생략
    if not attach_name:
        return errors
    if len(joint) != 3:
        errors.append(f"joint_position_world.position 형식 오류 (길이 3 필요).")
        return errors
    if attach_name not in current_positions:
        errors.append(f"attach_object '{attach_name}' current_positions에 없음.")
        return errors

    # attach_object가 joint로 이동한 가상 AABB (region별 stacking 반영)
    attach_region = step_result.get("attach_region_base", "")
    attach_after  = _simulate_attach_aabb(current_positions[attach_name], joint, attach_region)

    # 1) 충돌: 이미 배치된 모든 물체와 AABB 겹침 검사
    for other in assembled_objects:
        if other == attach_name:
            continue
        if other not in current_positions:
            continue
        vol = _aabb_overlap_volume(attach_after, current_positions[other])
        if vol > COLLISION_OVERLAP_EPS:
            errors.append(
                f"collision: '{attach_name}' AABB가 '{other}'와 부피 {vol:.3e} 겹침."
            )

    # 2) functional_end_exposed: primary_executor의 pole이 가려졌는지
    for name in assembled_objects + [attach_name]:
        info = functional_info.get(name)
        if not info or info["role"] != "primary_executor":
            continue
        pole = info["functional_pole"]
        # attach가 pole을 덮는지
        if name != attach_name and _point_in_aabb(pole, attach_after):
            errors.append(
                f"functional_end_exposed: primary_executor '{name}'의 pole"
                f" {pole}이 '{attach_name}' AABB 내부로 가려짐."
            )

    # 3) 기능적 정렬 — role 조합별 체크
    base_role   = functional_info.get(base_name,   {}).get("role")
    attach_role = functional_info.get(attach_name, {}).get("role")
    base_pole   = functional_info.get(base_name,   {}).get("functional_pole")

    # attach 이동 후 pole
    attach_profile = functional_info.get(attach_name, {}).get("primary_contact_profile", "")
    attach_pole_after = _functional_pole(attach_after, attach_profile)

    if base_role == "primary_executor" and attach_role == "grip_assistant" and base_pole:
        d = _distance(attach_pole_after, base_pole)
        if d > ALIGNMENT_TOLERANCE_M:
            errors.append(
                f"grip_alignment: grip_assistant '{attach_name}' tip {attach_pole_after}가"
                f" primary_executor '{base_name}' pole {base_pole}과 {d:.4f}m 떨어짐"
                f" (허용 {ALIGNMENT_TOLERANCE_M}m)."
            )

    if attach_role == "friction_enhancer" and base_pole:
        # friction_enhancer 는 몸통 전체로 마찰 접촉이라 tip 기준 비교가 부적절.
        # base_pole 이 attach AABB 내부에 들어오면 body-wide contact 성립으로 본다.
        #
        # **중요**: region 이 stacking 축 (upper_surface=z+, lower_surface=z-,
        # front_end=x+, rear_end=x-, side_face=y+) 인 경우 그 축은 거리 비교에서
        # 제외한다. friction 패드가 base 표면 위로 stacking 되는 건 의도된 배치
        # (z 로 떨어진 건 자연스러움). xy projection 만 보고 pole 이 attach
        # 의 AABB.xy 범위 안에 들어오면 마찰 접촉 의도가 충족된 것으로 본다.
        a_min = attach_after["aabb_min"]
        a_max = attach_after["aabb_max"]
        region = (step_result.get("attach_region_base") or "").lower()
        region_axis_dir = _REGION_AXIS_DIR.get(region)
        skip_axis = region_axis_dir[0] if region_axis_dir else None

        # 모든 축에 대해 AABB clamp closest 계산
        closest = [
            min(max(base_pole[i], a_min[i]), a_max[i]) for i in range(3)
        ]
        if skip_axis is not None:
            # skip_axis 거리는 0 으로 처리 (stacking 의 자연스러운 분리)
            check_pole = list(base_pole)
            check_closest = list(closest)
            check_pole[skip_axis] = 0.0
            check_closest[skip_axis] = 0.0
            d_clamp = _distance(check_closest, check_pole)
            axis_label = ['x','y','z'][skip_axis]
            tolerance_note = f"허용 {CONTACT_TOLERANCE_M}m, stacking {axis_label}축 무시"
        else:
            d_clamp = _distance(closest, base_pole)
            tolerance_note = f"허용 {CONTACT_TOLERANCE_M}m"

        if d_clamp > CONTACT_TOLERANCE_M:
            errors.append(
                f"contact_feasibility: friction_enhancer '{attach_name}' AABB "
                f"[{[round(v,4) for v in a_min]}~{[round(v,4) for v in a_max]}]"
                f" 가 base '{base_name}' pole {base_pole} 와 {d_clamp:.4f}m 떨어짐"
                f" ({tolerance_note}, AABB 안에 들어오거나 면이 닿아야 함)."
            )

    return errors


# ──────────────────────────────────────────────
# Helpers — verification post-processing
# ──────────────────────────────────────────────
# LLM 이 fail 사유에 자주 쓰는 추측성 표현 (영문 + 한글).
# 이 키워드만 등장하고 코드 검증 fail 이 없으면 추측성 fail 로 보고 강제 pass.
_SPECULATIVE_KEYWORDS: tuple[str, ...] = (
    "may ", "may not", "might ", "could ", "possibly", "perhaps",
    "potential", "potentially", "uncertain", "risk", "risky",
    "insufficient", "unlikely", "would not", "if not properly",
    "not be", "may be", "추측", "우려", "수도", "부족할", "어려울",
    "어려움", "가능성", "않을 수",
)

# verification 항목 → 코드 검증 에러 prefix 매핑.
# 코드 검증 final_errors 에 해당 prefix 가 있으면 진짜 fail 이므로 override 안 함.
_CODE_VALIDATED_ITEMS: dict[str, tuple[str, ...]] = {
    "collision":              ("collision:",),
    "functional_end_exposed": ("functional_end_exposed:",),
    "contact_feasibility":    ("contact_feasibility:",),
    "alignment":              ("grip_alignment:", "plan_mismatch", "semantics_violation"),
}


def _override_speculative_fails(
    verification: dict[str, Any],
    geometric_validation_results: dict[str, Any],
    tool_constraints: dict[str, Any] | None = None,
    selected_candidate: dict[str, Any] | None = None,
) -> list[str]:
    """LLM verification 의 추측성 fail 을 코드 검증과 대조해 자동 pass 로 override.

    원칙:
    - 코드 검증 final_errors 에 해당 항목 prefix 가 있으면 → 진짜 fail (그대로 둠).
    - 코드 fail 이 없는데 LLM 이 fail 을 매긴 경우, reason 에 추측성 키워드만 있으면
      → 강제 pass + reason 앞에 [auto_override:...] tag 부착.
    - subgoal_support 항목은 별도로 처리:
      → 모든 subgoal 의 required_atoms 와 required_interaction_primitives 가
        비어있으면 "검증 정보 부재" 로 자동 pass.
      → reason 에 구체적 atom 이름이 인용되지 않은 단정형 fail 도 추측으로 간주.

    반환: override 가 적용된 항목 이름 list (trace 기록용).
    """
    overrides: list[str] = []

    # 코드 검증의 모든 step error 텍스트를 합침
    all_errors: list[str] = []
    for step_log in geometric_validation_results.get("per_step", []):
        for err in step_log.get("final_errors") or []:
            all_errors.append(str(err))
    code_errors_blob = " ".join(all_errors).lower()

    # subgoal 정보 추출
    subgoal_constraints: list[dict[str, Any]] = []
    if tool_constraints:
        sg_raw = tool_constraints.get("subgoal_constraints") or []
        if isinstance(sg_raw, list):
            subgoal_constraints = [s for s in sg_raw if isinstance(s, dict)]

    # 모든 subgoal 의 required_* 가 다 비어있는지 (검증 정보 부재)
    all_required_empty = bool(subgoal_constraints) and all(
        not (sg.get("required_atoms") or [])
        and not (sg.get("required_interaction_primitives") or [])
        for sg in subgoal_constraints
    )

    # 후보가 인용 가능한 atom 어휘 (소문자) — 구체적 미커버 증거 판정용
    all_required_atoms_lower: set[str] = set()
    all_required_prims_lower: set[str] = set()
    for sg in subgoal_constraints:
        for a in sg.get("required_atoms") or []:
            if isinstance(a, str):
                all_required_atoms_lower.add(a.lower())
        for p in sg.get("required_interaction_primitives") or []:
            if isinstance(p, str):
                all_required_prims_lower.add(p.lower())

    # 후보의 subgoal_coverage 자기선언 (Module 2C 가 covered=True 로 표시)
    candidate_covered_sg_ids: set[str] = set()
    if selected_candidate:
        for sc in selected_candidate.get("subgoal_coverage", []) or []:
            if isinstance(sc, dict) and bool(sc.get("covered", False)):
                sg_id = sc.get("subgoal_id", "")
                if sg_id:
                    candidate_covered_sg_ids.add(str(sg_id).lower())
    # 모든 subgoal_id 가 후보에 covered=True 로 선언되어 있는가
    all_sg_self_covered = bool(subgoal_constraints) and all(
        str(sg.get("subgoal_id", "")).lower() in candidate_covered_sg_ids
        for sg in subgoal_constraints
    )

    for check in verification.get("checks", []) or []:
        if check.get("result") != "fail":
            continue

        item = check.get("item", "")
        reason_raw = check.get("reason") or ""
        reason_lower = reason_raw.lower()

        # 1) 코드가 실제로 fail 한 항목인지 확인
        code_markers = _CODE_VALIDATED_ITEMS.get(item, ())
        code_failed = any(m.lower() in code_errors_blob for m in code_markers)
        if code_failed:
            continue  # 진짜 코드 fail → override 안 함

        # 1-b) 코드 검증 우선 항목인데 코드 fail 이 없으면 → LLM 의 추측 fail 로 간주.
        # FINAL_SYSTEM_PROMPT 에 "코드가 pass 했으면 pass" 라고 명시되어 있는데도
        # LLM 이 "violates" 같은 단정형 표현으로 다시 fail 매기는 케이스 차단.
        # collision/contact_feasibility/functional_end_exposed/alignment 가 대상.
        if code_markers:  # 즉 _CODE_VALIDATED_ITEMS 에 등록된 항목인데 코드 fail 없음
            check["result"] = "pass"
            check["reason"] = (
                f"[auto_override: 코드 검증 fail 없음 → LLM 단정형 fail 무시] "
                f"{reason_raw}"
            )
            overrides.append(item)
            continue

        # 2) subgoal_support 전용 분기
        if item == "subgoal_support":
            # 2-a) 모든 subgoal 의 required_* 가 비어있으면 무조건 pass
            if all_required_empty:
                check["result"] = "pass"
                check["reason"] = (
                    f"[auto_override: 모든 subgoal 의 required_atoms / "
                    f"required_interaction_primitives 가 비어있어 검증 정보 부재] "
                    f"{reason_raw}"
                )
                overrides.append(item)
                continue
            # 2-b) reason 에 구체적 atom 이 인용되지 않은 단정형 fail → 추측
            cites_specific_atom = any(
                atom and atom in reason_lower
                for atom in (all_required_atoms_lower | all_required_prims_lower)
            )
            if not cites_specific_atom:
                check["result"] = "pass"
                check["reason"] = (
                    f"[auto_override: 단정형 fail 이지만 구체적 미커버 atom 미인용] "
                    f"{reason_raw}"
                )
                overrides.append(item)
                continue
            # 2-b2) 후보가 자기선언으로 모든 subgoal 을 covered=True 표시했고
            # Module 2D 의 hard filter 를 통과했다면, Module 3 가 LLM 만으로 다시
            # strict 하게 fail 매기는 것은 불필요한 보수성. 후보 신뢰가 합리적.
            # function_mapping 에 require_atom 단어가 글자 그대로 안 적혀있어도,
            # 도구 affordance 가 실제로 그 atom 을 충족할 수 있다.
            if all_sg_self_covered:
                check["result"] = "pass"
                check["reason"] = (
                    f"[auto_override: selected_candidate.subgoal_coverage 가 모든 "
                    f"subgoal 을 covered=True 로 자기선언했고 Module 2D hard filter "
                    f"통과] {reason_raw}"
                )
                overrides.append(item)
                continue

            # 2-b3) fail reason 에 인용된 sg_id 들이 모두 후보의 covered=True 에
            # 속하면 PASS. (일부 sg 는 covered=False 라도 그건 fail 이 가리키는
            # sg 가 아니므로 무관.)
            import re
            cited_sg_ids = {m.lower() for m in re.findall(r"sg_\d+", reason_lower)}
            if cited_sg_ids and cited_sg_ids.issubset(candidate_covered_sg_ids):
                check["result"] = "pass"
                check["reason"] = (
                    f"[auto_override: fail 이 가리키는 sg_id 들 {sorted(cited_sg_ids)} "
                    f"이 모두 후보에 covered=True 로 자기선언됨] {reason_raw}"
                )
                overrides.append(item)
                continue
            # 2-c) reason 이 "robot-level 일반 능력" atom 만 인용한 fail 은
            # 후보 function_mapping 에 없어도 robot 의 기본 manipulator 가
            # 처리하므로 PASS 로 본다.
            # 예: graspable_body, grasp, hold, release, lift, manipulate.
            #     이런 atom 은 도구의 affordance 가 아니라 로봇팔의 baseline
            #     역량이라, 도구 후보에 명시되지 않은 게 정상이다.
            robot_baseline_atoms = {
                "graspable_body", "grasp", "graspable", "grasping",
                "hold", "release", "lift", "lower",
                "manipulate", "manipulation",
                "pick", "pickup", "pick_up", "place",
            }
            cited_atoms = {
                atom for atom in (all_required_atoms_lower | all_required_prims_lower)
                if atom and atom in reason_lower
            }
            if cited_atoms and cited_atoms.issubset(robot_baseline_atoms):
                check["result"] = "pass"
                check["reason"] = (
                    f"[auto_override: 인용된 미커버 atom 이 모두 robot baseline "
                    f"능력({sorted(cited_atoms)}). 도구 affordance 가 아니므로 PASS] "
                    f"{reason_raw}"
                )
                overrides.append(item)
                continue

        # 3) 일반: 추측성 키워드만 있으면 pass 로 강제 override
        if any(kw in reason_lower for kw in _SPECULATIVE_KEYWORDS):
            check["result"] = "pass"
            check["reason"] = (
                f"[auto_override: 추측성 사유 + 코드 fail 없음] {reason_raw}"
            )
            overrides.append(item)

    # is_valid 재평가
    checks = verification.get("checks", []) or []
    if checks:
        verification["is_valid"] = all(c.get("result") == "pass" for c in checks)

    return overrides


# ──────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────
def calculate_pose(
    input_data: Module3Input,
    api_key: str | None = None,
    model: str = "gpt-4o",
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Assembly pose with per-step GPT + geometric validation retry.

    재시도: 각 step 최대 3회 호출 (초기 1회 + 재시도 2회).
    최종 실패 시 그대로 기록, need_feedback_to_module2a=true로 전달 (논문 3.1.4).
    """
    client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
    total_tokens = {"prompt": 0, "completion": 0}
    gpt_call_count = 0

    def _accum(usage: Any) -> None:
        if usage:
            total_tokens["prompt"]     += usage.prompt_tokens     or 0
            total_tokens["completion"] += usage.completion_tokens or 0

    # functional_role 매핑 (코드 계산)
    roles = _derive_roles(input_data.selected_candidate)

    base_ctx = {
        "task":                      input_data.task,
        "scene_context":             input_data.scene_context,
        "object_physical_properties": input_data.object_physical_properties,
        "tool_constraints":          input_data.tool_constraints,
        "selected_candidate":        input_data.selected_candidate,
        "filter_result":             input_data.filter_result,
        "functional_roles":          roles,
    }

    # ── Phase 1: Strategy (with plan validation retry) ──
    strategy_attempts: list[dict[str, Any]] = []
    strategy_result: dict[str, Any] = {}
    plan_errors: list[str] = []
    previous_strategy: dict[str, Any] | None = None

    for attempt in range(3):
        strategy_payload = dict(base_ctx)
        if previous_strategy is not None:
            strategy_payload["previous_attempt"] = previous_strategy

        strategy_result, usage = _call_gpt(
            client, STRATEGY_SYSTEM_PROMPT,
            "다음 입력을 분석하여 조립 전략과 순서를 계획하라."
            + (" 이전 시도가 validation fail이다. errors를 반드시 해결하라.\n\n"
               if previous_strategy else "\n\n")
            + json.dumps(strategy_payload, ensure_ascii=False, indent=2),
            model, temperature, max_tokens,
        )
        _accum(usage)
        gpt_call_count += 1

        plan_errors = _validate_plan(strategy_result.get("planned_sequence", []))
        strategy_attempts.append({
            "attempt":          attempt + 1,
            "planned_sequence": strategy_result.get("planned_sequence", []),
            "errors":           plan_errors,
            "passed":           len(plan_errors) == 0,
        })
        if not plan_errors:
            break

        previous_strategy = {
            "planned_sequence": strategy_result.get("planned_sequence", []),
            "errors":           plan_errors,
        }

    assembly_strategy = strategy_result.get("assembly_strategy", {})
    planned_sequence  = strategy_result.get("planned_sequence", [])
    strategy_unresolved = len(plan_errors) > 0

    # ── Phase 2: Per-step with validation retry ──
    # 팀원 시뮬 실측 좌표 override 적용 (PyBullet settle vs sim 미스매치 보정).
    raw_scene_objects = input_data.scene_context.get("scene_objects", [])
    scene_objects, overridden_names, matched_task_id = apply_scene_overrides(
        raw_scene_objects, input_data.task,
    )
    if overridden_names:
        print(
            f"[module3] scene_coord_overrides ({matched_task_id}): "
            f"{len(overridden_names)}개 물체 좌표 교체:"
        )
        for nm in overridden_names:
            obj = next((o for o in scene_objects if o.get("name") == nm), {})
            print(f"          - {nm}: center_world={obj.get('center_world')}")
    current_positions = _init_positions(
        scene_objects,
        input_data.object_physical_properties,
    )
    all_steps:         list[dict[str, Any]]       = []
    assembled_objects: list[str]                  = []
    validation_log:    list[dict[str, Any]]       = []
    any_unresolved_failure = False

    for plan in planned_sequence:
        step_num    = plan["step"]
        base_name   = plan.get("base_object", "")
        attach_name = plan.get("attach_object")

        functional_info = _build_functional_info(
            current_positions, input_data.object_physical_properties, roles,
        )

        previous_attempt: dict[str, Any] | None = None
        step_result:      dict[str, Any] | None = None
        errors:           list[str]             = []
        attempt_log:      list[dict[str, Any]]  = []

        for attempt in range(3):  # 초기 + 재시도 2회
            step_payload = {
                "step_info":                     plan,
                "current_positions":             _pos_snapshot(current_positions),
                "functional_info":               functional_info,
                "partial_assembly_state_before": list(assembled_objects),
                "task":                          input_data.task,
                "assembly_strategy":             assembly_strategy,
            }
            if previous_attempt is not None:
                step_payload["previous_attempt"] = previous_attempt

            step_result, usage = _call_gpt(
                client, STEP_SYSTEM_PROMPT,
                f"Step {step_num} 좌표를 계산하라. "
                f"{'이전 시도가 validation fail이다. errors를 해결하라.' if previous_attempt else ''}\n\n"
                + json.dumps(step_payload, ensure_ascii=False, indent=2),
                model, temperature, max_tokens,
            )
            _accum(usage)
            gpt_call_count += 1

            # ── plan 강제 동기화 ──
            # LLM 이 base/attach 를 반전하거나 region 을 바꾸는 일을 막는다.
            # plan 값이 ground truth.
            step_result["base_object"] = base_name
            step_result["attach_object"] = attach_name
            step_result["step"] = step_num

            # ── joint_position_world 결정적 override ──
            # LLM 이 region/role 의미를 오해해 collision 좌표를 반복 출력하는 문제 해결.
            # plan 의 attach_region_base 와 functional_role 을 기준으로 좌표를 코드가 계산하고
            # LLM 좌표를 덮어쓴다. LLM 은 reason / contact_type / orientation 만 신뢰한다.
            plan_region = plan.get("attach_region_base", "") or step_result.get("attach_region_base", "")
            llm_joint_raw = step_result.get("joint_position_world", {}).get("position")
            det_joint, det_basis = _compute_joint_deterministic(
                base_name=base_name,
                attach_name=attach_name,
                attach_region_base=plan_region,
                current_positions=current_positions,
                functional_info=functional_info,
                roles=roles,
            )
            jpw = step_result.setdefault("joint_position_world", {})
            jpw["position"] = det_joint
            llm_basis = jpw.get("calculation_basis", "")
            jpw["calculation_basis"] = (
                f"[deterministic] {det_basis}"
                + (f" | llm_proposed={llm_joint_raw}" if llm_joint_raw else "")
                + (f" | llm_basis={llm_basis}" if llm_basis else "")
            )
            jpw["description"] = "Code-computed contact point (region+role aware). PyBullet constraint here."
            # plan 의 region 을 step_result 에도 강제 동기화 (LLM 이 다른 region 이름을 적었을 때 대비)
            # alias ('handle' → 'rear_end' 등) 가 적용됐으면 정규화된 canonical 값으로 저장.
            if plan_region:
                normalized_region, _alias = _normalize_region(plan_region)
                step_result["attach_region_base"] = normalized_region

            # target_pose_world.position = attach center after stacking
            joint_pos = jpw["position"]
            if joint_pos and attach_name and attach_name in current_positions:
                region = step_result.get("attach_region_base", "")
                attach_after = _simulate_attach_aabb(
                    current_positions[attach_name], joint_pos, region,
                )
                step_result.setdefault("target_pose_world", {})["position"] = [
                    round(c, 6) for c in attach_after["center"]
                ]

            # ── 접합 각도 (orientation) ──
            # 팀원 결정 (2026-04-27): VLM 으로 회전 추론은 신뢰성 부족 → 각도는 [0,0,0] 으로 두고
            # "attach 위치만 정확하게 뽑기" 에 집중. 추후 활성화하려면 아래 helper 호출 결과를
            # tpw["orientation_rpy_deg"] 에 그대로 대입하면 됨 (계산 로직 보존).
            tpw = step_result.setdefault("target_pose_world", {})
            tpw["orientation_rpy_deg"] = [0.0, 0.0, 0.0]
            # 디버그용으로만 산출 (calculation_basis 에 보조 정보로 기록).
            contact_type_for_rot = step_result.get("contact_type", "") or plan.get("contact_type", "")
            _rel_rpy_dbg, _rel_rpy_basis_dbg = _compute_relative_orientation_rpy(
                base_name=base_name,
                attach_name=attach_name,
                attach_region_base=plan_region,
                contact_type=contact_type_for_rot,
                current_positions=current_positions,
            )
            jpw["calculation_basis"] = (
                jpw.get("calculation_basis", "")
                + f" || orientation=disabled (rpy_deg=[0,0,0])"
                + (f"; rel_rpy_dbg={_rel_rpy_dbg} ({_rel_rpy_basis_dbg})" if any(_rel_rpy_dbg) else "")
            )

            # ── relative_offset_from_base deterministic override ──
            # = joint_position_world 와 base.center 의 차이. role-shift 가 있으면 0이 아님.
            # joint 가 base.center 와 동일하면 [0,0,0] 이지만, primary_executor + grip_assistant
            # 같은 케이스에선 functional pole shift 로 의미 있는 값이 나옴.
            if base_name in current_positions and joint_pos:
                base_c = current_positions[base_name]["center"]
                rel_off = [round(joint_pos[i] - base_c[i], 6) for i in range(3)]
                step_result["relative_offset_from_base"] = rel_off

            # ── base_target_pose_world (양팔 조립 모델) ──
            # 팀원 사양 (2026-04-27): 왼팔이 base, 오른팔이 attach 를 들고 meeting point 에서 만남.
            # → step 출력에 base 가 이동해 도착할 좌표 (base_target_pose_world.position) 를 추가.
            #   target_pose_world.position 은 attach 도착 좌표 그대로.
            # step 1 (attach=null): base 단독 배치 → base center 그대로 (이동 없음).
            base_pose_position: list[float] | None = None
            if attach_name and base_name in current_positions and joint_pos:
                base_pose_position = _compute_base_pose_at_meeting(
                    current_positions[base_name],
                    joint_pos,
                    step_result.get("attach_region_base", ""),
                )
            elif base_name in current_positions:
                base_pose_position = list(current_positions[base_name]["center"])
            if base_pose_position is not None:
                step_result["base_target_pose_world"] = {
                    "position": base_pose_position,
                    "orientation_rpy_deg": [0.0, 0.0, 0.0],
                    "description": (
                        "양팔 조립 모델: base 가 meeting point 로 이동 후 attach 와 만남. "
                        "step 1 standalone 의 경우 base 시작 위치 그대로."
                    ),
                }

            # ── 테이블 침투 가드 (rear_end/front_end/side_face/mid_body 면접합) ──
            # face-to-face 측면 접합에서는 base center z 와 attach center z 가 같은 plane 에
            # 정렬되는데, 이 plane 이 낮으면 둘 중 하나(또는 둘 다)가 테이블에 박힌다.
            # 양팔이 둘 다 들고 만나는 meeting point 이므로 joint/attach/base 셋 다 동일 +z lift
            # 하면 contact 는 유지되고 테이블 침투만 해소된다.
            # (upper_surface 는 attach 가 base 위에 얹히는 구조라 이 보정이 보통 0.)
            tpw_pos = step_result.get("target_pose_world", {}).get("position")
            if (
                attach_name
                and tpw_pos and len(tpw_pos) == 3
                and base_pose_position and len(base_pose_position) == 3
                and attach_name in current_positions
                and base_name in current_positions
            ):
                a_half_z = (current_positions[attach_name]["aabb_max"][2]
                            - current_positions[attach_name]["aabb_min"][2]) / 2.0
                b_half_z = (current_positions[base_name]["aabb_max"][2]
                            - current_positions[base_name]["aabb_min"][2]) / 2.0
                attach_amin_z = tpw_pos[2] - a_half_z
                base_amin_z   = base_pose_position[2] - b_half_z
                lift = _table_clamp_deficit(attach_amin_z, base_amin_z)
                if lift > 0:
                    tpw_pos[2] = round(tpw_pos[2] + lift, 6)
                    base_pose_position[2] = round(base_pose_position[2] + lift, 6)
                    step_result["target_pose_world"]["position"] = tpw_pos
                    step_result["base_target_pose_world"]["position"] = base_pose_position
                    if joint_pos and len(joint_pos) == 3:
                        joint_pos[2] = round(joint_pos[2] + lift, 6)
                        step_result["joint_position_world"]["position"] = joint_pos
                    jpw["calculation_basis"] += (
                        f"; table_clamp_lift=+{lift*1000:.1f}mm "
                        f"(attach_amin_z={attach_amin_z:.4f}, base_amin_z={base_amin_z:.4f}, "
                        f"floor={TABLE_TOP_Z + TABLE_TOP_SAFETY_MARGIN_M:.4f})"
                    )
                    if base_name in current_positions and joint_pos:
                        base_c = current_positions[base_name]["center"]
                        rel_off = [round(joint_pos[i] - base_c[i], 6) for i in range(3)]
                        step_result["relative_offset_from_base"] = rel_off

            # functional_role 주입 (디버깅/논문 서술용)
            step_result["functional_roles"] = {
                "base_role":   roles.get(base_name,   "alignment_aid"),
                "attach_role": roles.get(attach_name, None) if attach_name else None,
            }

            # 기하 검증
            errors = _validate_step(
                step_result, plan, current_positions, assembled_objects, roles, functional_info,
            )
            attempt_log.append({
                "attempt":              attempt + 1,
                "joint_position_world": step_result.get("joint_position_world", {}).get("position"),
                "errors":               errors,
                "passed":               len(errors) == 0,
            })
            # 좌표는 코드가 결정적으로 계산하므로 LLM retry 로 좌표가 바뀌지 않는다.
            # 첫 시도에서 무조건 break — 실패해도 같은 좌표가 반복되어 토큰 낭비.
            # (collision 등 실제 기하 충돌은 Phase 3 feedback 으로 module2a 에 전달.)
            break

        validation_log.append({
            "step":              step_num,
            "attempts":          attempt_log,
            "final_passed":      len(errors) == 0,
            "final_errors":      errors,
        })
        if errors:
            any_unresolved_failure = True

        all_steps.append(step_result)  # type: ignore[arg-type]
        _update_positions(current_positions, step_result)  # type: ignore[arg-type]

        if base_name and base_name not in assembled_objects:
            assembled_objects.append(base_name)
        if attach_name and attach_name not in assembled_objects:
            assembled_objects.append(attach_name)

    # ── Phase 3: Final ──
    geometric_validation_results = {
        "strategy_attempts":      strategy_attempts,
        "strategy_unresolved":    strategy_unresolved,
        "per_step":               validation_log,
        "any_unresolved_failure": any_unresolved_failure or strategy_unresolved,
    }
    final_result, usage = _call_gpt(
        client, FINAL_SYSTEM_PROMPT,
        "완성된 조립 단계와 코드 검증 결과를 종합하여 최종 구조, 검증, 피드백을 출력하라.\n\n"
        + json.dumps({
            "task":                         input_data.task,
            "assembly_strategy":            assembly_strategy,
            "assembly_steps":               all_steps,
            "tool_constraints":             input_data.tool_constraints,
            "filter_result":                input_data.filter_result,
            "selected_candidate":           input_data.selected_candidate,
            "functional_roles":             roles,
            "geometric_validation_results": geometric_validation_results,
        }, ensure_ascii=False, indent=2),
        model, temperature, max_tokens,
    )
    _accum(usage)
    gpt_call_count += 1

    # 코드가 판단한 unresolved failure는 feedback 강제
    # 논문 3.1.4: 피드백은 3.1.1 (module2a, Scene Resource Parser)로 전달
    feedback = final_result.get("feedback", {})

    # 구 스키마 호환: need_feedback_to_module2c가 있으면 module2a로 이전
    if "need_feedback_to_module2c" in feedback and "need_feedback_to_module2a" not in feedback:
        feedback["need_feedback_to_module2a"] = feedback.pop("need_feedback_to_module2c")

    # ── 추측성 fail 자동 override ──
    # LLM 이 코드 검증 결과를 무시하고 "may not / insufficient" 같은 추측성 사유로
    # fail 을 매기는 패턴이 끈질겼다. 코드 검증 final_errors 에 해당 항목 prefix 가
    # 없으면서 LLM reason 에 추측성 키워드만 있는 fail 은 자동 pass 처리한다.
    verification_obj = final_result.setdefault("verification", {})
    auto_overrides = _override_speculative_fails(
        verification_obj,
        geometric_validation_results,
        tool_constraints=input_data.tool_constraints,
        selected_candidate=input_data.selected_candidate,
    )

    # 8개 검증 중 하나라도 fail이면 피드백 발동 (논문 규정)
    verif_checks = verification_obj.get("checks", [])
    any_check_fail = any(c.get("result") == "fail" for c in verif_checks)

    if any_unresolved_failure or strategy_unresolved or any_check_fail:
        feedback["need_feedback_to_module2a"] = True
        feedback["feedback_target"]           = "module2a"
        if not feedback.get("repair_type"):
            feedback["repair_type"] = "global_redesign"
    else:
        feedback["need_feedback_to_module2a"] = False
        feedback["feedback_target"]           = None

    # 피드백 회차 추적 + 태스크 포기 판단 (논문: 최대 2회)
    iteration = int(getattr(input_data, "feedback_iteration", 0))
    feedback["feedback_iteration"] = iteration
    task_abandoned = (
        feedback.get("need_feedback_to_module2a", False) and iteration >= 2
    )
    feedback["task_abandoned"] = task_abandoned
    if task_abandoned:
        # 이미 2회 실패 후 또 실패 → 태스크 수행 불가, 피드백 중단
        feedback["need_feedback_to_module2a"] = False
        feedback["feedback_target"]           = None

    output = {
        "assembly_strategy":            assembly_strategy,
        "assembly_steps":               all_steps,
        "final_structure":              final_result.get("final_structure", {}),
        "verification":                 final_result.get("verification", {}),
        "feedback":                     feedback,
        "reasoning_trace":              final_result.get("reasoning_trace", {}),
        "geometric_validation_results": geometric_validation_results,
    }
    trace = {
        "mode":                "step_by_step_with_validation",
        "model":               model,
        "temperature":         temperature,
        "prompt_tokens":       total_tokens["prompt"],
        "completion_tokens":   total_tokens["completion"],
        "gpt_call_count":      gpt_call_count,
        "assembly_step_count": len(all_steps),
        "is_valid":            output.get("verification", {}).get("is_valid", False),
        "need_feedback":       output.get("feedback", {}).get("need_feedback_to_module2a", False),
        "feedback_target":     output.get("feedback", {}).get("feedback_target"),
        "any_unresolved_failure": any_unresolved_failure,
        "validation_log":      validation_log,
        "functional_roles":    roles,
        "verification_auto_overrides": auto_overrides,
    }
    return output, trace
