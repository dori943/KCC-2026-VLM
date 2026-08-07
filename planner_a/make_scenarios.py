# -*- coding: utf-8 -*-
"""벤치마크 태스크 시나리오 생성.

태스크 정의 · 객체 규격 · 도구 후보 스펙은 벤치마크 명세표를 그대로 옮겼다.
SG 가 내야 할 판정(tool_effective)은 손으로 True/False 를 박지 않고
표의 수치에서 규칙으로 유도한다. 수치를 고치면 판정이 따라 바뀐다.
명세표가 적어 둔 의도 판정과 유도 결과가 어긋나면 실행이 멈춘다.

출력
    scenarios/t1_1_*, t1_2_*, t2_1_*, t2_2_*   벤치마크 태스크
    scenarios/u1_* … u4_*                       순서 규칙 회귀용 단위 시나리오

KG / SG 의 실제 출력 스키마가 확정되기 전이라 입력은 mock 이다.
확정되면 이 파일을 버리고 실제 출력을 scenarios/ 에 넣으면 된다.
"""
import json
import os

os.makedirs("scenarios", exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# 0. 로봇 · EE 상수
# ══════════════════════════════════════════════════════════════════════
# EE 랙 3슬롯. 초기 EE 는 실험 때 랜덤.
EE_RACK = ["2f", "3f", "vac"]
EE_PAYLOAD_KG = {"2f": 2.0, "3f": 2.0, "vac": 0.5}
EE_OPENING_MM = {"2f": 90, "3f": 160}


# ══════════════════════════════════════════════════════════════════════
# 1. SG 판정 규칙 — 수치에서 tool_effective 를 유도한다
# ══════════════════════════════════════════════════════════════════════
def graspable_ee(tool: dict) -> list:
    """이 도구를 집을 수 있는 EE.

    파지 폭은 bbox 전체가 아니라 실제로 손에 물리는 부위의 폭이다
    (망치는 손잡이, 막대는 지름). grip_width_mm 이 없으면 bbox 최소 폭을 쓴다.
    """
    ees = []
    w = tool.get("grip_width_mm", min(tool["bbox_mm"][0], tool["bbox_mm"][1]))
    for e in tool.get("ee_rack", EE_RACK):
        if e == "vac":
            if tool.get("flat_top", False):
                ees.append(e)
        elif w <= EE_OPENING_MM[e]:
            ees.append(e)
    return ees


def tool_verdict(tool: dict, target: dict, mode: str):
    """도구 하나가 목표 하나에 대해 유효한지. (판정, 사유)

    pull   목표까지 리치가 닿고, 뒤에 걸 lip/hook 이 있어야 한다
    push   리치만 닿으면 된다 (lip 불필요)
    sweep  리치가 닿고, 접촉 폭이 흩어진 범위를 덮어야 한다
    공통   그 도구를 실제로 들 수 있는 EE 가 있어야 한다
    """
    if tool["reach_mm"] < target["distance_mm"]:
        return False, f"유효 리치 {tool['reach_mm']}mm < 목표 거리 {target['distance_mm']}mm"
    if mode == "pull" and tool["lip_mm"] <= 0:
        return False, "lip/hook 없음 — 뒤에 걸 수 없어 당기기 불가"
    if mode == "sweep" and tool["contact_width_mm"] < target.get("span_mm", 0):
        return False, (f"접촉 폭 {tool['contact_width_mm']}mm "
                       f"< 흩어진 범위 {target['span_mm']}mm")
    ees = graspable_ee(tool)
    ok_ee = [e for e in ees if tool["mass_kg"] <= EE_PAYLOAD_KG[e]]
    if not ees:
        return False, "형상상 어떤 EE 로도 파지 불가"
    if not ok_ee:
        return False, (f"도구 질량 {tool['mass_kg']}kg 가 파지 가능한 EE "
                       f"{ees} 의 payload 초과")
    return True, f"리치·형상·질량 모두 만족 (파지 EE {ok_ee})"


def build_tool_effective(tools: dict, targets: dict, expect: dict) -> dict:
    """도구×목표 전조합을 유도하고, 명세표의 의도 판정과 대조한다."""
    out, why = {}, {}
    for tname, t in tools.items():
        for gname, g in targets.items():
            v, reason = tool_verdict(t, g, g["mode"])
            out[f"{tname}|{gname}"] = v
            why[f"{tname}|{gname}"] = reason
    for key, want in expect.items():
        got = out[key]
        if got != want:
            raise AssertionError(
                f"명세표 의도와 유도 결과가 다르다: {key} 기대 {want} / 유도 {got}"
                f"\n  사유: {why[key]}")
    return out, why


S = {}


# ══════════════════════════════════════════════════════════════════════
# 2. T1-1. 작업 반경 밖 물체 끌어오기          [카테고리 1 Tool Selection]
# ══════════════════════════════════════════════════════════════════════
# 명령 : "우유를 작업 환경 안으로 가져오기"
# 검증 : 주변 물체를 적절한 도구로 선택하는가
#
# 도구 후보 4개 중 하나만 성공. 나머지는 각각 다른 이유로 탈락한다.
#   A HammerObject     리치 충분 + claw 후킹 가능        → 주 성공 도구
#   B CylinderObject   리치 충분하지만 hook 없음         → push 만 가능
#   C BoxObject        넓은 판, push 가능 / pull 불가
#   D ShortHookObject  후킹 가능하지만 리치 부족
#
# ★ 가정 — 목표 거리 240mm.
#   명세표는 도구의 유효 길이만 주고 목표까지의 거리를 주지 않는다.
#   Hammer(250~300) 는 닿고 ShortHook(180) 은 못 닿는 값으로 잡았다.
T11_TARGET = {
    "MilkObject": {
        "bbox_mm": [40, 40, 144], "volume_cm3": 192, "density": 1500,
        "mass_kg": 0.288, "shape": "우유 카톤 (윗면 박공지붕)",
        "distance_mm": 240, "mode": "pull",
        "note": "바닥면이 평평해 끌기 시 마찰 확보는 양호. "
                "무게중심이 높아 옆으로 미는 힘이 크면 전도 위험",
    },
}
T11_TOOLS = {
    "HammerObject": {
        "bbox_mm": [300, 100, 40], "reach_mm": 275, "lip_mm": 40, "grip_width_mm": 30,
        "contact_width_mm": 100, "volume_cm3": 150, "density": 1330,
        "mass_kg": 0.20, "flat_top": False,
        "role": "긴 손잡이 + head + claw. 후킹/당기기",
        "robosuite": "robosuite.models.objects.HammerObject (Composite)",
    },
    "CylinderObject": {
        "bbox_mm": [30, 30, 300], "reach_mm": 300, "lip_mm": 0, "grip_width_mm": 30,
        "contact_width_mm": 30, "volume_cm3": 212, "density": 500,
        "mass_kg": 0.106, "flat_top": False,
        "role": "Ø30 직선 막대. 리치 충분 / hook 없음",
        "robosuite": "robosuite.models.objects.CylinderObject (primitive)",
    },
    "BoxObject": {
        "bbox_mm": [120, 15, 300], "reach_mm": 300, "lip_mm": 0, "grip_width_mm": 15,
        "contact_width_mm": 120, "volume_cm3": 540, "density": 250,
        "mass_kg": 0.135, "flat_top": True,
        "role": "얇고 긴 평판. broad push",
        "robosuite": "robosuite.models.objects.BoxObject (primitive)",
    },
    "ShortHookObject": {
        "bbox_mm": [120, 25, 180], "reach_mm": 180, "lip_mm": 30, "grip_width_mm": 25,
        "contact_width_mm": 120, "volume_cm3": 110, "density": 500,
        "mass_kg": 0.055, "flat_top": False,
        "role": "짧은 L-hook. 후킹 가능 / 리치 부족",
        "robosuite": "CompositeObject 로 새로 정의 (내장 클래스 아님)",
    },
}
T11_EXPECT = {
    "HammerObject|MilkObject": True,
    "CylinderObject|MilkObject": False,
    "BoxObject|MilkObject": False,
    "ShortHookObject|MilkObject": False,
}
t11_eff, t11_why = build_tool_effective(T11_TOOLS, T11_TARGET, T11_EXPECT)

S["t1_1_pull_milk"] = {
    "meta": {"task_id": "T1-1", "category": "1. Tool Selection",
             "verify": "주변 물체를 적절한 도구로 선택하는가",
             "scene_version": 1},
    "kg": {
        "scenario": "T1-1 작업 반경 밖 물체 끌어오기",
        "task": "우유를 작업 환경 안으로 가져오기",
        "subgoals": [{
            "subgoal_id": "SG1",
            "description": "우유를 작업 영역 안으로 가져온다",
            "target_hints": ["MilkObject"], "goal_region_hint": "work_zone",
            "required_capabilities": ["pull", "tool_use"],
        }],
        "must_precede": [],
    },
    "sg": {
        "objects": {
            **{k: {"reachable": True, "feasible_ee": graspable_ee(v), "at_rest": True}
               for k, v in T11_TOOLS.items()},
            "MilkObject": {"reachable": False, "feasible_ee": ["2f"], "at_rest": True},
        },
        "regions": {"work_zone": {"clear": True, "multi": True},
                    "tool_rest": {"clear": True, "multi": True}},
        "on_edges": [],
        "clearance": {"HammerObject|tool_rest": 100, "MilkObject|work_zone": 100},
        "tool_effective": t11_eff,
        "tool_effective_reason": t11_why,
        "object_specs": {**T11_TOOLS, **T11_TARGET},
        "per_subgoal": {"SG1": {
            "target": "MilkObject", "goal_region": "work_zone",
            "tool_required": True, "selected_tool": "HammerObject",
            "tool_mode": "pull", "tool_rest_region": "tool_rest",
            "feasible_ee": ["2f"], "ee_candidate": "2f"}},
    },
    "robot_state": {"in_hand": None, "current_ee": "2f", "ee_rack": EE_RACK},
}


# ---------------------------------------------------------------------
# T1-1 장면 구성 version2. 같은 태스크, 도구 후보 세트만 교체.
#   목표가 CerealObject 로 바뀌고 거리가 멀어진다.
# ★ 가정 — 목표 거리 380mm (리치 420 은 닿고 180 은 못 닿는 값)
# ---------------------------------------------------------------------
T11B_TARGET = {
    "CerealObject": {
        "bbox_mm": [100, 30, 150], "volume_cm3": 450, "density": 1110,
        "mass_kg": 0.50, "shape": "시리얼 박스. 넓은 면을 바닥에 두고 배치",
        "distance_mm": 380, "mode": "pull",
    },
}
T11B_TOOLS = {
    "LongHookObject": {
        "bbox_mm": [40, 70, 450], "reach_mm": 420, "lip_mm": 35, "grip_width_mm": 40,
        "contact_width_mm": 70, "volume_cm3": 180, "density": 1110,
        "mass_kg": 0.20, "flat_top": False,
        "role": "긴 L-hook. 리치 충분 + 후킹 가능. 주 성공 도구",
    },
    "ShortPaddleObject": {
        "bbox_mm": [100, 20, 180], "reach_mm": 180, "lip_mm": 0, "grip_width_mm": 20,
        "contact_width_mm": 100, "volume_cm3": 360, "density": 417,
        "mass_kg": 0.15, "flat_top": True,
        "role": "짧고 넓은 판. 접촉면은 넓지만 목표까지 리치 부족",
    },
    "ThinRodObject": {
        "bbox_mm": [20, 20, 420], "reach_mm": 420, "lip_mm": 0, "grip_width_mm": 20,
        "contact_width_mm": 20, "volume_cm3": 132, "density": 760,
        "mass_kg": 0.10, "flat_top": False,
        "role": "얇고 긴 직선 막대. 리치 충분하지만 hook 없음",
    },
    "ShortHookObject": {
        "bbox_mm": [40, 70, 210], "reach_mm": 180, "lip_mm": 30, "grip_width_mm": 40,
        "contact_width_mm": 70, "volume_cm3": 100, "density": 1000,
        "mass_kg": 0.10, "flat_top": False,
        "role": "짧은 L-hook. 후킹 가능하지만 리치 부족",
    },
}
T11B_EXPECT = {
    "LongHookObject|CerealObject": True,
    "ShortPaddleObject|CerealObject": False,
    "ThinRodObject|CerealObject": False,
    "ShortHookObject|CerealObject": False,
}
t11b_eff, t11b_why = build_tool_effective(T11B_TOOLS, T11B_TARGET, T11B_EXPECT)

S["t1_1b_pull_cereal"] = {
    "meta": {"task_id": "T1-1", "category": "1. Tool Selection",
             "verify": "주변 물체를 적절한 도구로 선택하는가",
             "scene_version": 2},
    "kg": {
        "scenario": "T1-1(v2) 작업 반경 밖 물체 끌어오기",
        "task": "시리얼 박스를 작업 환경 안으로 가져오기",
        "subgoals": [{
            "subgoal_id": "SG1",
            "description": "시리얼 박스를 작업 영역 안으로 가져온다",
            "target_hints": ["CerealObject"], "goal_region_hint": "work_zone",
            "required_capabilities": ["pull", "tool_use"],
        }],
        "must_precede": [],
    },
    "sg": {
        "objects": {
            **{k: {"reachable": True, "feasible_ee": graspable_ee(v), "at_rest": True}
               for k, v in T11B_TOOLS.items()},
            "CerealObject": {"reachable": False, "feasible_ee": ["2f"], "at_rest": True},
        },
        "regions": {"work_zone": {"clear": True, "multi": True},
                    "tool_rest": {"clear": True, "multi": True}},
        "on_edges": [],
        "clearance": {"LongHookObject|tool_rest": 100, "CerealObject|work_zone": 100},
        "tool_effective": t11b_eff,
        "tool_effective_reason": t11b_why,
        "object_specs": {**T11B_TOOLS, **T11B_TARGET},
        "per_subgoal": {"SG1": {
            "target": "CerealObject", "goal_region": "work_zone",
            "tool_required": True, "selected_tool": "LongHookObject",
            "tool_mode": "pull", "tool_rest_region": "tool_rest",
            "feasible_ee": ["2f"], "ee_candidate": "2f"}},
    },
    "robot_state": {"in_hand": None, "current_ee": "2f", "ee_rack": EE_RACK},
}


# ══════════════════════════════════════════════════════════════════════
# 3. T1-2. 작고 많은 물체 회수하기              [카테고리 1 Tool Selection]
# ══════════════════════════════════════════════════════════════════════
# 명령 : "레고를 담을 수거 트레이를 앞으로 끌어오고,
#         테이블에 흩어진 레고 블록을 수거 트레이에 쓸어 담아라."
#
# 서브골 2개가 서로 다른 도구를 요구한다.
#   SG1 부품 상자 pull   → LongHookObject  (리치 O + hook O)
#   SG2 레고 sweep       → Light Wide Box  (접촉 폭 O + Vac payload 이내)
#
# 오답 6개가 각각 다른 축에서 탈락한다.
#   HammerObject       hook O / 리치 부족
#   CylinderObject     리치 O / hook 없음, 접촉 폭 부족
#   Heavy Wide Box     sweep geometry 동일 / Vac payload 0.5kg 초과
#   BottleObject       리치·hook·폭 전부 부족한 순수 distractor
#
# ★ 가정 — 부품 상자 거리 400mm, 레고 흩어진 범위 180mm, 레고 거리 200mm
T12_TARGETS = {
    "PartsBoxObject": {
        "bbox_mm": [180, 180, 90], "mass_kg": 3.0,
        "distance_mm": 400, "mode": "pull",
        "role": "작업 반경 밖. 직접 grasp/lift 불가 → 도구 pull 필요",
        "goal_region": "blue_region",
    },
    "LegoBlockCluster": {
        "bbox_mm": [20, 20, 12], "mass_kg": 0.0024, "count": 12,
        "distance_mm": 200, "span_mm": 180, "mode": "sweep",
        "role": "개별 파지는 가능하나 step budget 초과 → sweep 필요",
        "goal_region": "collection_tray",
    },
}
T12_TOOLS = {
    "LongHookObject": {
        "bbox_mm": [40, 70, 450], "reach_mm": 420, "lip_mm": 35, "grip_width_mm": 40,
        "contact_width_mm": 70, "mass_kg": 0.20, "flat_top": False,
        "role": "긴 갈고리. reach + hooking 모두 만족. SG1 주 성공 후보",
    },
    "HammerObject": {
        "bbox_mm": [200, 100, 40], "reach_mm": 180, "lip_mm": 40, "grip_width_mm": 30,
        "contact_width_mm": 100, "mass_kg": 0.20, "flat_top": False,
        "role": "claw 로 hooking 가능하지만 유효 리치 부족. SG1 hard negative",
    },
    "CylinderObject": {
        "bbox_mm": [30, 30, 550], "reach_mm": 550, "lip_mm": 0, "grip_width_mm": 30,
        "contact_width_mm": 30, "volume_cm3": 389, "density": 514,
        "mass_kg": 0.20, "flat_top": False,
        "role": "긴 직선 막대. reach 충분하지만 hooking 불가. geometry distractor",
    },
    "LightWideBoxObject": {
        "bbox_mm": [200, 20, 220], "reach_mm": 220, "lip_mm": 25,
        "contact_width_mm": 200, "volume_cm3": 880, "density": 227,
        "mass_kg": 0.20, "flat_top": True,
        "role": "넓은 sweeping 판. 넓은 접촉 + 가벼운 질량. SG2 주 후보",
    },
    "HeavyWideBoxObject": {
        "bbox_mm": [200, 20, 220], "reach_mm": 220, "lip_mm": 25,
        "contact_width_mm": 200, "volume_cm3": 880, "density": 909,
        "mass_kg": 0.80, "flat_top": True,
        "role": "geometry 는 Light 와 동일. Vacuum payload 초과용 hard negative",
    },
    "BottleObject": {
        "bbox_mm": [55, 59, 160], "reach_mm": 120, "lip_mm": 0, "grip_width_mm": 55,
        "contact_width_mm": 55, "volume_cm3": 277, "mass_kg": 0.20,
        "flat_top": False,
        "role": "짧은 곡면 물체. reach·hook·sweep 모두 불리한 순수 오답",
    },
}
# 넓은 판은 두께 20mm 라 2F 로도 집히지만, sweeping 자세에서는 흡착만 성립한다.
T12_TOOLS["LightWideBoxObject"]["ee_rack"] = ["vac"]
T12_TOOLS["HeavyWideBoxObject"]["ee_rack"] = ["vac"]

T12_EXPECT = {
    "LongHookObject|PartsBoxObject": True,
    "HammerObject|PartsBoxObject": False,
    "CylinderObject|PartsBoxObject": False,
    "LightWideBoxObject|PartsBoxObject": False,
    "HeavyWideBoxObject|PartsBoxObject": False,
    "BottleObject|PartsBoxObject": False,
    "LightWideBoxObject|LegoBlockCluster": True,
    "HeavyWideBoxObject|LegoBlockCluster": False,   # Vac payload 초과
    "CylinderObject|LegoBlockCluster": False,       # 접촉 폭 부족
    "LongHookObject|LegoBlockCluster": False,       # 접촉 폭 부족
    "HammerObject|LegoBlockCluster": False,
    "BottleObject|LegoBlockCluster": False,
}
t12_eff, t12_why = build_tool_effective(T12_TOOLS, T12_TARGETS, T12_EXPECT)

S["t1_2_lego_sweep"] = {
    "meta": {"task_id": "T1-2", "category": "1. Tool Selection",
             "verify": "주변 물체를 적절한 도구로 선택하는가",
             "scene_version": 1},
    "kg": {
        "scenario": "T1-2 작고 많은 물체 회수하기",
        "task": "레고를 담을 수거 트레이를 앞으로 끌어오고, "
                "테이블에 흩어진 레고 블록을 수거 트레이에 쓸어 담아라",
        "subgoals": [
            {"subgoal_id": "SG1", "description": "부품 상자를 앞으로 끌어온다",
             "target_hints": ["PartsBoxObject"], "goal_region_hint": "blue_region",
             "required_capabilities": ["pull", "tool_use"]},
            {"subgoal_id": "SG2", "description": "레고 블록을 수거 트레이에 쓸어 담는다",
             "target_hints": ["LegoBlockCluster"], "goal_region_hint": "collection_tray",
             "required_capabilities": ["sweep", "tool_use"]},
        ],
        "must_precede": [],
    },
    "sg": {
        "objects": {
            **{k: {"reachable": True, "feasible_ee": graspable_ee(v), "at_rest": True}
               for k, v in T12_TOOLS.items()},
            # 3.0kg + 작업 반경 밖 → 직접 파지 불가
            "PartsBoxObject": {"reachable": False, "feasible_ee": [], "at_rest": True},
            "LegoBlockCluster": {"reachable": True, "feasible_ee": ["2f"], "at_rest": True},
        },
        "regions": {
            # 상자 중심이 x<=0.45m 안으로 진입하면 SG1 성공
            "blue_region": {"clear": True, "multi": True, "spec": "x <= 0.45 m"},
            # 레고가 트레이 내부로 진입하면 수거 판정
            "collection_tray": {"clear": True, "multi": True,
                                "spec": "CollectionTrayObject 250x180x35mm, 0.3kg"},
            "tool_rest": {"clear": True, "multi": True},
        },
        "on_edges": [],
        "clearance": {"LongHookObject|tool_rest": 100,
                      "LightWideBoxObject|tool_rest": 100,
                      "PartsBoxObject|blue_region": 100,
                      "LegoBlockCluster|collection_tray": 100},
        "tool_effective": t12_eff,
        "tool_effective_reason": t12_why,
        "object_specs": {**T12_TOOLS, **T12_TARGETS},
        "per_subgoal": {
            "SG1": {"target": "PartsBoxObject", "goal_region": "blue_region",
                    "tool_required": True, "selected_tool": "LongHookObject",
                    "tool_mode": "pull", "tool_rest_region": "tool_rest",
                    "feasible_ee": ["2f"], "ee_candidate": "2f"},
            "SG2": {"target": "LegoBlockCluster", "goal_region": "collection_tray",
                    "tool_required": True, "selected_tool": "LightWideBoxObject",
                    "tool_mode": "sweep", "tool_rest_region": "tool_rest",
                    "feasible_ee": ["vac"], "ee_candidate": "vac"},
        },
    },
    # EE 랙 3슬롯, 초기 EE 는 실험 때 랜덤
    "robot_state": {"in_hand": None, "current_ee": "2f", "ee_rack": EE_RACK},
}


# ══════════════════════════════════════════════════════════════════════
# 4. T2-1. 다중 객체 분류 및 운반        [카테고리 2 Efficient EE Selection]
# ══════════════════════════════════════════════════════════════════════
# 명령 : "테이블 위의 모든 물체를 각각 지정된 트레이로 옮겨라.
#         식품은 초록 트레이, 음료 용기는 파랑 트레이, 부품은 빨강 트레이이다"
# 검증 : 불필요한 EE 교체를 줄이는가
#
# 장면 구성 안 B "내장 + MimicGen 래퍼" (권장안) 을 따른다.
#   R1 Vac  유일   PlateWithHoleObject  340x340x20 대형 평판, 밀착
#   R2 3F   유일   BlenderObject        폭 94mm → 2F 개구 초과, 곡면 → Vac 밀봉 실패
#   R3 2F   유일   CanObject            슬롯에 꽂혀 있어 진입 폭 제약
#   R4 복수해      BreadObject          불규칙 유기형, F_crit 낮아 2F 는 파손
#
# Planner A 는 순서를 과하게 좁히지 않는 것이 정답이다.
# 4개는 서로 독립이므로 EE 교체 최소 순서 선택권을 Planner B 에 그대로 넘긴다.
T21_SPECS = {
    "PlateWithHoleObject": {
        "bbox_mm": [340, 340, 20], "mass_kg": 0.30,
        "class": "부품", "goal_region": "red_tray",
        "feasible_ee": ["vac"],
        "reason": "대형 평판 프레임, 밀착. 넓고 납작해 흡착만 성립. "
                  "테이블 대비 크므로 트레이도 커야 함",
        "robosuite": "robosuite 내장",
    },
    "BlenderObject": {
        "bbox_mm": [88, 94, 256], "mass_kg": 1.5,
        "class": "음료 용기", "goal_region": "blue_tray",
        "feasible_ee": ["3f"],
        "reason": "폭 94mm 가 2F 개구 90mm 초과. 곡면이라 Vac 밀봉도 실패",
        "robosuite": "내장 메시 + MimicGen 래퍼 "
                     "BlenderObject(bottle.xml, scale=1.6, density=1300)",
    },
    "CanObject": {
        "bbox_mm": [50, 50, 80], "mass_kg": 0.0146,
        "class": "음료 용기", "goal_region": "blue_tray",
        "feasible_ee": ["2f"],
        "reason": "Ø50x80. 슬롯 간격 95mm 라 2F(진입 90) 만 빠듯하게 진입",
        "robosuite": "robosuite 내장",
    },
    "BreadObject": {
        "bbox_mm": [48, 40, 48], "mass_kg": 0.0041,
        "class": "식품", "goal_region": "green_tray",
        "feasible_ee": ["3f", "vac"],
        "reason": "불규칙 유기형. F_crit(접촉력 임계)가 낮아 2F 는 파손 판정. "
                  "3F/Vac 둘 다 가능한 복수해",
        "robosuite": "robosuite 내장",
    },
}
# 실험 때 적용할 랜덤화. 정답이 뒤집히는 임계값을 함께 적는다.
T21_RANDOMIZATION = [
    {"param": "BlenderObject scale", "range": "1.0 - 1.8",
     "flip": "<= 1.45 (폭 85) 이면 2F 가능 -> {3F} 유일성 소멸"},
    {"param": "BlenderObject 질량 (런타임 오버라이드)", "range": "0.05 - 2.0 kg",
     "flip": "Vac payload 아래면 Vac 가능"},
    {"param": "슬롯 간격", "range": "95 - 200 mm", "flip": "> 170 이면 3F 가능"},
    {"param": "오버행 천장", "range": "100 - 350 mm", "flip": ">= 200 이면 Vac 가능"},
    {"param": "PlateWithHoleObject 부양 (받침 유무)", "range": "0 - 15 mm",
     "flip": ">= 8 mm 면 2F 핀치 가능"},
    {"param": "BreadObject F_crit", "range": "5 - 30 N", "flip": ">= 25 N 이면 2F 가능"},
    {"param": "물체 - 구역 배정", "range": "셔플", "flip": "'이 물체 = 이 EE' 암기 방지"},
    {"param": "트레이 색·위치", "range": "셔플", "flip": "분류 규칙 암기 방지"},
]

S["t2_1_sort_transport"] = {
    "meta": {"task_id": "T2-1", "category": "2. Efficient EE Selection",
             "verify": "불필요한 EE 교체를 줄이는가",
             "scene_version": "안 B (내장 + MimicGen 래퍼, 권장)"},
    "kg": {
        "scenario": "T2-1 다중 객체 분류 및 운반",
        "task": "테이블 위의 모든 물체를 각각 지정된 트레이로 옮겨라. "
                "식품은 초록 트레이, 음료 용기는 파랑 트레이, 부품은 빨강 트레이이다",
        "subgoals": [
            {"subgoal_id": f"SG{i+1}",
             "description": f"{name} 를 {spec['class']} 트레이로 옮긴다",
             "target_hints": [name], "goal_region_hint": spec["goal_region"],
             "required_capabilities": ["stable_grasp", "controlled_transport",
                                       "controlled_release"]}
            for i, (name, spec) in enumerate(T21_SPECS.items())
        ],
        "must_precede": [],
    },
    "sg": {
        "objects": {name: {"reachable": True, "feasible_ee": spec["feasible_ee"],
                           "at_rest": True}
                    for name, spec in T21_SPECS.items()},
        "regions": {
            "red_tray":   {"clear": True, "multi": True, "label": "부품 (빨강)"},
            "blue_tray":  {"clear": True, "multi": True, "label": "음료 용기 (파랑)"},
            "green_tray": {"clear": True, "multi": True, "label": "식품 (초록)"},
        },
        "on_edges": [],
        "clearance": {f"{n}|{s['goal_region']}": 50 for n, s in T21_SPECS.items()},
        "object_specs": T21_SPECS,
        "randomization": T21_RANDOMIZATION,
        "per_subgoal": {
            f"SG{i+1}": {"target": name, "goal_region": spec["goal_region"],
                         "tool_required": False,
                         "feasible_ee": spec["feasible_ee"],
                         "ee_candidate": spec["feasible_ee"][0]}
            for i, (name, spec) in enumerate(T21_SPECS.items())
        },
    },
    "robot_state": {"in_hand": None, "current_ee": "2f", "ee_rack": EE_RACK},
}


# ══════════════════════════════════════════════════════════════════════
# 5. T2-2. 다중 객체 탑 쌓기            [카테고리 2 Efficient EE Selection]
# ══════════════════════════════════════════════════════════════════════
# 명령 : "부품 3개를 작업대 위에 하나의 탑으로 쌓아 올려라. 큰 것이 아래로 가야한다"
#
# 장면에는 후보 8종이 놓여 있고 그중 3개만 대상이다. 나머지는 distractor.
# 유일하게 KG 태스크 논리 순서(must_precede)를 타는 태스크 —
# "큰 것이 아래로"는 명령문에서 나오는 순서라 관측 없이도 KG 가 낼 수 있다.
#
# ★ 가정 — 확인 필요
#   - 대상 3개를 부피 기준 CerealObject(450cm3) > MilkObject(192) > CanObject(146)
#     로 잡았다. BottleObject(277)는 목·뚜껑이 있어 위에 쌓을 수 없다고 보고 제외.
#   - CerealObject 를 세우면 밑면이 100x30 이라 그 위 40x40 이 한 축으로
#     삐져나온다. 다만 그건 시리얼을 놓은 뒤에야 잴 수 있는 값이라
#     계획 시점에 참/거짓 어느 쪽으로도 박지 않는다 (판정 유보).
T22_SPECS = {
    "CanObject":    {"bbox_mm": [50, 50, 80],    "volume_cm3": 146, "density": 100,
                     "mass_g": 14.6, "shape": "원통 (콜라캔)",
                     "feasible_ee": ["2f", "3f"]},
    "MilkObject":   {"bbox_mm": [40, 40, 144],   "volume_cm3": 192, "density": 100,
                     "mass_g": 19.2, "shape": "우유 카톤 (윗면 박공지붕)",
                     "feasible_ee": ["2f"]},
    "CerealObject": {"bbox_mm": [100, 30, 150],  "volume_cm3": 450, "density": 150,
                     "mass_g": 67.5, "shape": "얇은 박스",
                     "feasible_ee": ["2f", "vac"]},
    "BottleObject": {"bbox_mm": [55, 58.5, 160], "volume_cm3": 277, "density": 50,
                     "mass_g": 13.9, "shape": "유리병 (목·뚜껑)",
                     "feasible_ee": ["2f", "3f"]},
    "BreadObject":  {"bbox_mm": [48, 40, 48],    "volume_cm3": 83,  "density": 50,
                     "mass_g": 4.1, "shape": "불규칙 유기형",
                     "feasible_ee": ["3f"]},
    "LemonObject":  {"bbox_mm": [67.5, 40, 40],  "volume_cm3": 53,  "density": 50,
                     "mass_g": 2.7, "shape": "타원 구형",
                     "feasible_ee": ["3f"]},
    "PlateWithHoleObject": {"bbox_mm": [340, 340, 20], "volume_cm3": None,
                            "density": None, "mass_g": None,
                            "shape": "대형 평판 프레임 (박스 조합)",
                            "feasible_ee": ["vac"]},
    "SquareNutObject": {"bbox_mm": [110, 87, 20], "volume_cm3": None, "density": 100,
                        "mass_g": None, "shape": "손잡이 달린 너트 (박스 조합)",
                        "feasible_ee": ["2f"]},
}
# 쌓을 대상 3개 (아래 → 위)
T22_STACK = ["CerealObject", "MilkObject", "CanObject"]
T22_REGIONS = ["stack_base", "on_cereal", "on_milk"]

S["t2_2_stack_tower"] = {
    "meta": {"task_id": "T2-2", "category": "2. Efficient EE Selection",
             "verify": "불필요한 EE 교체를 줄이는가", "scene_version": 1},
    "kg": {
        "scenario": "T2-2 다중 객체 탑 쌓기",
        "task": "부품 3개를 작업대 위에 하나의 탑으로 쌓아 올려라. 큰 것이 아래로 가야한다",
        "subgoals": [
            {"subgoal_id": f"SG{i+1}",
             "description": (f"가장 큰 {o} 를 작업대에 놓는다" if i == 0
                             else f"{o} 를 {T22_STACK[i-1]} 위에 올린다"),
             "target_hints": [o], "goal_region_hint": T22_REGIONS[i],
             "required_capabilities": ["stable_grasp", "controlled_transport",
                                       "controlled_release"]}
            for i, o in enumerate(T22_STACK)
        ],
        # "큰 것이 아래로" — 명령문에서 나오는 순서. 관측으로는 도출 불가
        "must_precede": [["SG1", "SG2"], ["SG2", "SG3"]],
    },
    "sg": {
        "objects": {name: {"reachable": True, "feasible_ee": spec["feasible_ee"],
                           "at_rest": True}
                    for name, spec in T22_SPECS.items()},
        # stack_base 만 지금 관측되는 영역이다.
        # on_cereal / on_milk 는 받침 object 를 놓아야 생기는 영역이라
        # SG 가 지금 clear 나 clearance 를 보고할 수 없다.
        # supported_by 만 달아 보내고, 판정은 Planner A 가 유보한다.
        "regions": {
            "stack_base": {"clear": True},
            "on_cereal":  {"supported_by": "CerealObject"},
            "on_milk":    {"supported_by": "MilkObject"},
        },
        "on_edges": [],
        "clearance": {"CerealObject|stack_base": 100},
        "object_specs": T22_SPECS,
        "per_subgoal": {
            f"SG{i+1}": {"target": o, "goal_region": T22_REGIONS[i],
                         "tool_required": False,
                         "feasible_ee": T22_SPECS[o]["feasible_ee"],
                         "ee_candidate": "2f"}
            for i, o in enumerate(T22_STACK)
        },
    },
    "robot_state": {"in_hand": None, "current_ee": "2f", "ee_rack": EE_RACK},
}


# ---------------------------------------------------------------------
# T2-2 (SG 주석 없음). 판정기 대조군.
#
# supported_by 는 이 구현이 정의한 필드다. 실제 SG 가 그걸 내지 않으면
# 규칙 판정기는 "이 영역이 무엇의 윗면인지"를 알 방법이 없다.
# 그 상황을 그대로 만든 장면이다. SG 는 stack_base 만 보고했고
# on_cereal / on_milk 는 아예 보고하지 못했다.
#
# KG must_precede 도 뺐다. 순서가 오직 판정기에서만 나오게 하려는 것이다.
#
#   규칙 판정기  supported_by 가 없으므로 의존을 못 찾는다
#                -> 미관측 4, 관측 순서 엣지 0, 순서 수가 크게 벌어진다
#   VLM 판정기   계획과 영역 이름에서 의존을 추론할 수 있는가
#                -> 추론하면 관측 순서 엣지 4, 순서 수가 줄어든다
#
# 둘의 차이가 곧 VLM 이 규칙 위에 더한 몫이다.
S["t2_2c_stack_tower_nohint"] = {
    "meta": {"task_id": "T2-2", "category": "판정기 대조군",
             "verify": "SG 주석 없이 판정 가능 시점을 추론하는가",
             "scene_version": "주석 제거"},
    "kg": {
        "scenario": "T2-2(대조군) 다중 객체 탑 쌓기 — SG 주석 없음",
        "task": "부품 3개를 작업대 위에 하나의 탑으로 쌓아 올려라. 큰 것이 아래로 가야한다",
        "subgoals": S["t2_2_stack_tower"]["kg"]["subgoals"],
        "must_precede": [],          # 순서를 판정기에서만 나오게 한다
    },
    "sg": {
        "objects": S["t2_2_stack_tower"]["sg"]["objects"],
        # SG 가 실제로 본 영역은 작업대뿐이다
        "regions": {"stack_base": {"clear": True}},
        "on_edges": [],
        "clearance": {"CerealObject|stack_base": 100},
        "object_specs": T22_SPECS,
        "per_subgoal": S["t2_2_stack_tower"]["sg"]["per_subgoal"],
    },
    "robot_state": {"in_hand": None, "current_ee": "2f", "ee_rack": EE_RACK},
}


# ---------------------------------------------------------------------
# T2-2 (영역 이름까지 익명). 판정기 대조군 2.
#
# t2_2c 에서 VLM 이 의존을 찾아냈지만, 단서가 영역 이름(on_cereal)이었을
# 가능성이 있다. 이름을 R1/R2/R3 로 바꾸면 남는 단서는 명령문과 계획 구조뿐이다.
#
#   맞히면   이름 매칭이 아니라 태스크 이해에서 나온 판정이다
#   unknown  판정기가 모르는 것을 모른다고 말한 것이다 (이것도 정답 동작)
#   틀리면   근거 없이 순서를 지어낸 것이다 (가장 나쁜 결과)
_R = {"stack_base": "R1", "on_cereal": "R2", "on_milk": "R3"}
S["t2_2d_stack_tower_opaque"] = {
    "meta": {"task_id": "T2-2", "category": "판정기 대조군",
             "verify": "이름 단서 없이 판정 가능 시점을 추론하는가",
             "scene_version": "주석·이름 제거"},
    "kg": {
        "scenario": "T2-2(대조군2) 다중 객체 탑 쌓기 — 영역 이름 익명",
        "task": "부품 3개를 작업대 위에 하나의 탑으로 쌓아 올려라. 큰 것이 아래로 가야한다",
        "subgoals": [dict(sg, goal_region_hint=_R[sg["goal_region_hint"]])
                     for sg in S["t2_2_stack_tower"]["kg"]["subgoals"]],
        "must_precede": [],
    },
    "sg": {
        "objects": S["t2_2_stack_tower"]["sg"]["objects"],
        "regions": {"R1": {"clear": True}},
        "on_edges": [],
        "clearance": {"CerealObject|R1": 100},
        "object_specs": T22_SPECS,
        "per_subgoal": {k: dict(v, goal_region=_R[v["goal_region"]])
                        for k, v in S["t2_2_stack_tower"]["sg"]["per_subgoal"].items()},
    },
    "robot_state": {"in_hand": None, "current_ee": "2f", "ee_rack": EE_RACK},
}


# ══════════════════════════════════════════════════════════════════════
# 6. T3. Long-horizon 통합 태스크        [카테고리 3]
# ══════════════════════════════════════════════════════════════════════
# 대표 태스크 "작업대 정리 및 포장". 명세표에 장면 구성이 아직 비어 있어
# 시나리오를 만들지 않는다. 명세가 채워지면 여기에 추가한다.


# ══════════════════════════════════════════════════════════════════════
# 7. 순서 규칙 회귀용 단위 시나리오
# ══════════════════════════════════════════════════════════════════════
# 벤치마크 태스크가 아니다. 규칙 1~4 를 하나씩 겨냥한 최소 입력이다.
# 규칙을 고쳤을 때 어디가 깨졌는지 바로 알기 위해 남긴다.

# U1. 단일 물체 확보-운반-배치. 최소 케이스 (규칙 1)
S["u1_single_pick_place"] = {
    "meta": {"kind": "unit", "targets_rule": "규칙 1 인과링크"},
    "kg": {
        "scenario": "U1 단일 물체 확보-운반-배치",
        "task": "유리병을 상자에 담아라",
        "subgoals": [{
            "subgoal_id": "SG1", "description": "유리병을 상자로 옮긴다",
            "target_hints": ["BottleObject"], "goal_region_hint": "box_0",
            "required_capabilities": ["stable_grasp", "controlled_transport",
                                      "controlled_release"],
        }],
        "must_precede": [],
    },
    "sg": {
        "objects": {"BottleObject": {"reachable": True, "feasible_ee": ["2f", "3f"],
                                     "at_rest": True}},
        "regions": {"box_0": {"clear": True}},
        "on_edges": [],
        "clearance": {"BottleObject|box_0": 40},
        "per_subgoal": {"SG1": {"target": "BottleObject", "goal_region": "box_0",
                                "tool_required": False, "feasible_ee": ["2f"],
                                "ee_candidate": "2f"}},
    },
    "robot_state": {"in_hand": None, "current_ee": "2f"},
}

# U2. 스택 위 물체. 파생 조건으로 선행 순서가 도출되는지 (규칙 1 + 파생)
#     "무거운 상자 위에 너트가 얹혀 있다"는 관측에서만 알 수 있다.
#     KG 는 이 순서를 낼 수 없고 Planner A 만 낼 수 있다.
S["u2_stacked_object"] = {
    "meta": {"kind": "unit", "targets_rule": "파생 조건 top_exposed"},
    "kg": {
        "scenario": "U2 스택 포함 다중 물체 정리",
        "task": "너트를 작업대에 두고 상자를 팔레트로 옮겨라",
        "subgoals": [
            {"subgoal_id": "SG1", "description": "너트를 작업대에 둔다",
             "target_hints": ["SquareNutObject"], "goal_region_hint": "workbench",
             "required_capabilities": ["stable_grasp", "controlled_transport",
                                       "controlled_release"]},
            {"subgoal_id": "SG2", "description": "상자를 팔레트로 옮긴다",
             "target_hints": ["PartsBoxObject"], "goal_region_hint": "pallet",
             "required_capabilities": ["stable_grasp", "controlled_transport",
                                       "controlled_release"]},
        ],
        "must_precede": [],
    },
    "sg": {
        "objects": {
            "SquareNutObject": {"reachable": True, "feasible_ee": ["2f"], "at_rest": True},
            "PartsBoxObject":  {"reachable": True, "feasible_ee": ["2f"], "at_rest": True},
        },
        "regions": {"workbench": {"clear": True}, "pallet": {"clear": True}},
        "on_edges": [["SquareNutObject", "PartsBoxObject"]],
        "clearance": {"SquareNutObject|workbench": 100, "PartsBoxObject|pallet": 100},
        "per_subgoal": {
            "SG1": {"target": "SquareNutObject", "goal_region": "workbench",
                    "tool_required": False, "feasible_ee": ["2f"], "ee_candidate": "2f"},
            "SG2": {"target": "PartsBoxObject", "goal_region": "pallet",
                    "tool_required": False, "feasible_ee": ["2f"], "ee_candidate": "2f"},
        },
    },
    "robot_state": {"in_hand": None, "current_ee": "2f"},
}

# U3. clearance 부족 → 규칙 3(재분해 신호)이 발동하는지
S["u3_clearance_fail"] = {
    "meta": {"kind": "unit", "targets_rule": "규칙 3 재분해 신호"},
    "kg": {
        "scenario": "U3 상자 깊이보다 높은 병",
        "task": "병을 상자에 넣어라",
        "subgoals": [{
            "subgoal_id": "SG1", "description": "병을 상자에 넣는다",
            "target_hints": ["BottleObject"], "goal_region_hint": "box_shallow",
            "required_capabilities": ["stable_grasp", "controlled_transport",
                                      "controlled_release"],
        }],
        "must_precede": [],
    },
    "sg": {
        "objects": {"BottleObject": {"reachable": True, "feasible_ee": ["2f"],
                                     "at_rest": True}},
        "regions": {"box_shallow": {"clear": True}},
        "on_edges": [],
        # 병 높이 160 > 상자 깊이 90 → 음수
        "clearance": {"BottleObject|box_shallow": -70},
        "per_subgoal": {"SG1": {"target": "BottleObject", "goal_region": "box_shallow",
                                "tool_required": False, "feasible_ee": ["2f"],
                                "ee_candidate": "2f"}},
    },
    "robot_state": {"in_hand": None, "current_ee": "2f"},
}

# U4. 가반하중 초과 → 도구 경로(acquire tool → tool_act → 반환 → place)가 생성되는지
#     + 규칙 2b(인과링크 위협) 검증. 도구를 내려놓고 미는 순서가 걸러져야 한다.
S["u4_tool_push"] = {
    "meta": {"kind": "unit", "targets_rule": "도구 경로 + 규칙 2b 인과링크 위협"},
    "kg": {
        "scenario": "U4 가반하중 초과 물체 밀기",
        "task": "무거운 상자를 팔레트로 밀어라",
        "subgoals": [{
            "subgoal_id": "SG1", "description": "무거운 상자를 팔레트로 민다",
            "target_hints": ["PartsBoxObject"], "goal_region_hint": "pallet",
            "required_capabilities": ["push", "tool_use"],
        }],
        "must_precede": [],
    },
    "sg": {
        "objects": {
            "CylinderObject": {"reachable": True, "feasible_ee": ["2f"], "at_rest": True},
            "PartsBoxObject": {"reachable": True, "feasible_ee": [], "at_rest": True},
        },
        "regions": {"pallet": {"clear": True}, "tool_rest": {"clear": True}},
        "on_edges": [],
        "clearance": {"CylinderObject|tool_rest": 100, "PartsBoxObject|pallet": 100},
        "tool_effective": {"CylinderObject|PartsBoxObject": True},
        "per_subgoal": {"SG1": {
            "target": "PartsBoxObject", "goal_region": "pallet",
            "tool_required": True, "selected_tool": "CylinderObject",
            "tool_mode": "push", "tool_rest_region": "tool_rest",
            "feasible_ee": ["2f"], "ee_candidate": "2f"}},
    },
    "robot_state": {"in_hand": None, "current_ee": "2f"},
}


# ══════════════════════════════════════════════════════════════════════
# 8. 쓰기
# ══════════════════════════════════════════════════════════════════════
for name, case in S.items():
    with open(f"scenarios/{name}.json", "w", encoding="utf-8") as f:
        json.dump(case, f, ensure_ascii=False, indent=2)
    kind = case.get("meta", {}).get("task_id") or "단위"
    print(f"wrote scenarios/{name}.json   [{kind}]")

print("\n[SG 도구 판정 유도 결과]")
for label, eff, why in [("T1-1", t11_eff, t11_why), ("T1-1 v2", t11b_eff, t11b_why),
                        ("T1-2", t12_eff, t12_why)]:
    print(f"  {label}")
    for k, v in eff.items():
        print(f"    {'O' if v else 'X'}  {k.ljust(42)} {why[k]}")
