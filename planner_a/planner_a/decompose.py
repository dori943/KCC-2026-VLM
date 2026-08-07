"""KG 러프 서브골 -> 상세 서브골 재분해 + action_type 부착.

KG 는 "유리병을 상자로 옮긴다" 수준까지만 내고, 로봇팔 입장의
접근/파지/들어올림 단위로 쪼개는 것은 Planner A 가 한다.
어떤 object 를 어떤 EE 로 다루는지 알아야 쪼갤 수 있는데
그 정보가 KG 에 없기 때문이다.

분해 종료 조건은 정적 method 가 아니라 **관측 기반 실행가능성**이다.
SG 가 준 tool_required / feasible_ee 를 보고 깊이가 달라진다.
"""
from dataclasses import dataclass, field
from typing import Optional

from .templates import TEMPLATES

# capability 어휘 -> action_type 안전망.
# 구조적 분해가 정본이고, 이 부분집합 점수 매칭은 대조용으로만 쓴다.
# KG 의 capability 어휘와 분해 결과가 어긋나면 note 에 표시된다.
CAPABILITY_HINT = {
    "acquire":   {"stable_grasp", "grasp", "pick", "detection", "localization"},
    "transport": {"controlled_transport", "transport", "carry", "move"},
    "place":     {"controlled_release", "release", "place", "insert", "container_interior_localization"},
    "tool_act":  {"push", "pull", "sweep", "non_prehensile", "tool_use"},
}


@dataclass
class DetailedSubgoal:
    """상세 서브골 하나. Planner B 로 넘어가는 최소 단위."""
    sid: str
    action_type: str
    binding: dict                 # ?o -> object id 등
    group_id: str                 # 같은 object 의 확보/운반/배치를 묶는 키
    from_kg: str                  # 원래 KG 서브골 id
    mode: Optional[str] = None    # tool_act 일 때 pull/push/sweep
    note: str = ""

    def arg(self, var: str) -> str:
        return self.binding[var]


def _cap_score(caps, action_type) -> int:
    """capability 어휘와 action_type 사전의 교집합 크기 (안전망)."""
    hint = CAPABILITY_HINT[action_type]
    score = 0
    for c in caps or []:
        c = c.lower()
        for h in hint:
            if h in c or c in h:
                score += 1
                break
    return score


def decompose(kg: dict, sg: dict) -> list:
    """KG 서브골 목록을 상세 서브골 목록으로 편다."""
    out = []
    per_sg = sg.get("per_subgoal", {})

    for k in kg["subgoals"]:
        kid = k["subgoal_id"]
        info = per_sg.get(kid, {})
        target = info.get("target") or (k.get("target_hints") or [None])[0]
        region = info.get("goal_region") or k.get("goal_region_hint")
        caps = k.get("required_capabilities", [])

        tool_required = bool(info.get("tool_required"))
        tool = info.get("selected_tool")
        ee = info.get("ee_candidate") or (info.get("feasible_ee") or ["2f"])[0]

        if tool_required and tool:
            # 도구를 먼저 확보하고, 도구로 target 에 작용하고, 도구를 내려놓는다.
            mode = info.get("tool_mode", "pull")
            tool_rest = info.get("tool_rest_region", "tool_rest")
            out += [
                DetailedSubgoal(f"{kid}_d1", "acquire",
                                {"?o": tool, "?ee": ee},
                                group_id=f"G_{tool}", from_kg=kid,
                                note=f"도구 {tool} 확보"),
                DetailedSubgoal(f"{kid}_d2", "tool_act",
                                {"?t": tool, "?o": target, "?r": region},
                                group_id=f"G_{tool}", from_kg=kid, mode=mode,
                                note=f"{tool}로 {target}에 {mode}"),
                DetailedSubgoal(f"{kid}_d3", "place",
                                {"?o": tool, "?r": tool_rest},
                                group_id=f"G_{tool}", from_kg=kid,
                                note=f"도구 {tool} 내려놓기"),
            ]
            # tool_act 는 도구를 든 채로 하므로 above 사전조건이 필요 없다.
            # place(tool) 는 above 가 필요하므로 운반 한 단계를 끼운다.
            out.insert(len(out) - 1,
                       DetailedSubgoal(f"{kid}_d2b", "transport",
                                       {"?o": tool, "?r": tool_rest},
                                       group_id=f"G_{tool}", from_kg=kid,
                                       note="도구를 거치 위치로 운반"))
        else:
            if target is None:
                continue
            grp = f"G_{target}"
            out.append(DetailedSubgoal(f"{kid}_d1", "acquire",
                                       {"?o": target, "?ee": ee},
                                       group_id=grp, from_kg=kid,
                                       note=f"{target} 확보"))
            if region:
                out.append(DetailedSubgoal(f"{kid}_d2", "transport",
                                           {"?o": target, "?r": region},
                                           group_id=grp, from_kg=kid,
                                           note=f"{target}를 {region}로 운반"))
                out.append(DetailedSubgoal(f"{kid}_d3", "place",
                                           {"?o": target, "?r": region},
                                           group_id=grp, from_kg=kid,
                                           note=f"{target}를 {region}에 배치"))

    # 안전망: 구조적 분해 결과가 KG capability 와 심하게 어긋나면 표시만 남긴다.
    for d in out:
        k = next((x for x in kg["subgoals"] if x["subgoal_id"] == d.from_kg), None)
        if k and _cap_score(k.get("required_capabilities"), d.action_type) == 0:
            d.note += " [capability 어휘와 불일치 - 확인 필요]"
    return out
