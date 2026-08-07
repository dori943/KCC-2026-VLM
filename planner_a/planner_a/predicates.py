"""Predicate 어휘와 판정 주체(eval_by) 정의.

predicate 표는 늘리지 않고 조합으로 표현한다.
eval_by는 계획 시점에 누가 그 조건을 판정하는지를 정하며,
실행 시점에 실패했을 때 어디로 복귀할지도 같은 필드가 결정한다.
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Sequence

# ---------------------------------------------------------------- eval_by 표
# sg          : Scene Graph 관측값으로 판정
# planner_a   : Planner A가 기하/상태 계산으로 판정
# motion      : 계획 시점에 판정하지 않는다 (pass=None). 실행 시점에만 판정.
EVAL_BY = {
    # --- Scene Graph 관측
    "top_exposed":    "sg",
    "ee_feasible":    "sg",
    "fits_inside":    "sg",
    "tool_effective": "sg",   # 이 도구로 이 target에 실제 작용 가능한가
    # --- Planner A 계산
    "reachable":   "planner_a",
    "clear":       "planner_a",
    "holding":     "planner_a",
    "hand_empty":  "planner_a",
    "above":       "planner_a",
    "in_region":   "planner_a",
    "at_rest":     "planner_a",
    # --- Motion 이 판정 (계획 시점 유보)
    "path_clear":  "motion",
    "attached_ee": "motion",
}

# 인자 타입. 조건 인자에 엉뚱한 종류의 이름이 들어가는 것을 막는다.
#   object = 장면의 물체 / region = 영역·용기 / ee = 엔드이펙터 id / any = 둘 다 허용
ARG_TYPES = {
    "top_exposed":    ("object",),
    "ee_feasible":    ("object",),
    "fits_inside":    ("object", "region"),
    "tool_effective": ("object", "object"),
    "reachable":      ("object",),
    "clear":          ("region",),
    "holding":        ("object",),
    "hand_empty":     (),
    "above":          ("object", "region"),
    "in_region":      ("object", "region"),
    "at_rest":        ("object",),
    "path_clear":     ("any", "any"),
    "attached_ee":    ("ee",),
}

# 한 액션의 사전조건에 동시에 들어올 수 없는 술어 쌍.
# (술어A, 술어B, 인자까지 같아야 모순인가)
#
# 물체를 손에 들고 있는 동안에는 그 물체가 바닥에 놓여 있을 때의 조건이
# 성립할 수 없다. 이 조건들이 함께 들어가면 그 물체를 집는 액션과
# 양방향 엣지가 생겨 사이클이 된다.
CONTRADICTS = [
    ("hand_empty",  "holding",     False),   # 뭔가 들고 있는데 손이 비었다
    ("holding",     "top_exposed", True),    # 들고 있는데 위가 노출돼 있다
    ("holding",     "at_rest",     True),    # 들고 있는데 정지 상태다
]

# 자연어 설명 템플릿 (논문 표·디버깅용)
NL = {
    "top_exposed":    "{0} 위에 다른 object가 없다",
    "ee_feasible":    "현재 EE로 {0}를 다룰 수 있다",
    "fits_inside":    "{0}가 {1} 안에 들어간다",
    "tool_effective": "{0}로 {1}에 실제로 작용할 수 있다",
    "reachable":      "{0}가 작업 반경 안에 있다",
    "clear":          "{0}에 배치 공간이 있다",
    "holding":        "{0}를 들고 있다",
    "hand_empty":     "손이 비어 있다",
    "above":          "{0}가 {1} 위에 있다",
    "in_region":      "{0}가 {1} 안에 있다",
    "at_rest":        "{0}가 안정 상태다",
    "path_clear":     "{0}에서 {1}까지 경로가 비어 있다",
    "attached_ee":    "{0}가 장착되어 있다",
}

# check 식 (실행 시점 평가기가 그대로 쓰는 문자열)
CHECK = {
    "top_exposed":    "count(edges[type=on, to={0}]) == 0",
    "ee_feasible":    "{0} in sg.feasible_ee_objects[current_ee]",
    "fits_inside":    "sg.clearance({0}, {1}) >= 0",
    "tool_effective": "sg.tool_effective({0}, {1})",
    "reachable":      "dist(base, pose({0})) <= robot.reach",
    "clear":          "sg.free_volume({0}) > 0",
    "holding":        "state.in_hand == {0}",
    "hand_empty":     "state.in_hand is None",
    "above":          "xy_overlap({0}, {1})",
    "in_region":      "inside({0}, {1})",
    "at_rest":        "abs(vel({0})) < eps",
    "path_clear":     "motion.collision_free({0}, {1})",
    "attached_ee":    "robot.attached == {0}",
}


@dataclass
class Cond:
    """조건 레코드 하나.

    같은 레코드를 계획 시점 순서 도출과 실행 시점 검증에 모두 쓴다.
    조건을 두 벌 만들지 않는 것이 Planner A 설계의 핵심.
    """
    cond_id: str
    type: str
    args: tuple
    check: str
    eval_by: str
    nl: str
    pass_: Optional[bool] = None   # None = 계획 시점 판정 유보
    # 판정 유보의 이유. 셋 중 하나.
    #   None            계획 시점에 판정됨 (pass_ 가 True/False)
    #   "motion"        심볼 수준에서 원래 미결정. 실행 시점에만 알 수 있다
    #   "<서브골 id>"   그 서브골이 끝나야 관측 가능해진다. 그 전에는 판정 불가
    depends_on: Optional[str] = None
    # 그 시점이 되어도 SG 가 새로 재야 알 수 있는가.
    # depends_on 과 독립이다. 둘 다 참일 수 있다
    #   ("R2 는 SG1_d3 뒤에 생기고, 생긴 뒤에도 치수를 재야 fits_inside 를 안다")
    needs_observation: bool = False

    def key(self) -> tuple:
        """조건의 동일성 판정 키. (술어, 인자)"""
        return (self.type, tuple(self.args))

    def to_json(self) -> dict:
        return {
            "cond_id": self.cond_id,
            "type": self.type,
            "args": list(self.args),
            "check": self.check,
            "pass": self.pass_,
            "eval_by": self.eval_by,
            "depends_on": self.depends_on,
            "needs_observation": self.needs_observation,
            "nl": self.nl,
        }


def make_cond(cond_id: str, ptype: str, args: Sequence[str],
              pass_: Optional[bool] = None,
              depends_on: Optional[str] = None) -> Cond:
    if ptype not in EVAL_BY:
        raise KeyError(f"미정의 predicate: {ptype}. predicates.py의 표를 먼저 고칠 것.")
    eval_by = EVAL_BY[ptype]
    a = tuple(args)
    return Cond(
        cond_id=cond_id,
        type=ptype,
        args=a,
        check=CHECK[ptype].format(*a),
        eval_by=eval_by,
        # motion 조건은 계획 시점에 거짓으로 박지 않는다.
        # 거짓으로 박으면 실제로는 통과 가능한 순서까지 선후관계로 묶여 과제약이 생긴다.
        pass_=None if eval_by == "motion" else pass_,
        depends_on="motion" if eval_by == "motion" else depends_on,
        nl=NL[ptype].format(*a),
    )
