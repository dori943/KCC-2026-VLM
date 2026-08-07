"""action_type 템플릿 (lifted schema).

action_type은 KG가 아니라 Planner A가 부착한다.
KG가 내는 서브골은 러프해서 거기에 action_type을 달 수 없기 때문이다.

STRIPS operator schema를 그대로 씀 (Fikes & Nilsson 1971).
템플릿(domain)은 변수를 가진 채로 두고, 실제 object id를 채워
인스턴스(problem)를 만드는 것이 Planner A의 일이다 (PDDL의 lifted/grounded 구분).

`establish` = 이 액션이 참으로 만드는 조건  → POP의 causal link
`destroy`   = 이 액션이 거짓으로 만드는 조건 → POP의 threat(clobbering)
"""

# ?o = 대상 object, ?r = 목표 영역/용기, ?t = 도구 object, ?ee = end effector
TEMPLATES = {
    # 확보 ------------------------------------------------------------
    "acquire": {
        "params": ["?o", "?ee"],
        "pre": [
            ("reachable",   ["?o"]),
            ("top_exposed", ["?o"]),
            ("ee_feasible", ["?o"]),
            ("attached_ee", ["?ee"]),
            ("hand_empty",  []),
        ],
        "establish": [
            ("holding", ["?o"]),
        ],
        "destroy": [
            ("hand_empty",  []),
            ("at_rest",     ["?o"]),
            ("top_exposed", ["?o"]),
        ],
    },
    # 운반 ------------------------------------------------------------
    "transport": {
        "params": ["?o", "?r"],
        "pre": [
            ("holding",    ["?o"]),
            ("path_clear", ["?o", "?r"]),
        ],
        "establish": [
            ("above", ["?o", "?r"]),
        ],
        "destroy": [],
    },
    # 배치 ------------------------------------------------------------
    "place": {
        "params": ["?o", "?r"],
        "pre": [
            ("holding",     ["?o"]),
            ("above",       ["?o", "?r"]),   # 아래 NOTE 참조
            ("clear",       ["?r"]),
            ("fits_inside", ["?o", "?r"]),
        ],
        "establish": [
            ("in_region",  ["?o", "?r"]),
            ("at_rest",    ["?o"]),
            ("hand_empty", []),
        ],
        "destroy": [
            ("holding", ["?o"]),
            ("clear",   ["?r"]),
        ],
    },
    # 도구 작용 (push / pull / sweep) -----------------------------------
    # 도구로 대상에 작용하는 동작. 확보·운반·배치 3종만으로는 표현되지 않는다.
    # predicate 표는 늘리지 않고 조합으로만 정의했다.
    "tool_act": {
        "params": ["?t", "?o", "?r"],
        "pre": [
            ("holding",        ["?t"]),
            ("tool_effective", ["?t", "?o"]),
            ("path_clear",     ["?t", "?o"]),
        ],
        "establish": [
            ("in_region", ["?o", "?r"]),
            ("reachable", ["?o"]),
            ("at_rest",   ["?o"]),
        ],
        "destroy": [],
    },
}

# 도구 작용의 하위 모드. action_type을 늘리지 않고 파라미터로 둔다.
TOOL_MODES = ("pull", "push", "sweep")

# ---------------------------------------------------------------------------
# NOTE — 템플릿 정의에서 주의할 두 곳
#
# 1) place 의 사전조건에 `above(?o, ?r)` 가 필요하다.
#    이것이 없으면 transport 와 place 사이에 causal link 가 생기지 않아
#    "운반하기 전에 배치" 같은 순서가 합법이 된다. above 를 넣으면
#    acquire → transport → place 가 causal link 만으로 강제된다.
#
# 2) "손에서 놓는다"를 사후조건 `not_holding(?o)` 로 두지 않고
#    destroy: holding(?o) + establish: hand_empty 로 표현했다.
#    부정 술어를 establish 목록에 넣으면 threat 판정이 이중으로 걸린다.
#    의미는 같고 표는 늘지 않는다.
# ---------------------------------------------------------------------------


def consumed(action_type: str) -> set:
    """이 액션이 '사전조건으로 요구하면서 동시에 스스로 소비하는' 조건의 술어 집합.

    규칙 2의 예외 판정에 쓴다. 두 서브골이 같은 조건을 필요로 하면서
    동시에 소비하면 순서 제약이 아니라 상호배제다.
    """
    t = TEMPLATES[action_type]
    pre_keys = {p for p, _ in t["pre"]}
    des_keys = {p for p, _ in t["destroy"]}
    return pre_keys & des_keys
