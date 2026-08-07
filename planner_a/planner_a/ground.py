"""조건 인스턴스 생성 + 초기 상태 평가 + 판정 가능성 판정 + 파생 조건.

템플릿의 변수에 실제 object id 를 채워 조건 레코드를 만든다 (PDDL 의 problem 생성).

각 레코드에는 eval_by 와 함께 **판정 가능 시점**이 붙는다.
계획 시점에 참/거짓을 박을 수 있는 조건은 그렇게 하고, 그럴 수 없는 조건은
pass=None 으로 두되 왜 유보인지를 depends_on 에 남긴다.

  depends_on = None          계획 상태만으로 판정됨
  depends_on = "motion"      심볼 수준에서 원래 미결정
  depends_on = "<서브골 id>" 그 서브골이 끝나야 관측 가능해진다

  needs_observation          그 시점이 되어도 SG 가 새로 재야 안다

두 필드는 독립이다. 하나의 조건에 둘 다 붙을 수 있다.
  "R2 는 SG1_d3 뒤에 생기고, 생긴 뒤에도 치수를 재야 fits_inside 를 안다"
둘을 한 필드에 욱여넣으면 순서(앞의 것)나 관측 요청(뒤의 것) 중 하나를 잃는다.

액션이 세계를 바꾸면 아직 존재하지 않는 대상에 대한 조건이 생긴다.
그것을 지금 참으로 박으면 계획이 거짓말을 한다.
"""
from typing import Optional

from .conditions import (ConditionGenerator, TemplateConditionGenerator,
                         DecidabilityJudge, TemplateDecidabilityJudge,
                         validate)
from .predicates import make_cond, Cond
from .templates import TEMPLATES


class InitialState:
    """SG 관측값 + 로봇 초기 상태에서 술어의 초기 진리값을 판정.

    관측에 없는 것은 True 로 기본값을 주지 않고 None(미관측)을 돌려준다.
    기본값 True 는 SG 가 보지 못한 것을 봤다고 적는 것과 같다.
    """

    def __init__(self, sg: dict, robot: dict, task: str = ""):
        self.sg = sg
        self.robot = robot
        self.task = task            # 자연어 명령. 판정기가 참고한다.
        self.objects = sg.get("objects", {})
        self.regions = sg.get("regions", {})
        self.on = {}          # child -> parent
        self.supported = {}   # parent -> [children]
        for a, b in sg.get("on_edges", []):
            self.on[a] = b
            self.supported.setdefault(b, []).append(a)

    def is_multi(self, region: str) -> bool:
        """여러 object가 동시에 들어갈 수 있는 영역인가 (트레이·상자 등).

        용량이 1인 영역은 배치하면 clear 가 소진되지만, 트레이는 그렇지 않다.
        이걸 구분하지 않으면 같은 트레이에 넣는 서브골끼리 전부 상호배제로 잡힌다.
        """
        return bool(self.regions.get(region, {}).get("multi", False))

    def supported_by(self, region: str) -> Optional[str]:
        """이 영역이 어떤 object 의 윗면인가.

        SG 가 `supported_by` 를 달아 보내면, 그 object 가 제자리에 놓이기
        전까지 이 영역은 존재하지 않는다. 관측할 수 있는 대상이 아니다.
        """
        return self.regions.get(region, {}).get("supported_by")

    def observed_region(self, region: str) -> bool:
        return region in self.regions and self.supported_by(region) is None

    def eval(self, ptype: str, args) -> Optional[bool]:
        o = args[0] if len(args) > 0 else None
        r = args[1] if len(args) > 1 else None

        # --- 로봇 자신의 상태. 관측이 아니라 계획 상태다.
        if ptype == "hand_empty":
            return self.robot.get("in_hand") is None
        if ptype == "holding":
            return self.robot.get("in_hand") == o
        if ptype == "above":
            return False                       # 초기에는 어떤 것도 목표 위에 없다

        # --- 심볼 수준에서 원래 미결정
        if ptype in ("attached_ee", "path_clear"):
            return None

        # --- object 관측
        if ptype in ("reachable", "top_exposed", "at_rest", "ee_feasible"):
            if o not in self.objects:
                return None                    # 관측에 없는 object
            if ptype == "top_exposed":
                return len(self.supported.get(o, [])) == 0
            if ptype == "reachable":
                return bool(self.objects[o].get("reachable", True))
            if ptype == "at_rest":
                return bool(self.objects[o].get("at_rest", True))
            return bool(self.objects[o].get("feasible_ee"))

        if ptype == "in_region":
            if o not in self.objects:
                return None
            return self.objects[o].get("in_region") == r

        # --- region 관측
        if ptype == "clear":
            reg = r if r else o
            if not self.observed_region(reg):
                return None                    # 아직 존재하지 않거나 관측 밖
            return bool(self.regions[reg].get("clear", True))

        if ptype == "fits_inside":
            if not self.observed_region(r):
                return None                    # 영역이 아직 없으면 여유 공간도 없다
            key = f"{o}|{r}"
            if key not in self.sg.get("clearance", {}):
                return None                    # SG 가 재지 않았다
            return self.sg["clearance"][key] >= 0

        if ptype == "tool_effective":
            key = f"{o}|{r}"
            table = self.sg.get("tool_effective", {})
            if key not in table:
                return None                    # SG 가 판정하지 않았다
            return bool(table[key])

        raise KeyError(f"초기 상태 평가기가 모르는 predicate: {ptype}")


def _bind(args, binding):
    return [binding.get(a, a) for a in args]


def ground(dsubs: list, state: InitialState,
           generator: ConditionGenerator = None,
           judge: DecidabilityJudge = None) -> dict:
    """각 상세 서브골에 pre / establish / destroy 조건 레코드를 붙인다.

    조건 집합 자체를 어디서 얻을지는 generator 가 정한다 (기본: 템플릿).
    각 조건이 언제 판정 가능한지는 judge 가 정한다 (기본: 규칙 기반).
    VLM 을 쓰려면 conditions.VLMDecidabilityJudge 를 judge 로 넘긴다.
    """
    generator = generator or TemplateConditionGenerator()
    judge = judge or TemplateDecidabilityJudge()

    for d in dsubs:
        pre_spec, est_spec, des_spec = generator.generate(d, state)
        # 어휘·인자 검증. VLM 이 표에 없는 술어를 만들면 여기서 걸린다.
        validate((pre_spec, est_spec, des_spec))
        d.condition_source = generator.name
        d.pre, d.establish, d.destroy = [], [], []

        for i, (ptype, args) in enumerate(pre_spec):
            a = _bind(args, d.binding)
            d.pre.append(make_cond(f"C_{d.sid}_pre_{i}", ptype, a, state.eval(ptype, a)))

        for i, (ptype, args) in enumerate(est_spec):
            a = _bind(args, d.binding)
            d.establish.append(make_cond(f"C_{d.sid}_est_{i}", ptype, a, True))

        for i, (ptype, args) in enumerate(des_spec):
            a = _bind(args, d.binding)
            # 트레이처럼 여러 개가 들어가는 영역은 배치해도 clear 가 소진되지 않는다
            if ptype == "clear" and state.is_multi(a[0]):
                continue
            d.destroy.append(make_cond(f"C_{d.sid}_des_{i}", ptype, a, False))

        _add_derived(d, state)

    # ---- 판정 가능 시점 판정 -------------------------------------------
    return _decide(dsubs, state, judge)


def _decide(dsubs, state, judge) -> dict:
    """계획 시점에 판정되지 않은 조건마다 왜 유보인지를 채운다.

    두 가지를 따로 기록한다.
      deferred     앞선 서브골이 끝나야 판정 가능  -> 순서 엣지가 된다
      sg_requests  SG 가 새로 재야 판정 가능       -> SG 로 되돌릴 관측 요청
    하나의 조건이 양쪽에 모두 들어갈 수 있다.
    """
    deferred, sg_requests = [], []
    for d in dsubs:
        pending = [c for c in d.pre
                   if c.pass_ is None and c.depends_on != "motion"]
        if not pending:
            continue
        verdicts = judge.judge(d, pending, state, dsubs)
        for c in pending:
            v = verdicts.get(c.cond_id) or {}
            c.depends_on = v.get("depends_on")
            c.needs_observation = bool(v.get("needs_observation", not c.depends_on))
            rec = {"subgoal": d.sid, "condition": c.cond_id,
                   "type": c.type, "args": list(c.args), "nl": c.nl,
                   "depends_on": c.depends_on,
                   "needs_observation": c.needs_observation,
                   "why": v.get("why", "")}
            if c.depends_on:
                deferred.append(rec)
            if c.needs_observation:
                sg_requests.append(dict(
                    rec, request=(f"{c.depends_on} 이후에 재줄 것"
                                  if c.depends_on else "지금 재줄 것")))
    return {"deferred": deferred, "unobserved": sg_requests,
            "decidability_source": judge.name}


def _add_derived(d, state: InitialState) -> None:
    """파생 조건.

    object 를 집어 들면 그 밑 object 의 top_exposed 가 참이 된다.
    명시하지 않으면 "위에 있는 걸 먼저 치워라" 순서가 도출되지 않는다.
    같은 지지물 위에 다른 object 가 남아 있으면 추가하지 않는다.
    """
    if d.action_type != "acquire":
        return
    o = d.binding["?o"]
    under = state.on.get(o)
    if under is None:
        return
    siblings = [x for x in state.supported.get(under, []) if x != o]
    if siblings:
        return
    c = make_cond(f"C_{d.sid}_est_derived", "top_exposed", [under], True)
    c.nl += " (파생: 위의 object를 집어 들어 노출됨)"
    d.establish.append(c)
