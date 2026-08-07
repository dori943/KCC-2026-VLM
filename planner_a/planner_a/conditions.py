"""조건 집합 생성기 + 판정 가능성 판정기.

이 모듈에는 두 종류의 모듈이 있다.

  ConditionGenerator   액션 타입에 어떤 조건이 붙는지 (집합 생성)
  DecidabilityJudge    그 조건이 언제 판정 가능한지 (시점 판정)

앞의 것은 액션 타입 수준이라 장면과 거의 무관하고, 템플릿이 하한을 준다.
뒤의 것은 조건 인스턴스 수준이라 장면과 계획에 직접 달려 있다.
액션이 세계를 바꾸면 아직 존재하지 않는 대상에 대한 조건이 생기고,
그것을 지금 참으로 박으면 계획이 거짓말을 한다.


  TemplateConditionGenerator : 결정론적. 순서 도출 로직의 회귀 검증 전용.
  VLMConditionGenerator      : 장면을 보고 조건 집합을 생성.

회귀 테스트는 항상 Template 로 돌린다. VLM 을 끼우면 출력이 비결정적이 되어
순서 규칙 버그와 VLM 오류를 구분할 수 없다. 이 분리가 그대로
ablation 축(템플릿 조건 vs 생성 조건)이 된다.

생성 단위
---------
**action_type 스키마당 한 번만** 호출하고, 결과를 리프티드 형태로 캐시한 뒤
바인딩은 결정론적으로 한다. 서브골마다 호출하면 구조가 같은 서브골이 서로 다른
답을 받아 계획이 흔들린다. 호출 수도 서브골 수가 아니라 action_type 수로 줄어든다.

어휘 고정
---------
생성된 predicate 는 predicates.EVAL_BY 표 안에서만 허용하고, 인자 개수와
타입도 검사한다. 검사가 없으면 호출마다 다른 이름의 술어가 생성되어
조건 어휘가 폭증한다.
"""
import json
import os
import re
from typing import List, Optional, Tuple

from .predicates import ARG_TYPES, CONTRADICTS, EVAL_BY, NL
from .templates import TEMPLATES


class ConditionGenerator:
    """조건 집합 생성기 인터페이스.

    generate() 는 (pre, establish, destroy) 세 목록을 (술어, 인자) 쌍으로 반환.
    실제 조건 레코드로 만드는 것은 ground.py 가 한다.
    """

    name = "base"

    def generate(self, dsub, state) -> Tuple[List, List, List]:
        raise NotImplementedError


class TemplateConditionGenerator(ConditionGenerator):
    """action_type 템플릿에서 그대로 꺼낸다. 결정론적."""

    name = "template"

    def generate(self, dsub, state):
        t = TEMPLATES[dsub.action_type]
        return list(t["pre"]), list(t["establish"]), list(t["destroy"])



# ══════════════════════════════════════════════════════════════════════
# 판정 가능성 판정기 (Decidability Judge)
# ══════════════════════════════════════════════════════════════════════
class DecidabilityJudge:
    """계획 시점에 판정되지 않은 조건이 왜 유보인지를 정한다.

    judge() 는 {cond_id: {"depends_on": sid|None,
                          "needs_observation": bool, "why": str}} 를 반환.

      depends_on         그 서브골이 끝나야 대상이 생긴다 -> 순서 엣지가 된다
      needs_observation  그 시점이 되어도 SG 가 새로 재야 안다 -> 관측 요청이 된다

    두 필드는 독립이다. 둘 다 참인 조건이 있다.
    """

    name = "base"

    def judge(self, dsub, conds, state, dsubs) -> dict:
        raise NotImplementedError


def _producers_of(dsubs) -> dict:
    """object -> 그 object 를 제자리에 놓는 서브골 id."""
    out = {}
    for d in dsubs:
        if d.action_type in ("place", "tool_act"):
            o = d.binding.get("?o")
            if o:
                out[o] = d.sid
    return out


class TemplateDecidabilityJudge(DecidabilityJudge):
    """규칙 기반. 결정론적이라 회귀 검증에 쓴다.

    조건 인자에 '아직 존재하지 않는 영역'이 있으면, 그 영역을 만드는
    서브골에 의존한다고 본다. SG 가 영역에 supported_by 를 달아 보내면
    그 영역은 받침 object 가 제자리에 놓이기 전까지 관측 대상이 아니다.

    관측 요청 여부는 eval_by 로 정한다. 판정 주체가 sg 인데 값이 없으면
    SG 가 재야 하는 것이고, planner_a 면 계획 상태만으로 알 수 있다.
    """

    name = "rule"

    def judge(self, dsub, conds, state, dsubs) -> dict:
        prod = _producers_of(dsubs)
        out = {}
        for c in conds:
            dep, why = None, ""
            for a in c.args:
                sup = state.supported_by(a)
                if sup and prod.get(sup) and prod[sup] != dsub.sid:
                    dep = prod[sup]
                    why = f"{a} 는 {sup} 를 놓아야 생기는 영역이다"
                    break
            need = (c.eval_by == "sg") or dep is None
            if not why:
                why = "관측에 없음"
            out[c.cond_id] = {"depends_on": dep, "needs_observation": need,
                              "why": why}
        return out


JUDGE_SYSTEM = """\
너는 로봇 조작 플래너의 판정 가능성 판정 모듈이다.

플래너가 계획을 세우는 시점에, 각 사전조건이 지금 판정 가능한지를 정한다.
조건을 만들거나 고치는 것이 아니다. 이미 만들어진 조건의 판정 시점만 정한다.

**서로 독립인 두 가지**를 각각 답한다. 하나로 합치지 않는다.

[when]  대상이 언제 생기는가
  after:<서브골>  그 서브골이 끝나야 대상이 생긴다. 그 전에는 존재하지 않는다
  unknown        계획 안의 어떤 서브골도 이 대상을 만들지 않는다

[needs_observation]  그때가 되면 바로 알 수 있는가
  true   그 시점에 Scene Graph 가 새로 재야 값을 안다 (치수, 여유 공간, 형상 등)
  false  계획 상태만으로 알 수 있다 (무엇을 들고 있는지, 어디에 놓았는지 등)

둘은 동시에 성립할 수 있고, 그것이 흔한 경우다.
  "R2 는 SG1_d3 뒤에 생기고, 생긴 뒤에도 치수를 재야 들어가는지 안다"
  -> when = after:SG1_d3,  needs_observation = true

규칙
1. 액션이 세계를 바꿔서 생기는 대상(놓아야 생기는 윗면, 옮겨야 닿는 위치 등)에
   대한 조건은 after 다. 그 대상을 만드는 서브골 id 를 정확히 적는다.
2. 지금 판정하는 서브골 자신을 after 로 지목하지 않는다.
3. **대상을 만드는 서브골을 찾았다면, 값을 아직 모른다는 이유로 unknown 으로
   내리지 마라.** 그때는 after 를 적고 needs_observation 을 true 로 둔다.
   after 를 놓치면 계획이 순서를 통째로 잃는다.
4. 확신이 없으면 지금 판정된 것으로 취급하지 말고 unknown 으로 둔다.
   지금 참으로 박는 것이 가장 나쁜 오답이다.

JSON 으로만 답한다.
{"verdicts": [{"cond_id": "...", "when": "after|unknown",
               "after": "<서브골 id 또는 null>",
               "needs_observation": true, "why": "한 줄"}]}
"""

JUDGE_USER = """\
[명령]
{task}

[관측 — 이것이 지금 볼 수 있는 전부다]
objects  : {objects}
regions  : {regions}
on_edges : {on_edges}

관측 주석
- regions 의 supported_by 는 그 영역이 어떤 object 의 윗면이라는 뜻이다.
  받침 object 가 제자리에 놓이기 전에는 그 영역이 존재하지 않는다.

[계획 안의 서브골]
{plan}

[지금 판정할 서브골]
{sid}  {action_type}  {binding}

[판정할 조건]
{conds}

각 조건에 대해 when 과 needs_observation 을 각각 정하라.
"""


class VLMDecidabilityJudge(DecidabilityJudge):
    """조건 인스턴스마다 판정 가능 시점을 VLM 이 정한다.

    조건 집합 생성과 달리 여기서는 캐시하지 않는다.
    같은 action_type 이라도 인스턴스가 다르면 답이 달라야 하기 때문이다
    (place(A, 테이블) 은 now, place(B, A 의 윗면) 은 after).

    답의 공간이 now / after:<id> / unknown 셋뿐이라 자유 생성보다
    검증이 쉽고 흔들릴 여지가 적다.

    실패 처리
    ---------
    없는 서브골 지목, 자기 자신 지목 → 오류를 붙여 재시도
    재시도도 실패 → 규칙 판정기로 fallback
    """

    name = "vlm"

    def __init__(self, client=None, model: str = None,
                 fallback_to_rule: bool = True,
                 max_retry: int = 1,
                 log_path: str = None):
        self.client = client or _default_client()
        self.model = model or os.environ.get("PLANNER_A_VLM_MODEL", "gpt-4o")
        self.fallback = TemplateDecidabilityJudge() if fallback_to_rule else None
        self.max_retry = max_retry
        self.log_path = log_path
        self.stats = {"calls": 0, "retries": 0, "fallbacks": 0,
                      "prompt_tokens": 0, "completion_tokens": 0,
                      "after": 0, "unknown": 0, "needs_obs": 0}
        self.log = []

    # ------------------------------------------------------------------
    def judge(self, dsub, conds, state, dsubs):
        err = None
        for attempt in range(self.max_retry + 1):
            try:
                out = self._ask(dsub, conds, state, dsubs, err)
                self._record(dsub, out, None)
                for v in out.values():
                    self.stats["after" if v["depends_on"] else "unknown"] += 1
                    self.stats["needs_obs"] += bool(v["needs_observation"])
                return out
            except (JudgeError, ValueError, KeyError) as e:      # noqa: PERF203
                err = str(e)
                if attempt < self.max_retry:
                    self.stats["retries"] += 1
        self.stats["fallbacks"] += 1
        self._record(dsub, None, err)
        if self.fallback:
            return self.fallback.judge(dsub, conds, state, dsubs)
        return {c.cond_id: {"depends_on": None, "needs_observation": True,
                            "why": "판정 실패"} for c in conds}

    # ------------------------------------------------------------------
    def _ask(self, dsub, conds, state, dsubs, prev_err):
        plan = "\n".join(
            f"  {d.sid}  {d.action_type}  {json.dumps(d.binding, ensure_ascii=False)}"
            for d in dsubs)
        cond_lines = "\n".join(
            f"  {c.cond_id}  {c.type}({', '.join(c.args)})  — {c.nl}" for c in conds)
        user = JUDGE_USER.format(
            task=getattr(state, "task", ""),
            objects=json.dumps(state.objects, ensure_ascii=False),
            regions=json.dumps(state.regions, ensure_ascii=False),
            on_edges=json.dumps(state.sg.get("on_edges", []), ensure_ascii=False),
            plan=plan, sid=dsub.sid, action_type=dsub.action_type,
            binding=json.dumps(dsub.binding, ensure_ascii=False),
            conds=cond_lines)
        if prev_err:
            user += f"\n\n[직전 응답 오류 — 고쳐서 다시 답할 것]\n{prev_err}\n"

        content = [{"type": "text", "text": user}]
        img = state.sg.get("image")
        if img:
            content.append({"type": "image_url",
                            "image_url": {"url": _as_data_url(img)}})
        self.stats["calls"] += 1
        resp = self.client.chat.completions.create(
            model=self.model, temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": JUDGE_SYSTEM},
                      {"role": "user", "content": content}])
        usage = getattr(resp, "usage", None)
        if usage:
            self.stats["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
            self.stats["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        return self._parse(resp.choices[0].message.content, dsub, conds, dsubs)

    # ------------------------------------------------------------------
    @staticmethod
    def _parse(text, dsub, conds, dsubs):
        text = re.sub(r"^```(?:json)?|```$", "", text.strip(),
                      flags=re.MULTILINE).strip()
        data = json.loads(text)
        rows = data.get("verdicts", data if isinstance(data, list) else [])
        want = {c.cond_id for c in conds}
        sids = {d.sid for d in dsubs}
        out = {}
        for row in rows:
            cid = row.get("cond_id")
            if cid not in want:
                raise JudgeError(f"계획에 없는 조건 id: {cid}")
            when = (row.get("when") or "").strip()
            after = row.get("after") or None
            if isinstance(when, str) and when.startswith("after:"):
                after, when = when.split(":", 1)[1].strip(), "after"
            why = (row.get("why") or "")[:120]
            need = row.get("needs_observation")
            if when == "now":
                raise JudgeError(
                    f"{cid} 는 관측으로 판정되지 않아 유보된 조건이다. "
                    "now 로 답할 수 없다. after 또는 unknown 이어야 한다.")
            if when == "after":
                if after not in sids:
                    raise JudgeError(f"{cid} 의 after 가 계획에 없는 서브골이다: {after}")
                if after == dsub.sid:
                    raise JudgeError(f"{cid} 의 after 가 자기 자신이다: {after}")
                out[cid] = {"depends_on": after,
                            "needs_observation": bool(need), "why": why}
            elif when == "unknown":
                # 대상을 만드는 서브골을 못 찾았다는 뜻이므로 SG 가 재야 한다
                out[cid] = {"depends_on": None, "needs_observation": True,
                            "why": why}
            else:
                raise JudgeError(f"{cid} 의 when 이 now/after/unknown 이 아니다: {when}")
        missing = want - set(out)
        if missing:
            raise JudgeError(f"판정이 빠진 조건: {sorted(missing)}")
        return out

    def _record(self, dsub, out, err):
        self.log.append({"subgoal_id": dsub.sid, "action_type": dsub.action_type,
                         "verdicts": out, "error": err})
        self.flush()

    def flush(self):
        if not self.log_path:
            return
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump({"stats": self.stats, "log": self.log},
                      f, ensure_ascii=False, indent=2)


class JudgeError(ValueError):
    """판정 응답이 형식·참조 검사를 통과하지 못함."""


# --------------------------------------------------------------- 프롬프트
SYSTEM_PROMPT = """\
너는 로봇 조작 플래너의 조건 생성 모듈이다.
액션 스키마 하나를 받아, 그 동작이 성립하기 위한 사전조건과
동작 후 바뀌는 사후조건을 JSON 으로 낸다.

규칙
1) 주어진 어휘표의 술어만 쓴다. 새 술어를 만들면 거부된다.
2) 인자는 주어진 바인딩 변수(?o 등)만 쓰고, 어휘표에 적힌 타입을 지킨다.
   구체적인 물체 이름을 쓰지 않는다.
3) 기본 템플릿 조건은 그대로 유지한다.
4) 조건을 억지로 추가하지 마라. 장면을 보고 **이 액션이 실패할 수 있는 이유**가
   확인될 때만 더한다. 없으면 템플릿 그대로 낸다. 추가하지 않는 것이 기본이다.
5) 이미 손에 들고 있는 물체에는, 그 물체가 바닥에 놓여 있을 때만 성립하는
   조건(top_exposed, at_rest)이나 hand_empty 를 붙이지 않는다.
6) JSON 만 출력한다. 설명도 코드펜스도 붙이지 않는다.

출력 형식 — 인자가 없는 술어도 빈 배열을 반드시 붙인다
{"pre":       [["holding", ["?o"]], ["hand_empty", []]],
 "establish": [["above", ["?o", "?r"]]],
 "destroy":   []}
"""

FEWSHOT = """\
[예시 1] 다른 도메인(블록 쌓기)의 액션 스키마

action_type : unstack   바인딩 변수 : ?b, ?u
설명 : 다른 블록 위에 얹힌 블록을 떼어낸다

추론 : 떼어내려면 그 블록 위에 아무것도 없어야 하고 손이 비어 있어야 한다.
떼어내면 손에 들리고, 밑에 있던 블록의 위가 노출된다.
이 밖에 명시적으로도, 암묵적으로도, 상식적으로도 추가할 사전조건은 없다.

{"pre":[["top_exposed",["?b"]],["hand_empty",[]]],
 "establish":[["holding",["?b"]],["top_exposed",["?u"]]],
 "destroy":[["hand_empty",[]],["top_exposed",["?b"]]]}

[예시 2] 다른 도메인(블록 쌓기)의 액션 스키마

action_type : stack   바인딩 변수 : ?b, ?r
설명 : 이미 손에 든 블록을 자리에 내려놓는다

추론 : 놓으려면 그 블록을 들고 있어야 하고 놓을 자리가 비어 있어야 한다.
이미 들고 있으므로 그 블록이 바닥에 놓여 있을 때의 조건은 검사하지 않는다.
이 밖에 명시적으로도, 암묵적으로도, 상식적으로도 추가할 사전조건은 없다.

{"pre":[["holding",["?b"]],["clear",["?r"]]],
 "establish":[["in_region",["?b","?r"]],["hand_empty",[]]],
 "destroy":[["holding",["?b"]],["clear",["?r"]]]}

"""

USER_TEMPLATE = """\
[술어 어휘 — 이 밖의 술어는 금지]
{vocab}

{fewshot}[이번 액션 스키마]
action_type : {action_type}
바인딩 변수  : {params}
설명         : {note}
{stance}
[기본 템플릿 — 이 조건들은 그대로 유지한다]
pre       : {t_pre}
establish : {t_est}
destroy   : {t_des}

[장면]
objects : {objects}
regions : {regions}
on 관계 : {on_edges}

장면은 이 스키마에 조건을 더할 이유가 있는지 판단할 때만 참고한다.
이 장면 때문에 이 액션이 실패할 수 있는 이유가 보이면 그 조건만 추가하라.
보이지 않으면 템플릿 그대로 내라. 조건은 모두 합쳐 {max_conds}개를 넘기지 않는다.
JSON 만.
"""

ACTION_NOTE = {
    "acquire":   "대상 object 를 파지해 손에 든다.",
    "transport": "이미 손에 든 object 를 목표 영역 위로 옮긴다.",
    "place":     "이미 손에 든 object 를 목표 영역에 내려놓는다.",
    "tool_act":  "이미 손에 든 도구로 대상 object 를 끌기·밀기·쓸기 한다.",
}


def _vocab_block() -> str:
    """술어 목록에 뜻과 타입을 같이 준다.

    이름과 타입만 주면 모델이 뜻을 짐작해서 엉뚱한 자리에 쓴다.
    """
    lines = []
    for p, ev in EVAL_BY.items():
        types = ARG_TYPES.get(p, ())
        sig = f"{p}({', '.join(types)})"
        try:
            meaning = NL[p].format(*types)
        except (IndexError, KeyError):
            meaning = ""
        lines.append(f"  {sig:34} {meaning}")
    lines.append("")
    lines.append("  타입: object = 장면의 물체 / region = 영역·용기 / ee = 엔드이펙터 id")
    return "\n".join(lines)


def _stance(action_type: str) -> str:
    """이 액션 시작 시점에 이미 참인 것과, 결과로 만들어지는 것을 못박는다.

    이걸 안 적으면 두 가지 오류가 난다.
      - 운반·배치처럼 물체를 이미 든 액션에 hand_empty / top_exposed 를 붙임
      - 이 액션이 만들어낼 결과(예: tool_act 의 reachable)를 전제로 요구함
    """
    t = TEMPLATES[action_type]
    lines = []
    held = next((a[0] for p, a in t["pre"] if p == "holding"), None)
    if held:
        lines.append(
            f"로봇은 이미 {held} 를 손에 들고 있다고 가정한다. 따라서 {held} 가\n"
            f"바닥에 놓여 있을 때만 성립하는 조건(top_exposed, at_rest)이나\n"
            f"hand_empty 는 사전조건이 될 수 없다.")
    est = sorted({p for p, _ in t["establish"]})
    if est:
        lines.append(
            f"이 액션의 결과는 {', '.join(est)} 다. 결과를 사전조건으로\n"
            f"요구하지 않는다. 장면에서 그 조건이 지금 거짓이더라도,\n"
            f"그것을 참으로 만드는 것이 이 액션의 목적이다.")
    if not lines:
        return ""
    return "\n[이 액션의 전제와 결과]\n" + "\n".join(lines) + "\n"


class VLMConditionGenerator(ConditionGenerator):
    """장면을 보고 조건 집합을 생성한다.

    설계 결정 (생성자 인자로 변경 가능)
    -----------------------------------
    ⓐ **action_type 스키마당 1회 호출**(`schema_level`). 결과를 리프티드로 캐시하고
       바인딩은 결정론적으로 한다. 서브골 단위 호출은 같은 구조에 다른 답을 준다.
    ⓑ 템플릿 조건을 **하한**으로 둔다(`union_with_template`). 누락이 가장 위험하다.
    ⓒ 조건 개수 상한(`max_conds`). 넘으면 줄이라고 되돌린다.
    ⓓ few-shot 2개는 **다른 도메인**을 쓴다. 같은 도메인 예시는 답을 편향시킨다.
    ⓔ 이미지는 선택. `sg["image"]` 가 있으면 함께 보낸다.

    실패 처리
    ---------
    어휘·타입·모순·개수 검사 실패 → 오류를 프롬프트에 붙여 재시도
    재시도도 실패 → 템플릿으로 fallback
    """

    name = "vlm"

    def __init__(self, client=None, model: str = None,
                 union_with_template: bool = True,
                 fallback_to_template: bool = True,
                 image_path_key: str = "image",
                 max_retry: int = 1,
                 schema_level: bool = True,
                 fewshot: bool = True,
                 max_conds: int = 12,
                 log_path: str = None):
        self.client = client or _default_client()
        self.model = model or os.environ.get("PLANNER_A_VLM_MODEL", "gpt-4o")
        self.union_with_template = union_with_template
        self.fallback = TemplateConditionGenerator() if fallback_to_template else None
        self.image_path_key = image_path_key
        self.max_retry = max_retry
        self.schema_level = schema_level
        self.fewshot = fewshot
        self.max_conds = max_conds
        self.log_path = log_path
        self._cache = {}
        self.stats = {"calls": 0, "retries": 0, "fallbacks": 0, "cache_hits": 0,
                      "prompt_tokens": 0, "completion_tokens": 0}
        self.log = []

    # ------------------------------------------------------------------
    def generate(self, dsub, state):
        t = TEMPLATES[dsub.action_type]
        key = dsub.action_type
        if self.schema_level and key in self._cache:
            self.stats["cache_hits"] += 1
            spec = self._cache[key]
        else:
            spec = self._ask(dsub, state, t)
            if self.schema_level:
                self._cache[key] = spec
        return tuple(_bind_spec(g, dsub.binding) for g in spec)

    def _ask(self, dsub, state, t):
        t_pre = list(t["pre"])
        t_est = list(t["establish"])
        t_des = list(t["destroy"])
        user = USER_TEMPLATE.format(
            vocab=_vocab_block(),
            fewshot=FEWSHOT if self.fewshot else "",
            action_type=dsub.action_type,
            params=", ".join(t["params"]),
            note=ACTION_NOTE.get(dsub.action_type, "-"),
            stance=_stance(dsub.action_type),
            t_pre=_fmt(t_pre), t_est=_fmt(t_est), t_des=_fmt(t_des),
            objects=json.dumps(state.objects, ensure_ascii=False),
            regions=json.dumps(state.regions, ensure_ascii=False),
            on_edges=json.dumps(state.sg.get("on_edges", []), ensure_ascii=False),
            max_conds=self.max_conds,
        )
        image = state.sg.get(self.image_path_key)
        # 리프티드 스키마이므로 허용 인자는 템플릿에 등장하는 변수뿐이다
        variables = set(t["params"]) | {
            a for grp in ("pre", "establish", "destroy")
            for _, args in t[grp] for a in args}

        err = None
        for attempt in range(self.max_retry + 1):
            prompt = user if err is None else f"{user}\n\n[직전 시도 오류 — 고쳐서 다시]\n{err}"
            try:
                raw = self._call(prompt, image)
                gen = _parse(raw)
                _check(gen, variables, None,
                       own_establish={p for p, _ in t["establish"]})
                n = sum(len(gen[k]) for k in ("pre", "establish", "destroy"))
                if n > self.max_conds:
                    raise VocabularyError(
                        f"조건이 {n}개다. 관련된 것만 남겨 {self.max_conds}개 이하로 줄여라.")
            except Exception as e:                       # noqa: BLE001
                err = str(e)
                if attempt < self.max_retry:
                    self.stats["retries"] += 1
                    continue
                if self.fallback:
                    self.stats["fallbacks"] += 1
                self._record(dsub, None, err)
                if self.fallback:
                    return (t_pre, t_est, t_des)
                raise
            pre, est, des = gen["pre"], gen["establish"], gen["destroy"]
            if self.union_with_template:
                pre = _union(t_pre, pre)
                est = _union(t_est, est)
                des = _union(t_des, des)
            self._record(dsub, gen, None)
            return (pre, est, des)

    # ------------------------------------------------------------------
    def _call(self, prompt: str, image: Optional[str]) -> str:
        content = [{"type": "text", "text": prompt}]
        if image:
            content.append({"type": "image_url",
                            "image_url": {"url": _as_data_url(image)}})
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": content}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        self.stats["calls"] += 1
        usage = getattr(resp, "usage", None)
        if usage:
            self.stats["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
            self.stats["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        return resp.choices[0].message.content

    def _record(self, dsub, gen, err):
        self.log.append({"action_type": dsub.action_type, "subgoal_id": dsub.sid,
                         "generated": gen, "error": err})
        self.flush()

    def flush(self):
        """현재까지의 통계·로그를 파일로 쓴다.

        호출 시점마다 덮어쓰므로, 마지막 호출 이후에 오르는 값
        (캐시 히트 등)을 반영하려면 실행이 끝난 뒤 한 번 더 불러야 한다.
        """
        if not self.log_path:
            return
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump({"stats": self.stats, "log": self.log},
                      f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------------ 유틸
def _default_client():
    """OpenAI 클라이언트. 키가 없으면 호출 시점에 에러가 난다."""
    try:
        from openai import OpenAI
    except ImportError as e:                              # noqa: BLE001
        raise ImportError("pip install openai 필요") from e
    return OpenAI()


def _as_data_url(path: str) -> str:
    import base64
    import mimetypes
    if path.startswith(("http://", "https://", "data:")):
        return path
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def _bind_spec(spec, binding):
    """리프티드 스펙의 변수를 실제 id 로 치환한다."""
    return [(p, [binding.get(a, a) for a in args]) for p, args in spec]


def _fmt(spec) -> str:
    return ", ".join(f"{p}({', '.join(a)})" if a else f"{p}()" for p, a in spec)


def _union(base, extra):
    out = list(base)
    have = {(p, tuple(a)) for p, a in base}
    for p, a in extra:
        if (p, tuple(a)) not in have:
            out.append((p, list(a)))
            have.add((p, tuple(a)))
    return out


def _parse(raw: str) -> dict:
    s = raw.strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.M).strip()
    try:
        d = json.loads(s)
    except json.JSONDecodeError as e:
        raise VocabularyError(f"JSON 파싱 실패: {e}") from e
    if not isinstance(d, dict):
        raise VocabularyError("최상위가 객체가 아니다")
    out = {}
    for k in ("pre", "establish", "destroy"):
        items = d.get(k) or []
        if not isinstance(items, list):
            raise VocabularyError(f"'{k}' 가 배열이 아니다")
        out[k] = [_one(x, k) for x in items]
    return out


def _one(x, field):
    """조건 하나를 (술어, 인자목록) 으로 정규화한다.

    모델이 내는 형태가 일정하지 않다. 특히 인자가 없는 술어를
    ["hand_empty"] 처럼 뒤의 빈 배열 없이 쓰는 경우가 잦다.

        ["hand_empty"]                   -> ("hand_empty", [])
        ["holding", "?o"]                -> ("holding", ["?o"])
        ["fits_inside", "?o", "?r"]      -> ("fits_inside", ["?o","?r"])
        "holding(?o)"                    -> ("holding", ["?o"])
        {"predicate":"holding","args":["?o"]}
                                         -> ("holding", ["?o"])
    """
    if isinstance(x, str):
        m = re.match(r"\s*([A-Za-z_]\w*)\s*\((.*)\)\s*$", x)
        if m:
            args = [a.strip().strip("'\"") for a in m.group(2).split(",") if a.strip()]
            return (m.group(1), args)
        return (x.strip(), [])

    if isinstance(x, dict):
        p = x.get("predicate") or x.get("type") or x.get("name")
        if not p:
            raise VocabularyError(f"'{field}' 항목에 술어 이름이 없다: {x}")
        a = x.get("args", [])
        return (p, list(a) if isinstance(a, (list, tuple)) else [a])

    if isinstance(x, (list, tuple)):
        if not x:
            raise VocabularyError(f"'{field}' 에 빈 항목이 있다")
        p = x[0]
        if not isinstance(p, str):
            raise VocabularyError(f"'{field}' 항목의 술어 이름이 문자열이 아니다: {x}")
        rest = list(x[1:])
        if len(rest) == 1 and isinstance(rest[0], (list, tuple)):
            return (p, list(rest[0]))
        return (p, [str(r) for r in rest])

    raise VocabularyError(f"'{field}' 항목 형식을 모르겠다: {x!r}")


# ------------------------------------------------------------------ 검증기
class VocabularyError(ValueError):
    pass


def entity_kinds(state) -> dict:
    """이름 -> 종류(object / region / ee) 사전. 그라운디드 검사용."""
    kinds = {}
    for o in state.objects:
        kinds[o] = "object"
    for r in state.regions:
        kinds[r] = "region"
    for o in state.objects.values():
        for e in o.get("feasible_ee") or []:
            kinds.setdefault(e, "ee")
    for ps in (state.sg.get("per_subgoal") or {}).values():
        for e in ps.get("feasible_ee") or []:
            kinds.setdefault(e, "ee")
        if ps.get("ee_candidate"):
            kinds.setdefault(ps["ee_candidate"], "ee")
    cur = state.robot.get("current_ee")
    if cur:
        kinds.setdefault(cur, "ee")
    return kinds


def _check(gen: dict, allowed, kinds=None, own_establish=None) -> None:
    _validate_groups((gen["pre"], gen["establish"], gen["destroy"]), allowed, kinds)
    # 이 액션이 만들어내는 결과를 전제로 요구할 수 없다.
    # 예: tool_act 는 닿지 않는 대상을 끌어오는 액션이므로
    #     reachable 은 결과이지 사전조건이 아니다.
    if own_establish:
        bad = {p for p, _ in gen["pre"]} & set(own_establish)
        if bad:
            b = sorted(bad)[0]
            raise VocabularyError(
                f"'{b}' 는 이 액션이 만들어내는 결과다. 사전조건으로 요구할 수 없다.")
    by_pred = {}
    for p, args in gen["pre"]:
        by_pred.setdefault(p, []).append(tuple(args))
    for a, b, same_args in CONTRADICTS:
        if a not in by_pred or b not in by_pred:
            continue
        if not same_args:
            raise VocabularyError(
                f"사전조건에 '{a}' 와 '{b}' 가 같이 있다. 동시에 성립할 수 없다.")
        shared = {x[0] for x in by_pred[a] if x} & {x[0] for x in by_pred[b] if x}
        if shared:
            o = sorted(shared)[0]
            raise VocabularyError(
                f"사전조건에 '{a}({o})' 와 '{b}({o})' 가 같이 있다. "
                f"손에 든 대상이 동시에 그 상태일 수 없다.")


def validate(generated, allowed=None, kinds=None) -> None:
    """생성된 조건 집합이 고정 어휘와 허용 인자를 벗어나지 않는지 검사."""
    _validate_groups(generated, allowed, kinds)


def _validate_groups(groups, allowed, kinds=None) -> None:
    for group in groups:
        for ptype, args in group:
            if ptype not in EVAL_BY:
                raise VocabularyError(
                    f"어휘표에 없는 predicate: '{ptype}'. "
                    f"predicates.EVAL_BY 를 먼저 고치거나 조합으로 표현할 것.")
            expected = len(ARG_TYPES.get(ptype, ()))
            if len(args) != expected:
                raise VocabularyError(
                    f"'{ptype}' 인자 개수가 {len(args)}개다. {expected}개여야 한다.")
            if allowed is not None:
                for a in args:
                    if a not in allowed:
                        raise VocabularyError(
                            f"쓸 수 없는 이름을 조건 인자로 씀: '{a}' ({ptype}). "
                            f"허용: {', '.join(sorted(allowed))}")
            if kinds:
                want = ARG_TYPES.get(ptype, ())
                for i, a in enumerate(args):
                    if i >= len(want) or want[i] == "any":
                        continue
                    got = kinds.get(a)
                    if got and got != want[i]:
                        raise VocabularyError(
                            f"'{ptype}' 의 {i+1}번째 인자는 {want[i]} 여야 하는데 "
                            f"'{a}' 는 {got} 다.")
