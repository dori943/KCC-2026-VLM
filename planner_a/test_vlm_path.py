# -*- coding: utf-8 -*-
"""VLM 조건 생성 경로 배선 검증.

실제 API 키 없이 가짜 클라이언트로 다음 네 가지를 확인한다.
  1. 정상 응답 → 조건이 생성되고 파이프라인이 끝까지 돈다
  2. 어휘표 밖 술어 → 거부되고 재시도가 걸린다
  3. 장면에 없는 object → 거부된다
  4. 계속 실패 → 템플릿으로 fallback 하고 결과가 나온다

  python3 test_vlm_path.py
"""
import json
import re
import sys
from types import SimpleNamespace

from planner_a.conditions import (VLMConditionGenerator, VLMDecidabilityJudge,
                                  VocabularyError, validate)
from planner_a.planner import run


class FakeClient:
    """OpenAI 클라이언트 흉내. responses 를 순서대로 뱉는다.

    responses 원소가 callable 이면 프롬프트를 넘겨 호출한다
    (서브골마다 다른 답을 내야 할 때).
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        body = self._responses.pop(0) if self._responses else self._responses_default()
        if callable(body):
            body = body(kw["messages"][1]["content"][0]["text"])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=body))],
            usage=SimpleNamespace(prompt_tokens=700, completion_tokens=90),
        )

    @staticmethod
    def _responses_default():
        return json.dumps({"pre": [], "establish": [], "destroy": []})


def load(name):
    with open(f"scenarios/{name}.json", encoding="utf-8") as f:
        return json.load(f)


def good(prompt):
    """action_type 을 읽어 리프티드 스키마를 낸다. 정상 응답 흉내."""
    # few-shot 예시에도 action_type 줄이 있으므로 실제 질의 구역부터 찾는다
    body = prompt[prompt.index("[이번 액션 스키마]"):]
    at = re.search(r"action_type\s*:\s*(\w+)", body).group(1)
    if at == "acquire":
        return json.dumps({
            "pre": [["reachable", ["?o"]], ["top_exposed", ["?o"]],
                    ["ee_feasible", ["?o"]], ["hand_empty"]],
            "establish": [["holding", ["?o"]]],
            "destroy": [["hand_empty", []]]})
    if at == "transport":
        return json.dumps({"pre": [["holding", ["?o"]]],
                           "establish": [["above", ["?o", "?r"]]], "destroy": []})
    return json.dumps({
        "pre": [["holding", ["?o"]], ["above", ["?o", "?r"]],
                ["fits_inside", ["?o", "?r"]]],
        "establish": [["in_region", ["?o", "?r"]], ["hand_empty", []]],
        "destroy": [["holding", ["?o"]]]})


# 들고 있는 대상에 top_exposed 를 붙이는 잘못된 응답 (모순 검사용)
CONTRADICT = json.dumps({
    "pre": [["holding", ["?o"]], ["top_exposed", ["?o"]]],
    "establish": [["above", ["?o", "?r"]]], "destroy": [],
})
BAD_VOCAB = json.dumps({
    "pre": [["gripper_is_ready", ["?o"]]],            # 어휘표에 없는 술어
    "establish": [], "destroy": [],
})
BAD_OBJ = json.dumps({
    "pre": [["reachable", ["BananaObject"]]],         # 스키마에 없는 이름
    "establish": [], "destroy": [],
})
BAD_ARITY = json.dumps({
    "pre": [["holding", ["?o", "?r"]]],               # 인자 개수 틀림
    "establish": [], "destroy": [],
})

FAIL = 0


def check(label, cond, detail=""):
    global FAIL
    mark = "OK  " if cond else "FAIL"
    if not cond:
        FAIL += 1
    print(f"  [{mark}] {label}" + (f"  — {detail}" if detail else ""))


print("1) 어휘 검증기 단위 테스트")
for label, payload in [("어휘표 밖 술어", BAD_VOCAB),
                       ("장면에 없는 object", BAD_OBJ),
                       ("인자 개수 불일치", BAD_ARITY)]:
    d = json.loads(payload)
    try:
        validate((d["pre"], d["establish"], d["destroy"]), {"?o", "?r", "?ee"})
        check(label + " 거부", False, "통과해버림")
    except VocabularyError as e:
        check(label + " 거부", True, str(e)[:52])

print("\n2) 정상 응답 — 파이프라인 끝까지")
case = load("t2_2_stack_tower")
gen = VLMConditionGenerator(client=FakeClient([good] * 9), model="fake")
res = run(case["kg"], case["sg"], case.get("robot_state"), generator=gen)
check("상세 서브골 9개", len(res["detailed_subgoals"]) == 9,
      f'{len(res["detailed_subgoals"])}개')
check("condition_source == vlm", res["condition_source"] == "vlm",
      res["condition_source"])
check("VLM 호출 3회 (action_type 당 1회)", gen.stats["calls"] == 3, f'{gen.stats["calls"]}회')
check("캐시 히트 6회", gen.stats["cache_hits"] == 6, f'{gen.stats["cache_hits"]}회')
check("사이클 없음", res["stats"]["n_cycles"] == 0)
check("순서 수 DP == 완전탐색", res["stats"]["orders_agree"])
check("토큰 집계됨", gen.stats["prompt_tokens"] > 0,
      f'{gen.stats["prompt_tokens"]}+{gen.stats["completion_tokens"]}')

print("\n3) 첫 응답이 어휘 위반 → 재시도 후 성공")
gen = VLMConditionGenerator(client=FakeClient([BAD_VOCAB] + [good] * 12), model="fake")
res = run(case["kg"], case["sg"], case.get("robot_state"), generator=gen)
check("재시도 1회 발생", gen.stats["retries"] == 1, f'{gen.stats["retries"]}회')
check("fallback 안 씀", gen.stats["fallbacks"] == 0)
check("결과 정상", res["stats"]["n_cycles"] == 0)

print("\n3-b) 들고 있는 대상에 top_exposed 를 붙이면 모순으로 거부된다")
gen = VLMConditionGenerator(client=FakeClient([CONTRADICT] * 20), model="fake")
res = run(case["kg"], case["sg"], case.get("robot_state"), generator=gen)
check("fallback 발생", gen.stats["fallbacks"] > 0, f'{gen.stats["fallbacks"]}회')
check("사이클 없음 (템플릿으로 방어)", res["stats"]["n_cycles"] == 0)

print("\n4) 계속 실패 → 템플릿 fallback")
gen = VLMConditionGenerator(client=FakeClient([BAD_VOCAB] * 40), model="fake")
res = run(case["kg"], case["sg"], case.get("robot_state"), generator=gen)
check("fallback 3회 (스키마 단위)", gen.stats["fallbacks"] == 3, f'{gen.stats["fallbacks"]}회')
check("결과가 템플릿과 동일", res["stats"]["n_orders_dp"] == 1,
      f'순서 {res["stats"]["n_orders_dp"]}개')

print("\n5) 템플릿 하한 보장 — VLM 이 조건을 빠뜨려도 템플릿 조건은 남는다")
EMPTY = json.dumps({"pre": [], "establish": [], "destroy": []})
gen = VLMConditionGenerator(client=FakeClient([EMPTY] * 9), model="fake")
res = run(case["kg"], case["sg"], case.get("robot_state"), generator=gen)
first = res["detailed_subgoals"][0]
check("빈 응답에도 pre 가 채워짐", len(first["pre"]) >= 4, f'{len(first["pre"])}개')
check("순서 정상 도출", res["stats"]["n_orders_dp"] == 1)

print("\n6) 프롬프트에 어휘표·템플릿·장면이 들어갔는지")
gen = VLMConditionGenerator(client=FakeClient([good] * 9), model="fake")
run(case["kg"], case["sg"], case.get("robot_state"), generator=gen)
sent = gen.client.calls[0]["messages"][1]["content"][0]["text"]
for token in ["top_exposed(object)", "acquire", "?o", "블록 쌓기"]:
    check(f"프롬프트에 '{token}' 포함", token in sent)
check("temperature=0", gen.client.calls[0]["temperature"] == 0)
check("JSON 강제", gen.client.calls[0]["response_format"]["type"] == "json_object")

print("\n7) 판정 가능 시점 판정 — 유보 조건에 depends_on 이 붙는지")


def judge_ok(prompt):
    """유보된 조건을 전부 '앞선 서브골 이후'로 판정하는 정상 응답."""
    cids = re.findall(r"(C_\w+)\s+(\w+)\(([^)]*)\)", prompt.split("[판정할 조건]")[1])
    dep = "SG1_d3" if "SG2_d3" in prompt.split("[지금 판정할 서브골]")[1][:40] else "SG2_d3"
    return json.dumps({"verdicts": [
        {"cond_id": c, "when": "after", "after": dep,
         "needs_observation": _t == "fits_inside", "why": "받침을 놓아야 생긴다"}
        for c, _t, _a in cids]})


j = VLMDecidabilityJudge(client=FakeClient([judge_ok] * 8), model="fake")
res = run(case["kg"], case["sg"], case.get("robot_state"), judge=j)
check("판정기 호출 2회 (유보가 있는 서브골만)", j.stats["calls"] == 2, f'{j.stats["calls"]}회')
check("유보 조건 4개", len(res["deferred_conditions"]) == 4,
      f'{len(res["deferred_conditions"])}개')
check("관측 순서 엣지 생성", res["stats"]["n_edges_observability"] > 0,
      f'{res["stats"]["n_edges_observability"]}개')
check("사이클 없음", res["stats"]["n_cycles"] == 0)
check("decidability_source == vlm", res["decidability_source"] == "vlm",
      res["decidability_source"])
check("SG 관측 요청 2건 (fits_inside 만)", res["stats"]["n_sg_requests"] == 2,
      f'{res["stats"]["n_sg_requests"]}건')
check("유보 + 관측요청이 겹칠 수 있음",
      any(x["depends_on"] and x["needs_observation"]
          for x in res["deferred_conditions"]))

print("\n7-b) 계획에 없는 서브골을 지목하면 거부 -> 규칙 판정으로 fallback")
BAD_AFTER = json.dumps({"verdicts": [{"cond_id": "C_SG2_d3_pre_2", "when": "after",
                                      "after": "SG9_d9", "why": "없는 서브골"}]})
j = VLMDecidabilityJudge(client=FakeClient([BAD_AFTER] * 20), model="fake")
res = run(case["kg"], case["sg"], case.get("robot_state"), judge=j)
check("fallback 발생", j.stats["fallbacks"] > 0, f'{j.stats["fallbacks"]}회')
check("규칙 판정으로 결과 유지", len(res["deferred_conditions"]) == 4,
      f'{len(res["deferred_conditions"])}개')
check("사이클 없음", res["stats"]["n_cycles"] == 0)

print("\n7-c) 유보 조건을 now 로 답하면 거부된다")
NOW = json.dumps({"verdicts": [{"cond_id": "C_SG2_d3_pre_2", "when": "now",
                                "after": None, "why": "지금 보인다"}]})
j = VLMDecidabilityJudge(client=FakeClient([NOW] * 20), model="fake")
run(case["kg"], case["sg"], case.get("robot_state"), judge=j)
check("now 응답 거부", j.stats["fallbacks"] > 0, f'fallback {j.stats["fallbacks"]}회')

print()
print("=" * 56)
print("전체 통과" if FAIL == 0 else f"실패 {FAIL}건")
sys.exit(0 if FAIL == 0 else 1)
