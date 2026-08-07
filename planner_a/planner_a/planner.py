"""Planner A 파이프라인.

입력  : KG 출력 JSON + SG 출력 JSON + robot_spec
출력  : 상세 서브골 + 조건 레코드 + 부분순서 DAG (Planner B 로 넘김)
"""
import json

from .decompose import decompose
from .ground import InitialState, ground
from .order import (derive, count_linear_extensions_dp,
                    count_linear_extensions_bruteforce)


def run(kg: dict, sg: dict, robot: dict = None,
        generator=None, judge=None) -> dict:
    robot = robot or sg.get("robot_state", {"in_hand": None})

    # S2 재분해 + action_type 부착
    dsubs = decompose(kg, sg)
    # S3 조건 인스턴스화 + 판정 가능 시점 판정
    state = InitialState(sg, robot, kg.get("task", ""))
    dec = ground(dsubs, state, generator, judge)
    # S4/S5 부분순서 + 사이클/상호배제
    rel = derive(dsubs, kg)

    nodes = [d.sid for d in dsubs]
    n_dp = count_linear_extensions_dp(nodes, rel["edges"]) if not rel["cycles"] else 0
    n_bf = count_linear_extensions_bruteforce(nodes, rel["edges"]) if not rel["cycles"] else 0

    # KG must_precede 가 결과 DAG 에 보존되는지, 역방향 모순이 없는지
    kg_preserved, kg_violation = _check_kg(dsubs, kg, rel["edges"])
    kg_audit = audit_kg_order(dsubs, kg, rel["edges"])

    return {
        "scenario": kg.get("scenario", ""),
        "task": kg.get("task", ""),
        "condition_source": getattr(dsubs[0], "condition_source", "template") if dsubs else "template",
        "decidability_source": dec["decidability_source"],
        "detailed_subgoals": [_dsub_json(d) for d in dsubs],
        "edges": [
            {"from": f, "to": t, "reason": r, "via_condition": c, "source": s}
            for f, t, r, c, s in rel["edges"]
        ],
        "mutex": rel["mutex"],
        "deferred_conditions": dec["deferred"],
        "sg_observation_requests": dec["unobserved"],
        "open_conditions": rel["open_conditions"],
        "disjunctive_threats": rel["disjunctive_threats"],
        "cycles": rel["cycles"],
        "redecompose_signals": rel["redecompose"],
        "kg_order_audit": kg_audit,
        "stats": {
            "n_kg_subgoals": len(kg["subgoals"]),
            "n_detailed_subgoals": len(dsubs),
            "n_edges": len(rel["edges"]),
            "n_edges_from_kg": sum(1 for e in rel["edges"] if e[4] == "kg"),
            "n_edges_from_planner_a": sum(1 for e in rel["edges"] if e[4] == "planner_a"),
            "n_edges_observability": sum(1 for e in rel["edges"] if e[2] == "observability"),
            "n_deferred": len(dec["deferred"]),
            "n_sg_requests": len(dec["unobserved"]),
            "n_mutex": len(rel["mutex"]),
            "n_cycles": len(rel["cycles"]),
            "n_orders_dp": n_dp,
            "n_orders_bruteforce": n_bf,
            "orders_agree": (n_bf == -1) or (n_dp == n_bf),
            "orders_note": "DAG 의 선형확장 수. mutex(자원 배타)는 반영하지 않았으므로 Planner B 가 실제로 고를 수 있는 계획 수는 이보다 적다.",
            "kg_must_precede_preserved": kg_preserved,
            "kg_violation": kg_violation,
        },
    }


def _dsub_json(d) -> dict:
    return {
        "subgoal_id": d.sid,
        "action_type": d.action_type,
        "mode": d.mode,
        "binding": d.binding,
        "group_id": d.group_id,
        "from_kg": d.from_kg,
        "condition_source": getattr(d, "condition_source", "template"),
        "note": d.note,
        "pre": [c.to_json() for c in d.pre],
        "establish": [c.to_json() for c in d.establish],
        "destroy": [c.to_json() for c in d.destroy],
    }


def _check_kg(dsubs, kg, edges):
    """KG 의 태스크 논리 순서가 결과 DAG 에서 살아 있는지, 역방향이 없는지."""
    by_kid = {}
    for d in dsubs:
        by_kid.setdefault(d.from_kg, []).append(d.sid)
    reach = _reachability([d.sid for d in dsubs], edges)
    ok, bad = True, []
    for a, b in kg.get("must_precede", []):
        A, B = by_kid.get(a, []), by_kid.get(b, [])
        if not A or not B:
            continue
        if not all(y in reach[x] for x in A for y in B):
            ok = False
            bad.append({"pair": [a, b], "issue": "DAG 에 보존되지 않음"})
        if any(x in reach[y] for x in A for y in B):
            ok = False
            bad.append({"pair": [a, b], "issue": "역방향 경로 존재 (모순)"})
    return ok, bad


def audit_kg_order(dsubs, kg, edges) -> dict:
    """KG 가 준 순서가 관측과 맞는지 판정한다.

    KG 는 문장만 보고 순서를 낸다. Planner A 는 관측에서 순서를 도출한다.
    KG 엣지를 빼고 관측 유래 엣지만으로 도달성을 계산하면, KG 의 각 순서가
    어느 쪽인지 알 수 있다.

      모순  관측은 반대 순서를 강제한다. KG 순서가 틀렸다
      중복  관측만으로도 같은 순서가 나온다. KG 가 없어도 됐다
      필요  관측만으로는 안 나온다. KG 만 낼 수 있는 태스크 논리 순서

    반대로 관측에서만 나오고 KG 가 주지 못한 순서는 `kg_missing` 에 담는다.
    """
    nodes = [d.sid for d in dsubs]
    pa_edges = [e for e in edges if e[4] != "kg"]
    reach = _reachability(nodes, pa_edges)
    by_kid = {}
    for d in dsubs:
        by_kid.setdefault(d.from_kg, []).append(d.sid)

    verdicts = []
    for a, b in kg.get("must_precede", []):
        A, B = by_kid.get(a, []), by_kid.get(b, [])
        if not A or not B:
            continue
        # must_precede 는 "A 의 모든 단위가 B 의 모든 단위보다 먼저"를 뜻한다.
        # 일부만 앞서는 것으로는 KG 순서를 대체하지 못한다.
        forward = all(y in reach[x] for x in A for y in B)
        backward = any(x in reach[y] for x in A for y in B)
        if backward and not forward:
            v = "모순"
        elif forward:
            v = "중복"
        else:
            v = "필요"
        verdicts.append({"pair": [a, b], "verdict": v})

    given = {(a, b) for a, b in kg.get("must_precede", [])}
    kid_of = {d.sid: d.from_kg for d in dsubs}
    missing = set()
    for f, t, *_rest in pa_edges:
        ka, kb = kid_of.get(f), kid_of.get(t)
        if ka and kb and ka != kb and (ka, kb) not in given:
            missing.add((ka, kb))

    return {"verdicts": verdicts,
            "kg_missing": [list(x) for x in sorted(missing)],
            "counts": {v: sum(1 for x in verdicts if x["verdict"] == v)
                       for v in ("필요", "중복", "모순")}}


def _reachability(nodes, edges):
    adj = {n: set() for n in nodes}
    for f, t, *_ in edges:
        if f in adj and t in adj:
            adj[f].add(t)
    reach = {n: set() for n in nodes}

    def dfs(u, seen):
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                dfs(v, seen)
    for n in nodes:
        s = set()
        dfs(n, s)
        reach[n] = s
    return reach


def load_and_run(path: str, generator=None, judge=None) -> dict:
    with open(path, encoding="utf-8") as f:
        case = json.load(f)
    return run(case["kg"], case["sg"], case.get("robot_state"), generator, judge)
