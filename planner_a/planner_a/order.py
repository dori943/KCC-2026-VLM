"""부분 선후관계 도출 + 사이클/상호배제 검사.

용어는 부분순서계획(POP)의 표준을 따른다.
  - causal link : A 가 B 의 사전조건을 참으로 만들어 A 가 B 보다 먼저여야 하는 관계
                  (McAllester & Rosenblitt 1991, Penberthy & Weld 1992, Weld 1994)
  - threat      : 초기에 참인 B 의 사전조건을 C 가 거짓으로 만드는 관계.
                  B 를 C 앞으로 보내 해소한다 (promotion).
  - observability : B 의 사전조건이 A 가 끝나야 관측 가능해지는 관계.
                  고전 deordering 은 모든 사전조건이 심볼 수준에서 판정된다고
                  가정하므로 이 관계가 없다. 관측에서 조건을 얻는 경우에만 생긴다.

규칙 1 = causal link 생성, 규칙 2 = threat 해소, 규칙 3/4 = 재분해 신호.

출력은 완전한 순서가 아니라 **부분 순서**다.
"D 는 B 보다 먼저" 수준만 내고 교환 가능한 것에는 순서를 두지 않는다.
상세 순서와 EE 배정은 Planner B 가 통합 상태공간에서 결정한다.
"""
from itertools import permutations

from .templates import consumed


def derive(dsubs: list, kg: dict) -> dict:
    idx = {d.sid: d for d in dsubs}
    est_by = {}      # 조건 key -> [sid]
    des_by = {}
    for d in dsubs:
        for c in d.establish:
            est_by.setdefault(c.key(), []).append(d.sid)
        for c in d.destroy:
            des_by.setdefault(c.key(), []).append(d.sid)

    edges = []          # (from, to, reason, cond_id, source)
    mutex = []          # (a, b, cond_id)
    open_conds = []     # 생산자가 여럿이라 Planner B 가 고를 것
    redecompose = []    # 규칙 3/4 신호
    links = []          # causal link (A, B, 조건key, cond_id) — 규칙 2b 에서 씀
    disjunctive = []    # 승격/강등 둘 다 가능한 threat. Planner B 가 선택.

    for b in dsubs:
        for p in b.pre:
            # ---- 판정 유보 조건
            if p.pass_ is None:
                # 심볼 수준 미결정(motion)은 순서를 만들지 않는다.
                # 반면 "앞선 서브골이 끝나야 관측 가능"한 조건은 실제 선후관계다.
                # 이 엣지가 없으면 관측할 수 없는 조건을 먼저 쓰는 순서가 허용된다.
                if p.depends_on and p.depends_on not in ("motion", b.sid):
                    edges.append((p.depends_on, b.sid, "observability",
                                  p.cond_id, "planner_a"))
                continue

            # ---- 규칙 1 : 초기에 거짓인 사전조건 -> causal link
            if p.pass_ is False:
                producers = [s for s in est_by.get(p.key(), []) if s != b.sid]
                if not producers:
                    redecompose.append({
                        "rule": 3,
                        "subgoal": b.sid,
                        "condition": p.cond_id,
                        "type": p.type,
                        "args": list(p.args),
                        "nl": p.nl,
                        "reason": "이 사전조건을 참으로 만드는 상세 서브골이 없음 -> 재분해 필요",
                    })
                elif len(producers) == 1:
                    edges.append((producers[0], b.sid, "causal_link",
                                  p.cond_id, "planner_a"))
                    links.append((producers[0], b.sid, p.key(), p.cond_id))
                else:
                    open_conds.append({
                        "subgoal": b.sid, "condition": p.cond_id,
                        "candidates": producers,
                        "note": "생산자가 여럿. 하드 제약을 걸지 않고 Planner B 가 선택",
                    })
                continue

            # ---- 규칙 2 : 초기에 참인 사전조건을 누가 깨뜨림 -> threat
            for c_sid in des_by.get(p.key(), []):
                if c_sid == b.sid:
                    continue
                if p.type in consumed(b.action_type):
                    # 예외: B 도 그 조건을 스스로 소비한다.
                    # 순서 제약이 아니라 상호배제다. 강제는 Planner B 가 한다.
                    pair = tuple(sorted((b.sid, c_sid)))
                    if not any(m["a"] == pair[0] and m["b"] == pair[1]
                               and m["condition"] == p.cond_id for m in mutex):
                        # 상호배제만 기록하면 Planner B 가 "둘 사이에 재확립이
                        # 필요하다"는 것을 알 수 없다. 해소 방법을 같이 넘긴다.
                        mutex.append({
                            "a": pair[0], "b": pair[1],
                            "condition": p.cond_id,
                            "predicate": p.type,
                            "args": list(p.args),
                            "reestablished_by": sorted(
                                s for s in est_by.get(p.key(), []) if s not in pair),
                            "resolutions": [
                                f"{pair[0]} -> {pair[1]} 사이에 {p.type} 재확립을 끼움",
                                f"{pair[1]} -> {pair[0]} 사이에 {p.type} 재확립을 끼움",
                            ],
                            "note": ("순서 제약이 아니라 자원 배타. 둘 다 이 조건을 "
                                     "소비하므로 사이에 재확립 서브골이 반드시 들어가야 "
                                     "한다. 강제는 Planner B 의 통합 상태공간이 담당."),
                        })
                else:
                    edges.append((b.sid, c_sid, "threat", p.cond_id, "planner_a"))

    # ---- 규칙 2b : causal link 에 대한 threat 해소 ------------------------
    # 규칙 2 는 "초기에 참인 사전조건"만 다루는데,
    # A --p--> B 라는 causal link 가 걸린 뒤에도 제3의 서브골 C 가 p 를 깨면
    # A 와 B 사이에 C 가 끼어들 수 있다. POP 의 표준 해소는 두 가지다.
    #   promotion : C 를 A 앞으로   (edge C -> A)
    #   demotion  : C 를 B 뒤로     (edge B -> C)
    # 한쪽만 사이클 없이 가능하면 그것을 하드 제약으로 걸고,
    # 둘 다 가능하면 순서를 정하지 않고 Planner B 에 선택으로 넘긴다.
    node_ids = [d.sid for d in dsubs]
    changed = True
    while changed:
        changed = False
        for (a_sid, b_sid, pkey, cond_id) in links:
            for c_sid in des_by.get(pkey, []):
                if c_sid in (a_sid, b_sid):
                    continue
                if _has_path(edges, node_ids, b_sid, c_sid) or \
                   _has_path(edges, node_ids, c_sid, a_sid):
                    continue                      # 이미 해소되어 있음
                promo_ok = not _would_cycle(edges, node_ids, c_sid, a_sid)
                demo_ok = not _would_cycle(edges, node_ids, b_sid, c_sid)
                if demo_ok and not promo_ok:
                    edges.append((b_sid, c_sid, "threat_demotion", cond_id, "planner_a"))
                    changed = True
                elif promo_ok and not demo_ok:
                    edges.append((c_sid, a_sid, "threat_promotion", cond_id, "planner_a"))
                    changed = True
                elif promo_ok and demo_ok:
                    key = (a_sid, b_sid, c_sid, cond_id)
                    if not any(x["key"] == list(key) for x in disjunctive):
                        disjunctive.append({
                            "key": list(key),
                            "link": [a_sid, b_sid], "threat": c_sid,
                            "condition": cond_id,
                            "options": [f"{c_sid} -> {a_sid} (promotion)",
                                        f"{b_sid} -> {c_sid} (demotion)"],
                            "note": "둘 다 가능. 하드 제약을 걸지 않고 Planner B 가 비용으로 선택",
                        })
                else:
                    redecompose.append({
                        "rule": "2b", "link": [a_sid, b_sid], "threat": c_sid,
                        "condition": cond_id,
                        "reason": "해소 불가능한 threat (승격·강등 모두 사이클) -> 재분해 필요",
                    })

    # ---- KG 가 준 태스크 논리 순서를 상세 서브골 수준으로 내린다
    order_in_group = {d.sid: i for i, d in enumerate(dsubs)}
    for a_kid, b_kid in kg.get("must_precede", []):
        a_last = max([d for d in dsubs if d.from_kg == a_kid],
                     key=lambda d: order_in_group[d.sid], default=None)
        b_first = min([d for d in dsubs if d.from_kg == b_kid],
                      key=lambda d: order_in_group[d.sid], default=None)
        if a_last and b_first:
            edges.append((a_last.sid, b_first.sid, "kg_must_precede", None, "kg"))

    edges = _dedup(edges)
    cycles = find_cycles(edges, [d.sid for d in dsubs])
    for cyc in cycles:
        redecompose.append({"rule": 4, "cycle": cyc,
                            "reason": "사이클 -> 재분해 필요"})

    return {"edges": edges, "mutex": mutex, "cycles": cycles,
            "open_conditions": open_conds, "redecompose": redecompose,
            "disjunctive_threats": disjunctive}


def _has_path(edges, nodes, src, dst) -> bool:
    adj = {n: [] for n in nodes}
    for f, t, *_ in edges:
        if f in adj and t in adj:
            adj[f].append(t)
    seen, stack = set(), [src]
    while stack:
        u = stack.pop()
        if u == dst:
            return True
        for v in adj.get(u, []):
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return False


def _would_cycle(edges, nodes, f, t) -> bool:
    """엣지 f->t 를 추가하면 사이클이 생기는가 (= 이미 t->f 경로가 있는가)."""
    return f == t or _has_path(edges, nodes, t, f)


def _dedup(edges):
    seen, out = set(), []
    for e in edges:
        k = (e[0], e[1], e[2], e[3])
        if k not in seen:
            seen.add(k)
            out.append(e)
    return out


def find_cycles(edges, nodes):
    adj = {n: [] for n in nodes}
    for f, t, *_ in edges:
        if f in adj and t in adj:
            adj[f].append(t)
    WHITE, GREY, BLACK = 0, 1, 2
    color = {n: WHITE for n in nodes}
    stack, cycles = [], []

    def dfs(u):
        color[u] = GREY
        stack.append(u)
        for v in adj[u]:
            if color[v] == GREY:
                cycles.append(stack[stack.index(v):] + [v])
            elif color[v] == WHITE:
                dfs(v)
        stack.pop()
        color[u] = BLACK

    for n in nodes:
        if color[n] == WHITE:
            dfs(n)
    return cycles


# --------------------------------------------------------------- 순서 수 계산
def count_linear_extensions_dp(nodes, edges) -> int:
    """부분집합 DP 로 실행 가능한 전체 순서의 수를 센다."""
    n = len(nodes)
    if n == 0:
        return 0
    if n > 20:
        return -1
    pos = {s: i for i, s in enumerate(nodes)}
    need = [0] * n                       # need[i] = i 보다 앞서야 하는 노드 비트마스크
    for f, t, *_ in edges:
        if f in pos and t in pos:
            need[pos[t]] |= (1 << pos[f])
    dp = [0] * (1 << n)
    dp[0] = 1
    for mask in range(1 << n):
        if not dp[mask]:
            continue
        for i in range(n):
            if mask & (1 << i):
                continue
            if need[i] & ~mask:
                continue
            dp[mask | (1 << i)] += dp[mask]
    return dp[(1 << n) - 1]


def count_linear_extensions_bruteforce(nodes, edges) -> int:
    """전체 순열 완전탐색. DP 결과와 대조하기 위한 것."""
    n = len(nodes)
    if n > 8:
        return -1
    pos = {s: i for i, s in enumerate(nodes)}
    cons = [(pos[f], pos[t]) for f, t, *_ in edges if f in pos and t in pos]
    cnt = 0
    for perm in permutations(range(n)):
        rank = {v: i for i, v in enumerate(perm)}
        if all(rank[a] < rank[b] for a, b in cons):
            cnt += 1
    return cnt
