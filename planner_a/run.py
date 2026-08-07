# -*- coding: utf-8 -*-
"""전체 시나리오 실행 + 회귀 검증 요약.

사용법:
    python run.py                     # 규칙 판정 (결정론적, 회귀 검증용)
    python run.py t2_2                # 특정 시나리오만
    python run.py --real-vlm          # 판정 가능 시점을 VLM 이 정한다
    python run.py --real-vlm t2_2     # 둘 다
    python run.py --vlm-conditions    # (ablation) 조건 집합 자체를 VLM 이 생성

VLM 의 자리는 조건 집합 생성이 아니라 **판정 가능 시점 판정**이다.
조건 집합은 액션 타입 수준이라 장면과 거의 무관해서 템플릿이 하한을 주면 되고,
판정 시점은 조건 인스턴스 수준이라 장면과 계획에 직접 달려 있다.
--vlm-conditions 는 그 대조군으로만 남겨 둔다.

회귀 검증은 항상 규칙 판정으로 돌린다. VLM 은 출력이 비결정적이라
순서 규칙 버그와 VLM 오류를 구분할 수 없기 때문이다.
"""
import glob
import json
import os
import sys

from planner_a.planner import load_and_run

# 규칙 판정과 VLM 판정은 출력 폴더를 나눈다.
# 한 폴더를 쓰면 회귀 검증 한 번에 VLM 실행 결과가 덮어써진다.
OUT_RULE = "outputs"        # 규칙 판정 (결정론적, 회귀 검증용)
OUT_VLM = "outputs_vlm"     # VLM 판정 (실험 결과. 재실행 비용이 든다)
OUT = OUT_RULE


def main(argv):
    global OUT
    real_vlm = "--real-vlm" in argv
    vlm_conds = "--vlm-conditions" in argv
    argv = [a for a in argv if not a.startswith("--")]
    OUT = OUT_VLM if (real_vlm or vlm_conds) else OUT_RULE
    os.makedirs(OUT, exist_ok=True)
    paths = sorted(glob.glob("scenarios/*.json"))
    if argv:
        paths = [p for p in paths if any(a in p for a in argv)]

    if real_vlm or vlm_conds:
        os.makedirs("vlm_logs", exist_ok=True)
    if real_vlm:
        print("[VLM 모드] 조건의 판정 가능 시점을 VLM 이 정합니다. "
              "출력은 비결정적입니다.\n")
    print(f"[출력] {OUT}/\n")
    if vlm_conds:
        print("[대조군] 조건 집합 자체를 VLM 이 생성합니다.\n")

    rows, all_ok = [], True
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        gen = judge = None
        if vlm_conds:
            from planner_a.conditions import VLMConditionGenerator
            gen = VLMConditionGenerator(log_path=f"vlm_logs/cond_{name}.json")
        if real_vlm:
            from planner_a.conditions import VLMDecidabilityJudge
            judge = VLMDecidabilityJudge(log_path=f"vlm_logs/{name}.json")
        res = load_and_run(p, generator=gen, judge=judge)
        for tag, m in (("조건생성", gen), ("시점판정", judge)):
            if m is None:
                continue
            m.flush()
            res.setdefault("vlm_stats", {})[tag] = m.stats
            extra = (f", 캐시 히트 {m.stats['cache_hits']}"
                     if "cache_hits" in m.stats else
                     f", 유보 {m.stats['after']} / 미해결 {m.stats['unknown']}"
                     f" / 관측요청 {m.stats['needs_obs']}")
            print(f"  {name} [{tag}]: 호출 {m.stats['calls']}회{extra}, "
                  f"재시도 {m.stats['retries']}, fallback {m.stats['fallbacks']}, "
                  f"토큰 {m.stats['prompt_tokens']}+{m.stats['completion_tokens']}")
            errs = [e for e in m.log if e.get("error")]
            if errs:
                by_msg = {}
                for e in errs:
                    by_msg.setdefault(e["error"][:110], []).append(e["subgoal_id"])
                print(f"      fallback {len(errs)}건, 사유 {len(by_msg)}종:")
                for msg, sids in by_msg.items():
                    print(f"        {len(sids)}건  {msg}")
                    print(f"              -> {', '.join(sids)}")
        with open(f"{OUT}/{name}.json", "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

        s = res["stats"]
        ok = (s["orders_agree"] and s["kg_must_precede_preserved"]
              and s["n_cycles"] == 0 and s["n_orders_dp"] > 0)
        all_ok &= ok
        rows.append((name, res["scenario"], s, ok))

    # ---------------------------------------------------------- 요약 표
    hdr = ("시나리오", "KG", "상세", "엣지", "(kg/pa)", "관측순서", "mutex",
           "사이클", "순서수", "유보/SG요청", "재분해", "검증")
    print("\n" + " | ".join(hdr))
    print("-" * 108)
    for name, scen, s, ok in rows:
        print(" | ".join([
            name.ljust(20),
            str(s["n_kg_subgoals"]),
            str(s["n_detailed_subgoals"]),
            str(s["n_edges"]),
            f'{s["n_edges_from_kg"]}/{s["n_edges_from_planner_a"]}',
            str(s["n_edges_observability"]),
            str(s["n_mutex"]),
            str(s["n_cycles"]),
            str(s["n_orders_dp"]),
            f'{s["n_deferred"]}/{s["n_sg_requests"]}',
            str(len([r for r in _sig(name)])),
            "OK" if ok else "FAIL",
        ]))

    print("\n[검증] 실행 순서 수 = 부분집합 DP vs 전체 순열 완전탐색 대조")
    for name, scen, s, ok in rows:
        bf = s["n_orders_bruteforce"]
        bf_s = "n>8 생략" if bf == -1 else str(bf)
        print(f"  {name.ljust(20)} DP={s['n_orders_dp']:<6} BF={bf_s:<10} "
              f"일치={'O' if s['orders_agree'] else 'X'}  "
              f"KG순서보존={'O' if s['kg_must_precede_preserved'] else 'X'}")

    print("\n[재분해 신호]")
    any_sig = False
    for name, *_ in rows:
        sigs = _sig(name)
        for g in sigs:
            any_sig = True
            r = g.get("rule")
            if r == 3:
                print(f"  {name}: 규칙3 {g['subgoal']} <- {g['nl']} ({g['condition']})")
            elif r == 4:
                print(f"  {name}: 규칙4 사이클 {' -> '.join(g['cycle'])}")
            else:
                print(f"  {name}: 규칙{r} 해소 불가 threat "
                      f"link{g['link']} <- {g['threat']} ({g['condition']})")
    if not any_sig:
        print("  없음")

    print("\n[판정 유보] 앞선 서브골이 끝나야 대상이 생기는 조건 -> 순서 엣지")
    any_def = False
    for name, *_ in rows:
        with open(f"{OUT}/{name}.json", encoding="utf-8") as f:
            d = json.load(f)
        for x in d["deferred_conditions"]:
            any_def = True
            obs = " + SG 재측정 필요" if x["needs_observation"] else ""
            print(f"  {name}: {x['subgoal']} <- {x['type']}({', '.join(x['args'])})"
                  f"  {x['depends_on']} 이후{obs}")
    if not any_def:
        print("  없음")

    print("\n[SG 관측 요청] Planner A 가 SG 로 되돌리는 항목")
    any_req = False
    for name, *_ in rows:
        with open(f"{OUT}/{name}.json", encoding="utf-8") as f:
            d = json.load(f)
        for x in d["sg_observation_requests"]:
            any_req = True
            print(f"  {name}: {x['type']}({', '.join(x['args'])})  {x['request']}"
                  f"   <- {x['subgoal']} 이 씀")
    if not any_req:
        print("  없음")

    print("\n[KG 순서 감사] 관측 유래 엣지만으로 KG 순서를 다시 판정")
    any_audit = False
    for name, *_ in rows:
        with open(f"{OUT}/{name}.json", encoding="utf-8") as f:
            a = json.load(f)["kg_order_audit"]
        if not a["verdicts"] and not a["kg_missing"]:
            continue
        any_audit = True
        c = a["counts"]
        print(f"  {name.ljust(22)} 필요 {c['필요']} / 중복 {c['중복']} / 모순 {c['모순']}")
        for v in a["verdicts"]:
            print(f"        {v['verdict']}  {v['pair'][0]} -> {v['pair'][1]}")
        for m in a["kg_missing"]:
            print(f"        KG누락  {m[0]} -> {m[1]}  (관측에서만 도출됨)")
    if not any_audit:
        print("  KG 가 준 순서 없음")

    print(f"\n총 {len(rows)}개 시나리오, 전체 검증 {'통과' if all_ok else '실패'}"
          f"  ->  {OUT}/")
    return 0 if all_ok else 1


def _sig(name):
    with open(f"{OUT}/{name}.json", encoding="utf-8") as f:
        return json.load(f)["redecompose_signals"]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
