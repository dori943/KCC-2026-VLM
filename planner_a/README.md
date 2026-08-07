# Planner A

KG가 낸 러프한 서브골과 Scene Graph 관측을 받아, 로봇이 실행할 수 있는 단위로 쪼개고
각 단위의 사전조건·사후조건을 만든 다음, 조건이 맞물리는 관계에서 **부분 순서**를 도출한다.

완전한 실행 순서와 EE 배정은 내지 않는다. "반드시 이것보다 먼저"인 것만 하드 제약으로
못 박고 나머지는 열어둔 채 Planner B로 넘긴다.

```
python3 make_scenarios.py         # scenarios/*.json 생성
python3 run.py                    # 규칙 판정 -> outputs/       (결정론적, 회귀 검증용)
python3 run.py t2_2               # 특정 시나리오만
python3 test_vlm_path.py          # VLM 경로 배선 검증 (API 키 불필요)

export OPENAI_API_KEY=sk-...
export PLANNER_A_VLM_MODEL=gpt-4o   # 선택, 기본값 gpt-4o
python3 run.py --real-vlm         # VLM 판정 -> outputs_vlm/
python3 run.py --real-vlm t2_2    # 둘 다
python3 run.py --vlm-conditions   # (대조군) 조건 집합 자체를 VLM이 생성
```

**출력 폴더가 모드별로 나뉜다.** 규칙 판정은 `outputs/`, VLM 판정은 `outputs_vlm/`.
한 폴더를 쓰면 회귀 검증 한 번에 VLM 실행 결과가 덮어써진다.
Planner B 로 넘기는 것은 `outputs_vlm/` 쪽이다.

템플릿 경로는 의존성 없음. VLM 경로만 `pip install openai` 필요.

---

## ⚠️ 입력 스키마는 가정값이다

`scenarios/*.json`의 KG · SG 입력은 **실제 모듈 출력이 아니라 직접 가정해서 만든 mock**이다.
실제 스키마가 확정되면 맞춰야 하고, 어긋나면 `decompose.py`와 `ground.py` 두 곳만 고치면 된다.
순서 도출 로직은 스키마와 무관하다.

**KG 쪽 가정**

```
  {
    "subgoals": [
      { "subgoal_id":            "SG1",
        "description":           "유리병을 상자로 옮긴다",
        "target_hints":          ["glass_bottle"],
        "goal_region_hint":      "box_0",
        "required_capabilities": ["stable_grasp", ...] }
    ],
    "must_precede": [["SG1", "SG2"]]
  }
```

* 서브골에 `action_type`이 없다고 가정한다. action_type은 Planner A가 붙인다
* `must_precede`를 **하드 제약으로 받는다.** KG가 순서를 힌트로만 준다면 이 처리를 바꿔야 한다
* KG가 완전순서를 준다면, 관측에서만 알 수 있는 순서를 Planner A가 뒤집을 수 없게 된다

**SG 쪽 가정**

```
  {
    "objects":  { "<id>": { "reachable": bool,
                            "feasible_ee": ["2f","3f","vac"],
                            "at_rest": bool,
                            "in_region": "<region id>" } },
    "regions":  { "<id>": { "clear": bool, "multi": bool } },
    "on_edges": [["위 object", "아래 object"]],
    "clearance":       { "<object>|<region>": mm },
    "tool_effective":  { "<tool>|<target>":   bool },
    "per_subgoal": {
      "SG1": { "target": "...", "goal_region": "...",
               "tool_required": bool, "selected_tool": "...",
               "tool_mode": "pull|push|sweep", "tool_rest_region": "...",
               "feasible_ee": [...], "ee_candidate": "2f" } },
    "image": "<RGB 크롭 경로>"          # 선택. VLM 조건 생성 시에만 사용
  }
```

| 항목 | 가정 | 확인 필요 |
|---|---|---|
| `on_edges` | SG가 제공 | `near`만 준다면 `top_exposed`를 bbox의 z 겹침으로 직접 계산해야 함 |
| `regions` | SG가 목표 영역을 제공 | 영역 개념이 SG에 있는지 |
| `regions.multi` | 트레이처럼 여러 개 들어가는 영역 구분용 | SG에 용량 개념이 있는지 |
| `clearance` | `object\|region` 문자열 키 | 실제 제공 형태 |
| `tool_effective` | SG가 도구–대상 유효성을 판정 | 이 판정 주체가 SG가 맞는지 |
| `per_subgoal` | 서브골별 `feasible_ee` / `tool_required` 제공 | 키 이름 대조 |
| `image` | RGB 크롭 경로 | SG가 크롭을 주는지, Planner A가 직접 잘라야 하는지 |

**내부 id 규칙도 임의로 정했다.**

```
  상세 서브골 id   SG1_d1, SG1_d2, SG1_d2b, SG1_d3
  group_id        G_<object 이름>
  조건 id          C_<서브골id>_pre_<번호> / _est_ / _des_
```

---

## 파이프라인

```
KG JSON + SG JSON (+ RGB 크롭)
  → decompose.py    러프 서브골 → 상세 서브골 + action_type
  → conditions.py   조건 집합 생성 (템플릿 또는 VLM)
    ground.py       조건 인스턴스화 + eval_by + 초기 진리값 + 파생 조건
  → order.py        causal link / threat → 부분 순서, 사이클, 상호배제
  → planner.py      DAG JSON 출력
```

| 파일 | 역할 |
|---|---|
| `predicates.py` | 조건 레코드 정의, eval_by 표, check식 |
| `templates.py` | action_type 템플릿 4종 |
| `conditions.py` | 조건 생성기 인터페이스, 어휘 검증 |
| `decompose.py` | 상세 서브골 재분해, action_type 부착 |
| `ground.py` | 조건 인스턴스화, 초기 상태 평가, 파생 조건 |
| `order.py` | 순서 도출 규칙, 사이클 검사, 선형확장 계산 |
| `planner.py` | 파이프라인 결합, JSON 직렬화 |

---

## action_type 템플릿

```
  acquire     pre:       reachable(?o), top_exposed(?o), ee_feasible(?o),
                         attached_ee(?ee), hand_empty()
              establish: holding(?o)
              destroy:   hand_empty(), at_rest(?o), top_exposed(?o)

  transport   pre:       holding(?o), path_clear(?o, ?r)
              establish: above(?o, ?r)

  place       pre:       holding(?o), above(?o, ?r), clear(?r), fits_inside(?o, ?r)
              establish: in_region(?o, ?r), at_rest(?o), hand_empty()
              destroy:   holding(?o), clear(?r)

  tool_act    pre:       holding(?t), tool_effective(?t, ?o), path_clear(?t, ?o)
              establish: in_region(?o, ?r), reachable(?o), at_rest(?o)
              mode:      pull / push / sweep
```

새 동작이 나와도 predicate 표를 늘리지 않고 조합으로 표현한다.
`tool_act`의 끌기·밀기·쓸기는 action_type을 늘리는 대신 `mode` 인자로 구분한다.

**`place`의 `above` 사전조건이 없으면** transport와 place 사이에 causal link가 생기지 않아
"운반하기 전에 배치"가 합법이 된다.

**"손에서 놓는다"를 `not_holding`으로 두지 않고** `destroy: holding` + `establish: hand_empty`로
표현한다. 부정 술어를 establish에 넣으면 threat 판정이 이중으로 걸린다.

---

## 조건 레코드와 `eval_by`

```json
{ "cond_id": "C_SG2_d1_pre_1",
  "type": "top_exposed",
  "args": ["heavy_box"],
  "check": "count(edges[type=on, to=heavy_box]) == 0",
  "pass": false,
  "eval_by": "sg",
  "nl": "heavy_box 위에 다른 object가 없다" }
```

같은 레코드를 **계획 시점 순서 도출과 실행 시점 검증에 모두 쓴다.** 조건을 두 벌 만들지 않는다.

`eval_by`가 판정 주체이자 실행 시점 실패의 복귀 지점이다.

| eval_by | 조건 | 계획 시점 | 실패 시 복귀 |
|---|---|---|---|
| `sg` | top_exposed, ee_feasible, fits_inside, tool_effective | SG 관측값으로 판정 | Scene Graph 재관측 후 재인스턴스화 |
| `planner_a` | reachable, clear, holding, hand_empty, above, in_region, at_rest | 기하 · 상태 계산 | 서브골 재분해 |
| `motion` | path_clear, attached_ee | **판정하지 않음 (`pass: null`)** | 내부 재시도 → 소진 시 Planner B 재계획 |

**`motion` 조건을 계획 시점에 거짓으로 박으면** 실제로는 통과 가능한 순서까지 선후관계로
묶여 과제약이 생긴다. 그래서 `null`로 유보한다.

**파생 조건** — object를 집어 들면 그 밑 object의 `top_exposed`가 참이 된다.
명시하지 않으면 "위에 있는 걸 먼저 치워라" 순서가 도출되지 않는다.
같은 지지물 위에 다른 object가 남아 있으면 추가하지 않는다.

---

## 순서 도출 규칙

용어는 부분순서계획(POP)의 표준을 따른다. `establish` = causal link, `destroy` = threat.

```
  규칙 1   B의 사전조건 p가 초기에 거짓, A가 p를 참으로 만듦
           → A → B                        (causal link)
           생산자가 여럿이면 하드 제약을 걸지 않고 open_conditions 로 넘김

  규칙 2   B의 사전조건 p가 초기에 참, C가 p를 깨뜨림
           → B → C                        (threat)
           예외: B도 p를 스스로 소비하면 순서가 아니라 상호배제(mutex)

  규칙 2b  A --p--> B causal link를 제3의 C가 위협
           → promotion (C → A) 또는 demotion (B → C)
           한쪽만 사이클 없이 가능하면 하드 제약
           둘 다 가능하면 disjunctive_threats 로 Planner B에 위임
           둘 다 불가능하면 재분해 신호

  규칙 3   아무도 establish 못 하는 사전조건 → 재분해 신호
  규칙 4   사이클                          → 재분해 신호
```

**규칙 2b가 없으면** 순서가 한 번 정해진 뒤 제3의 서브골이 사이에 끼어들 수 있다.
실제로 도구를 내려놓은 뒤에 그 도구로 미는 순서가 합법으로 나온다.

**mutex는 해소 방법을 함께 기록한다.** `(a, b, cond)` 세 값만 넘기면
"둘 사이에 재확립 서브골이 반드시 들어가야 한다"는 정보가 사라져, Planner B가
도구를 든 채로 다른 것을 집는 계획을 낼 수 있다.

```json
{ "a": "SG1_d1", "b": "SG2_d1",
  "predicate": "hand_empty",
  "reestablished_by": ["SG1_d3", "SG2_d3"],
  "resolutions": ["SG1_d1 -> SG2_d1 사이에 hand_empty 재확립을 끼움",
                  "SG2_d1 -> SG1_d1 사이에 hand_empty 재확립을 끼움"] }
```

---

## VLM 의 자리 — 판정 가능 시점

VLM 이 하는 일은 **조건을 만드는 것이 아니라, 각 조건이 언제 판정 가능한지를 정하는 것**이다.

두 가지를 나눠야 한다.

```
조건 집합 생성    "acquire 에 어떤 술어가 붙는가"
                 -> 액션 타입 수준. 장면과 거의 무관. 템플릿이 하한을 준다

판정 가능 시점    "fits_inside(MilkObject, on_cereal) 을 지금 판정할 수 있는가"
                 -> 조건 인스턴스 수준. 장면과 계획에 직접 달려 있다
```

`--vlm-conditions` 로 앞의 것을 실제로 돌려 봤을 때 VLM 은 **템플릿과 동일한 답**을 냈다.
장면이 달라도 리프티드 스키마는 달라지지 않으므로 당연한 결과다. 대조군으로만 남긴다.

### 세 갈래 판정

계획 시점에 참/거짓을 박을 수 없는 조건은 `pass: null` 로 두고, **왜** 유보인지를
`depends_on` 에 남긴다.

**서로 독립인 두 필드**로 기록한다. 하나로 합치면 순서나 관측 요청 중 하나를 잃는다.

| `depends_on` | 뜻 | 예 |
|---|---|---|
| `null` | 계획 상태만으로 판정됨 | `top_exposed(CerealObject)` |
| `"motion"` | 심볼 수준에서 원래 미결정 | `path_clear(...)` |
| `"<서브골 id>"` | 그 서브골이 끝나야 대상이 생긴다 → **순서 엣지** | `clear(on_cereal)` → `SG1_d3` 이후 |

| `needs_observation` | 뜻 |
|---|---|
| `false` | 계획 상태만으로 값을 안다 (무엇을 들고 있는지, 어디에 놓았는지) |
| `true` | 그 시점에 SG 가 새로 재야 안다 → **SG 관측 요청** |

둘 다 참인 조건이 흔하다.

```
fits_inside(MilkObject, on_cereal)
  depends_on        = SG1_d3   시리얼을 놓아야 그 윗면이 생긴다      -> 순서 엣지
  needs_observation = true     생긴 뒤에도 치수를 재야 들어가는지 안다 -> SG 요청
```

세 번째가 핵심이다. **액션이 세계를 바꾸면 아직 존재하지 않는 대상에 대한 조건이 생긴다.**
`on_cereal` 은 시리얼을 놓아야 생기는 영역이라, SG 가 지금 `clear` 를 보고할 수 없다.
그것을 기본값 `true` 로 채우면 계획이 통과했다고 말하고 실행에서 무너진다.

`InitialState` 는 관측에 없는 것에 대해 `true` 를 주지 않고 `None`(미관측)을 돌려준다.

### 관측 순서 (observability) 엣지

`depends_on` 이 서브골을 가리키면 그것은 실제 선후관계다. `A -> B (observability)` 엣지가
생긴다. 이 엣지가 없으면 관측할 수 없는 조건을 먼저 쓰는 순서가 허용된다.

고전 plan deordering (Kambhampati & Kedar 1994, Bäckström 1998) 에는 이 관계가 없다.
**모든 사전조건이 심볼 수준에서 판정 가능하다**고 가정하기 때문이다.
관측에서 조건을 얻는 경우에만 생기는 관계고, 이 파이프라인이 그 가정을 깨는 지점이다.

### 판정기

```
  DecidabilityJudge
   ├── TemplateDecidabilityJudge   규칙 기반. 결정론적. 회귀 검증 전용 (기본값)
   └── VLMDecidabilityJudge        실제 운용
```

* **캐시하지 않는다.** 같은 `place` 라도 인스턴스가 다르면 답이 달라야 한다
  (`place(A, 테이블)` 은 now, `place(B, A 의 윗면)` 은 after)
* **답의 공간이 `now` / `after:<id>` / `unknown` 셋뿐**이라 자유 생성보다 흔들릴 여지가 적다
* **참조 검증.** 계획에 없는 서브골이나 자기 자신을 지목하면 거부, 1회 재시도, 규칙 판정으로 fallback
* **유보된 조건을 `now` 로 답하면 거부.** 관측으로 판정 안 된 것을 판정됐다고 할 수 없다
* **`when` 과 `needs_observation` 을 따로 묻는다.** 값을 모른다는 이유로 `unknown` 으로
  내려버리면 순서를 통째로 잃는다. 대상을 만드는 서브골을 찾았으면 `after` 를 적고
  `needs_observation` 을 켠다
* **확신 없으면 `unknown`.** 지금 참으로 박는 것이 가장 나쁜 오답이다

### 각 모듈에 되돌아가는 것

Planner A 는 계획을 만드는 모듈이 아니라, 계획이 왜 그 순서여야 하는지를
각 모듈이 검증 가능한 형태로 되돌리는 모듈이다.

```
    KG   에게   네가 준 순서는 필요 / 중복 / 모순 중 뭐다     kg_order_audit
    SG   에게   무엇을 언제 재야 하는지                     sg_observation_requests
Planner B 에게   순서를 안 좁힌 채로 + 배타 조건 + 선택지     edges / mutex / disjunctive
   실행기 에게   실패하면 어디로 돌아가야 하는지              eval_by / depends_on
```

`test_vlm_path.py` 가 가짜 클라이언트로 배선을 검증한다 (API 키 불필요).

```
  1. 어휘 검증기          표 밖 술어 / 없는 object / 인자 개수 전부 거부
  2. 정상 응답            서브골 9개, action_type 당 1회 호출, 사이클 0
  3. 어휘 위반 후 재시도    재시도 1회 후 성공, fallback 미사용
  3-b. 모순 조건          들고 있는 대상에 top_exposed -> 거부
  4. 연속 실패            템플릿 fallback, 결과는 템플릿과 동일
  5. 빈 응답             템플릿 하한 덕에 사전조건 유지
  6. 프롬프트 내용         어휘표·action_type·object·region 포함, temperature 0
  7. 시점 판정            depends_on + needs_observation 분리, 관측 순서 엣지
  7-b. 없는 서브골 지목     거부 -> 규칙 판정 fallback, 결과 동일
  7-c. now 응답           거부
```

---

## 출력

`outputs/<scenario>.json` (규칙 판정) · `outputs_vlm/<scenario>.json` (VLM 판정)

| 필드 | 내용 |
|---|---|
| `detailed_subgoals[]` | `subgoal_id`, `action_type`, `mode`, `binding`, `group_id`, `from_kg`, `condition_source`, `pre` / `establish` / `destroy` |
| `edges[]` | `from`, `to`, `reason`, `via_condition`, `source` |
| `mutex[]` | 자원 배타 + 해소 정보 |
| `disjunctive_threats[]` | 승격 · 강등 둘 다 가능. Planner B가 비용으로 선택 |
| `open_conditions[]` | 생산자가 여럿인 사전조건. 하드 제약 미부여 |
| `redecompose_signals[]` | 규칙 3 / 2b / 4 발동 |
| `deferred_conditions[]` | 앞선 서브골이 끝나야 관측 가능한 조건 + 어느 서브골인지 |
| `sg_observation_requests[]` | SG 가 재야 알 수 있는 조건 + 언제 재야 하는지 → **SG 로 되돌릴 항목** |
| `kg_order_audit` | KG 가 준 순서의 필요 / 중복 / 모순 판정 → **KG 로 되돌릴 항목** |
| `stats` | 검증 수치 |

```
  reason    causal_link / threat / threat_promotion /
            threat_demotion / observability / kg_must_precede
  source    kg = 태스크 논리 순서 / planner_a = 관측 유래 순서
```

조건 레코드에는 `pass` 와 함께 `depends_on` 이 붙는다.
`null` = 지금 판정됨 / `"motion"` = 심볼 수준 미결정 / `"<서브골 id>"` = 그 뒤에 판정.
`needs_observation` 이 참이면 그 시점에 SG 가 새로 재야 한다.

`stats.n_orders_dp`는 **DAG의 선형확장 수이지 실행 가능한 계획 수가 아니다.**
mutex를 반영하지 않았으므로 실제 계획 수는 이보다 적다.

`group_id`는 같은 object의 확보 · 운반 · 배치를 하나로 묶는다.
EE를 서브골 단위가 아니라 그룹 단위로 바인딩해야 교체 횟수가 줄어든다.

---

## 검증

시나리오 9종. 벤치마크 태스크 5종 + 순서 규칙 회귀용 단위 시나리오 4종.

### 벤치마크 태스크

객체 규격·도구 후보 스펙은 벤치마크 명세표를 그대로 옮겼다.
SG 가 내야 할 `tool_effective` 판정은 손으로 박지 않고 리치·lip·접촉 폭·EE payload
수치에서 유도하며, 명세표의 의도 판정과 어긋나면 생성이 멈춘다.

| 시나리오 | 태스크 | 겨냥한 것 | KG | 상세 | 엣지 | mutex | 사이클 | 선형확장 |
|---|---|---|---|---|---|---|---|---|
| `t1_1_pull_milk` | T1-1 | 도구 후보 4개 중 1개만 유효 | 1 | 4 | 5 | 0 | 0 | 2 |
| `t1_1b_pull_cereal` | T1-1 v2 | 같은 태스크·도구 세트 교체 | 1 | 4 | 5 | 0 | 0 | 2 |
| `t1_2_lego_sweep` | T1-2 | 도구 2개 (pull + sweep) · 자원 배타 | 2 | 8 | 10 | 2 | 0 | 280 |
| `t2_1_sort_transport` | T2-1 | 순서를 과하게 좁히지 않는지 | 4 | 12 | 12 | 12 | 0 | 369,600 |
| `t2_2_stack_tower` | T2-2 | KG 논리 순서 보존 + 판정 유보 4 | 3 | 9 | 15 | 6 | 0 | 1 |
| `t2_2c_stack_tower_nohint` | 판정기 대조군 | SG 주석 없이 의존을 찾는가 | 3 | 9 | 9 | 6 | 0 | 1,680 |
| `t2_2d_stack_tower_opaque` | 판정기 대조군 | 이름 단서 없이 찾는가 | 3 | 9 | 9 | 6 | 0 | 1,680 |

T3(Long-horizon 통합)은 명세표에 장면 구성이 아직 비어 있어 만들지 않았다.

도구 후보의 탈락 사유가 후보마다 다르다는 것이 T1 계열의 설계 의도다.

```
  T1-1   Hammer         리치 275 >= 240, lip 40  -> 유효
         Cylinder       리치는 닿지만 lip 0      -> 당길 수 없다
         Box            넓은 판, lip 0           -> 밀 수만 있다
         ShortHook      hook 있지만 리치 180     -> 못 닿는다

  T1-2   LongHook       리치 420 >= 400, lip 35  -> SG1 유효
         Hammer         hook 있지만 리치 180     -> 못 닿는다
         Cylinder       리치 550 이지만 lip 0    -> 당길 수 없다
         LightWideBox   접촉 폭 200 >= 180, 0.20kg <= Vac 0.5kg -> SG2 유효
         HeavyWideBox   형상 동일, 0.80kg > Vac 0.5kg           -> 도구를 들 수 없다
         Bottle         리치 120                 -> 순수 오답
```

### 단위 시나리오 (회귀용)

| 시나리오 | 겨냥한 것 | KG | 상세 | 엣지 | mutex | 사이클 | 선형확장 | 재분해 |
|---|---|---|---|---|---|---|---|---|
| `u1_single_pick_place` | 최소 케이스 (규칙 1) | 1 | 3 | 3 | 0 | 0 | 1 | – |
| `u2_stacked_object` | 파생 조건 -> 선행 순서 | 2 | 6 | 7 | 2 | 0 | 10 | – |
| `u3_clearance_fail` | 규칙 3 재분해 신호 | 1 | 3 | 3 | 0 | 0 | 1 | O |
| `u4_tool_push` | 도구 경로 · 규칙 2b | 1 | 4 | 5 | 0 | 0 | 2 | – |

검증 방법
- 선형확장 수를 **부분집합 DP**와 **전체 순열 완전탐색** 두 방법으로 계산해 일치 확인
- KG `must_precede`가 결과 DAG에 보존되는지
- 역방향 경로가 생기지 않는지
- 관측 유래 엣지만으로 KG 순서를 다시 판정 (필요 / 중복 / 모순)
- `must_precede` 는 "A 의 모든 단위가 B 의 모든 단위보다 먼저"로 검사한다.
  일부만 앞서는 것으로는 KG 순서를 대체하지 못한다

세 시나리오가 각각 다른 것을 보인다.

```
  u2     SG1_d1 -> SG2_d1 엣지가 파생 조건에서 나온다
         너트를 집어야 상자 위가 노출된다는 사실은 관측에만 있다
         -> 관측이 없는 KG는 낼 수 없는 순서 (KG 감사에서 kg_missing 으로 잡힌다)

  t2_2   유일하게 source: kg 엣지를 탄다
         "큰 것이 아래로"는 명령문에서 나오는 논리 순서
         -> 실행 순서가 1개로 확정. KG 감사 판정도 필요 2 / 중복 0 / 모순 0
         조건 4개가 판정 유보된다. on_cereal / on_milk 는 받침을 놓아야 생기는
         영역이라 SG 가 지금 clear 도 clearance 도 보고할 수 없다
         -> 관측 순서 엣지 4개. 유보하지 않고 true 로 박으면 계획이 거짓말을 한다

  t2_1   반대 극단. 하드 제약 없이 mutex 12개로만 묶여 선형확장 369,600
         -> 순서를 좁히지 않고 EE 교체 최적화 여지를 Planner B에 남긴다
```

**이 검증이 증명하지 않는 것** — 조건이 실제 장면에서 제대로 생성되는지.
시나리오의 관측값은 손으로 작성한 mock이며 기본 경로에서 VLM 호출은 0회다.

---

## 알려진 제한

* KG · SG 실제 출력 스키마 미확정. `scenarios/*.json`은 mock
* KG `must_precede`를 하드 제약으로 처리 중. 힌트로 봐야 한다면 변경 필요
* `stats.n_orders_dp`는 mutex를 반영하지 않은 선형확장 수
* minimal 순서 제약이지 minimum(제약 개수 최소)이 아님
* 같은 거치 영역에 도구가 둘 이상 가면 `clear` mutex가 생긴다. 도구별 슬롯 분리 필요
* `VLMConditionGenerator`는 OpenAI Chat Completions 형식에 맞춰져 있다
