# 추론 파이프라인 — 모듈 입출력 + 피드백 연결 (KCC 2026)

대상 논문: 김도희, 정수빈, 임다인, 이민수, **"창발적 도구 조합을 위한 기능 중심
어포던스 추론 및 후보 생성 프레임워크"** (KCC 2026)

이 문서는 추론 모듈의 경계·입출력을 정리하고, 논문에 서술되어 있으나 코드로
연결되지 않았던 **피드백 경로 F1·F2를 논문 서술 그대로 구현·배선**한 결과를
기록한다. 시뮬레이션(모듈 5, `main_simulation_2.py` 등)은 범위 밖이며 읽기만 한다.

---

## 1. 논문 모듈 ↔ 레포 모듈 대응

| 논문 모듈 | 역할 | 레포 위치 | 피드백 |
|---|---|---|---|
| 1. Scene Resource Parser | 물체별 물리속성·부위 단위 어포던스 카드 | `app/runners/module1_runner.py`, `app/providers/vision_provider.py`, `app/models/module1_*` | — |
| 2. Task-to-Function Reasoner | 태스크→기능 세부목표 분해, 필요 기능/어포던스 추출, 수치 제약 구체화 | `app/module2a`(서브골 분해) + `app/module2b`(환경 제약·수치화) | **F1 완화 대상** |
| 3. Tool Composition Generator | 어포던스 카드+환경 제약→도구 조합 후보 생성·3단 필터링 | `app/module2c`(후보 생성) + `app/module2d`(3단 필터) | **★F1 위치** |
| 4. Tool Assembly Generator | 조립 자세 계산, 8항목 검증 | `app/module3`(pose 계산 + 검증) | **★F2 위치** |
| 5. Assembly Simulation | dual-arm PyBullet, 파지 예측 | `../main_simulation_2.py`, `../robot_controller_2.py`, `../VL-Grasp/` | ✕ (읽기만) |

> **주의 — 레포 모듈 번호 ≠ 논문 모듈 번호.** 레포는 추론 후반부를 2A/2B/2C/2D/3으로
> 잘게 나눴다. 논문의 "모듈 3(Tool Composition Generator)"은 레포의 **2C+2D**,
> 논문의 "모듈 4(Tool Assembly Generator)"는 레포의 **3**에 대응한다.

---

## 2. 모듈별 입출력 표 (경계 인터페이스)

| 모듈 | 입력 | 출력(핵심 파일) | 핵심 필드 |
|---|---|---|---|
| **M1** Scene Resource Parser | RGB 이미지 (+선택 `scene_info_case*.json`의 AABB) | `module2_common_input_template.json` | 부위 단위 어포던스 카드: `resource_inventory[]`, 부위별 물리속성/접촉단면/삽입두께/형상 + **3 인식수준(직접관측/추론/가정)·불확실성** |
| **M2A** Task-to-Function Reasoner (분해) | `module2_common_input_template.json` + `--user-goal`(task) | `module2a_output.json`, `module2_common_input.json` | 기능적 세부목표(subgoal), 필요 기능(파지부/작용부/연결부), `required_atoms`, `required_interaction_primitives` |
| **M2B** Task-to-Function Reasoner (환경·수치화) | `module2_common_input.json` + `module2a_output.json` | `module2b_output.json`, `module3_handoff_preview.json` | 환경 구조(`access_path_profile`), `numeric_estimates`, `derived_constraints`, `subgoal_constraints` |
| **M2C** Tool Composition Generator (후보) | M2B output + Material Reasoner(이미지+타겟) | `module2c_output.json` | `candidate_tools[]`: `used_objects`, `structure_description`, `function_mapping[]{object,function}` |
| **M2D** Tool Composition Generator (3단 필터) ★F1 | `module2c_output.json` | `module2d_output.json` | `evaluated_candidates[]{pass,failed_stage,stage_scores,weak_points}`, `selected_candidate_id`, `feedback_decision{branch,feedback_target}`, **`filter_counts`** |
| **M3** Tool Assembly Generator ★F2 | `module2d_output.json` | `module3_output.json` | `assembly_strategy`, `assembly_steps[]{pose,joint_position_world}`, `final_structure{functional_end,handle_end}`, `verification{is_valid,checks[8]}`, `feedback{feedback_iteration,task_abandoned}` |
| **M5** Assembly Simulation (범위 밖) | `module3_output.json` | 시뮬 로그/파지 결과 | — |

**출력 스키마 통일 원칙:** 시나리오별로 출력 파일을 가르지 않는다. 동일 스키마를
쓰고 `scenario`(=task 이름)를 필드로 구분한다. 실패 로그(§5)도 한 경로로 통합한다.
(`main_simulation_*`가 시나리오별로 나뉜 것은 실험 구조상 정당하므로 건드리지 않음.)

---

## 3. 시나리오 대응 (논문 ↔ 레포 task)

| 논문 시나리오 | 로그 `scenario` | 레포 preset | 설명 |
|---|---|---|---|
| S1 풍선 꺼내기 | `balloon` | `task3` (`suspended_target`) | 나뭇가지에 걸린 풍선 |
| S2 하수구 차키 회수 | `chain` | `task2` (`deep_hole_reach`) | 하수구 격자 속 자동차 키 |
| S3 화재 반려견 케이지 | `pet` | `task6` (`pet_cage_fire_rescue`) | 케이지 문 고리 걸어 당기기 |

> 현재 오케스트레이터는 로그 `scenario` 필드에 **task 이름**을 넣는다. 논문 표기
> (balloon/chain/pet)로 집계할 때는 위 대응표로 매핑한다.

---

## 4. F1·F2 피드백 연결 (구현)

### 4.1 왜 "연결"이 필요했나

논문은 피드백 구조를 서술했지만, 기존 오케스트레이터(`scripts/run_pipeline.py`)는
`1→2a→2b→2c→2d→3` **완전 선형**이었다. 각 모듈이 피드백 '결정'을 JSON에 **기록만**
하고 아무도 그것을 소비해 상위 모듈을 재호출하지 않았다. 그래서 실험에서 100%
통과가 나오면 §3.3의 "통과 후보 없음" 분기와 §3.4의 "검증 실패" 분기가 한 번도
발동하지 않았다. 이번 작업은 그 결정을 **실제로 소비·재호출하는 연결 계층**을 만든다.

- 연결 계층: `app/feedback/`
  - `controller.py` — `FeedbackController`: 분기 판정 + 재시도 상한 강제
  - `loop.py` — `FeedbackRunner`: 판정→재호출→로그 루프(엔진 독립, mock 테스트 가능)
  - `log_record.py` — 공통 실패 로그(§5)
  - `verification_items.py` — F2 8항목 **단일 정의 소스**
- 오케스트레이터 배선: `scripts/run_pipeline.py` (`--feedback` 기본 on, `--no-feedback`로 기존 선형)

### 4.2 F1 — Tool Composition Generator (논문 §3.3)

> "통과 후보가 없을 경우, 환경 제약 일괄 탈락 시 상위 제약 생성 단계에 완화를
> 요청하고, 그 외 실패는 후보 재생성 및 직전 추론 단계로 피드백되어 재시도된다."

**두 갈래로 분기하며, 판정은 `module2d`의 `filter_counts`(3단 필터 순차 in/out)에 의존한다.**

| 조건 | branch | action | 재호출 대상 |
|---|---|---|---|
| 환경 제약 필터(1단)에서 후보 **일괄 탈락** (`env_constraint.in == out`, in>0) | `env_constraint_wipeout` | `relax_request_to_module2` | **module2b** (상위 수치 제약 완화 재생성) |
| 그 외 이유로 통과 후보 0 | `other` | `regenerate_candidates` | **module2c** (후보 재생성) + 직전 단계 |

3단 필터 구조와 `filter_counts` 매핑(어느 필터에서 몇 개가 떨어졌는지 로그로 남긴다):

| 논문 필터 | filter_counts 키 | 코드 판정 근거 (`failed_stage`) |
|---|---|---|
| (1) 환경 제약 필터 (진입 공간·허용 하중) | `env_constraint` | `environment` |
| (2) 접합 구조 필터 (비호환 결합·극단 크기차) | `joint_structure` | `assembly`, `handle_feasibility`, `hook_feasibility` |
| (3) 기하·물리·상식 다차원 점수 | `score_eval` | `geometry/physics/commonsense/task_fit/emergence/necessity/role_contribution/total/subgoal_coverage_critical` |

### 4.3 F2 — Tool Assembly Generator (논문 §3.4)

> "조립 결과는 충돌, 기능부 노출, 힘 전달 등 8개 항목으로 검증되며, 실패 시 직전
> 단계로 피드백이 전달되어 재수행한다."

- 8항목 검증 → `verification.checks[]`, 하나라도 `fail`이면 `is_valid=false`.
- 실패 시 **직전 단계 = Tool Composition Generator(레포 `module2c`)로 피드백** 후
  2c→2d→3 재수행.
- **8항목 검증 목록** (단일 소스: `app/feedback/verification_items.py`):

  | # | item | 논문 명시 | 의미 |
  |---|---|---|---|
  | 1 | `alignment` | | 물체가 task 방향으로 정렬됐는가 |
  | 2 | `collision` | ★ | 비의도적 충돌이 없는가 |
  | 3 | `functional_end_exposed` | ★ | 기능단(tip/edge)이 노출됐는가 |
  | 4 | `handle_region_free` | | 파지 영역이 확보됐는가 |
  | 5 | `force_transfer` | ★ | 힘이 도구 끝까지 전달되는가 |
  | 6 | `weak_point_mitigation` | | 이전 단계 취약 항목이 조립에 반영됐는가 |
  | 7 | `subgoal_support` | | subgoal의 `required_atoms`가 지원되는가 |
  | 8 | `contact_feasibility` | | 도구가 타겟과 실제 접촉 가능한가 |

  논문 본문이 이름을 밝힌 항목은 2·3·5(충돌·기능부 노출·힘 전달) 3개이고, 나머지
  5개는 코드(`FINAL_SYSTEM_PROMPT`, `Module3OutputValidator`)에서 확인해 문서화했다.

> **라우팅 표기 불일치 메모(팀 전달).** `module3` 코드 내부는 역사적으로
> `feedback_target="module2a"`로 적어 왔다(코드 주석의 '논문 3.1.4' 번호 기준).
> 본 정리의 전달 문서(§3.4)는 F2의 직전 단계를 **Tool Composition Generator**로
> 규정하므로, 연결 계층(`FeedbackController`)은 **module2c**로 라우팅하고 그 사실을
> 로그·본 문서에 남긴다. `module3` 출력 스키마(need_feedback_to_module2a 등)는 하위
> 호환을 위해 그대로 두었다. 최종 표기는 팀에서 논문과 맞춰 확정 필요.

### 4.4 재시도 상한 (논문 미명시 — 구현 기본값)

논문은 F1·F2 재계획/재수행의 재시도 상한을 명시하지 않는다. 임의 하드코딩을
피하기 위해 설정값으로 분리했다.

- 파일: `configs/feedback_policy.yaml` → `retry_caps: {F1: 2, F2: 2}`
- 기본값 **2**는 `module3`의 기존 하드코딩(`feedback_iteration >= 2 → task_abandoned`)과
  일치시킨 것이며, 이제 그 값도 이 config에서 읽는다(`pose_calculator._f2_retry_cap`).
- 의미: `attempt`는 0부터. cap회까지 재시도하고 초과 시 `task_abandoned`.

---

## 5. 공통 실패 로그 형식 (논문 §4.5)

F1·F2 실패 1건당 한 레코드. `outputs/<task>/feedback_failures.jsonl`에 append.

```json
{"scenario": "chain", "failed_at": "F1", "module": "ToolCompositionGenerator",
 "action": "relax_request_to_module2", "attempt": 1, "result": "success",
 "target_module": "module2b",
 "filter_counts": {"env_constraint": {"in": 12, "out": 12}, "joint_structure": {"in": 0, "out": 0}, "score_eval": {"in": 0, "out": 0}},
 "branch": "env_constraint_wipeout"}
```

- `failed_at`: `F1` | `F2`
- F1은 `filter_counts` + `branch`, F2는 `violated_checks`(실패한 8항목 이름) 필드를 쓴다.
- `result`: `pending`→ 재시도로 해소되면 `success`, 상한 초과면 `abandoned`, 그 외 `failed`.

---

## 6. F3·F4 (시뮬 쪽 — 수정하지 않음, 위치만 기록)

| 경로 | 논문 | 위치 | 상태 |
|---|---|---|---|
| F3 grip force 루프 (contact/slip→grip force 증가) | §3.5 | 시뮬 코드(`../robot_controller_2.py` 계열) | 이미 구현, 수정 안 함 |
| F4 파지 시도 재시도 (모델 추론+AABB 폴백 2단계×3회=최대 6회) | §4.3 | 시뮬 코드 | 상한값 논문 일치 여부는 담당자 확인 필요 |

---

## 7. 실행 / 검증

### 전체 파이프라인 (피드백 on)
```bash
cd module1-2B
export OPENAI_API_KEY="sk-..."
python scripts/run_pipeline.py --preset task2            # 피드백 기본 on
python scripts/run_pipeline.py --preset task2 --no-feedback   # 기존 선형
```
실패 로그: `outputs/<task>/feedback_failures.jsonl`

### 피드백 연결 검증 (GPT 불필요 — mock 실패 주입)
종료 조건 ①(mock 실패 → 논문대로 분기 + 로그)을 GPT 호출 없이 검증한다.
```bash
python -m pytest tests/test_feedback_controller.py tests/test_feedback_loop.py -q
```
- mock 실패 입력: `fixtures/feedback_cases/{f1_env_wipeout,f1_other,f2_check_fail}/`
- F1은 환경 제약 일괄 탈락 케이스와 그 외 케이스 두 가지를 모두 포함.

---

## 8. 엔진 독립성 메모

추론 모듈(`app/module*`, `app/feedback`)은 PyBullet에 의존하지 않는다. `import
pybullet`은 `app/pybullet/runner.py` 한 곳에만 격리돼 있어 어댑터 경계가 유지된다.
좌표 변환·물리 조회를 추론 모듈로 끌어들이지 말 것.
