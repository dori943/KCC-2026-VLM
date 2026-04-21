# module1-2B (Module 1 -> 2-A -> 2-B -> 2-C -> 2-D -> 3)

이 디렉토리는 KCC 2026 논문 "기능 추론 및 물체 조합을 통한 창발적 도구 생성"의
전체 도구 생성 모듈(3.1.1 ~ 3.1.4)을 단일 파이프라인으로 실행/검증하기 위한
실험 하네스입니다.

## 1. 현재 구현 범위

### Module 1 (Scene Resource Parser — 논문 3.1.1)
- 입력 이미지 → OpenAI Vision(gpt-4.1-mini) 호출로 `module1_raw_output` 생성
- 검증 및 정규화 → `scene_resources_from_module1`
- `module2_common_input_template` 브리지 생성

### Module 2-A (Task-to-Function Reasoner — 논문 3.1.2)
- `module2_common_input` 기반 subgoal 분해
- subgoal별 요구사항(required/preferred/risk atoms) 생성
- `module2a_output` 산출
- **현 구현은 rule-based keyword matching** (LLM 전환은 향후 과제)

### Module 2-B (환경 제약 생성 — 논문 3.1.2 일부)
- env-only target/environment binding
- numeric estimates → derived constraints 생성
- `module2b_output` 및 `module3_handoff_preview` 산출
- **현 구현은 rule-based YAML rule engines**

### Module 2-C (Tool Candidate Generator — 논문 3.1.3.1)
- 물체 어포던스 + 환경 제약 + 서브골 입력
- GPT-4o로 도구 조합 후보 생성 (Material Reasoner 포함)
- `module2c_output` (candidate_tools[] 구조)

### Module 2-D (Candidate Filter — 논문 3.1.3.2)
- 생성된 후보를 3단계 필터링
  - 환경 제약 필터
  - 접합 구조 필터
  - 기하/물리/상식 체크리스트 (18개 항목)
- `module2d_output` (selected_candidate_id + evaluated_candidates[])

### Module 3 (Tool Assembly Generator — 논문 3.1.4)
- 선택된 후보 + scene + 물리 속성 + 제약 입력
- GPT-4o로 조립 순서/접합 위치/상대 위치/회전/월드 좌표 단계적 계산
- 8개 검증 항목 체크 (alignment/collision/functional_end_exposed/
  handle_region_free/force_transfer/weak_point_mitigation/subgoal_support/
  contact_feasibility)
- 코드 기반 기하 검증 + 재시도 루프 (최대 2회)
- 실패 시 피드백을 Module 2a(Scene Resource Parser)로 전달
- 최대 2회 피드백 후에도 실패 시 `task_abandoned=true`

## 2. 파이프라인 흐름

```
이미지 (cases/images/<task>.png)
  ↓
Module 1     (VisionProvider, OpenAI API)
  ↓ module2_common_input_template.json
Module 2-A   (rule-based, subgoal 분해)
  ↓ module2a_output.json
Module 2-B   (rule-based, 환경 제약)
  ↓ module2b_output.json
Module 2-C   (GPT-4o, 후보 생성)
  ↓ module2c_output.json
Module 2-D   (GPT-4o, 필터링 + 선택)
  ↓ module2d_output.json
Module 3     (GPT-4o, 조립 pose)
  ↓ module3_output.json
(→ Module 3.2 Assembly Simulation)
```

## 3. 디렉토리 구성

- `app/`: CLI, runners, pipelines, reasoners, providers, validators
- `configs/`: 룰/변형/registry 설정 + task preset
  - `task_presets.yaml` (task 1~5 preset)
- `scripts/`: 파이프라인 오케스트레이터
  - `run_pipeline.py` (Module 1→3 end-to-end)
- `cases/images/`: 각 task별 scene 이미지
- `schemas/`: 실행 계약(JSON schema)
- `fixtures/`: 테스트 케이스/번들/샘플 입력
- `specs/`: 각 모듈 명세 문서
- `tests/`: 단위/통합 테스트
- `outputs/`: `<task_name>/<module>_<timestamp>_<suffix>/` 구조로 저장

## 4. 파이프라인 한 번에 실행 (권장)

```bash
export OPENAI_API_KEY="sk-..."

# task preset 기반 — configs/task_presets.yaml 참조
python scripts/run_pipeline.py --preset task1          # card_from_gap
python scripts/run_pipeline.py --preset task2          # deep_hole_reach
python scripts/run_pipeline.py --preset task3          # suspended_target
python scripts/run_pipeline.py --preset task4          # blocked_door_handle
python scripts/run_pipeline.py --preset task5          # glass_shard_extract
```

### 특정 단계까지만
```bash
python scripts/run_pipeline.py --preset task3 --stop-at 2d     # 2d까지
python scripts/run_pipeline.py --preset task3 --stop-at 2a     # 2a까지
```

### 중간부터 이어서 (이미 전 단계 결과 있을 때)
```bash
python scripts/run_pipeline.py --preset task3 \
  --start-from 2c \
  --module2b-dir outputs/suspended_target/module2b_20260421_.../
```

### 개별 옵션으로 task 지정
```bash
python scripts/run_pipeline.py \
  --image cases/images/my_scene.png \
  --task-name my_scene \
  --target-name "target object" \
  --task-description "설명 문구"
```

## 5. 모듈별 CLI (개별 실행)

### Module 1
```bash
python -m app.cli run-experiments \
  --provider vision --image cases/images/task1.png \
  --task-name card_from_gap
```

### Module 2-A / 2-B
```bash
python -m app.cli run-module2a \
  --module2-input outputs/<task>/run_.../module2_common_input_template.json \
  --task-name <task>

python -m app.cli run-module2b \
  --module2-common outputs/<task>/module2a_.../module2_common_input.json \
  --module2a-output outputs/<task>/module2a_.../module2a_output.json \
  --task-name <task>
```

### Module 2-C / 2-D / 3
```bash
python -m app.module2c_cli run-module2c \
  --provider module2b --bundle outputs/<task>/module2b_.../

python -m app.module2d_cli run-module2d \
  --provider module2c --bundle outputs/<task>/module2c_.../

python -m app.module3_cli run-module3 \
  --provider module2d --bundle outputs/<task>/module2d_.../
```

## 6. 핵심 산출물

### Module 1
- `raw_module1_output.json`
- `normalized_module1_output.json`
- `scene_resources_from_module1.json`
- `module2_common_input_template.json`

### Module 2-A
- `module2_common_input.json`
- `module2a_output.json`

### Module 2-B
- `raw_input_bundle.json`, `normalized_context.json`
- `module2b_output.json`, `module3_handoff_preview.json`
- `target_binding_candidates.json`, `environment_structure_candidates.json`
- `numeric_estimates_trace.json`, `derived_constraints_trace.json`

### Module 2-C
- `module2c_output.json` (candidate_tools[])
- `generation_trace.json`
- `summary.json`

### Module 2-D
- `module2d_output.json` (selected_candidate_id, evaluated_candidates[])
- `filter_trace.json`

### Module 3
- `module3_output.json`
  - `assembly_strategy`, `assembly_steps[]`, `final_structure`
  - `verification` (8 checks), `feedback`, `reasoning_trace`
  - `geometric_validation_results` (코드 검증 + 재시도 로그)
- `pose_trace.json`

## 7. task preset

`configs/task_presets.yaml`의 5개 task:

| ID | name | 설명 |
|---|---|---|
| task1 | card_from_gap | 좁은 틈에 끼인 명함 꺼내기 |
| task2 | deep_hole_reach | 깊은 구멍 바닥 물체 꺼내기 |
| task3 | suspended_target | 실에 매달린 물체 회수 |
| task4 | blocked_door_handle | 장애물 너머 문 손잡이 회전 |
| task5 | glass_shard_extract | 유리 파편 속 부품 꺼내기 |

이미지는 `cases/images/<name>.png`에 위치. task 이름 = 이미지 파일 stem으로 일치시키면 자동 인식.

## 8. 출력 경로 구조

모든 모듈의 output은 task별 폴더로 분리:

```
outputs/
├── card_from_gap/
│   ├── run_20260421_.../                      # module 1
│   ├── module2a_20260421_.../
│   ├── module2b_20260421_.../
│   ├── module2c_20260421_.../
│   ├── module2d_20260421_.../
│   └── module3_20260421_.../
├── deep_hole_reach/
├── suspended_target/
├── blocked_door_handle/
└── glass_shard_extract/
```

task 자동 추출 우선순위:
1. `--task-name` 명시 옵션
2. bundle path에서 `module2b_<task>` 패턴 추출
3. `image_path.stem`
4. `case_id`
5. `default`

## 9. 테스트

```bash
pytest -q
```

## 10. 연구 재현성 메모

- Module 1: `VisionProvider`가 OpenAI API 호출 (API 키 env: `OPENAI_API_KEY`)
- Module 2-A/2-B: 현재 rule-based. 논문 서술과 일치시키려면 LLM 전환 필요 (향후 과제)
- Module 2-C/2-D/3: GPT-4o 호출. temperature 0.1~0.3으로 낮춰 재현성 확보
- `outputs/`는 실행 결과 디렉토리로 .gitignore 대상
- `fixtures/`, `schemas/`, `specs/`는 재현성 자산
