# module1-2B (Module 1 -> Module 2-A -> Module 2-B)

이 디렉토리는 Module 1, Module 2-A, Module 2-B 연구 파이프라인을 단일 CLI로 실행/검증하기 위한 실험 하네스입니다.

## 1. 현재 구현 범위

### Module 1
- 입력(raw Module 1 output) 검증 및 정규화
- `scene_resources_from_module1` 생성
- `module2_common_input_template` 브리지 생성
- PyBullet surrogate 시나리오 실행(선택)

### Module 2-A
- `module2_common_input` 기반 subgoal 분해
- subgoal별 요구사항(reasoning) 생성
- `module2a_output` 산출

### Module 2-B
- 입력 번들(`module2_common_input + module2a_output`) 검증
- env-only target/environment binding
- numeric estimates -> derived constraints 생성
- `module2b_output` 및 `module3_handoff_preview` 산출

## 2. 파이프라인 흐름

1. Module 1: raw 출력 검증/정규화
2. Bridge: Module 1 -> Module 2 입력 계약 생성
3. Module 2-A: subgoal 및 요구사항 추론
4. Module 2-B: 환경 제약(env-only) 추론 및 handoff 생성

## 3. 디렉토리 구성

- `app/`: CLI, runners, reasoners, validators, pipelines
- `configs/`: 룰/변형/registry 설정
- `schemas/`: 실행 계약(JSON schema)
- `fixtures/`: 테스트 케이스/번들/샘플 입력
- `specs/`: Module 1/2-A/2-B 명세 문서
- `tests/`: 단위/통합 테스트
- `outputs/`: 실행 산출물

## 4. 주요 CLI 명령

### Module 1 전체 실행
```bash
python -m app.cli run-experiments --provider mock --case-id wooden_block_like_object --scenarios all
```

### Module 1 -> Module 2 Bridge만 추출
```bash
python -m app.cli export-module2-bridge --provider mock --case-id mug_or_container_like_object
```

### Module 2-A 실행
```bash
python -m app.cli run-module2a --provider mock --case-id wooden_block_like_object
```

### Module 2-B 입력 검증
```bash
python -m app.cli validate-module2b-input --bundle fixtures/bundles/coin_in_narrow_gap_case.json
```

### Module 2-B 실행
```bash
python -m app.cli run-module2b --provider mock --case-id coin_in_narrow_gap_case
```

### Module 2-B 반복 실행/비교
```bash
python -m app.cli batch-module2b --cases all --provider mock --repeats 2
python -m app.cli compare-module2b-runs --run-a outputs/<run_a_dir> --run-b outputs/<run_b_dir>
```

## 5. 핵심 산출물

### Module 1
- `raw_module1_output.json`
- `normalized_module1_output.json`
- `scene_resources_from_module1.json`
- `module2_common_input_template.json`
- `module2_bridge_diagnostics.json`

### Module 2-A
- `module2_common_input.json`
- `module2a_output.json`

### Module 2-B
- `raw_input_bundle.json`
- `normalized_context.json`
- `module2b_output.json`
- `module3_handoff_preview.json`
- `target_binding_candidates.json`
- `environment_structure_candidates.json`
- `numeric_estimates_trace.json`
- `derived_constraints_trace.json`

## 6. 테스트

```bash
pytest -q
```

선택 실행:
```bash
pytest tests/test_e2e_smoke.py -q
pytest tests/test_module2a_reasoner.py -q
pytest tests/test_module2b_pipeline.py -q
```

## 7. 연구 재현성 메모

- `fixtures/`, `schemas/`, `specs/`는 재현성 자산으로 유지합니다.
- `outputs/`는 실행 결과 디렉토리이므로 별도 보관/정리 정책을 적용할 수 있습니다.
- Module 2-B는 현재 env-only baseline이며, material/state/damage 계열은 후속 merge 대상으로 남겨둡니다.
