# module1-2B (Module 1 -> Module 2-A -> Module 2-B)

Deterministic research harness for the grasp pipeline:
- Module 1: object/resource extraction + normalization
- Bridge: Module 1 -> Module 2 common contract
- Module 2-A: subgoal decomposition and requirement reasoning
- Module 2-B: env-only target/environment constraint reasoning
- PyBullet: repeatable surrogate simulation for Module 1 outputs

This repository is designed for reproducible local experiments and does not call external LLM/VLM APIs.

## Scope

### Included now
- End-to-end CLI for Module 1, Module 2-A, and Module 2-B
- Fixture-based `mock` providers and file-based `file` providers
- Schema-validated artifacts under deterministic run directories
- Repeatability and comparison runners for Module 1/2-A/2-B

### Not included yet
- Full Module 3 planner/recommender
- Real provider-backed online inference
- Material/state/damage family merge reasoners

## Pipeline View

1. Module 1 ingests raw output (from fixture or file), validates and normalizes it.
2. Bridge exports `scene_resources_from_module1` and `module2_common_input_template`.
3. Module 2-A creates subgoals + task-level requirement summaries.
4. Module 2-B consumes `(module2_common_input + module2a_output)` and emits env-only constraints + Module 3 handoff preview.
5. Optional PyBullet scenarios measure deterministic surrogate dynamics for Module 1 objects.

## Repository Layout

- `app/`: CLI, runners, pipelines, reasoners, providers, validators
- `configs/`: rule sets, run variants, prompt registry, vocabulary
- `schemas/`: executable contracts for Module 1/2-A/2-B artifacts
- `fixtures/`: stable test cases, bundle inputs, images, expected outputs
- `specs/`: prompt/spec references for Module 1/2-A/2-B
- `tests/`: unit/integration/smoke tests
- `outputs/`: generated experiment artifacts (can grow large)

## Quick Start

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -e .[dev]
```

## CLI Commands

### Module 1 full run (validation + normalization + bridge + pybullet)
```bash
python -m app.cli run-experiments --provider mock --case-id wooden_block_like_object --scenarios all
```

### Bridge-only export (Module 1 -> Module 2 input contract)
```bash
python -m app.cli export-module2-bridge --provider mock --case-id mug_or_container_like_object
```

### Module 2-A run
```bash
python -m app.cli run-module2a --provider mock --case-id wooden_block_like_object
```

Or from an explicit Module 2 input:
```bash
python -m app.cli run-module2a --module2-input outputs/<run_dir>/module2_common_input_template.json
```

### Module 2-B input validation
```bash
python -m app.cli validate-module2b-input --bundle fixtures/bundles/coin_in_narrow_gap_case.json
```

### Module 2-B run (env-only)
```bash
python -m app.cli run-module2b --provider mock --case-id coin_in_narrow_gap_case
```

Or from split files:
```bash
python -m app.cli run-module2b \\
  --module2-common fixtures/module2b_cases/coin_in_narrow_gap_case/module2_common_input.json \\
  --module2a-output fixtures/module2b_cases/coin_in_narrow_gap_case/module2a_output.json
```

### Batch and repeatability
```bash
python -m app.cli batch --cases all --provider mock
python -m app.cli evaluate-module1-core --cases all --provider mock --repeats 5
python -m app.cli evaluate-module2a-reasoner --cases all --repeats 5
python -m app.cli evaluate-module2b-reasoner --cases all --repeats 5
python -m app.cli batch-module2b --cases all --provider mock --repeats 2
```

### Module 2-B run-to-run comparison
```bash
python -m app.cli compare-module2b-runs --run-a outputs/<run_a_dir> --run-b outputs/<run_b_dir>
```

## Key Artifacts

### Module 1 run
- `raw_module1_output.json`
- `normalized_module1_output.json`
- `scene_resources_from_module1.json`
- `module2_common_input_template.json`
- `module2_bridge_diagnostics.json`
- `pybullet_surrogate_params.json`
- `pybullet_proxy_spec.json`
- `applied_dynamics.json`
- `metrics.json`
- `summary.json`
- `run_manifest.json`

### Module 2-A run
- `module2_common_input.json`
- `module2a_output.json`
- `run_manifest.json`

### Module 2-B run
- `raw_input_bundle.json`
- `normalized_context.json`
- `validation_report.json`
- `target_binding_candidates.json`
- `environment_structure_candidates.json`
- `numeric_estimates_trace.json`
- `derived_constraints_trace.json`
- `module2b_output.json`
- `module3_handoff_preview.json`
- `summary.json`
- `run_manifest.json`

## Determinism and Contracts

- Schema checks are performed at key boundaries (`schemas/*.json`).
- Rule behavior is controlled by `configs/module2b_*` and bridge mapping configs.
- Deterministic IDs/orderings are used to support repeatability and diff-based comparison.

## Testing

```bash
pytest -q
```

Focused examples:
```bash
pytest tests/test_e2e_smoke.py -q
pytest tests/test_module2a_reasoner.py -q
pytest tests/test_module2b_pipeline.py -q
```

## Notes for Research Use

- `fixtures/`, `specs/`, and `schemas/` are treated as reproducibility assets.
- `outputs/` is generated data and can be archived or cleaned separately.
- Module 2-B is intentionally env-only and emits `module3_handoff_preview.json` as a partial handoff.
