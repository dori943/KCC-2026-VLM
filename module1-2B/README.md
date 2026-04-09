# Module 2-B Env-Only Experiment Framework (Python)

## 1. Current Scope vs Future Scope

### Current scope (implemented now)
- Module 2-B only: Target object + environment constraints (env-only)
- Local/mock/file-based end-to-end execution
- Optional Module 1 direct image parsing via OpenAI vision provider (`provider=vision`)
- Strict layered artifacts:
  - raw upstream bundle
  - normalized internal context
  - env-only reasoning trace
  - strict Module 2-B output + Module 3 handoff preview
- Deterministic rule-based baseline for:
  - target binding
  - environment structure synthesis
  - numeric estimate derivation
  - derived constraint generation
  - Module 3 handoff packaging

### Future scope (not implemented in this phase)
- Module 3 reasoning/candidate ranking/planning
- tool recommendation/combo generation
- rollout/simulation planning
- material/state/damage merge reasoning
- external provider-backed LLM/VLM inference
- GT annotation-based evaluation

## 2. Why 4 Layers Are Separated

Layer separation prevents hidden coupling and silent mutation:
- Layer 1 raw preserves upstream evidence exactly.
- Layer 2 normalized creates typed, stable internal data for deterministic reasoning.
- Layer 3 trace records every rule hit/fallback/omission/confidence component.
- Layer 4 strict output gives downstream-stable contract + handoff view.

This lets us swap reasoners/providers later without losing provenance or repeatability.

## 3. Derived Minimal Contract Rationale

Official full `module2_common_input` may not exist yet, so this repo uses a derived minimal schema for Module 2-B only:
- `schemas/module2_common_input_for_module2b_derived_min.schema.json`
- `schema_name = module2_common_input_for_module2b_derived_min`
- `schema_version = 0.1`

Validation strategy:
- Input validator enforces required minimum fields and referential integrity.
- Raw layer can still keep richer upstream fields.
- Normalized layer stores only Module 2-B-consumed fields.

## 4. Target Means Task Object (Not Tool)

Module 2-B enforces:
- target candidates only from `scene_resources.resource_inventory`
- no invented `object_id`
- tool-like object types are penalized in target scoring
- `primary_targets[].object_id` must exist in inventory

## 5. Deterministic Target Binding Heuristic

Implemented in `app/module2b/reasoners/target_binding.py`:
- evidence text sources:
  - `task_brief.user_goal`, `success_criteria`, `task_notes`
  - `task_model.task_restatement`, `primary_success_condition`
  - each subgoal `objective`, `success_condition`
- score components (configurable):
  - semantic match
  - target state-change alignment
  - visibility
  - accessibility
  - relation evidence
- output mode/status:
  - `target_mode`: `single | multiple | implicit | ambiguous | none`
  - `binding_status`: `resolved | partially_resolved | ambiguous | deferred`

Config:
- `configs/module2b_target_binding_rules.yaml`
- `configs/target_alias_registry.yaml`

## 6. Environment Structure Synthesis Without 1:1 Inventory Mapping

Environment structures may exist even without a dedicated inventory object.

Implemented in `app/module2b/reasoners/environment_binding.py`:
- relation/accessibility/geometry/task-keyword evidence fusion
- deterministic `environment_structure_id`: `env_01`, `env_02`, ...
- complete `access_path_profile` generation:
  - `entry_mode`
  - `rotation_clearance`
  - `requires_pass_through_opening`
  - `requires_deep_reach`
  - `available_support_surface`
  - `slip_hazard_present`
  - `confinement_level` (1/2/3)

## 7. Why `numeric_estimates` and `derived_constraints` Are Split

Two-step logic is explicit by design:
1. `numeric_estimates` quantify environment facts.
2. `derived_constraints` translate those facts into comparison-ready constraints.

Benefits:
- clearer provenance
- safer fallback (ordinal ranges when scale anchor is weak)
- easier future replacement by learned/LLM components

## 8. Env-Only Limitation and Pending Merge Policy

This baseline intentionally omits material/state/damage families.

`module3_handoff` therefore includes:
- `pending_merge_sources`: typically `material_reasoner`
- `omitted_constraint_families`:
  - `risk_limit`
  - `target_material_state`
  - `damage_sensitivity`
  - `contact_style_preference`

`handoff_status` is usually `partial` in env-only mode.

## 9. How to Attach Real Module 3 Later

When Module 3 is added:
1. Read `module2b_output.json`
2. Merge pending source families into constraint space
3. Use `module3_handoff.handoff_constraint_ids` for downstream ranking/planning
4. Preserve current IDs and ordering to keep compatibility

## 10. How to Attach Real LLM Providers Later

Provider seam is separated now:
- `app/module2b/providers.py`
  - `MockBundleProvider`
  - `FileBundleProvider`
  - `Module2BBundleProvider` protocol

A future provider can implement the same interface and return the same bundle contract.

## 11. How to Change Rules/Thresholds/Vocab

- prompt variant registry: `configs/prompt_registry.yaml`
- run variant registry: `configs/module2b_run_variants.yaml`
- target binding rules: `configs/module2b_target_binding_rules.yaml`
- environment rules: `configs/module2b_environment_rules.yaml`
- numericization rules: `configs/module2b_numericization_rules.yaml`
- constraint rules: `configs/module2b_constraint_rules.yaml`
- allowed enums/vocab: `configs/vocab_registry.json`

## 12. How to Add Fixture Cases

Add a directory under `fixtures/module2b_cases/<case_id>/` with:
- `module2_common_input.json`
- `module2a_output.json`
- `bundle.json`
- `expected.json`

Then update:
- `fixtures/module2b_cases/index.json`
- optional bundled shortcut under `fixtures/bundles/<case_id>.json`

## 13. Deterministic ID + Ordering Rules

Implemented deterministic rules:
- target candidates: score descending, inventory order tie-break
- `environment_structure_id`: `env_XX` by first evidence order + structure role priority
- `measurement_id`: `m_XX` by `(environment_structure_id, parameter_name, bound_type)`
- `constraint_id`: `c_XX` by `(subgoal_order, priority, category, parameter_name, applies_to)`
- `subgoal_bindings`: exact Module 2-A subgoal order
- dedup policy documented in diagnostics artifact and run manifest

## 14. Repeatability and Comparison Harness

Implemented:
- repeated batch runs
- deterministic run comparison
- structural/value diff artifacts

Outputs:
- `comparison_summary.json`
- `structural_diff.json`
- `value_diff.json`

## Repository Structure (Module 2-B additions)

- `specs/module2a_prompt_spec.md`
- `specs/module2b_prompt_spec.md`
- `schemas/module2_common_input_for_module2b_derived_min.schema.json`
- `schemas/module2b_input_bundle.schema.json`
- `schemas/module2b_output_env_only.schema.json`
- `schemas/module2b_diagnostics.schema.json`
- `configs/module2b_*.yaml`
- `configs/target_alias_registry.yaml`
- `fixtures/module2b_cases/*`
- `fixtures/bundles/*`
- `app/module2b/*`
- `app/pipelines/module2b_pipeline.py`
- `app/runners/module2b_runner.py`
- `tests/test_module2b_pipeline.py`

## CLI Examples

### Run Module 1 directly from image (OpenAI vision provider)
```bash
set OPENAI_API_KEY=YOUR_KEY
python -m app.cli run-experiments --provider vision --image path/to/input.png
```

### Validate input bundle
```bash
python -m app.cli validate-module2b-input --bundle fixtures/bundles/coin_in_narrow_gap_case.json
```

### Run Module 2-B from bundle
```bash
python -m app.cli run-module2b --bundle fixtures/bundles/mug_under_overhang_case.json
```

### Run Module 2-B from split files
```bash
python -m app.cli run-module2b \
  --module2-common fixtures/module2b_cases/bottle_in_deep_recess_case/module2_common_input.json \
  --module2a-output fixtures/module2b_cases/bottle_in_deep_recess_case/module2a_output.json
```

### Run Module 2-B with mock provider by case id
```bash
python -m app.cli run-module2b --provider mock --case-id coin_in_narrow_gap_case
```

### Export normalized context only
```bash
python -m app.cli export-module2b-normalized --provider mock --case-id mug_under_overhang_case
```

### Batch run + repeatability
```bash
python -m app.cli batch-module2b --cases all --provider mock --repeats 2
```

### Compare two run outputs
```bash
python -m app.cli compare-module2b-runs --run-a outputs/module2b_... --run-b outputs/module2b_...
```

## Run Artifacts Per Module 2-B Execution

Each run stores at least:
- `run_manifest.json`
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

## Tests

Run Module 2-B tests:
```bash
pytest tests/test_module2b_pipeline.py -q
```

## Important Assumptions

- APPENDIX_A/B prompt blocks in the task request were placeholders; this repo stores those literal placeholders in spec files.
- Strict executable contracts are enforced through JSON schemas and validators in this repository.
- Module 2-B reasoner itself is deterministic; optional external API usage applies only when `provider=vision` is selected for Module 1 input generation.
