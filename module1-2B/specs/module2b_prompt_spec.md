# Module 2-B Prompt Spec (Authoritative Placeholder)

Source: user-provided APPENDIX_B in this repository task request.

This repository currently stores the exact literal appendix payload as received:

```
너는 Module 2-B: Target Object 및 환경 제약 반영기(env-only)이다.

[PASTE THE EXACT MODULE 2-B PROMPT SPEC HERE VERBATIM]
```

This implementation enforces:
- env-only scope (no Module 3 reasoning/planning)
- strict `module2b_output_env_only` schema
- deterministic IDs and ordering
- layered artifacts: raw / normalized / reasoning trace / strict output + handoff
