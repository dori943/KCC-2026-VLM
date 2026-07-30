"""Candidate filter for Module 2-D using GPT-4o."""

from __future__ import annotations

import copy
import json
import os
import re
from typing import Any

from openai import OpenAI

from app.feedback.controller import compute_filter_counts
from app.module2d.models import (
    EvaluatedCandidate, FeedbackDecision, Module2DInput,
    RepairAnalysis, StageScores, WeakPoint,
)


# Emergence margin for level_1_to_5 constraints.
# We relax only the copy sent to the LLM, not the original input object.
# This reduces over-pruning of creative but plausible candidates.
_EMERGENCE_DELTA = 1.0  # Relax level_1_to_5 bounds by plus_or_minus 1.0 for LLM-side filtering.
_LEVEL_MIN = 1.0
_LEVEL_MAX = 5.0
# Matches only "level_1_to_5" bounds such as:
# "lower_bound:4.5level_1_to_5" or "upper_bound 2.5level_1_to_5"
_BOUND_RE = re.compile(
    r"(?P<kind>lower_bound|upper_bound)\s*[:\s]\s*"
    r"(?P<val>\d+(?:\.\d+)?)\s*level_1_to_5",
    re.IGNORECASE,
)

# Handle role hard gate:
# - allowlist objects pass immediately
# - non-allowlist objects are rejected unless geometry/rigidity numeric exception passes
_HANDLE_ALLOWLIST: set[str] = {
    "phillips_screwdriver",
    "flat_screwdriver",
    "large_marker",
    "small_marker",
    "fork",
    "spoon",
    "knife",
    "spatula",
    "adjustable_wrench",
    "hammer",
    "power_drill",
    "key",
}
_HANDLE_FUNC_EXACT: set[str] = {
    "handle",
    "graspable_body",
    "elongated_reach",
    "long_reach",
    "reach_extension",
    "shaft",
}
_HANDLE_FUNC_KEYWORDS: tuple[str, ...] = (
    "handle",
    "grip",
    "reach",
    "shaft",
    "lever",
)
# Numeric exception thresholds for non-allowlist handle candidates.
_HANDLE_MIN_ASPECT_RATIO = 2.6
_HANDLE_MIN_EFFECTIVE_REACH_M = 0.12
_HANDLE_MAX_CROSS_SECTION_WIDTH_M = 0.08

# Hook role hard gate:
# - allowlist objects pass immediately
# - non-allowlist objects need thin/pointed contact profile for reliable latch hooking
_HOOK_ALLOWLIST: set[str] = {
    "fork",
    "key",
    "scissors",
    "small_clamp",
    "medium_clamp",
    "large_clamp",
    "extra_large_clamp",
}
_HOOK_FUNC_EXACT: set[str] = {
    "hook",
    "hookable_gap",
    "hook_tip",
    "curved_tip",
}
_HOOK_FUNC_KEYWORDS: tuple[str, ...] = (
    "hook",
    "gap",
    "latch",
)
_HOOK_MAX_LOCAL_THICKNESS_M = 0.006
_HOOK_MAX_TIP_RADIUS_M = 0.0025

# Primitive coverage aliases:
# map high-level interaction primitives to candidate function labels.
_PRIMITIVE_FUNCTION_ALIASES: dict[str, set[str]] = {
    "reach": {
        "reach",
        "elongated_reach",
        "long_reach",
        "reach_extension",
        "shaft",
        "handle",
        "graspable_body",
    },
    "hook": {
        "hook",
        "hookable_gap",
        "hook_tip",
        "curved_tip",
        "insert",
    },
    "pull": {
        "pull",
        "hook",
        "hookable_gap",
        "elongated_reach",
        "reach_extension",
        "graspable_body",
    },
    "align": {
        "align",
        "insert",
        "thin_edge",
        "rigid_tip",
        "support",
    },
    "clamp": {
        "clamp",
        "capture",
        "container_cavity",
        "support",
        "compliant_pad",
    },
    "grasp": {
        "grasp",
        "graspable_body",
        "handle",
        "elongated_reach",
        "rigid_tip",
        "support",
    },
}

# Coverage policy:
# - missing critical primitives => hard fail
# - missing optional atoms/primitives => soft penalty
_COVERAGE_SOFT_PENALTY_PER_PRIMITIVE = 0.45
_COVERAGE_SOFT_PENALTY_PER_ATOM = 0.25
_COVERAGE_SOFT_PENALTY_MAX = 2.0


def _safe_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _normalize_func_name(s: str) -> str:
    return (s or "").strip().lower().replace("-", "_").replace(" ", "_")


def _is_handle_role_function(func_name: str) -> bool:
    f = _normalize_func_name(func_name)
    if not f:
        return False
    if f in _HANDLE_FUNC_EXACT:
        return True
    return any(k in f for k in _HANDLE_FUNC_KEYWORDS)


def _is_hook_role_function(func_name: str) -> bool:
    f = _normalize_func_name(func_name)
    if not f:
        return False
    if f in _HOOK_FUNC_EXACT:
        return True
    return any(k in f for k in _HOOK_FUNC_KEYWORDS)


def _estimate_handle_metrics(
    obj_name: str,
    prop_map: dict[str, dict[str, Any]],
    geo_map: dict[str, dict[str, Any]],
) -> dict[str, float | None]:
    """Estimate handle viability metrics from available profiles.

    Priority:
    1) numeric_profile explicit values
    2) AABB-derived proxy values
    """
    prop = prop_map.get(obj_name, {}) or {}
    np_ = prop.get("numeric_profile", {}) or {}
    aspect_ratio = _safe_float(np_.get("aspect_ratio"))
    effective_reach = _safe_float(np_.get("effective_reach"))
    cross_width = _safe_float(np_.get("cross_section_width"))

    geo = geo_map.get(obj_name, {}) or {}
    amin = geo.get("aabb_min")
    amax = geo.get("aabb_max")
    if (
        isinstance(amin, list)
        and isinstance(amax, list)
        and len(amin) >= 3
        and len(amax) >= 3
    ):
        extents = [abs(float(amax[i]) - float(amin[i])) for i in range(3)]
        sorted_ext = sorted(extents, reverse=True)
        longest = sorted_ext[0]
        second = sorted_ext[1] if len(sorted_ext) > 1 else None
        if effective_reach is None:
            effective_reach = longest
        if cross_width is None and second is not None:
            cross_width = second
        if aspect_ratio is None and second is not None and second > 1e-9:
            aspect_ratio = longest / second

    return {
        "aspect_ratio": aspect_ratio,
        "effective_reach": effective_reach,
        "cross_section_width": cross_width,
    }


def _estimate_hook_metrics(
    obj_name: str,
    prop_map: dict[str, dict[str, Any]],
    geo_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    prop = prop_map.get(obj_name, {}) or {}
    gp = prop.get("geometry_profile", {}) or {}
    np_ = prop.get("numeric_profile", {}) or {}
    rigid = str(prop.get("rigidity", "")).lower()
    local_thickness = _safe_float(np_.get("local_thickness_m"))
    tip_radius = _safe_float(np_.get("tip_radius_m"))
    pointed_or_thin = bool(gp.get("has_pointed_or_thin_end", False))
    has_open_cavity = bool(gp.get("has_open_cavity", False))
    primary_contact = _normalize_func_name(str(gp.get("primary_contact_profile") or ""))
    thickness_class = _normalize_func_name(str(gp.get("thickness_class") or ""))

    # AABB fallback for thickness when numeric profile is sparse.
    geo = geo_map.get(obj_name, {}) or {}
    amin = geo.get("aabb_min")
    amax = geo.get("aabb_max")
    cross_section_width = None
    if (
        isinstance(amin, list)
        and isinstance(amax, list)
        and len(amin) >= 3
        and len(amax) >= 3
    ):
        extents = [abs(float(amax[i]) - float(amin[i])) for i in range(3)]
        sorted_ext = sorted(extents, reverse=True)
        cross_section_width = sorted_ext[1] if len(sorted_ext) > 1 else None
        if local_thickness is None and len(sorted_ext) > 2:
            local_thickness = sorted_ext[2]

    return {
        "rigidity": rigid,
        "local_thickness_m": local_thickness,
        "tip_radius_m": tip_radius,
        "has_pointed_or_thin_end": pointed_or_thin,
        "has_open_cavity": has_open_cavity,
        "primary_contact_profile": primary_contact,
        "thickness_class": thickness_class,
        "cross_section_width_m": cross_section_width,
    }


def _check_handle_role_feasibility(
    candidate: dict[str, Any],
    prop_map: dict[str, dict[str, Any]],
    geo_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Hard-gate handle-role assignments.

    Returns a list of violations. Empty list means pass.
    """
    violations: list[dict[str, Any]] = []
    for fm in candidate.get("function_mapping", []) or []:
        obj_name = (fm.get("object") or "").strip()
        func_name = fm.get("function") or ""
        if not obj_name or not _is_handle_role_function(func_name):
            continue

        if obj_name in _HANDLE_ALLOWLIST:
            continue

        prop = prop_map.get(obj_name, {}) or {}
        rigidity = str(prop.get("rigidity", "")).lower()
        metrics = _estimate_handle_metrics(obj_name, prop_map, geo_map)
        ar = metrics.get("aspect_ratio")
        reach = metrics.get("effective_reach")
        cross = metrics.get("cross_section_width")

        # Numeric exception path for non-allowlist objects.
        rigidity_ok = rigidity in {"rigid", "semi-rigid"}
        ar_ok = ar is not None and ar >= _HANDLE_MIN_ASPECT_RATIO
        reach_ok = reach is not None and reach >= _HANDLE_MIN_EFFECTIVE_REACH_M
        cross_ok = cross is not None and cross <= _HANDLE_MAX_CROSS_SECTION_WIDTH_M
        numeric_exception_ok = rigidity_ok and ar_ok and reach_ok and cross_ok

        if numeric_exception_ok:
            continue

        reason = (
            f"non-allowlist handle role rejected: object='{obj_name}', function='{func_name}', "
            f"rigidity='{rigidity or 'unknown'}', "
            f"aspect_ratio={ar}, effective_reach={reach}, cross_section_width={cross}; "
            f"required: rigidity in {{rigid,semi-rigid}}, aspect_ratio>={_HANDLE_MIN_ASPECT_RATIO}, "
            f"effective_reach>={_HANDLE_MIN_EFFECTIVE_REACH_M}m, "
            f"cross_section_width<={_HANDLE_MAX_CROSS_SECTION_WIDTH_M}m."
        )
        violations.append({
            "object": obj_name,
            "function": func_name,
            "reason": reason,
            "metrics": metrics,
        })

    return violations


def _check_hook_role_feasibility(
    candidate: dict[str, Any],
    prop_map: dict[str, dict[str, Any]],
    geo_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Hard-gate hook-role assignments for latch/pull type tasks."""
    violations: list[dict[str, Any]] = []
    for fm in candidate.get("function_mapping", []) or []:
        obj_name = (fm.get("object") or "").strip()
        func_name = fm.get("function") or ""
        if not obj_name or not _is_hook_role_function(func_name):
            continue

        if obj_name in _HOOK_ALLOWLIST:
            continue

        metrics = _estimate_hook_metrics(obj_name, prop_map, geo_map)
        rigidity = str(metrics.get("rigidity") or "")
        local_thickness = _safe_float(metrics.get("local_thickness_m"))
        tip_radius = _safe_float(metrics.get("tip_radius_m"))
        pointed_or_thin = bool(metrics.get("has_pointed_or_thin_end"))
        has_open_cavity = bool(metrics.get("has_open_cavity"))
        primary_contact = str(metrics.get("primary_contact_profile") or "")

        rigid_ok = rigidity in {"rigid", "semi-rigid"}
        pointed_ok = pointed_or_thin or primary_contact in {"tip", "edge"}
        thickness_ok = (
            local_thickness is not None
            and local_thickness <= _HOOK_MAX_LOCAL_THICKNESS_M
        )
        tip_ok = tip_radius is None or tip_radius <= _HOOK_MAX_TIP_RADIUS_M
        cavity_ok = not has_open_cavity or pointed_or_thin
        numeric_exception_ok = rigid_ok and pointed_ok and thickness_ok and tip_ok and cavity_ok

        if numeric_exception_ok:
            continue

        reason = (
            f"non-allowlist hook role rejected: object='{obj_name}', function='{func_name}', "
            f"rigidity='{rigidity or 'unknown'}', pointed_or_thin={pointed_or_thin}, "
            f"primary_contact_profile='{primary_contact or 'unknown'}', has_open_cavity={has_open_cavity}, "
            f"local_thickness_m={local_thickness}, tip_radius_m={tip_radius}; required: "
            f"rigidity in {{rigid,semi-rigid}}, thin/pointed contact (tip/edge), "
            f"local_thickness<={_HOOK_MAX_LOCAL_THICKNESS_M}m, tip_radius<={_HOOK_MAX_TIP_RADIUS_M}m, "
            "and open-cavity objects must still provide a pointed/thin hooking edge."
        )
        violations.append({
            "object": obj_name,
            "function": func_name,
            "reason": reason,
            "metrics": metrics,
        })

    return violations


def _relax_bound_string(s: str) -> tuple[str, str | None]:
    """Relax the first level_1_to_5 lower/upper bound found in a string.

    Returns: (relaxed_string, change_summary or None)
    """
    if not isinstance(s, str):
        return s, None
    m = _BOUND_RE.search(s)
    if not m:
        return s, None
    kind = m.group("kind").lower()
    old_val = float(m.group("val"))
    if kind == "lower_bound":
        new_val = max(_LEVEL_MIN, old_val - _EMERGENCE_DELTA)
    else:  # upper_bound
        new_val = min(_LEVEL_MAX, old_val + _EMERGENCE_DELTA)
    if new_val == old_val:
        return s, None
    # Keep integer formatting when possible (4.0 -> 4, 3.5 -> 3.5).
    new_str = f"{int(new_val)}" if new_val == int(new_val) else f"{new_val:g}"
    relaxed = s[: m.start("val")] + new_str + s[m.end("val") :]
    return relaxed, f"{kind} {old_val} -> {new_val}"


def _relax_tool_constraints(
    tc: Any,
) -> tuple[Any, list[str]]:
    """Apply emergence-margin relaxation to tool constraints.

    Supports:
    - global_constraints: list[str]
    - derived_constraints: dict[str, str]

    Returns a deep-copied relaxed object and change summaries.
    """
    if not isinstance(tc, dict):
        return tc, []
    relaxed = copy.deepcopy(tc)
    changes: list[str] = []

    # global_constraints: list[str]
    gc = relaxed.get("global_constraints")
    if isinstance(gc, list):
        for i, item in enumerate(gc):
            new_item, change = _relax_bound_string(item)
            if change:
                gc[i] = new_item
                changes.append(f"global_constraints[{i}] {change}")

    # derived_constraints: dict[str, str]
    dc = relaxed.get("derived_constraints")
    if isinstance(dc, dict):
        for key, val in list(dc.items()):
            new_val, change = _relax_bound_string(val)
            if change:
                dc[key] = new_val
                changes.append(f"derived_constraints[{key}] {change}")

    return relaxed, changes

CANDIDATE_SYSTEM_PROMPT = """You are Module 2-D, a strict candidate evaluator.
Evaluate exactly one candidate tool assembly and return JSON only.

Input fields:
- task
- tool_constraints (including derived_constraints and subgoal_constraints)
- candidate (used_objects, structure_description, function_mapping)
- object_physical_properties
- scene_objects

Process:
1. pre_analysis (required)
- Classify task_type_classification into one of A/B/C/D/E with one short reason.
- For each used object, provide object_profiles with:
  name, functional_end, mechanism, attach_zone, blockable (high|medium|low with reason)

2. Hard filters
- environment_filter:
  pass true/false, violated_constraints list, and one concise reason.
- assembly_filter:
  pass true/false, violated_reasons list, and one concise reason.

3. Checklist scoring evidence (binary only)
Return checklist arrays for all required items:
- geometry: G1..G7
- physics: P1..P6
- commonsense: C1..C5
- task_fit: TF1..TF3
Each item must include: item, result, false_reason.
Set false_reason only when result=false.

4. repair_analysis
- repair_type: local_fix | global_redesign | none
- target_issue, reason, repair_strategy

Output rules:
- Output JSON only.
- Do not output stage_scores, total_score, pass, failed_stage, or weak_points.
- Include every checklist item exactly once.

Output schema:
{
  "candidate_id": "...",
  "pre_analysis": {
    "task_type_classification": "A|B|C|D|E with brief reason",
    "object_profiles": [
      {
        "name": "object name",
        "functional_end": "...",
        "mechanism": "...",
        "attach_zone": "...",
        "blockable": "high | medium | low with reason"
      }
    ]
  },
  "environment_filter": {
    "pass": true,
    "violated_constraints": [],
    "reason": "..."
  },
  "assembly_filter": {
    "pass": true,
    "violated_reasons": [],
    "reason": "..."
  },
  "checklist": {
    "geometry": [
      {"item": "G1", "result": true, "false_reason": null},
      {"item": "G2", "result": true, "false_reason": null},
      {"item": "G3", "result": true, "false_reason": null},
      {"item": "G4", "result": true, "false_reason": null},
      {"item": "G5", "result": true, "false_reason": null},
      {"item": "G6", "result": true, "false_reason": null},
      {"item": "G7", "result": true, "false_reason": null}
    ],
    "physics": [
      {"item": "P1", "result": true, "false_reason": null},
      {"item": "P2", "result": true, "false_reason": null},
      {"item": "P3", "result": true, "false_reason": null},
      {"item": "P4", "result": true, "false_reason": null},
      {"item": "P5", "result": true, "false_reason": null},
      {"item": "P6", "result": false, "false_reason": "reason"}
    ],
    "commonsense": [
      {"item": "C1", "result": true, "false_reason": null},
      {"item": "C2", "result": true, "false_reason": null},
      {"item": "C3", "result": true, "false_reason": null},
      {"item": "C4", "result": false, "false_reason": "reason"},
      {"item": "C5", "result": true, "false_reason": null}
    ],
    "task_fit": [
      {"item": "TF1", "result": true, "false_reason": null},
      {"item": "TF2", "result": true, "false_reason": null},
      {"item": "TF3", "result": true, "false_reason": null}
    ]
  },
  "repair_analysis": {
    "repair_type": "local_fix | global_redesign | none",
    "target_issue": "...",
    "reason": "...",
    "repair_strategy": ["..."]
  }
}
"""


def _collect_candidate_functions(
    candidate: dict[str, Any],
    prop_map: dict[str, dict[str, Any]],
) -> set[str]:
    used_names = candidate.get("used_objects", []) or []
    funcs: set[str] = set()
    for fm in candidate.get("function_mapping", []) or []:
        f = _normalize_func_name(str(fm.get("function") or ""))
        if f:
            funcs.add(f)
    for n in used_names:
        prop = prop_map.get(n, {}) or {}
        for f in prop.get("inferred_functions", []) or []:
            if isinstance(f, str) and f.strip():
                funcs.add(_normalize_func_name(f))
    return funcs


def _expand_capabilities_with_aliases(candidate_funcs: set[str]) -> set[str]:
    expanded = set(candidate_funcs)
    for primitive, aliases in _PRIMITIVE_FUNCTION_ALIASES.items():
        normalized_aliases = {_normalize_func_name(x) for x in aliases}
        if expanded.intersection(normalized_aliases | {primitive}):
            expanded.add(primitive)
    return expanded


def _infer_critical_primitives(
    subgoal_constraints: list[dict[str, Any]],
) -> set[str]:
    required_atoms: set[str] = set()
    required_primitives: set[str] = set()
    for sg in subgoal_constraints or []:
        if not isinstance(sg, dict):
            continue
        for a in sg.get("required_atoms", []) or []:
            if isinstance(a, str) and a.strip():
                required_atoms.add(_normalize_func_name(a))
        for p in sg.get("required_interaction_primitives", []) or []:
            if isinstance(p, str) and p.strip():
                required_primitives.add(_normalize_func_name(p))

    if "hookable_gap" in required_atoms and {"hook", "pull", "reach"}.intersection(required_primitives):
        return {p for p in {"reach", "hook", "pull"} if p in required_primitives}
    return set()


def _check_subgoal_coverage(
    candidate: dict[str, Any],
    subgoal_constraints: list[dict[str, Any]],
    prop_map: dict[str, dict[str, Any]],
    critical_primitives: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deterministically validate required atoms/primitives coverage for each subgoal."""
    critical_primitives = {_normalize_func_name(p) for p in (critical_primitives or set())}
    candidate_funcs = _collect_candidate_functions(candidate, prop_map)
    candidate_caps = _expand_capabilities_with_aliases(candidate_funcs)

    def _covered(label: str) -> bool:
        target = _normalize_func_name(label)
        if not target:
            return True
        if target in candidate_caps:
            return True
        for f in candidate_caps:
            if target == f or target in f or f in target:
                return True
        return False

    issues: list[dict[str, Any]] = []
    required_atoms_total = 0
    required_primitives_total = 0
    missing_atoms_total = 0
    missing_primitives_total = 0
    required_primitives_unique: set[str] = set()
    missing_primitives_unique: set[str] = set()
    missing_atoms_unique: set[str] = set()

    for sg in subgoal_constraints or []:
        if not isinstance(sg, dict):
            continue
        sg_id = sg.get("subgoal_id", "")
        required_atoms = [
            _normalize_func_name(a)
            for a in (sg.get("required_atoms") or [])
            if isinstance(a, str) and a.strip()
        ]
        required_prims = [
            _normalize_func_name(p)
            for p in (sg.get("required_interaction_primitives") or [])
            if isinstance(p, str) and p.strip()
        ]
        if not required_atoms and not required_prims:
            continue

        required_atoms_total += len(required_atoms)
        required_primitives_total += len(required_prims)
        required_primitives_unique.update(required_prims)

        missing_atoms = [a for a in required_atoms if not _covered(a)]
        missing_prims = [p for p in required_prims if not _covered(p)]

        missing_atoms_total += len(missing_atoms)
        missing_primitives_total += len(missing_prims)
        missing_atoms_unique.update(missing_atoms)
        missing_primitives_unique.update(missing_prims)

        if missing_atoms or missing_prims:
            issues.append({
                "subgoal_id": sg_id,
                "missing_atoms": missing_atoms,
                "missing_primitives": missing_prims,
                "candidate_functions": sorted(candidate_caps),
            })

    required_total = required_atoms_total + required_primitives_total
    missing_total = missing_atoms_total + missing_primitives_total
    coverage_complete_ratio = (
        round((required_total - missing_total) / required_total, 4)
        if required_total > 0 else 1.0
    )
    critical_missing = sorted(missing_primitives_unique.intersection(critical_primitives))
    optional_missing_primitives = sorted(
        p for p in missing_primitives_unique if p not in critical_primitives
    )
    summary = {
        "required_atoms_total": required_atoms_total,
        "required_primitives_total": required_primitives_total,
        "missing_atoms_total": missing_atoms_total,
        "missing_primitives_total": missing_primitives_total,
        "required_primitives_unique": sorted(required_primitives_unique),
        "missing_primitives_unique": sorted(missing_primitives_unique),
        "missing_atoms_unique": sorted(missing_atoms_unique),
        "critical_primitives": sorted(critical_primitives),
        "critical_missing_primitives": critical_missing,
        "optional_missing_primitives": optional_missing_primitives,
        "coverage_complete_ratio": coverage_complete_ratio,
        "candidate_functions": sorted(candidate_caps),
    }
    return issues, summary


EMERGENCE_SCORING_PROMPT = """

# Additional scoring: emergent-tool plausibility

In addition to the existing filters/checklists, output an `emergence_analysis`
object. These scores are used by code for final candidate selection, so do not
give every feasible assembly a perfect score.

Use 1-5 integer or float scores:
- emergence_score: Does the combination create a new task-relevant affordance
  that the individual objects do not provide alone?
- necessity_score: Is the combination actually needed in this scenario, rather
  than one object already solving the task by itself?
- role_contribution_score: Does every used object have a clear, non-redundant,
  physically plausible role in the final tool?

Important penalties:
- Penalize broad, bulky, or flat objects when they are described as long
  handles or reach extenders without geometry that can transmit force well.
- Penalize safety, stability, or cushioning attachments if they do not directly
  support the stated task constraints.
- Penalize combinations where the primary task can be solved by one object and
  the added objects only make the assembly look more complex.

High-scoring emergent-tool pattern:
- a primary interaction feature that can contact/manipulate the target,
- a transmission or reach feature that enables the task constraints,
- optional safety/contact modifier only when it is functionally needed.

Add this field to the JSON root:
"emergence_analysis": {
  "emergence_score": 1-5,
  "necessity_score": 1-5,
  "role_contribution_score": 1-5,
  "single_object_baseline_gap": "why no single object is enough",
  "synergy_reason": "what new combined affordance appears",
  "role_contribution_reason": "why each object is necessary"
}
"""


def _build_hard_rejected_candidate(
    candidate_id: str,
    failed_stage: str,
    reasons: list[str],
) -> EvaluatedCandidate:
    weak_points = [
        WeakPoint(
            stage=failed_stage,
            item=f"HR{i + 1}",
            score=0.0,
            reason=r,
        )
        for i, r in enumerate(reasons)
    ]
    return EvaluatedCandidate(
        candidate_id=candidate_id,
        stage_scores=StageScores(
            geometry=0.0,
            physics=0.0,
            commonsense=0.0,
            task_fit=0.0,
            emergence=0.0,
            necessity=0.0,
            role_contribution=0.0,
        ),
        total_score=0.0,
        passed=False,
        failed_stage=failed_stage,
        weak_points=weak_points,
        repair_analysis=RepairAnalysis(
            repair_type="global_redesign",
            target_issue=failed_stage,
            reason="hard gate rejection",
            repair_strategy=["handle role object reassignment"],
        ),
        environment_filter={"pass": True},
        assembly_filter={"pass": True},
        checklist={},
        pre_analysis={"hard_rejected": True},
        emergence_analysis={},
    )


def filter_candidates(
    input_data: Module2DInput,
    api_key: str | None = None,
    model: str = "gpt-4o",
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> tuple[list[EvaluatedCandidate], str | None, FeedbackDecision, dict[str, Any]]:
    """Evaluate each candidate with an independent GPT call, then aggregate.

    Returns:
        (evaluated_candidates, selected_candidate_id, feedback_decision, trace)
    """
    client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    # Build lookup maps shared across all candidates
    prop_map = {p["name"]: p for p in input_data.object_physical_properties}
    geo_map = {o["name"]: o for o in input_data.scene_objects}

    # Extract subgoal constraints generated by Module 2-B.
    subgoal_constraints: list[dict[str, Any]] = []
    tc = input_data.tool_constraints or {}
    sg_raw = tc.get("subgoal_constraints") if isinstance(tc, dict) else None
    if isinstance(sg_raw, list):
        subgoal_constraints = [s for s in sg_raw if isinstance(s, dict)]
    critical_primitives = _infer_critical_primitives(subgoal_constraints)

    # Apply emergence-margin relaxation only to the LLM input copy.
    relaxed_tool_constraints, emergence_changes = _relax_tool_constraints(
        input_data.tool_constraints
    )
    if emergence_changes:
        print(
            f"[2D emergence_margin] applied to {len(emergence_changes)} constraints "
            f"(delta={_EMERGENCE_DELTA}):"
        )
        for ch in emergence_changes:
            print(f"             - {ch}")

    # Scene-name allowlist for hard validation.
    allowed_scene_names: set[str] = {
        (o.get("name") or "").strip() for o in (input_data.scene_objects or [])
        if (o.get("name") or "").strip()
    }

    evaluated: list[EvaluatedCandidate] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    raw_previews: dict[str, str] = {}
    coverage_issues_by_cid: dict[str, list[dict[str, Any]]] = {}
    coverage_summary_by_cid: dict[str, dict[str, Any]] = {}
    rejected_for_unknown_names: list[dict[str, Any]] = []
    rejected_for_handle_role: list[dict[str, Any]] = []
    rejected_for_hook_role: list[dict[str, Any]] = []

    for candidate in input_data.candidate_tools:
        cid = candidate["candidate_id"]
        used_names = candidate.get("used_objects", [])

        # Hard filter: every used object must exist in scene_objects names.
        unknown_names = [n for n in used_names if n not in allowed_scene_names]
        if unknown_names:
            rejected_for_unknown_names.append({
                "candidate_id": cid,
                "unknown_names": unknown_names,
                "used_objects": used_names,
            })
            continue

        # Hard gate: handle-role object feasibility
        handle_violations = _check_handle_role_feasibility(candidate, prop_map, geo_map)
        if handle_violations:
            rejected_for_handle_role.append({
                "candidate_id": cid,
                "violations": handle_violations,
                "used_objects": used_names,
            })
            evaluated.append(_build_hard_rejected_candidate(
                candidate_id=cid,
                failed_stage="handle_feasibility",
                reasons=[v.get("reason", "") for v in handle_violations],
            ))
            continue

        # Hard gate: hook-role object feasibility
        hook_violations = _check_hook_role_feasibility(candidate, prop_map, geo_map)
        if hook_violations:
            rejected_for_hook_role.append({
                "candidate_id": cid,
                "violations": hook_violations,
                "used_objects": used_names,
            })
            evaluated.append(_build_hard_rejected_candidate(
                candidate_id=cid,
                failed_stage="hook_feasibility",
                reasons=[v.get("reason", "") for v in hook_violations],
            ))
            continue

        # Idea 3: Python-built object profile summary (geometry_profile + numeric_profile)
        profile_lines: list[str] = []
        for name in used_names:
            prop = prop_map.get(name)
            if not prop:
                continue
            line_parts = [
                f"[{name}]",
                f"  shape={prop.get('shape_category','?')}",
                f"  rigidity={prop.get('rigidity','?')}",
                f"  friction={prop.get('surface_friction','?')}",
                f"  functions={prop.get('inferred_functions',[])}",
                f"  connection_modes={prop.get('connection_modes',[])}",
            ]
            gp = prop.get("geometry_profile", {})
            if gp:
                gp_str = ", ".join(f"{k}={v}" for k, v in gp.items())
                line_parts.append(f"  geometry_profile: {gp_str}")
            np_ = prop.get("numeric_profile", {})
            if np_:
                np_str = ", ".join(f"{k}={v}" for k, v in np_.items())
                line_parts.append(f"  numeric_profile: {np_str}")
            geo = geo_map.get(name, {})
            aabb = geo.get("aabb_dimensions_m") or geo.get("aabb_size_m")
            if aabb:
                line_parts.append(f"  aabb_m: {aabb}")
            profile_lines.append("\n".join(line_parts))
        object_profiles_text = "\n".join(profile_lines) if profile_lines else "(none)"

        user_payload = {
            "task": input_data.task,
            "tool_constraints": relaxed_tool_constraints,
            "candidate": candidate,
            "object_physical_properties": [prop_map[n] for n in used_names if n in prop_map],
            "scene_objects": [geo_map[n] for n in used_names if n in geo_map],
        }
        user_message = (
            f"Evaluate the following single candidate. candidate_id: {cid}\n\n"
            f"=== Object Profile Summary (Python-derived) ===\n{object_profiles_text}\n\n"
            "=== Full Input Payload ===\n"
            + json.dumps(user_payload, ensure_ascii=False, indent=2)
        )

        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": CANDIDATE_SYSTEM_PROMPT + EMERGENCE_SCORING_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
        )

        raw_text = response.choices[0].message.content or ""
        usage = response.usage
        total_prompt_tokens += usage.prompt_tokens if usage else 0
        total_completion_tokens += usage.completion_tokens if usage else 0
        raw_previews[cid] = raw_text[:300]

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Module 2-D [{cid}]: failed to parse GPT JSON response: {e}\n"
                f"response: {raw_text[:200]}"
            )

        new_evals = _parse_evaluated_candidates([parsed])

        # Deterministic coverage check (primitive/atom matching + critical coverage gate)
        coverage_issues, coverage_summary = _check_subgoal_coverage(
            candidate,
            subgoal_constraints,
            prop_map,
            critical_primitives=critical_primitives,
        )
        coverage_summary_by_cid[cid] = coverage_summary
        if coverage_issues:
            coverage_issues_by_cid[cid] = coverage_issues

        for ev in new_evals:
            if ev.candidate_id != cid:
                continue
            for issue in coverage_issues:
                sg_id = issue.get("subgoal_id", "")
                miss_a = issue.get("missing_atoms", []) or []
                miss_p = issue.get("missing_primitives", []) or []
                parts: list[str] = []
                if miss_a:
                    parts.append(f"missing_atoms={miss_a}")
                if miss_p:
                    parts.append(f"missing_primitives={miss_p}")
                reason = (
                    f"[deterministic_subgoal_coverage] subgoal={sg_id} "
                    + ", ".join(parts)
                    + f" | candidate_functions={issue.get('candidate_functions', [])}"
                )
                ev.weak_points.append(WeakPoint(
                    stage="commonsense",
                    item="C3",
                    score=0.0,
                    reason=reason,
                ))

            critical_missing = coverage_summary.get("critical_missing_primitives", []) or []
            if critical_missing:
                ev.weak_points.append(WeakPoint(
                    stage="commonsense",
                    item="C3_critical",
                    score=0.0,
                    reason=f"critical_missing_primitives={critical_missing}",
                ))

            _apply_subgoal_coverage_policy(ev, coverage_summary)

        evaluated.extend(new_evals)

    selected_id = _select_best_candidate(
        evaluated,
        input_data.candidate_tools,
        coverage_summary_by_cid=coverage_summary_by_cid,
    )

    # 논문 §3.3 3단 필터의 순차 in/out 카운트. F1 분기 판정의 근거이며,
    # "어느 필터에서 몇 개가 떨어졌는지"를 로그로 남기기 위함이다.
    filter_counts = compute_filter_counts(
        [c.to_dict() for c in evaluated],
        rejected_for_unknown_names=rejected_for_unknown_names,
    )
    feedback = _compute_feedback_decision(evaluated, input_data, filter_counts)

    passed_count = sum(1 for c in evaluated if c.passed)
    is_fallback = selected_id is not None and passed_count == 0
    selected_passed = next(
        (c.passed for c in evaluated if c.candidate_id == selected_id), False
    ) if selected_id else False

    trace = {
        "model": model,
        "temperature": temperature,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "evaluated_count": len(evaluated),
        "passed_count": passed_count,
        "filter_counts": filter_counts,
        "feedback_branch": feedback.branch,
        "selected_candidate_id": selected_id,
        "selected_via_fallback": is_fallback,
        "selected_passed": selected_passed,
        "fallback_note": (
            "All candidates failed hard filters; selecting one best-effort candidate. "
            "Module 3 may still attempt execution, but feedback should be sent to Module 2-A."
            if is_fallback else None
        ),
        "raw_response_previews": raw_previews,
        "subgoal_coverage_issues": coverage_issues_by_cid,
        "subgoal_coverage_summary": coverage_summary_by_cid,
        "critical_primitives": sorted(critical_primitives),
        "subgoal_coverage_checked_count": len(subgoal_constraints),
        "rejected_for_unknown_names": rejected_for_unknown_names,
        "rejected_for_handle_role": rejected_for_handle_role,
        "rejected_for_hook_role": rejected_for_hook_role,
        "allowed_scene_names_count": len(allowed_scene_names),
        "emergence_margin": {
            "delta": _EMERGENCE_DELTA,
            "applied_changes": emergence_changes,
            "applied_count": len(emergence_changes),
        },
    }

    return evaluated, selected_id, feedback, trace


def _select_best_candidate(
    evaluated: list[EvaluatedCandidate],
    candidate_tools: list[dict[str, Any]],
    coverage_summary_by_cid: dict[str, dict[str, Any]] | None = None,
) -> str | None:
    """Select best candidate.

    Priority:
    1) among passed=True candidates, maximize total_score.
    Tie-break:
    a) higher coverage/emergence/necessity/role_contribution/task_fit
    b) fewer weak points
    c) fewer used objects
    Fallback: if no candidate passed, select the most recoverable best-effort candidate.
    """
    obj_count = {c["candidate_id"]: len(c.get("used_objects", [])) for c in candidate_tools}
    coverage_summary_by_cid = coverage_summary_by_cid or {}
    near_tie_margin = 0.25
    near_tie_emergence_weight = 0.2

    def _coverage_ratio(cid: str) -> float:
        summary = coverage_summary_by_cid.get(cid, {}) or {}
        try:
            return float(summary.get("coverage_complete_ratio", 0.0))
        except (TypeError, ValueError):
            return 0.0

    passed = [c for c in evaluated if c.passed]
    if passed:
        max_score = max(c.total_score for c in passed)
        top = [c for c in passed if c.total_score == max_score]
        near_top = [c for c in passed if (max_score - c.total_score) <= near_tie_margin]

        def _tie_key(c: EvaluatedCandidate) -> tuple:
            scores = c.stage_scores
            if not scores:
                return (
                    _coverage_ratio(c.candidate_id),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -len(c.weak_points),
                    -obj_count.get(c.candidate_id, 0),
                )
            return (
                _coverage_ratio(c.candidate_id),
                scores.emergence,
                scores.necessity,
                scores.role_contribution,
                scores.task_fit,
                -len(c.weak_points),
                -obj_count.get(c.candidate_id, 0),
            )

        def _near_tie_key(c: EvaluatedCandidate) -> tuple:
            scores = c.stage_scores
            if not scores:
                return (
                    c.total_score,
                    _coverage_ratio(c.candidate_id),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -len(c.weak_points),
                    -obj_count.get(c.candidate_id, 0),
                )
            emergence_boost = near_tie_emergence_weight * (scores.emergence / 5.0)
            return (
                c.total_score + emergence_boost,
                _coverage_ratio(c.candidate_id),
                scores.emergence,
                scores.necessity,
                scores.role_contribution,
                scores.task_fit,
                -len(c.weak_points),
                -obj_count.get(c.candidate_id, 0),
            )

        if len(near_top) > 1:
            return max(near_top, key=_near_tie_key).candidate_id

        return max(top, key=_tie_key).candidate_id

    # Fallback: passed=0 -> choose one best-effort candidate.
    if not evaluated:
        return None

    def _fallback_key(c: EvaluatedCandidate) -> tuple:
        env_pass = bool((c.environment_filter or {}).get("pass", False))
        asm_pass = bool((c.assembly_filter or {}).get("pass", False))
        # Priority:
        # 1) environment_filter pass
        # 2) assembly_filter pass
        # 3) higher coverage ratio
        # 4) fewer weak points
        # 5) higher total_score
        # 6) more used_objects (prefer fuller assembly for fallback only)
        return (
            int(env_pass),
            int(asm_pass),
            _coverage_ratio(c.candidate_id),
            -len(c.weak_points),
            c.total_score,
            obj_count.get(c.candidate_id, 0),
        )

    return max(evaluated, key=_fallback_key).candidate_id


_PASS_THRESHOLDS = {
    "geometry": 2.0,
    "physics": 1.5,
    "commonsense": 1.5,
    "task_fit": 1.0,  # At least one TF item should pass.
    "emergence": 2.0,
    "necessity": 2.0,
    "role_contribution": 2.0,
    "total": 3.5,
}


def _score_from_checklist(items: list[dict[str, Any]]) -> float:
    if not items:
        return 0.0
    yes = sum(1 for i in items if i.get("result") is True)
    return round(yes / len(items) * 5, 4)


def _bounded_score(value: Any, default: float = 3.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default
    return round(max(1.0, min(5.0, score)), 4)


def _normalize_emergence_analysis(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    return {
        "emergence_score": _bounded_score(raw.get("emergence_score")),
        "necessity_score": _bounded_score(raw.get("necessity_score")),
        "role_contribution_score": _bounded_score(raw.get("role_contribution_score")),
        "single_object_baseline_gap": str(raw.get("single_object_baseline_gap", "")),
        "synergy_reason": str(raw.get("synergy_reason", "")),
        "role_contribution_reason": str(raw.get("role_contribution_reason", "")),
    }


def _compute_failed_stage_from_scores(stage_scores: StageScores) -> str | None:
    g = stage_scores.geometry
    p = stage_scores.physics
    c = stage_scores.commonsense
    tf = stage_scores.task_fit
    em = stage_scores.emergence
    ne = stage_scores.necessity
    rc = stage_scores.role_contribution
    total = stage_scores.total()

    if g < _PASS_THRESHOLDS["geometry"]:
        return "geometry"
    if p < _PASS_THRESHOLDS["physics"]:
        return "physics"
    if c < _PASS_THRESHOLDS["commonsense"]:
        return "commonsense"
    if tf < _PASS_THRESHOLDS["task_fit"]:
        return "task_fit"
    if em < _PASS_THRESHOLDS["emergence"]:
        return "emergence"
    if ne < _PASS_THRESHOLDS["necessity"]:
        return "necessity"
    if rc < _PASS_THRESHOLDS["role_contribution"]:
        return "role_contribution"
    if total < _PASS_THRESHOLDS["total"]:
        return "total"
    return None


def _apply_subgoal_coverage_policy(
    ev: EvaluatedCandidate,
    coverage_summary: dict[str, Any],
) -> None:
    critical_missing = coverage_summary.get("critical_missing_primitives", []) or []
    optional_missing_prims = coverage_summary.get("optional_missing_primitives", []) or []
    missing_atoms_unique = coverage_summary.get("missing_atoms_unique", []) or []

    if critical_missing and ev.failed_stage is None:
        ev.failed_stage = "subgoal_coverage_critical"
        ev.passed = False

    # Soft penalty applies only when candidate is still potentially feasible.
    if ev.failed_stage in {None, "total", "commonsense", "task_fit"}:
        raw_penalty = (
            _COVERAGE_SOFT_PENALTY_PER_PRIMITIVE * len(optional_missing_prims)
            + _COVERAGE_SOFT_PENALTY_PER_ATOM * len(missing_atoms_unique)
        )
        penalty = min(_COVERAGE_SOFT_PENALTY_MAX, raw_penalty)
        if penalty > 0:
            ev.stage_scores.commonsense = round(max(0.0, ev.stage_scores.commonsense - penalty), 4)
            ev.stage_scores.task_fit = round(max(0.0, ev.stage_scores.task_fit - penalty * 0.5), 4)

    if ev.failed_stage != "subgoal_coverage_critical":
        failed_stage = _compute_failed_stage_from_scores(ev.stage_scores)
        ev.failed_stage = failed_stage
        ev.passed = failed_stage is None

    ev.total_score = ev.stage_scores.total()

def _parse_evaluated_candidates(raw: list[dict[str, Any]]) -> list[EvaluatedCandidate]:
    result = []
    for item in raw:
        env_filter = item.get("environment_filter") or {}
        asm_filter = item.get("assembly_filter") or {}

        # Hard filter failures bypass quantitative scoring
        env_failed = not bool(env_filter.get("pass", True))
        asm_failed = not bool(asm_filter.get("pass", True))

        checklist = item.get("checklist") or {}
        geo_items = checklist.get("geometry", [])
        phys_items = checklist.get("physics", [])
        comm_items = checklist.get("commonsense", [])
        tf_items = checklist.get("task_fit", [])  # LLM-provided task-type fit checklist.
        emergence_analysis = _normalize_emergence_analysis(
            item.get("emergence_analysis")
        )
        em = _bounded_score(emergence_analysis.get("emergence_score"))
        ne = _bounded_score(emergence_analysis.get("necessity_score"))
        rc = _bounded_score(emergence_analysis.get("role_contribution_score"))

        if env_failed or asm_failed:
            stage_scores = StageScores(
                geometry=0.0,
                physics=0.0,
                commonsense=0.0,
                task_fit=0.0,
                emergence=0.0,
                necessity=0.0,
                role_contribution=0.0,
            )
            passed = False
            failed_stage = "environment" if env_failed else "assembly"
        else:
            # Compute scores from checklist (Python-controlled, not GPT)
            g = _score_from_checklist(geo_items)
            p = _score_from_checklist(phys_items)
            c = _score_from_checklist(comm_items)
            tf = _score_from_checklist(tf_items) if tf_items else 0.0
            stage_scores = StageScores(
                geometry=g,
                physics=p,
                commonsense=c,
                task_fit=tf,
                emergence=em,
                necessity=ne,
                role_contribution=rc,
            )
            failed_stage = _compute_failed_stage_from_scores(stage_scores)
            passed = failed_stage is None

        # Extract weak_points from checklist false items (score always 0.0)
        weak_points: list[WeakPoint] = []
        for stage_name, stage_items in [
            ("geometry", geo_items),
            ("physics", phys_items),
            ("commonsense", comm_items),
            ("task_fit", tf_items),  # NEW
        ]:
            for ci in stage_items:
                if ci.get("result") is False:
                    weak_points.append(WeakPoint(
                        stage=stage_name,
                        item=ci.get("item", ""),
                        score=0.0,
                        reason=ci.get("false_reason") or "",
                    ))

        for stage_name, item_name, score_key, reason_key in [
            ("emergence", "EM1", "emergence_score", "synergy_reason"),
            ("necessity", "NE1", "necessity_score", "single_object_baseline_gap"),
            ("role_contribution", "RC1", "role_contribution_score", "role_contribution_reason"),
        ]:
            score = _bounded_score(emergence_analysis.get(score_key))
            threshold = _PASS_THRESHOLDS[stage_name]
            if score < threshold:
                weak_points.append(WeakPoint(
                    stage=stage_name,
                    item=item_name,
                    score=score,
                    reason=str(emergence_analysis.get(reason_key, "")),
                ))

        ra = item.get("repair_analysis") or {}
        repair_analysis = RepairAnalysis(
            repair_type=ra.get("repair_type", "none"),
            target_issue=ra.get("target_issue", ""),
            reason=ra.get("reason", ""),
            repair_strategy=ra.get("repair_strategy", []),
        )
        result.append(EvaluatedCandidate(
            candidate_id=item["candidate_id"],
            stage_scores=stage_scores,
            total_score=stage_scores.total(),
            passed=passed,
            failed_stage=failed_stage,
            weak_points=weak_points,
            repair_analysis=repair_analysis,
            environment_filter=item.get("environment_filter"),
            assembly_filter=item.get("assembly_filter"),
            checklist=checklist,
            pre_analysis=item.get("pre_analysis"),
            emergence_analysis=emergence_analysis,
        ))
    return result


def _compute_feedback_decision(
    evaluated: list[EvaluatedCandidate],
    input_data: Module2DInput,
    filter_counts: dict[str, Any] | None = None,
) -> FeedbackDecision:
    """Determine feedback target using rule-based logic across all evaluated candidates.

    논문 §3.3 F1의 두 갈래를 branch 필드로 명시한다:
    - env_constraint_wipeout: 환경 제약 필터(1단)에서 후보 전부 탈락
        → 상위 수치 제약 생성 단계(module2b)에 완화 요청
    - other: 그 외 통과 후보 없음
        → 후보 재생성(module2c) + 직전 추론 단계 피드백
    분기 판정은 FeedbackController와 동일하게 filter_counts에 의존한다.
    """
    from collections import Counter

    filter_counts = filter_counts or {}
    env_fc = filter_counts.get("env_constraint", {}) or {}
    env_in = int(env_fc.get("in", 0) or 0)
    env_out = int(env_fc.get("out", 0) or 0)
    env_wipeout = env_in > 0 and env_out == env_in

    passed = [c for c in evaluated if c.passed]
    if passed:
        return FeedbackDecision(
            need_feedback=False,
            feedback_target=None,
            reason=f"{len(passed)}/{len(evaluated)} candidates passed.",
            dominant_failure_pattern=[],
            suggested_relaxations=[],
            branch=None,
        )

    # Tally failure stages
    stage_counter: Counter[str] = Counter()
    for c in evaluated:
        if c.failed_stage:
            stage_counter[c.failed_stage] += 1

    # Tally weak point items across all candidates
    item_counter: Counter[str] = Counter()
    for c in evaluated:
        for wp in c.weak_points:
            item_counter[wp.item] += 1

    dominant = [item for item, cnt in item_counter.most_common(3) if cnt >= 2]
    dominant_stages = [s for s, cnt in stage_counter.most_common(3)]

    # Determine feedback target — 논문 §3.3 두 갈래.
    # env_constraint_wipeout: 1단 환경 제약 필터에서 후보 전부 탈락
    #   → 상위 수치 제약 생성 단계(module2b)에 완화 요청.
    # filter_counts가 비어 있을 때를 대비해 environment_filter로도 폴백 판정한다.
    all_env_failed = env_wipeout or (
        bool(evaluated)
        and all(not (c.environment_filter or {}).get("pass", True) for c in evaluated)
    )
    c3_failures = sum(
        1 for c in evaluated
        for wp in c.weak_points if wp.item == "C3"
    )

    if all_env_failed:
        branch = "env_constraint_wipeout"
        feedback_target = "module2b"
        reason = (
            "환경 제약 필터에서 후보가 일괄 탈락했다. 상위 수치 제약 생성 단계"
            "(module2b)에 제약 완화를 요청한다."
        )
        relaxations = ["relax_numeric_constraints", "relax_derived_constraint_ranges"]
    elif c3_failures >= len(evaluated) // 2:
        branch = "other"
        feedback_target = "module2a"
        reason = "Many candidates missed required_atoms/interaction_primitives coverage."
        relaxations = ["downgrade_some_required_atoms_to_preferred_atoms"]
    else:
        branch = "other"
        feedback_target = "module2c"
        reason = (
            f"Dominant failure stages: {', '.join(dominant_stages)}. "
            "Candidate composition should be improved in Module 2-C."
        )
        relaxations = ["generate_candidates_with_diverse_connection_modes"]

    return FeedbackDecision(
        need_feedback=True,
        feedback_target=feedback_target,
        reason=reason,
        dominant_failure_pattern=dominant or dominant_stages,
        suggested_relaxations=relaxations,
        branch=branch,
    )

