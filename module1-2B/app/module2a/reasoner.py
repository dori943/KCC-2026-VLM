"""Deterministic Module 2-A reasoner (task decomposition + requirement extraction)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


AVAILABILITY_LABEL_BY_CODE = {
    0: "none",
    1: "trace",
    2: "sparse",
    3: "present",
    4: "strong",
    5: "abundant",
}

COVERAGE_LABEL_BY_CODE = {
    0: "blocked",
    1: "very_weak",
    2: "weak",
    3: "usable",
    4: "strong",
    5: "robust",
}

SUFFICIENCY_LABEL_BY_CODE = {
    0: "insufficient",
    1: "very_limited",
    2: "partial",
    3: "usable",
    4: "strong",
    5: "sufficient",
}


@dataclass
class SubgoalSpec:
    """Static pattern for one subgoal template."""

    name: str
    objective: str
    success_condition: str
    primitives: list[str]
    required_atom_candidates: list[list[str]]
    preferred_atoms: list[str]
    risk_atoms: list[str]
    physical_rationale: str
    failure_risks: list[str]


@dataclass(frozen=True)
class TaskProfile:
    """Task-level constraints inferred from the brief and notes."""

    narrow_gap: bool = False
    exposed_edge_only: bool = False
    low_force_bias: bool = False
    avoid_sharp_contact: bool = False
    avoid_excessive_compression: bool = False
    fragile_sheet_target: bool = False
    preserve_surrounding_surface: bool = False


def generate_module2a_output(
    module2_common_input: dict[str, Any], vocab_registry: dict[str, Any]
) -> dict[str, Any]:
    """Generate strict Module 2-A output from module2_common_input."""
    task_brief = module2_common_input.get("task_brief", {})
    scene_resources = module2_common_input.get("scene_resources", {})
    resource_summary = scene_resources.get("resource_summary", {})
    histogram_raw = resource_summary.get("affordance_histogram", {})
    inventory = scene_resources.get("resource_inventory", [])

    affordance_histogram = {
        str(k): max(0, int(v)) for k, v in histogram_raw.items() if isinstance(v, (int, float))
    }
    interaction_vocab: dict[str, int] = vocab_registry["module2a"]["interaction_primitives"]
    atom_vocab: dict[str, int] = vocab_registry["module2a"]["affordance_atoms"]
    risk_vocab: dict[str, int] = vocab_registry["module2a"]["risk_atoms"]

    user_goal = task_brief.get("user_goal")
    goal_text = str(user_goal).strip() if user_goal else ""
    success_criteria = [str(item) for item in task_brief.get("success_criteria", [])]
    task_notes = [str(item) for item in task_brief.get("task_notes", [])]

    task_profile = _build_task_profile(
        goal_text=goal_text,
        success_criteria=success_criteria,
        task_notes=task_notes,
    )
    intent = _infer_intent(
        goal_text=goal_text,
        success_criteria=success_criteria,
        task_notes=task_notes,
        task_profile=task_profile,
    )
    subgoal_specs = _build_subgoal_specs(intent=intent, task_profile=task_profile)
    atom_stats = _build_atom_stats(
        atom_vocab=atom_vocab,
        affordance_histogram=affordance_histogram,
        inventory=inventory,
    )

    subgoals: list[dict[str, Any]] = []
    required_atoms_union: list[str] = []
    preferred_atoms_union: list[str] = []
    risk_atoms_union: list[str] = []
    required_primitives_union: list[str] = []
    gap_hypotheses: list[str] = []
    subgoal_coverage_codes: list[int] = []
    subgoal_uncertainty_scores: list[int] = []

    for idx, spec in enumerate(subgoal_specs, start=1):
        subgoal_id = f"sg_{idx:02d}"
        depends_on = [f"sg_{idx - 1:02d}"] if idx > 1 else []
        depends_on_indices = [idx - 1] if idx > 1 else []

        required_atoms = _resolve_required_atoms(spec.required_atom_candidates, atom_stats)
        preferred_atoms = _dedupe_keep_order([a for a in spec.preferred_atoms if a in atom_vocab])
        risk_atoms = _dedupe_keep_order([a for a in spec.risk_atoms if a in risk_vocab])
        primitives = _dedupe_keep_order([p for p in spec.primitives if p in interaction_vocab])

        final_availability_codes = [atom_stats.get(atom, _empty_atom_stat()).get("availability_code", 0) for atom in required_atoms]
        required_uncertainties = [atom_stats.get(atom, _empty_atom_stat()).get("uncertainty_score", 5) for atom in required_atoms]
        uncertainty_score = max(required_uncertainties) if required_uncertainties else 5
        coverage_code = _coverage_code(required_codes=final_availability_codes, uncertainty_score=uncertainty_score)
        coverage_label = COVERAGE_LABEL_BY_CODE[coverage_code]
        supporting_atoms = [
            atom for atom in required_atoms if atom_stats.get(atom, _empty_atom_stat())["scene_count"] > 0
        ]
        gap_atoms = [
            atom for atom in required_atoms if atom_stats.get(atom, _empty_atom_stat())["availability_code"] == 0
        ]
        weak_support_atoms = [
            atom
            for atom in required_atoms
            if atom_stats.get(atom, _empty_atom_stat())["availability_code"] in (1, 2)
        ]

        required_codes = [interaction_vocab[p] for p in primitives]
        required_atom_codes = [atom_vocab[a] for a in required_atoms]
        preferred_atom_codes = [atom_vocab[a] for a in preferred_atoms]
        risk_atom_codes = [risk_vocab[a] for a in risk_atoms]
        supporting_atom_codes = [atom_vocab[a] for a in supporting_atoms]
        gap_atom_codes = [atom_vocab[a] for a in gap_atoms]

        subgoal = {
            "subgoal_id": subgoal_id,
            "subgoal_name": spec.name,
            "objective": spec.objective,
            "success_condition": spec.success_condition,
            "depends_on": depends_on,
            "required_interaction_primitives": primitives,
            "function_requirements": {
                "required_atoms": required_atoms,
                "preferred_atoms": preferred_atoms,
                "risk_atoms_to_avoid": risk_atoms,
            },
            "physical_rationale": spec.physical_rationale,
            "resource_feasibility_hint": {
                "coverage_status": coverage_label,
                "supporting_atoms_seen_in_scene": supporting_atoms,
                "gap_atoms": gap_atoms,
            },
            "failure_risks": spec.failure_risks,
            "pybullet_bridge": {
                "subgoal_index": idx,
                "depends_on_indices": depends_on_indices,
                "required_interaction_primitive_codes": required_codes,
                "required_atom_codes": required_atom_codes,
                "preferred_atom_codes": preferred_atom_codes,
                "risk_atom_codes_to_avoid": risk_atom_codes,
                "supporting_atom_codes_seen_in_scene": supporting_atom_codes,
                "gap_atom_codes": gap_atom_codes,
                "coverage_status_code": coverage_code,
                "uncertainty_score": uncertainty_score,
                "required_interaction_primitive_count": len(required_codes),
                "required_atom_count": len(required_atom_codes),
                "preferred_atom_count": len(preferred_atom_codes),
                "risk_atom_count": len(risk_atom_codes),
                "required_atom_missing_count": len(gap_atom_codes),
                "has_blocking_gap_flag": 1 if coverage_code == 0 else 0,
            },
        }
        subgoals.append(subgoal)

        required_atoms_union.extend(required_atoms)
        preferred_atoms_union.extend(preferred_atoms)
        risk_atoms_union.extend(risk_atoms)
        required_primitives_union.extend(primitives)
        subgoal_coverage_codes.append(coverage_code)
        subgoal_uncertainty_scores.append(uncertainty_score)
        for atom in gap_atoms:
            gap_hypotheses.append(
                f"Capability shortage in {atom} for subgoal {subgoal_id}."
            )
        for atom in weak_support_atoms:
            if atom in gap_atoms:
                continue
            support_label = AVAILABILITY_LABEL_BY_CODE[
                atom_stats.get(atom, _empty_atom_stat())["availability_code"]
            ]
            gap_hypotheses.append(
                f"Only {support_label} support for {atom} in subgoal {subgoal_id}; execution may remain fragile."
            )

    required_atoms_union = _dedupe_keep_order(required_atoms_union)
    preferred_atoms_union = _dedupe_keep_order(preferred_atoms_union)
    risk_atoms_union = _dedupe_keep_order(risk_atoms_union)
    required_primitives_union = _dedupe_keep_order(required_primitives_union)
    gap_hypotheses = _dedupe_keep_order(gap_hypotheses)

    overall_sufficiency_code = _overall_sufficiency_code(subgoal_coverage_codes)
    overall_uncertainty_score = max(subgoal_uncertainty_scores) if subgoal_uncertainty_scores else 5

    available_affordance_atoms = []
    for atom, stats in atom_stats.items():
        if stats["scene_count"] <= 0:
            continue
        available_affordance_atoms.append(
            {
                "atom": atom,
                "availability": AVAILABILITY_LABEL_BY_CODE[stats["availability_code"]],
                "pybullet_bridge": {
                    "atom_code": atom_vocab[atom],
                    "scene_count": stats["scene_count"],
                    "raw_availability_code": stats["raw_availability_code"],
                    "uncertainty_score": stats["uncertainty_score"],
                    "uncertainty_penalty": stats["uncertainty_penalty"],
                    "availability_code": stats["availability_code"],
                },
            }
        )

    high_uncertainty_areas = [
        atom
        for atom, stats in atom_stats.items()
        if stats["scene_count"] > 0 and stats["uncertainty_score"] >= 4
    ]
    high_risk_areas = _high_risk_areas(scene_resources=scene_resources)
    if task_profile.avoid_sharp_contact and "sharp_contact_risk" in high_risk_areas:
        gap_hypotheses.append(
            "Scene includes sharp-contact risk that conflicts with the requested low-damage extraction path."
        )
    if task_profile.fragile_sheet_target and "break_prone" in high_risk_areas:
        gap_hypotheses.append(
            "Scene includes break-prone capabilities that can destabilize thin-sheet extraction."
        )
    gap_hypotheses = _dedupe_keep_order(gap_hypotheses)

    return {
        "schema_name": "module2a_output",
        "schema_version": "0.2",
        "stage": "task_decomposition_and_function_requirement_extraction",
        "task_model": {
            "task_restatement": _task_restatement(goal_text),
            "primary_success_condition": success_criteria[0] if success_criteria else _primary_success(goal_text),
            "secondary_success_conditions": success_criteria[1:] if len(success_criteria) > 1 else [],
            "decomposition_principle": _decomposition_principle(
                intent=intent,
                task_profile=task_profile,
            ),
            "assumptions": _task_assumptions(
                goal_text=goal_text,
                task_notes=task_notes,
                task_profile=task_profile,
            ),
            "deferred_items": {
                "needs_target_binding_later": True,
                "needs_environment_constraint_binding_later": True,
                "notes": [
                    "Target instance binding is deferred to downstream stage.",
                    "Environment constraints are deferred to downstream stage."
                ],
            },
        },
        "subgoals": subgoals,
        "task_level_requirement_summary": {
            "required_atoms_union": required_atoms_union,
            "preferred_atoms_union": preferred_atoms_union,
            "risk_atoms_to_avoid_union": risk_atoms_union,
            "required_interaction_primitives_union": required_primitives_union,
            "resource_gap_hypotheses": gap_hypotheses,
            "overall_resource_sufficiency": SUFFICIENCY_LABEL_BY_CODE[overall_sufficiency_code],
            "pybullet_bridge": {
                "required_atoms_union_codes": sorted(atom_vocab[a] for a in required_atoms_union),
                "preferred_atoms_union_codes": sorted(atom_vocab[a] for a in preferred_atoms_union),
                "risk_atoms_to_avoid_union_codes": sorted(risk_vocab[a] for a in risk_atoms_union),
                "required_interaction_primitives_union_codes": sorted(
                    interaction_vocab[p] for p in required_primitives_union
                ),
                "resource_gap_atom_codes": sorted(
                    {
                        atom_vocab[atom]
                        for atom in required_atoms_union
                        if atom_stats.get(atom, _empty_atom_stat())["availability_code"] == 0
                    }
                ),
                "overall_resource_sufficiency_code": overall_sufficiency_code,
                "overall_uncertainty_score": overall_uncertainty_score,
                "required_atoms_union_count": len(required_atoms_union),
                "preferred_atoms_union_count": len(preferred_atoms_union),
                "risk_atoms_to_avoid_union_count": len(risk_atoms_union),
                "required_interaction_primitives_union_count": len(required_primitives_union),
                "blocked_subgoal_count": sum(1 for code in subgoal_coverage_codes if code == 0),
                "weak_or_worse_subgoal_count": sum(1 for code in subgoal_coverage_codes if code <= 2),
                "usable_or_better_subgoal_count": sum(1 for code in subgoal_coverage_codes if code >= 3),
            },
        },
        "scene_resource_readout": {
            "available_affordance_atoms": available_affordance_atoms,
            "high_risk_capability_areas": high_risk_areas,
            "high_uncertainty_capability_areas": high_uncertainty_areas,
            "pybullet_bridge": {
                "available_atom_codes": sorted(
                    atom_vocab[item["atom"]] for item in available_affordance_atoms
                ),
                "available_atom_count": len(available_affordance_atoms),
                "high_risk_area_count": len(high_risk_areas),
                "high_uncertainty_area_count": len(high_uncertainty_areas),
            },
        },
        "pybullet_bridge": {
            "format_version": 2,
            "subgoal_count": len(subgoals),
            "needs_target_binding_later_flag": 1,
            "needs_environment_constraint_binding_later_flag": 1,
            "task_target_is_tool_flag": 0,
            "task_target_is_state_change_object_flag": 1,
        },
    }


def _build_task_profile(
    goal_text: str,
    success_criteria: list[str],
    task_notes: list[str],
) -> TaskProfile:
    merged = " ".join([goal_text] + success_criteria + task_notes).lower()

    def has_any(keywords: list[str]) -> bool:
        return any(keyword in merged for keyword in keywords)

    return TaskProfile(
        narrow_gap=has_any(
            [
                "narrow gap",
                "tight gap",
                "small gap",
                "slot",
                "drawer slot",
                "좁은 틈",
                "틈",
            ]
        ),
        exposed_edge_only=has_any(
            [
                "exposed edge",
                "small edge",
                "only a small edge",
                "edge is exposed",
                "small edge is exposed",
                "only a small edge of the card is exposed",
            ]
        ),
        low_force_bias=has_any(
            [
                "low-force",
                "low force",
                "gentle",
                "minimize force",
                "prefer low-force",
                "avoid excessive force",
            ]
        ),
        avoid_sharp_contact=has_any(
            [
                "avoid sharp",
                "avoid sharp contact",
                "no sharp contact",
                "surface is not damaged",
                "surrounding surface is not damaged",
                "avoid surface damage",
                "avoid gouging",
                "avoid scratching",
            ]
        ),
        avoid_excessive_compression=has_any(
            [
                "avoid excessive compression",
                "without bending or tearing",
                "not bent or torn",
                "do not bend",
                "do not tear",
                "avoid bending",
                "avoid tearing",
                "avoid crease",
            ]
        ),
        fragile_sheet_target=has_any(
            [
                "business card",
                "card",
                "paper",
                "sheet",
                "document",
                "crease",
                "tear",
                "bent",
            ]
        ),
        preserve_surrounding_surface=has_any(
            [
                "surrounding surface",
                "surface is not damaged",
                "avoid surface damage",
                "avoid scratching",
                "avoid gouging",
                "avoid marking",
            ]
        ),
    )


def _infer_intent(
    goal_text: str,
    success_criteria: list[str],
    task_notes: list[str],
    task_profile: TaskProfile,
) -> str:
    merged = " ".join([goal_text] + success_criteria + task_notes).lower()
    if not merged.strip():
        return "generic_state_change"

    if (
        task_profile.narrow_gap
        and task_profile.fragile_sheet_target
        and any(keyword in merged for keyword in ["extract", "remove", "pull out", "retrieve", "꺼내"])
    ):
        return "extract_from_narrow_gap"

    intent_keywords: list[tuple[str, list[str]]] = [
        ("transfer_liquid", ["pour", "funnel", "transfer liquid", "붓", "따르", "옮겨 담"]),
        ("cut_or_separate", ["cut", "slice", "trim", "separate", "자르", "절단", "분리"]),
        ("insert_or_thread", ["insert", "thread", "fit into", "끼우", "삽입", "관통"]),
        ("hang_or_hook", ["hang", "hook", "suspend", "걸", "매달"]),
        ("stack_or_place", ["stack", "place on top", "쌓", "올려놓"]),
        ("pry_or_wedge", ["pry", "wedge", "lever", "벌리", "지렛"]),
        ("cover_or_block", ["cover", "block", "shield", "덮", "막"]),
    ]
    for intent, keywords in intent_keywords:
        if any(keyword in merged for keyword in keywords):
            return intent
    return "generic_state_change"


def _build_subgoal_specs(intent: str, task_profile: TaskProfile) -> list[SubgoalSpec]:
    if intent == "extract_from_narrow_gap":
        low_profile_preferred = ["edge_followable", "elongated_reach", "frictional_contact"]
        if not task_profile.avoid_sharp_contact:
            low_profile_preferred.append("rigid_tip_contact")

        purchase_preferred = ["frictional_contact", "clampable_span", "edge_followable"]
        if task_profile.avoid_excessive_compression:
            purchase_preferred.append("compliant_pad")
        else:
            purchase_preferred.append("graspable_body")

        extraction_preferred = ["frictional_contact", "elongated_reach", "stable_support_face"]
        if task_profile.low_force_bias:
            extraction_preferred.append("edge_followable")
        else:
            extraction_preferred.append("graspable_body")

        return [
            SubgoalSpec(
                name="establish_low_profile_entry",
                objective="Establish a low-profile entry contact at the exposed card edge without gouging the surrounding slot surface.",
                success_condition="A low-profile engagement reaches the exposed edge with enough purchase to begin extraction while keeping local damage bounded.",
                primitives=["reach", "orient", "probe", "contact", "insert", "guide"],
                required_atom_candidates=[["thin_insertable"], ["edge_followable"]],
                preferred_atoms=low_profile_preferred,
                risk_atoms=["sharp_contact_risk", "deform_prone"],
                physical_rationale="Narrow-gap extraction begins with a thin, guided entry that can access the exposed edge before larger pulling forces are applied.",
                failure_risks=["Entry contact scratches the surrounding slot surface.", "The exposed card edge crumples before purchase is established."],
            ),
            SubgoalSpec(
                name="secure_extraction_purchase",
                objective="Create controllable tangential purchase on the exposed card edge so extraction can begin without local tearing.",
                success_condition="The exposed edge can sustain controlled pull initiation without slip or visible crease growth.",
                primitives=["contact", "align", "pull", "stabilize"],
                required_atom_candidates=[["frictional_contact"], ["clampable_span"]],
                preferred_atoms=purchase_preferred,
                risk_atoms=["slip_prone", "deform_prone"],
                physical_rationale="Once entry is achieved, extraction depends on stable tangential traction that does not over-compress the thin target edge.",
                failure_risks=["Tangential contact slips off the exposed edge.", "Local compression initiates a bend or tear at the extraction point."],
            ),
            SubgoalSpec(
                name="extract_card_incrementally",
                objective="Draw the card out of the narrow gap in controlled increments until it can be removed cleanly.",
                success_condition="The card exits the gap without permanent bend, tear, or surrounding surface damage.",
                primitives=["pull", "slide", "guide", "stabilize"],
                required_atom_candidates=[["frictional_contact"], ["elongated_reach"]],
                preferred_atoms=extraction_preferred,
                risk_atoms=["deform_prone", "sharp_contact_risk", "slip_prone"],
                physical_rationale="Incremental extraction reduces peak bending and allows the target to transition from constrained sliding to free removal under bounded force.",
                failure_risks=["Extraction stalls and turns into uncontrolled jerk motion.", "Contact concentration causes a permanent crease or edge tear during pull-out."],
            ),
        ]

    if intent == "transfer_liquid":
        return [
            SubgoalSpec(
                name="prepare_flow_path",
                objective="Prepare an oriented transfer path from source opening to target opening.",
                success_condition="A stable and directed flow geometry is established at the target interface.",
                primitives=["reach", "approach", "orient", "contact", "align"],
                required_atom_candidates=[
                    ["concave_containment", "pour_channel"],
                    ["pour_channel"],
                ],
                preferred_atoms=["frictional_contact", "elongated_reach", "graspable_body"],
                risk_atoms=["slip_prone", "placement_unstable"],
                physical_rationale="Fluid transfer needs cavity-guided flow geometry and controlled orientation to reduce spill probability.",
                failure_risks=["Flow breakup at rim transition.", "Loss of orientation control during transfer."],
            ),
            SubgoalSpec(
                name="execute_controlled_transfer",
                objective="Drive the target state transition by controlled pouring and stabilization.",
                success_condition="The transfer reaches the intended amount while keeping the target state stable.",
                primitives=["lift", "orient", "pour", "lower", "stabilize"],
                required_atom_candidates=[["pour_channel"], ["concave_containment"]],
                preferred_atoms=["frictional_contact", "graspable_body"],
                risk_atoms=["slip_prone", "break_prone"],
                physical_rationale="Continuous transfer requires stable handoff of momentum and friction-consistent control.",
                failure_risks=["Sudden slip causes over-pour.", "Brittle contact zone fractures under transient load."],
            ),
        ]

    if intent == "cut_or_separate":
        return [
            SubgoalSpec(
                name="establish_separation_interface",
                objective="Create and stabilize an engagement interface that can initiate material separation.",
                success_condition="The target interface is engaged with sufficient geometric concentration for separation onset.",
                primitives=["reach", "approach", "orient", "contact", "clamp"],
                required_atom_candidates=[
                    ["prying_edge"],
                    ["thin_insertable"],
                    ["rigid_tip_contact"],
                ],
                preferred_atoms=["graspable_body", "clampable_span", "frictional_contact"],
                risk_atoms=["sharp_contact_risk", "slip_prone"],
                physical_rationale="Separation onset depends on local stress concentration and stable normal/tangential control.",
                failure_risks=["Edge skids off the contact line.", "Uncontrolled sharp contact causes unsafe collision."],
            ),
            SubgoalSpec(
                name="propagate_separation",
                objective="Propagate the target split until the requested state change is reached.",
                success_condition="The target transitions to a separated state without uncontrolled fracture spread.",
                primitives=["press", "pull", "separate"],
                required_atom_candidates=[["prying_edge"], ["thin_insertable"]],
                preferred_atoms=["frictional_contact", "rigid_tip_contact"],
                risk_atoms=["break_prone", "sharp_contact_risk"],
                physical_rationale="Separation propagation requires sustained directional force with controlled contact progression.",
                failure_risks=["Excess force causes brittle failure.", "Contact escapes due to insufficient friction."],
            ),
        ]

    if intent == "insert_or_thread":
        return [
            SubgoalSpec(
                name="align_insertion_axis",
                objective="Align the moving geometry with the target entry axis before insertion.",
                success_condition="Relative pose error is reduced so axial insertion can begin without jamming.",
                primitives=["reach", "approach", "orient", "align", "contact"],
                required_atom_candidates=[["axis_insertable"], ["thin_insertable"]],
                preferred_atoms=["elongated_reach", "edge_followable", "frictional_contact"],
                risk_atoms=["slip_prone", "placement_unstable"],
                physical_rationale="Insertion success is sensitive to angular error and edge guidance at first contact.",
                failure_risks=["Misalignment creates jam.", "Low friction causes lateral drift at entry."],
            ),
            SubgoalSpec(
                name="advance_and_seat",
                objective="Advance insertion until the target reaches the required seated state.",
                success_condition="The target reaches seated depth and remains stable under nominal disturbance.",
                primitives=["insert", "guide", "press", "hold"],
                required_atom_candidates=[["axis_insertable"], ["thin_insertable"]],
                preferred_atoms=["frictional_contact", "stable_support_face"],
                risk_atoms=["break_prone", "deform_prone"],
                physical_rationale="Progressive insertion requires guided contact and bounded force escalation.",
                failure_risks=["Over-constraint causes buckle or crack.", "Compliance mismatch causes permanent deformation."],
            ),
        ]

    if intent == "hang_or_hook":
        return [
            SubgoalSpec(
                name="capture_support_edge",
                objective="Engage the support feature that can transition the target to a suspended state.",
                success_condition="A mechanically valid hanging capture is established on the target support feature.",
                primitives=["reach", "approach", "orient", "hook", "contact"],
                required_atom_candidates=[["hookable_gap"], ["hangable_edge"]],
                preferred_atoms=["elongated_reach", "frictional_contact", "graspable_body"],
                risk_atoms=["slip_prone", "placement_unstable"],
                physical_rationale="Suspension initiation needs a stable geometric catch and enough tangential resistance.",
                failure_risks=["Hook misses or slips during capture.", "Target oscillation destabilizes initial hang."],
            ),
            SubgoalSpec(
                name="stabilize_suspended_state",
                objective="Release and settle the target into the requested stable suspended configuration.",
                success_condition="The target remains suspended without immediate detachment or unstable swing growth.",
                primitives=["hang", "release", "stabilize"],
                required_atom_candidates=[["hangable_edge"], ["hookable_gap"]],
                preferred_atoms=["stable_support_face", "inertial_anchor"],
                risk_atoms=["placement_unstable", "break_prone"],
                physical_rationale="Suspended equilibrium depends on support geometry, inertia, and disturbance damping.",
                failure_risks=["Detachment due to insufficient retention.", "Support fracture from concentrated load."],
            ),
        ]

    if intent == "stack_or_place":
        return [
            SubgoalSpec(
                name="prepare_stable_placement_pair",
                objective="Establish mating geometry so the target can be placed in the intended stacked state.",
                success_condition="Placement surfaces are aligned to support stable target stacking.",
                primitives=["reach", "approach", "orient", "align", "contact"],
                required_atom_candidates=[
                    ["stable_support_face", "stackable_top"],
                    ["flat_mating_face", "stable_support_face"],
                ],
                preferred_atoms=["frictional_contact", "graspable_body"],
                risk_atoms=["placement_unstable", "slip_prone"],
                physical_rationale="Stable placement requires supportable flat interfaces and bounded slip tendency.",
                failure_risks=["Placement settles into unstable tilt.", "Low friction initiates post-release slide."],
            ),
            SubgoalSpec(
                name="place_and_verify_stability",
                objective="Complete placement and maintain the target in the requested final stable state.",
                success_condition="Target remains placed with no immediate topple, roll, or gross displacement.",
                primitives=["lift", "place", "release", "stabilize"],
                required_atom_candidates=[["stable_support_face"], ["stackable_top"]],
                preferred_atoms=["inertial_anchor", "frictional_contact"],
                risk_atoms=["placement_unstable", "break_prone"],
                physical_rationale="Final stability depends on support geometry, inertia, and perturbation tolerance.",
                failure_risks=["Residual momentum causes topple.", "Fragile support feature fails under load."],
            ),
        ]

    if intent == "pry_or_wedge":
        return [
            SubgoalSpec(
                name="insert_leverage_interface",
                objective="Create an initial wedge insertion that enables controlled leverage on the target interface.",
                success_condition="A leverage-capable engagement is seated without immediate slip-out.",
                primitives=["reach", "orient", "contact", "insert", "wedge"],
                required_atom_candidates=[["prying_edge"], ["thin_insertable"]],
                preferred_atoms=["rigid_tip_contact", "elongated_reach"],
                risk_atoms=["sharp_contact_risk", "break_prone"],
                physical_rationale="Leverage initiation needs thin engagement geometry with enough stiffness retention.",
                failure_risks=["Edge chips at insertion point.", "Unsafe sharp contact under misalignment."],
            ),
            SubgoalSpec(
                name="apply_controlled_leverage",
                objective="Apply leverage until the target achieves the intended separated or opened state.",
                success_condition="Target state transition completes with controlled force growth and no abrupt recoil.",
                primitives=["pry", "pull", "separate", "stabilize"],
                required_atom_candidates=[["prying_edge"], ["rigid_tip_contact"]],
                preferred_atoms=["inertial_anchor", "frictional_contact"],
                risk_atoms=["slip_prone", "break_prone"],
                physical_rationale="Leverage propagation depends on contact persistence and reaction-force control.",
                failure_risks=["Slip-out induces sudden recoil.", "Excess force causes brittle crack propagation."],
            ),
        ]

    if intent == "cover_or_block":
        return [
            SubgoalSpec(
                name="align_coverage_interface",
                objective="Align a coverage geometry with the target area that needs shielding or blocking.",
                success_condition="Coverage geometry overlaps the intended target region with acceptable margin.",
                primitives=["reach", "approach", "orient", "align", "contact"],
                required_atom_candidates=[["cover_area"], ["flat_mating_face"]],
                preferred_atoms=["stable_support_face", "frictional_contact"],
                risk_atoms=["placement_unstable", "deform_prone"],
                physical_rationale="Coverage tasks require area overlap and stable mating at the contact interface.",
                failure_risks=["Coverage shifts due to low contact stability.", "Compliant cover buckles under load."],
            ),
            SubgoalSpec(
                name="place_and_secure_cover",
                objective="Settle the cover/block state so the target remains in the requested protected configuration.",
                success_condition="Coverage remains in place and preserves the intended blocked or shielded state.",
                primitives=["place", "press", "hold", "release"],
                required_atom_candidates=[["cover_area"], ["stable_support_face"]],
                preferred_atoms=["inertial_anchor", "frictional_contact"],
                risk_atoms=["slip_prone", "placement_unstable"],
                physical_rationale="Retention quality is governed by contact area, friction, and disturbance rejection.",
                failure_risks=["Cover drifts after release.", "Insufficient normal support creates collapse."],
            ),
        ]

    return [
        SubgoalSpec(
            name="establish_controlled_contact",
            objective="Establish controllable contact with the target interface that must change state.",
            success_condition="A repeatable and stable contact orientation is achieved on the target interface.",
            primitives=["reach", "approach", "orient", "contact", "grasp"],
            required_atom_candidates=[
                ["graspable_body"],
                ["frictional_contact"],
                ["elongated_reach"],
            ],
            preferred_atoms=["clampable_span", "stable_support_face"],
            risk_atoms=["slip_prone", "sharp_contact_risk"],
            physical_rationale="Any physical state transition first requires a controllable contact manifold.",
            failure_risks=["Insufficient grip authority causes drift.", "Unsafe contact concentration appears during approach."],
        ),
        SubgoalSpec(
            name="apply_state_transition",
            objective="Apply physical action that moves the target toward the requested final state.",
            success_condition="Target state change is initiated and remains controllable without immediate reversal.",
            primitives=["align", "press", "push", "stabilize", "release"],
            required_atom_candidates=[
                ["broad_push_face"],
                ["flat_mating_face"],
                ["rigid_tip_contact"],
            ],
            preferred_atoms=["frictional_contact", "stable_support_face"],
            risk_atoms=["break_prone", "placement_unstable"],
            physical_rationale="State transition quality depends on directional force transfer and support stability.",
            failure_risks=["Force vector mismatch causes ineffective transition.", "Placement instability reverses the achieved state."],
        ),
    ]


def _build_atom_stats(
    atom_vocab: dict[str, int],
    affordance_histogram: dict[str, int],
    inventory: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    for atom in atom_vocab.keys():
        count = int(max(0, affordance_histogram.get(atom, 0)))
        uncertainty_samples = []
        for item in inventory:
            if not isinstance(item, dict):
                continue
            if str(item.get("atom", "")) != atom:
                continue
            value = item.get("uncertainty_overall")
            if isinstance(value, (int, float)):
                uncertainty_samples.append(float(value))

        uncertainty_score = _atom_uncertainty_score(
            count=count,
            uncertainty_samples=uncertainty_samples,
        )
        raw_availability_code = _availability_code(count)
        uncertainty_penalty = _uncertainty_penalty(uncertainty_score)
        availability_code = max(0, raw_availability_code - uncertainty_penalty)
        stats[atom] = {
            "scene_count": count,
            "raw_availability_code": raw_availability_code,
            "uncertainty_score": uncertainty_score,
            "uncertainty_penalty": uncertainty_penalty,
            "availability_code": availability_code,
        }
    return stats


def _atom_uncertainty_score(count: int, uncertainty_samples: list[float]) -> int:
    if count <= 0:
        return 5
    if not uncertainty_samples:
        if count >= 6:
            return 1
        if count >= 4:
            return 2
        if count >= 2:
            return 3
        return 4

    mean_sample = sum(uncertainty_samples) / len(uncertainty_samples)
    max_sample = max(uncertainty_samples)

    if mean_sample <= 0.15:
        score = 0
    elif mean_sample <= 0.25:
        score = 1
    elif mean_sample <= 0.40:
        score = 2
    elif mean_sample <= 0.60:
        score = 3
    elif mean_sample <= 0.80:
        score = 4
    else:
        score = 5

    if count == 1 and score < 2:
        score = 2
    if max_sample >= 0.75 and score < 4:
        score = 4
    return max(0, min(score, 5))


def _availability_code(count: int) -> int:
    if count <= 0:
        return 0
    if count == 1:
        return 1
    if count <= 3:
        return 2
    if count <= 5:
        return 3
    if count <= 8:
        return 4
    return 5


def _uncertainty_penalty(uncertainty_score: int) -> int:
    if uncertainty_score <= 1:
        return 0
    if uncertainty_score <= 3:
        return 1
    return 2


def _coverage_code(required_codes: list[int], uncertainty_score: int) -> int:
    if not required_codes:
        return 0
    if any(code == 0 for code in required_codes):
        return 0
    low_or_trace = sum(1 for code in required_codes if code <= 1)
    if low_or_trace * 2 >= len(required_codes):
        return 1
    if any(code == 2 for code in required_codes):
        return 2

    if all(code >= 4 for code in required_codes) and uncertainty_score <= 1:
        return 5
    if any(code >= 4 for code in required_codes) and uncertainty_score <= 2:
        return 4
    return 3


def _overall_sufficiency_code(subgoal_coverage_codes: list[int]) -> int:
    if not subgoal_coverage_codes:
        return 0
    if any(code == 0 for code in subgoal_coverage_codes):
        return 0
    if any(code == 1 for code in subgoal_coverage_codes):
        return 1
    if any(code == 2 for code in subgoal_coverage_codes):
        return 2
    if all(code == 5 for code in subgoal_coverage_codes):
        return 5
    if all(code >= 4 for code in subgoal_coverage_codes):
        return 4
    return 3


def _resolve_required_atoms(
    candidates: list[list[str]],
    atom_stats: dict[str, dict[str, int]],
) -> list[str]:
    cleaned_candidates: list[list[str]] = []
    for candidate in candidates:
        cleaned = _dedupe_keep_order([atom for atom in candidate if atom in atom_stats])
        if cleaned:
            cleaned_candidates.append(cleaned)

    if not cleaned_candidates:
        sorted_atoms = sorted(
            atom_stats.items(),
            key=lambda item: (
                item[1]["availability_code"],
                item[1]["scene_count"],
            ),
            reverse=True,
        )
        return [sorted_atoms[0][0]] if sorted_atoms else []

    best_index = 0
    best_score: tuple[int, int, int] | None = None
    for idx, candidate in enumerate(cleaned_candidates):
        codes = [atom_stats[atom]["availability_code"] for atom in candidate]
        candidate_score = (min(codes), sum(codes), -len(candidate))
        if best_score is None or candidate_score > best_score:
            best_score = candidate_score
            best_index = idx
    return cleaned_candidates[best_index]


def _task_restatement(goal_text: str) -> str:
    if goal_text:
        return (
            "Change the target object state to satisfy the user goal while keeping the solution "
            f"object-independent: {goal_text}"
        )
    return (
        "Change the target object state according to future user goal details, "
        "using only scene capability resources."
    )


def _primary_success(goal_text: str) -> str:
    if goal_text:
        return "The target object reaches the requested final state with stable physical feasibility."
    return "A target state change can be executed with usable capability coverage."


def _decomposition_principle(intent: str, task_profile: TaskProfile) -> str:
    if intent == "extract_from_narrow_gap":
        return "damage-aware narrow-gap extraction decomposition with object-independent requirements"
    if task_profile.low_force_bias or task_profile.avoid_sharp_contact:
        return "capability-minimal decomposition with damage-aware soft constraints"
    return "capability-minimal decomposition with object-independent requirements"


def _task_assumptions(
    goal_text: str,
    task_notes: list[str],
    task_profile: TaskProfile,
) -> list[str]:
    assumptions = [
        "The task target is a state-change object, not a tool instance.",
        "No external resources outside the provided scene_resources are assumed.",
    ]
    if not goal_text:
        assumptions.append("Task objective details are incomplete and will be refined when user_goal is provided.")
    if task_notes:
        assumptions.append("Task notes are interpreted as soft constraints unless they conflict with feasibility.")
    if task_profile.narrow_gap or task_profile.exposed_edge_only:
        assumptions.append("Initial access is limited to a narrow-gap entry and a small exposed edge region.")
    if task_profile.fragile_sheet_target:
        assumptions.append("Thin-sheet deformation and tear avoidance are treated as primary execution constraints.")
    return assumptions


def _high_risk_areas(scene_resources: dict[str, Any]) -> list[str]:
    summary = scene_resources.get("resource_summary", {})
    risk_histogram = summary.get("risk_histogram", {})
    if not isinstance(risk_histogram, dict):
        return []
    risk_items: list[tuple[str, int]] = []
    for key, value in risk_histogram.items():
        if isinstance(value, (int, float)) and int(value) > 0:
            risk_items.append((str(key), int(value)))
    risk_items.sort(key=lambda item: (-item[1], item[0]))
    return [item[0] for item in risk_items]


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _empty_atom_stat() -> dict[str, int]:
    return {
        "scene_count": 0,
        "raw_availability_code": 0,
        "uncertainty_score": 5,
        "uncertainty_penalty": 2,
        "availability_code": 0,
    }
