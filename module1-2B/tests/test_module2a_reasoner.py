from __future__ import annotations

from app.bridges.module1_to_module2a import build_module2_bridge_package
from app.models.module1_normalizer import normalize_module1_raw
from app.module2a.reasoner import generate_module2a_output
from app.utils import load_json, load_yaml, project_root
from app.validators.schema_validator import validate_with_schema


def test_module2a_reasoner_generates_strict_shape(fixture_raw_outputs):
    raw = fixture_raw_outputs["mug_or_container_like_object"]
    normalized = normalize_module1_raw(raw)
    rule_cfg = load_yaml(project_root() / "configs" / "module1_to_module2a_atom_rules.yaml")
    vocab = load_json(project_root() / "configs" / "vocab_registry.json")
    bridge = build_module2_bridge_package(
        normalized=normalized,
        rule_cfg=rule_cfg,
        vocab_registry=vocab,
    )
    module2_input = bridge["module2_common_input_template"]
    module2_input["task_brief"]["user_goal"] = "Pour liquid from source to target container."
    module2_input["task_brief"]["success_criteria"] = [
        "Target container receives liquid without major spill."
    ]

    output = generate_module2a_output(module2_input, vocab)
    assert output["schema_name"] == "module2a_output"
    assert output["schema_version"] == "0.2"
    assert output["stage"] == "task_decomposition_and_function_requirement_extraction"
    assert len(output["subgoals"]) >= 1

    atom_vocab = vocab["module2a"]["affordance_atoms"]
    primitive_vocab = vocab["module2a"]["interaction_primitives"]
    for index, subgoal in enumerate(output["subgoals"], start=1):
        bridge_info = subgoal["pybullet_bridge"]
        assert bridge_info["subgoal_index"] == index
        if index == 1:
            assert subgoal["depends_on"] == []
            assert bridge_info["depends_on_indices"] == []
        else:
            assert subgoal["depends_on"] == [f"sg_{index - 1:02d}"]
            assert bridge_info["depends_on_indices"] == [index - 1]

        required_atoms = subgoal["function_requirements"]["required_atoms"]
        required_primitives = subgoal["required_interaction_primitives"]
        assert required_atoms
        assert required_primitives
        assert all(atom in atom_vocab for atom in required_atoms)
        assert all(primitive in primitive_vocab for primitive in required_primitives)
        assert bridge_info["required_atom_count"] == len(bridge_info["required_atom_codes"])
        assert bridge_info["required_interaction_primitive_count"] == len(
            bridge_info["required_interaction_primitive_codes"]
        )

    schema_errors = validate_with_schema(
        output,
        project_root() / "schemas" / "module2a_output.schema.json",
    )
    assert schema_errors == []


def test_module2a_reasoner_handles_empty_histogram():
    vocab = load_json(project_root() / "configs" / "vocab_registry.json")
    module2_input = {
        "schema_name": "module2_common_input_template_derived_min",
        "schema_version": "0.1",
        "task_brief": {
            "user_goal": "Insert one part into another slot.",
            "success_criteria": ["Insertion is completed."],
            "task_notes": [],
        },
        "scene_resources": {
            "resource_summary": {"affordance_histogram": {}},
            "resource_inventory": [],
        },
    }
    output = generate_module2a_output(module2_input, vocab)
    coverage_codes = [
        subgoal["pybullet_bridge"]["coverage_status_code"] for subgoal in output["subgoals"]
    ]
    assert any(code == 0 for code in coverage_codes)
    assert (
        output["task_level_requirement_summary"]["overall_resource_sufficiency"]
        == "insufficient"
    )


def test_module2a_reasoner_handles_narrow_gap_card_extraction():
    vocab = load_json(project_root() / "configs" / "vocab_registry.json")
    module2_input = {
        "schema_name": "module2_common_input_template_derived_min",
        "schema_version": "0.1",
        "task_brief": {
            "user_goal": "Pry the exposed edge and extract a business card from a narrow gap without bending or tearing it.",
            "success_criteria": [
                "The business card is removed from the gap.",
                "The card is not bent or torn.",
                "The surrounding surface is not damaged.",
            ],
            "task_notes": [
                "Prefer low-force contact paths.",
                "Assume only a small edge of the card is exposed.",
                "Avoid sharp contact and excessive compression.",
            ],
        },
        "scene_resources": {
            "resource_summary": {
                "affordance_histogram": {
                    "thin_insertable": 3,
                    "edge_followable": 3,
                    "frictional_contact": 9,
                    "elongated_reach": 3,
                    "clampable_span": 13,
                    "stable_support_face": 7,
                },
                "risk_histogram": {
                    "break_prone": 3,
                    "sharp_contact_risk": 2,
                    "slip_prone": 2,
                },
                "primitive_histogram": {
                    "insert": 2,
                    "guide": 4,
                    "pull": 3,
                    "stabilize": 5,
                },
            },
            "resource_inventory": [
                {"atom": "thin_insertable", "uncertainty_overall": 0.12},
                {"atom": "edge_followable", "uncertainty_overall": 0.12},
                {"atom": "frictional_contact", "uncertainty_overall": 0.12},
                {"atom": "elongated_reach", "uncertainty_overall": 0.12},
                {"atom": "clampable_span", "uncertainty_overall": 0.12},
                {"atom": "stable_support_face", "uncertainty_overall": 0.12},
            ],
        },
    }

    output = generate_module2a_output(module2_input, vocab)

    assert [subgoal["subgoal_name"] for subgoal in output["subgoals"]] == [
        "establish_low_profile_entry",
        "secure_extraction_purchase",
        "extract_card_incrementally",
    ]
    assert output["task_model"]["decomposition_principle"].startswith(
        "damage-aware narrow-gap extraction"
    )
    assert output["task_level_requirement_summary"]["required_atoms_union"] == [
        "thin_insertable",
        "frictional_contact",
    ]
    assert "sharp_contact_risk" in output["scene_resource_readout"]["high_risk_capability_areas"]
    assert any(
        "thin_insertable" in hypothesis
        for hypothesis in output["task_level_requirement_summary"]["resource_gap_hypotheses"]
    )
