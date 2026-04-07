from __future__ import annotations

from app.bridges.module1_to_module2a import build_module2_bridge_package
from app.models.module1_normalizer import normalize_module1_raw
from app.utils import load_json, load_yaml, project_root


def test_module2_bridge_generation(fixture_raw_outputs):
    raw = fixture_raw_outputs["mug_or_container_like_object"]
    normalized = normalize_module1_raw(raw)
    rule_cfg = load_yaml(project_root() / "configs" / "module1_to_module2a_atom_rules.yaml")
    vocab = load_json(project_root() / "configs" / "vocab_registry.json")

    artifacts = build_module2_bridge_package(normalized=normalized, rule_cfg=rule_cfg, vocab_registry=vocab)
    scene_resources = artifacts["scene_resources_from_module1"]
    template = artifacts["module2_common_input_template"]

    assert scene_resources["schema_name"] == "scene_resources_from_module1"
    assert "concave_containment" in scene_resources["resource_summary"]["affordance_histogram"]
    assert "risk_histogram" in template["scene_resources"]["resource_summary"]
    assert "primitive_histogram" in template["scene_resources"]["resource_summary"]
    assert template["task_brief"]["user_goal"] is None
    assert template["task_brief"]["success_criteria"] == []
    assert template["task_brief"]["task_notes"] == []


def test_capability_counting_consistency(fixture_raw_outputs):
    raw = fixture_raw_outputs["wooden_block_like_object"]
    normalized = normalize_module1_raw(raw)
    rule_cfg = load_yaml(project_root() / "configs" / "module1_to_module2a_atom_rules.yaml")
    vocab = load_json(project_root() / "configs" / "vocab_registry.json")
    artifacts = build_module2_bridge_package(normalized=normalized, rule_cfg=rule_cfg, vocab_registry=vocab)

    scene_resources = artifacts["scene_resources_from_module1"]
    histogram_total = sum(scene_resources["resource_summary"]["affordance_histogram"].values())
    inventory_count = len(scene_resources["resource_inventory"])
    assert histogram_total == inventory_count
