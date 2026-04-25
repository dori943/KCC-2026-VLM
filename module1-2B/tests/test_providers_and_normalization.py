from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.bridges.module1_to_module2a import build_module2_bridge_package
from app.module2b.providers import VisionBundleProvider
from app.module2a.reasoner import generate_module2a_output
from app.models.module1_normalizer import normalize_module1_raw
from app.providers.base import ProviderResult
from app.providers.file_provider import FileProvider
from app.providers.mock_provider import MockProvider
from app.providers.vision_provider import VisionProvider
from app.providers import vision_provider
from app.module2b.providers import FileBundleProvider
from app.utils import load_json, load_yaml, project_root


def test_mock_provider_loading():
    provider = MockProvider(fixtures_root=project_root() / "fixtures")
    result = provider.get_module1_output(case_id="rubber_ball_like_object")
    assert result.metadata["provider"] == "mock"
    assert result.raw_output["schema_name"] == "module1_raw_output_lite"


def test_file_provider_loading(repo_root: Path):
    provider = FileProvider()
    path = repo_root / "fixtures" / "module1_raw_outputs" / "wooden_block_like_object.json"
    result = provider.get_module1_output(module1_output_path=path)
    assert result.metadata["provider"] == "file"
    assert result.raw_output["objects"][0]["object_type_canonical"] == "wooden_block"


def test_normalization_preserves_traceability(fixture_raw_outputs):
    raw = fixture_raw_outputs["mug_or_container_like_object"]
    normalized = normalize_module1_raw(raw)
    assert normalized.schema_name == "module1_normalized_internal"
    assert normalized.objects[0].raw_object_id == "obj_01"
    assert normalized.objects[0].provenance["raw_path"] == "objects[0]"
    assert normalized.objects[0].usable_parts[0].part_name == "cavity"


def test_vision_provider_requires_image(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider = VisionProvider()
    with pytest.raises(ValueError):
        provider.get_module1_output()


def test_vision_provider_requires_api_key(repo_root: Path, monkeypatch):
    image_path = repo_root / "outputs" / "test_vision_provider_dummy.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    try:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        provider = VisionProvider()
        with pytest.raises(ValueError):
            provider.get_module1_output(image_path=image_path)
    finally:
        if image_path.exists():
            image_path.unlink()


def test_module2b_vision_bundle_provider_requires_image():
    provider = VisionBundleProvider()
    with pytest.raises(ValueError):
        provider.get_bundle()


def test_module2b_vision_bundle_provider_uses_module2a_bridge(
    repo_root: Path,
    fixture_raw_outputs,
    monkeypatch,
):
    captured: dict[str, object] = {}

    def _fake_get_module1_output(
        self,
        image_path: Path | None = None,
        case_id: str | None = None,
        module1_output_path: Path | None = None,
    ) -> ProviderResult:
        captured["image_path"] = image_path
        captured["case_id"] = case_id
        captured["module1_output_path"] = module1_output_path
        return ProviderResult(
            raw_output=fixture_raw_outputs["mug_or_container_like_object"],
            metadata={"provider": "vision", "model": "stub-model"},
        )

    monkeypatch.setattr(VisionProvider, "get_module1_output", _fake_get_module1_output)

    provider = VisionBundleProvider()
    image_path = repo_root / "outputs" / "test_module2b_vision_bundle_provider_input.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    try:
        result = provider.get_bundle(
            image_path=image_path,
            case_id="test_case",
            user_goal="Pick up the coin",
            success_criteria=["coin is extracted"],
        )
    finally:
        if image_path.exists():
            image_path.unlink()

    common = result.bundle["module2_common_input"]
    assert common["schema_name"] == "module2_common_input_for_module2b_derived_min"
    assert isinstance(common.get("task_id"), str) and common["task_id"]
    object_ids = [item["object_id"] for item in common["scene_resources"]["resource_inventory"]]
    assert len(object_ids) == len(set(object_ids))
    assert result.bundle["module2a_output"]["schema_name"] == "module2a_output"
    assert result.metadata["provider"] == "vision"
    assert result.metadata["mode"] == "image_to_bundle"
    assert result.metadata["case_id"] == "test_case"
    assert result.metadata["module1_provider_metadata"]["provider"] == "vision"
    assert common["task_brief"]["user_goal"] == "Pick up the coin"
    assert captured["image_path"] == image_path
    assert captured["case_id"] == "test_case"
    assert captured["module1_output_path"] is None


def test_module2b_file_bundle_provider_adapts_module2_template(
    repo_root: Path,
    fixture_raw_outputs,
):
    raw = fixture_raw_outputs["wooden_block_like_object"]
    normalized = normalize_module1_raw(raw)
    rule_cfg = load_yaml(project_root() / "configs" / "module1_to_module2a_atom_rules.yaml")
    vocab = load_json(project_root() / "configs" / "vocab_registry.json")
    artifacts = build_module2_bridge_package(
        normalized=normalized,
        rule_cfg=rule_cfg,
        vocab_registry=vocab,
    )
    template = artifacts["module2_common_input_template"]
    module2a_output = generate_module2a_output(
        module2_common_input=template,
        vocab_registry=vocab,
    )

    output_dir = repo_root / "outputs" / "test_module2b_file_bundle_provider"
    output_dir.mkdir(parents=True, exist_ok=True)
    module2_common_path = output_dir / "module2_common_input_template.json"
    module2a_output_path = output_dir / "module2a_output.json"
    module2_common_path.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
    module2a_output_path.write_text(json.dumps(module2a_output, ensure_ascii=False, indent=2), encoding="utf-8")

    provider = FileBundleProvider()
    bundle = provider.get_bundle(
        module2_common_path=module2_common_path,
        module2a_output_path=module2a_output_path,
        case_id="wooden_block_like_object",
    ).bundle

    common = bundle["module2_common_input"]
    assert common["schema_name"] == "module2_common_input_for_module2b_derived_min"
    assert common["schema_version"] == "0.1"
    assert common["task_id"] == "task_wooden_block_like_object"

    inventory = common["scene_resources"]["resource_inventory"]
    object_ids = [item["object_id"] for item in inventory]
    assert len(object_ids) == len(set(object_ids))
    assert len(inventory) == 1

    required_keys = {
        "object_id",
        "object_name",
        "object_type_canonical",
        "coarse_location_hint",
        "visibility",
        "accessibility",
        "state",
        "scene_relations",
        "geometry_cues",
        "functional_parts",
        "target_mode_numeric_summary",
        "uncertainty",
    }
    assert required_keys.issubset(set(inventory[0].keys()))

    if module2_common_path.exists():
        module2_common_path.unlink()
    if module2a_output_path.exists():
        module2a_output_path.unlink()


def test_build_object_entry_replaces_unknown_placeholder_name():
    entry = vision_provider._build_object_entry(  # noqa: SLF001
        raw_entry={"object_name": "unknown_object_1", "position": "center"},
        index=0,
    )
    assert entry["object_name"] == "inferred_object_1"
    assert entry["object_type_canonical"] == "inferred_object_1"


def test_infer_missing_object_names_with_llm_filters_unknown_like_labels(monkeypatch):
    def _fake_post_json(*, url, api_key, body, timeout_seconds):  # noqa: ARG001
        payload = {
            "object_names": [
                {"index": 0, "object_name": "unknown_object_1"},
                {"index": 1, "object_name": "metal_hook"},
            ]
        }
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    monkeypatch.setattr(vision_provider, "_post_json", _fake_post_json)
    overrides = vision_provider._infer_missing_object_names_with_llm(  # noqa: SLF001
        entries=[{"name": ""}, {"name": ""}],
        api_url="https://api.openai.com/v1/chat/completions",
        api_key="test-key",
        model="gpt-4.1-mini",
        timeout_seconds=30.0,
        image_b64="ZmFrZQ==",
        mime_type="image/png",
    )
    assert overrides[0] == "inferred_object_1"
    assert overrides[1] == "metal_hook"
