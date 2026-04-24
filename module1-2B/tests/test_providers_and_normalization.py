from __future__ import annotations

import json
from pathlib import Path

from app.models.module1_normalizer import normalize_module1_raw
from app.providers.file_provider import FileProvider
from app.providers.mock_provider import MockProvider
from app.providers import vision_provider
from app.utils import project_root


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
