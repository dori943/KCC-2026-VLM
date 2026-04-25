from __future__ import annotations

from pathlib import Path

<<<<<<< HEAD
import pytest

from app.models.module1_normalizer import normalize_module1_raw
from app.providers.file_provider import FileProvider
from app.providers.mock_provider import MockProvider
from app.providers.vision_provider import VisionProvider
=======
from app.models.module1_normalizer import normalize_module1_raw
from app.providers.file_provider import FileProvider
from app.providers.mock_provider import MockProvider
>>>>>>> origin/subin/module2c-3-pipeline
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
<<<<<<< HEAD


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
=======
>>>>>>> origin/subin/module2c-3-pipeline
