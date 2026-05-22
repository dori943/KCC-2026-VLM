from __future__ import annotations

from pathlib import Path

from app.models.module1_normalizer import normalize_module1_raw
from app.providers.mock_provider import MockProvider
from app.utils import project_root
from app.validators.module1_validator import Module1Validator


def test_multi_object_image_case_validation_and_count(fixture_raw_outputs):
    payload = fixture_raw_outputs["Gemini_Generated_Image_gvc8a5gvc8a5gvc8"]
    report = Module1Validator().validate(payload)
    assert report.valid, report.errors
    assert len(payload["objects"]) == 6
    normalized = normalize_module1_raw(payload)
    assert len(normalized.objects) == 6
    assert [obj.raw_object_id for obj in normalized.objects] == [
        "obj_01",
        "obj_02",
        "obj_03",
        "obj_04",
        "obj_05",
        "obj_06",
    ]


def test_mock_provider_can_resolve_by_image_stem():
    provider = MockProvider(fixtures_root=project_root() / "fixtures")
    image = Path(r"C:\Users\SAMSUNG\Downloads\Gemini_Generated_Image_gvc8a5gvc8a5gvc8.png")
    result = provider.get_module1_output(image_path=image)
    assert result.metadata["case_id"] == "Gemini_Generated_Image_gvc8a5gvc8a5gvc8"
    assert len(result.raw_output["objects"]) == 6
