from __future__ import annotations

import copy

from app.validators.module1_validator import Module1Validator


def test_module1_schema_validation_pass(fixture_raw_outputs):
    validator = Module1Validator()
    for payload in fixture_raw_outputs.values():
        report = validator.validate(payload)
        assert report.valid, report.errors


def test_enum_validation_fail(fixture_raw_outputs):
    validator = Module1Validator()
    payload = copy.deepcopy(fixture_raw_outputs["rubber_ball_like_object"])
    payload["objects"][0]["visibility"] = "visible"
    report = validator.validate(payload)
    assert not report.valid
    assert any("visibility" in err for err in report.errors)


def test_uncertainty_overall_consistency(fixture_raw_outputs):
    validator = Module1Validator()
    payload = copy.deepcopy(fixture_raw_outputs["wooden_block_like_object"])
    payload["objects"][0]["uncertainty"]["overall"] = 0.99
    report = validator.validate(payload)
    assert not report.valid
    assert any("uncertainty.overall" in err for err in report.errors)
