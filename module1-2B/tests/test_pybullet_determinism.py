from __future__ import annotations

import pytest

from app.bridges.module1_to_pybullet import map_module1_to_pybullet
from app.models.module1_normalizer import normalize_module1_raw
from app.pybullet.proxy_generator import generate_proxy_specs
from app.pybullet.runner import run_pybullet_experiments
from app.utils import load_yaml, project_root


pytest.importorskip("pybullet")


def test_pybullet_run_deterministic_within_tolerance(fixture_raw_outputs):
    raw = fixture_raw_outputs["rubber_ball_like_object"]
    normalized = normalize_module1_raw(raw)
    cfg = load_yaml(project_root() / "configs" / "module1_to_pybullet_map.yaml")
    surrogate = map_module1_to_pybullet(normalized=normalized, mapping_cfg=cfg)
    proxy = generate_proxy_specs(normalized=normalized, mapping_cfg=cfg)

    result_a = run_pybullet_experiments(
        proxy_spec=proxy,
        surrogate_spec=surrogate,
        mapping_cfg=cfg,
        scenarios=["drop_test", "slide_test", "force_response_test"],
        seed=123,
    )
    result_b = run_pybullet_experiments(
        proxy_spec=proxy,
        surrogate_spec=surrogate,
        mapping_cfg=cfg,
        scenarios=["drop_test", "slide_test", "force_response_test"],
        seed=123,
    )

    metrics_a = result_a["metrics"]["obj_01"]
    metrics_b = result_b["metrics"]["obj_01"]
    for scenario_name, metric_map in metrics_a.items():
        for key, value_a in metric_map.items():
            value_b = metrics_b[scenario_name][key]
            assert abs(float(value_a) - float(value_b)) <= 1e-6
