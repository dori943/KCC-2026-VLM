from __future__ import annotations

from app.bridges.module1_to_pybullet import map_module1_to_pybullet
from app.models.module1_normalizer import normalize_module1_raw
from app.utils import load_yaml, project_root


def test_pybullet_mapping_has_provenance(fixture_raw_outputs):
    raw = fixture_raw_outputs["rubber_ball_like_object"]
    normalized = normalize_module1_raw(raw)
    cfg = load_yaml(project_root() / "configs" / "module1_to_pybullet_map.yaml")
    mapped = map_module1_to_pybullet(normalized=normalized, mapping_cfg=cfg)

    assert mapped["schema_name"] == "module1_to_pybullet_surrogate"
    entry = mapped["objects"][0]
    params = entry["surrogate_parameters"]
    assert params["mass_kg"] > 0.0
    assert 0.0 <= params["restitution"] <= 0.95
    assert "mapping_provenance" in entry["provenance"]
    assert len(entry["provenance"]["mapping_provenance"]) >= 3
