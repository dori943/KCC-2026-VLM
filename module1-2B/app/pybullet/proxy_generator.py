"""Conservative primitive proxy generation for PyBullet experiments."""

from __future__ import annotations

from typing import Any

from app.models.module1_models import Module1Normalized, Module1ObjectNormalized
from app.utils import clamp, get_path, to_float


def generate_proxy_specs(
    normalized: Module1Normalized, mapping_cfg: dict[str, Any]
) -> dict[str, Any]:
    """Generate conservative primitive proxies per object.

    This does not attempt CAD reconstruction. It intentionally favors simple,
    reproducible primitives with explicit fallback reasons.
    """
    objects = []
    for obj in normalized.objects:
        objects.append(_generate_proxy_for_object(obj=obj, cfg=mapping_cfg))
    return {
        "schema_name": "pybullet_proxy_spec",
        "schema_version": "0.1",
        "map_version": mapping_cfg["map_version"],
        "objects": objects,
    }


def _generate_proxy_for_object(
    obj: Module1ObjectNormalized, cfg: dict[str, Any]
) -> dict[str, Any]:
    selected_rule_id: str | None = None
    selected_primitive: str | None = None
    selection_reason = ""

    context = {
        "geometry": obj.geometry,
    }

    for rule in cfg.get("proxy_rules", []):
        if _rule_match(context=context, rule=rule):
            selected_rule_id = rule["rule_id"]
            selected_primitive = rule["primitive"]
            selection_reason = "matched_proxy_rule"
            break

    if selected_primitive is None:
        selected_primitive = cfg["default_proxy"]["primitive"]
        selection_reason = cfg["default_proxy"]["reason"]

    dimensions, dim_provenance = _build_dimensions(
        obj=obj, primitive=selected_primitive, cfg=cfg
    )
    return {
        "object_id": obj.raw_object_id,
        "primitive": selected_primitive,
        "dimensions": dimensions,
        "selection": {
            "rule_id": selected_rule_id,
            "reason": selection_reason,
        },
        "provenance": dim_provenance,
    }


def _rule_match(context: dict[str, Any], rule: dict[str, Any]) -> bool:
    cond = rule["when"]
    path = cond["path"]
    op = cond["op"]
    value = get_path(context, path)
    if op == "eq":
        return value == cond["value"]
    if op == "contains_any":
        text = str(value).lower()
        return any(token.lower() in text for token in cond["values"])
    return False


def _build_dimensions(
    obj: Module1ObjectNormalized, primitive: str, cfg: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    defaults = cfg["dimensions_defaults"][primitive]
    span_reference = float(cfg["dimension_scale_rules"]["usable_span_reference_m"])
    min_scale = float(cfg["dimension_scale_rules"]["min_scale"])
    max_scale = float(cfg["dimension_scale_rules"]["max_scale"])

    representative_part = _pick_representative_usable_part(obj)
    raw_span = (
        representative_part.target_mode_numeric.usable_span_m
        if representative_part is not None
        else None
    )
    if raw_span is None:
        scale_factor = 1.0
        span_reason = "usable_span_missing_default_scale"
    else:
        raw_scale = to_float(raw_span, default=span_reference) / span_reference
        scale_factor, was_clamped = clamp(raw_scale, min_scale, max_scale)
        span_reason = "usable_span_scaled"
        if was_clamped:
            span_reason = "usable_span_scaled_clamped"

    thickness_label = _canonicalize_thickness_label(obj.geometry.get("thickness_class"))
    thickness_radius = float(cfg["thickness_to_radius_m"].get(thickness_label, 0.03))

    dims: dict[str, Any]
    if primitive in {"box", "thin_box"}:
        base = defaults["half_extents_m"]
        dims = {
            "half_extents_m": [
                round(float(base[0]) * scale_factor, 6),
                round(float(base[1]) * scale_factor, 6),
                round(float(base[2]) * scale_factor, 6),
            ]
        }
    elif primitive == "sphere":
        radius = to_float(defaults["radius_m"]) * scale_factor
        # keep sphere conservative for thin classes
        radius = min(radius, max(thickness_radius, radius))
        dims = {"radius_m": round(radius, 6)}
    elif primitive in {"cylinder", "capsule"}:
        radius_default = to_float(defaults["radius_m"])
        height_default = to_float(defaults["height_m"])
        radius = max(radius_default, thickness_radius * 0.8)
        dims = {
            "radius_m": round(radius * min(scale_factor, 1.4), 6),
            "height_m": round(height_default * scale_factor, 6),
        }
    else:
        dims = {}

    provenance = {
        "dimension_rule": "conservative_primitive_scaling",
        "scale_factor": round(scale_factor, 6),
        "scale_reason": span_reason,
        "thickness_label": thickness_label,
        "thickness_radius_reference_m": thickness_radius,
    }
    return dims, provenance


def _pick_representative_usable_part(obj: Module1ObjectNormalized) -> Any:
    if not obj.usable_parts:
        return None
    return max(
        obj.usable_parts,
        key=lambda p: sum(p.affordance_scores.values()) + sum(p.interaction_primitives.values()),
    )


def _canonicalize_thickness_label(raw: Any) -> str:
    text = str(raw or "unknown").strip().lower()
    if text in {"very_thin", "thin", "medium", "thick", "bulky", "unknown"}:
        return text
    if "thin" in text:
        return "thin"
    if "thick" in text:
        return "thick"
    if "bulk" in text:
        return "bulky"
    if "medium" in text:
        return "medium"
    return "unknown"
