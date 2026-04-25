"""Bridge from normalized Module 1 output to PyBullet surrogate parameters."""

from __future__ import annotations

from typing import Any

from app.models.module1_models import Module1Normalized, Module1ObjectNormalized
from app.utils import clamp


def map_module1_to_pybullet(
    normalized: Module1Normalized, mapping_cfg: dict[str, Any]
) -> dict[str, Any]:
    """Map qualitative Module 1 properties to conservative numeric surrogates.

    This function never claims physical ground truth. It only generates a
    reproducible experimental surrogate parameterization.
    """
    objects: list[dict[str, Any]] = []

    for obj in normalized.objects:
        mapped = _map_object(obj=obj, cfg=mapping_cfg)
        objects.append(mapped)

    return {
        "schema_name": "module1_to_pybullet_surrogate",
        "schema_version": "0.1",
        "map_version": mapping_cfg["map_version"],
        "notes": list(mapping_cfg.get("notes", [])),
        "objects": objects,
    }


def _map_object(obj: Module1ObjectNormalized, cfg: dict[str, Any]) -> dict[str, Any]:
    friction_map = cfg["friction_map"]
    slip_penalty = cfg["slip_penalty"]
    restitution_map = cfg["restitution_map"]
    mass_base = cfg["mass_base_kg_by_category"]
    density_multiplier = cfg["density_multiplier"]
    size_multiplier = cfg["size_multiplier"]
    clamp_ranges = cfg["clamp_ranges"]

    provenance: list[dict[str, Any]] = []
    clamp_events: list[dict[str, Any]] = []
    assumptions: list[str] = []
    defaults: list[dict[str, Any]] = []

    surface_friction_label = obj.physical["surface_friction"].label
    slip_label = obj.physical["slip_tendency"].label
    restitution_label = obj.physical["restitution"].label
    mass_category = obj.physical["mass_category"].label
    density_category = obj.physical["density_category"].label
    size_label = _canonicalize_size_label(obj.geometry.get("size_relative"))

    friction_base = float(friction_map.get(surface_friction_label, friction_map["medium"]))
    if surface_friction_label not in friction_map:
        defaults.append(
            {
                "field": "surface_friction",
                "value_used": "medium",
                "reason": "unknown label in mapping config",
            }
        )
    slip_adjust = float(slip_penalty.get(slip_label, slip_penalty["medium"]))
    if slip_label not in slip_penalty:
        defaults.append(
            {
                "field": "slip_tendency",
                "value_used": "medium",
                "reason": "unknown label in mapping config",
            }
        )
    lateral_friction_raw = friction_base + slip_adjust
    lateral_friction, clamped = clamp(
        lateral_friction_raw,
        float(clamp_ranges["lateral_friction"][0]),
        float(clamp_ranges["lateral_friction"][1]),
    )
    if clamped:
        clamp_events.append(
            {
                "field": "lateral_friction",
                "raw": lateral_friction_raw,
                "clamped": lateral_friction,
                "range": clamp_ranges["lateral_friction"],
            }
        )

    restitution_raw = float(restitution_map.get(restitution_label, restitution_map["medium"]))
    if restitution_label not in restitution_map:
        defaults.append(
            {
                "field": "restitution",
                "value_used": "medium",
                "reason": "unknown label in mapping config",
            }
        )
    restitution, clamped = clamp(
        restitution_raw,
        float(clamp_ranges["restitution"][0]),
        float(clamp_ranges["restitution"][1]),
    )
    if clamped:
        clamp_events.append(
            {
                "field": "restitution",
                "raw": restitution_raw,
                "clamped": restitution,
                "range": clamp_ranges["restitution"],
            }
        )

    mass_base_value = float(mass_base.get(mass_category, mass_base["medium"]))
    if mass_category not in mass_base:
        defaults.append(
            {
                "field": "mass_category",
                "value_used": "medium",
                "reason": "unknown label in mapping config",
            }
        )
    density_mul = float(density_multiplier.get(density_category, density_multiplier["medium"]))
    if density_category not in density_multiplier:
        defaults.append(
            {
                "field": "density_category",
                "value_used": "medium",
                "reason": "unknown label in mapping config",
            }
        )
    size_mul = float(size_multiplier.get(size_label, size_multiplier["unknown"]))
    if size_label not in size_multiplier:
        defaults.append(
            {
                "field": "size_relative",
                "value_used": "unknown",
                "reason": "unknown size label in mapping config",
            }
        )
    mass_raw = mass_base_value * density_mul * size_mul
    mass_kg, clamped = clamp(
        mass_raw, float(clamp_ranges["mass_kg"][0]), float(clamp_ranges["mass_kg"][1])
    )
    if clamped:
        clamp_events.append(
            {
                "field": "mass_kg",
                "raw": mass_raw,
                "clamped": mass_kg,
                "range": clamp_ranges["mass_kg"],
            }
        )

    deformability = obj.physical["deformability"].label
    linear_damping_raw = {"low": 0.02, "medium": 0.05, "high": 0.08}.get(deformability, 0.05)
    angular_damping_raw = {"low": 0.01, "medium": 0.03, "high": 0.06}.get(deformability, 0.03)
    if deformability not in {"low", "medium", "high"}:
        defaults.append(
            {
                "field": "deformability",
                "value_used": "medium",
                "reason": "deformability not in low/medium/high",
            }
        )

    linear_damping, clamped = clamp(
        linear_damping_raw,
        float(clamp_ranges["linear_damping"][0]),
        float(clamp_ranges["linear_damping"][1]),
    )
    if clamped:
        clamp_events.append(
            {
                "field": "linear_damping",
                "raw": linear_damping_raw,
                "clamped": linear_damping,
                "range": clamp_ranges["linear_damping"],
            }
        )

    angular_damping, clamped = clamp(
        angular_damping_raw,
        float(clamp_ranges["angular_damping"][0]),
        float(clamp_ranges["angular_damping"][1]),
    )
    if clamped:
        clamp_events.append(
            {
                "field": "angular_damping",
                "raw": angular_damping_raw,
                "clamped": angular_damping,
                "range": clamp_ranges["angular_damping"],
            }
        )

    provenance.extend(
        [
            {
                "target_field": "lateral_friction",
                "rule": "friction_map(surface_friction) + slip_penalty(slip_tendency)",
                "source_fields": [
                    "physical.surface_friction.label",
                    "physical.slip_tendency.label",
                ],
                "source_values": {
                    "surface_friction": surface_friction_label,
                    "slip_tendency": slip_label,
                },
            },
            {
                "target_field": "restitution",
                "rule": "restitution_map(restitution)",
                "source_fields": ["physical.restitution.label"],
                "source_values": {"restitution": restitution_label},
            },
            {
                "target_field": "mass_kg",
                "rule": "mass_base(mass_category) * density_multiplier(density_category) * size_multiplier(size_relative)",
                "source_fields": [
                    "physical.mass_category.label",
                    "physical.density_category.label",
                    "geometry.size_relative",
                ],
                "source_values": {
                    "mass_category": mass_category,
                    "density_category": density_category,
                    "size_relative": obj.geometry.get("size_relative"),
                },
            },
        ]
    )

    non_rigid_metadata = []
    for entry in cfg.get("non_rigid_metadata_rules", []):
        prop = entry["property"]
        physical_entry = obj.physical.get(prop)
        non_rigid_metadata.append(
            {
                "property": prop,
                "strategy": entry["strategy"],
                "note": entry["note"],
                "label": physical_entry.label if physical_entry else None,
                "confidence": physical_entry.confidence if physical_entry else None,
            }
        )

    assumptions.append(
        "Rigid-body approximation used. Deformation/failure are preserved as diagnostics, not direct simulated mechanics."
    )

    return {
        "object_id": obj.raw_object_id,
        "surrogate_parameters": {
            "mass_kg": round(mass_kg, 6),
            "lateral_friction": round(lateral_friction, 6),
            "restitution": round(restitution, 6),
            "linear_damping": round(linear_damping, 6),
            "angular_damping": round(angular_damping, 6),
        },
        "uncertainty_overall": obj.uncertainty["overall"],
        "non_rigid_metadata": non_rigid_metadata,
        "provenance": {
            "mapping_provenance": provenance,
            "assumptions": assumptions,
            "defaults_applied": defaults,
            "clamps_applied": clamp_events,
        },
    }


def _canonicalize_size_label(raw_size: Any) -> str:
    text = str(raw_size or "unknown").strip().lower()
    if text in {"very_small", "small", "medium", "large", "very_large", "unknown"}:
        return text
    if "small" in text:
        return "small"
    if "large" in text:
        return "large"
    if "medium" in text:
        return "medium"
    return "unknown"
