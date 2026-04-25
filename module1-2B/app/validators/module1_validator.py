"""Strict validator for Module 1 raw output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.utils import load_json, project_root

try:  # pragma: no cover - optional dependency
    from jsonschema import Draft202012Validator  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Draft202012Validator = None


VISIBILITY = {"full", "partial", "heavily_occluded"}
ACCESSIBILITY = {"clear", "partial", "occluded", "entangled", "nested"}
POSE_CLASS = {"lying", "upright", "leaning", "stacked", "inside_container", "unknown"}
SUPPORT_CONTEXT = {"on_surface", "in_container", "against_object", "held_by_group", "unknown"}
RELATION = {
    "on_surface",
    "inside",
    "partially_inside",
    "against",
    "leaning_on",
    "adjacent_to",
    "touching",
    "overlapping",
    "stacked_on",
    "clipped_to",
    "between_surfaces",
}
ASPECT_RATIO = {"compact", "elongated", "sheet_like", "blocky", "unknown"}
CONTACT_PROFILE = {"tip", "edge", "broad_flat_face", "curved_side", "cavity_rim", "mixed", "unknown"}
ROLL_RISK = {"none", "curved_side", "round_cross_section", "joint_instability", "unknown"}
ROLE_CANONICAL = {
    "rigid_tip",
    "thin_edge",
    "flat_face",
    "support_base",
    "hook_region",
    "container_cavity",
    "compliant_pad",
    "grip_body",
    "hinge_joint",
    "unknown",
}

LOW_MED_HIGH = {"low", "medium", "high"}
MASS_CATEGORY = {"very_light", "light", "medium", "heavy"}
DENSITY_CATEGORY = {"very_low", "low", "medium", "high"}
PRESS_RESPONSE = {
    "negligible_deformation",
    "elastic_deformation",
    "plastic_deformation",
    "compressible",
}
FAILURE_MODE = {"none_likely", "bend", "compress", "dent", "crack", "shatter", "buckle", "slip"}
SCALE_ANCHOR = {"anchored", "weak_prior", "unknown"}


@dataclass
class ValidationReport:
    """Validation result with explicit error and warning traces."""

    valid: bool
    errors: list[str]
    warnings: list[str]


class Module1Validator:
    """Validate raw payload against schema and additional strict rules."""

    def __init__(self, schema_path: Path | None = None) -> None:
        schema_file = (
            schema_path
            if schema_path is not None
            else project_root() / "schemas" / "module1_raw_output_lite.schema.json"
        )
        self.schema = load_json(schema_file)
        self.validator = (
            Draft202012Validator(self.schema) if Draft202012Validator is not None else None
        )

    def validate(self, payload: dict[str, Any]) -> ValidationReport:
        """Run strict validation rules."""
        errors: list[str] = []
        warnings: list[str] = []

        if self.validator is not None:
            schema_errors = sorted(self.validator.iter_errors(payload), key=lambda e: e.path)
            for err in schema_errors:
                path = "$"
                if err.path:
                    path += "." + ".".join(str(item) for item in err.path)
                errors.append(f"{path}: {err.message}")
        else:
            self._manual_structure_check(payload, errors)

        objects = payload.get("objects", [])
        if not objects:
            warnings.append(
                "objects array is empty. Runner will continue with explicit no-object diagnostics."
            )

        object_ids: set[str] = set()
        for idx, obj in enumerate(objects):
            object_id = obj.get("object_id")
            if object_id in object_ids:
                errors.append(f"objects[{idx}].object_id duplicated: {object_id}")
            object_ids.add(object_id)

            self._validate_uncertainty_consistency(obj=obj, index=idx, errors=errors)
            self._validate_score_thresholds(obj=obj, index=idx, errors=errors)

        return ValidationReport(valid=(len(errors) == 0), errors=errors, warnings=warnings)

    def validate_or_raise(self, payload: dict[str, Any]) -> ValidationReport:
        """Validate and raise ValueError on any failure."""
        report = self.validate(payload)
        if not report.valid:
            joined = "\n".join(report.errors)
            raise ValueError(f"Module 1 validation failed:\n{joined}")
        return report

    def _manual_structure_check(self, payload: dict[str, Any], errors: list[str]) -> None:
        """Fallback strict checks when jsonschema dependency is unavailable."""
        self._exact_keys(
            payload,
            {"schema_name", "schema_version", "scene_summary", "objects"},
            "$",
            errors,
        )
        if payload.get("schema_name") != "module1_raw_output_lite":
            errors.append("$.schema_name must be module1_raw_output_lite")
        if payload.get("schema_version") != "0.4":
            errors.append("$.schema_version must be 0.4")

        summary = payload.get("scene_summary")
        if not isinstance(summary, dict):
            errors.append("$.scene_summary must be object")
            return
        self._exact_keys(
            summary,
            {"selection_policy", "ordering_rule", "notes", "coverage_caveats"},
            "$.scene_summary",
            errors,
        )
        if summary.get("selection_policy") != "all visible movable tool-usable object instances":
            errors.append("$.scene_summary.selection_policy is fixed by spec")
        if summary.get("ordering_rule") != "left_to_right_then_front_to_back":
            errors.append("$.scene_summary.ordering_rule is fixed by spec")
        if not isinstance(summary.get("coverage_caveats"), list):
            errors.append("$.scene_summary.coverage_caveats must be array")

        objects = payload.get("objects")
        if not isinstance(objects, list):
            errors.append("$.objects must be array")
            return
        for index, obj in enumerate(objects):
            self._manual_object_check(obj=obj, index=index, errors=errors)

    def _manual_object_check(self, obj: dict[str, Any], index: int, errors: list[str]) -> None:
        path = f"$.objects[{index}]"
        required = {
            "object_id",
            "object_name",
            "object_type_canonical",
            "grouped",
            "quantity_estimate",
            "coarse_location_hint",
            "visibility",
            "accessibility",
            "state",
            "scene_relations",
            "observed_vs_inferred",
            "geometry_cues",
            "scale_anchor_status",
            "material_hypotheses",
            "functional_parts",
            "affordance_card",
            "physical_properties",
            "uncertainty",
        }
        self._exact_keys(obj, required, path, errors)
        self._enum_check(obj.get("visibility"), VISIBILITY, f"{path}.visibility", errors)
        self._enum_check(obj.get("accessibility"), ACCESSIBILITY, f"{path}.accessibility", errors)
        self._enum_check(
            obj.get("scale_anchor_status"), SCALE_ANCHOR, f"{path}.scale_anchor_status", errors
        )
        if not isinstance(obj.get("grouped"), bool):
            errors.append(f"{path}.grouped must be boolean")
        if not isinstance(obj.get("quantity_estimate"), int) or obj["quantity_estimate"] < 1:
            errors.append(f"{path}.quantity_estimate must be integer >= 1")

        state = obj.get("state")
        if isinstance(state, dict):
            self._exact_keys(
                state,
                {"pose_class", "orientation_note", "support_context"},
                f"{path}.state",
                errors,
            )
            self._enum_check(state.get("pose_class"), POSE_CLASS, f"{path}.state.pose_class", errors)
            self._enum_check(
                state.get("support_context"), SUPPORT_CONTEXT, f"{path}.state.support_context", errors
            )
        else:
            errors.append(f"{path}.state must be object")

        relations = obj.get("scene_relations")
        if isinstance(relations, list):
            for rel_idx, rel in enumerate(relations):
                rel_path = f"{path}.scene_relations[{rel_idx}]"
                if not isinstance(rel, dict):
                    errors.append(f"{rel_path} must be object")
                    continue
                self._exact_keys(rel, {"relation", "object_ref", "relation_note"}, rel_path, errors)
                self._enum_check(rel.get("relation"), RELATION, f"{rel_path}.relation", errors)
        else:
            errors.append(f"{path}.scene_relations must be array")

        observed = obj.get("observed_vs_inferred")
        if isinstance(observed, dict):
            self._exact_keys(
                observed,
                {"observed_cues", "inferred_aspects", "assumed_aspects"},
                f"{path}.observed_vs_inferred",
                errors,
            )
        else:
            errors.append(f"{path}.observed_vs_inferred must be object")

        geometry = obj.get("geometry_cues")
        if isinstance(geometry, dict):
            self._exact_keys(
                geometry,
                {
                    "shape_class",
                    "aspect_ratio_hint",
                    "size_relative",
                    "thickness_class",
                    "primary_contact_profile",
                    "has_pointed_or_thin_end",
                    "has_flat_contact_face",
                    "has_open_cavity",
                    "roll_risk_source",
                },
                f"{path}.geometry_cues",
                errors,
            )
            self._enum_check(
                geometry.get("aspect_ratio_hint"),
                ASPECT_RATIO,
                f"{path}.geometry_cues.aspect_ratio_hint",
                errors,
            )
            self._enum_check(
                geometry.get("primary_contact_profile"),
                CONTACT_PROFILE,
                f"{path}.geometry_cues.primary_contact_profile",
                errors,
            )
            self._enum_check(
                geometry.get("roll_risk_source"),
                ROLL_RISK,
                f"{path}.geometry_cues.roll_risk_source",
                errors,
            )
        else:
            errors.append(f"{path}.geometry_cues must be object")

        material_hyp = obj.get("material_hypotheses")
        if isinstance(material_hyp, list):
            for mat_idx, item in enumerate(material_hyp):
                mpath = f"{path}.material_hypotheses[{mat_idx}]"
                if not isinstance(item, dict):
                    errors.append(f"{mpath} must be object")
                    continue
                self._exact_keys(item, {"material", "probability"}, mpath, errors)
                self._range_check(item.get("probability"), mpath + ".probability", errors)
        else:
            errors.append(f"{path}.material_hypotheses must be array")

        parts = obj.get("functional_parts")
        if isinstance(parts, list):
            for part_idx, part in enumerate(parts):
                ppath = f"{path}.functional_parts[{part_idx}]"
                if not isinstance(part, dict):
                    errors.append(f"{ppath} must be object")
                    continue
                self._exact_keys(
                    part,
                    {
                        "part_name",
                        "role",
                        "role_canonical",
                        "contact_profile",
                        "local_material",
                        "local_property_note",
                        "local_property_tags",
                    },
                    ppath,
                    errors,
                )
                self._enum_check(part.get("role_canonical"), ROLE_CANONICAL, ppath + ".role_canonical", errors)
                self._enum_check(part.get("contact_profile"), CONTACT_PROFILE, ppath + ".contact_profile", errors)
        else:
            errors.append(f"{path}.functional_parts must be array")

        self._manual_affordance_card_check(obj.get("affordance_card"), path, errors)
        self._manual_physical_properties_check(obj.get("physical_properties"), path, errors)
        self._manual_uncertainty_check(obj.get("uncertainty"), path, errors)

    def _manual_affordance_card_check(self, card: Any, object_path: str, errors: list[str]) -> None:
        path = f"{object_path}.affordance_card"
        if not isinstance(card, dict):
            errors.append(f"{path} must be object")
            return
        self._exact_keys(
            card,
            {
                "object_name",
                "observed_visual_features",
                "inferred_physical_properties",
                "usable_parts",
                "connection_modes",
                "weaknesses_or_risks",
                "uncertain_points",
                "confidence",
            },
            path,
            errors,
        )
        self._range_check(card.get("confidence"), path + ".confidence", errors)
        usable_parts = card.get("usable_parts")
        if not isinstance(usable_parts, list):
            errors.append(f"{path}.usable_parts must be array")
            return
        for idx, part in enumerate(usable_parts):
            ppath = f"{path}.usable_parts[{idx}]"
            if not isinstance(part, dict):
                errors.append(f"{ppath} must be object")
                continue
            self._exact_keys(
                part, {"part_name", "affordance_scores", "interaction_primitives", "target_mode_numeric"}, ppath, errors
            )
            self._score_object_check(part.get("affordance_scores"), ppath + ".affordance_scores", errors)
            self._score_object_check(part.get("interaction_primitives"), ppath + ".interaction_primitives", errors)
            target = part.get("target_mode_numeric")
            if not isinstance(target, dict):
                errors.append(f"{ppath}.target_mode_numeric must be object")
            else:
                self._exact_keys(
                    target,
                    {
                        "point_score",
                        "edge_score",
                        "face_score",
                        "rim_score",
                        "cavity_score",
                        "axis_score",
                        "hook_gap_score",
                        "exposure_ratio",
                        "clearance_ratio",
                        "usable_span_m",
                        "local_thickness_m",
                        "tip_radius_m",
                        "flat_patch_m2",
                        "approach_directions_count",
                    },
                    ppath + ".target_mode_numeric",
                    errors,
                )
                for key in [
                    "point_score",
                    "edge_score",
                    "face_score",
                    "rim_score",
                    "cavity_score",
                    "axis_score",
                    "hook_gap_score",
                    "exposure_ratio",
                    "clearance_ratio",
                ]:
                    value = target.get(key)
                    if value is not None:
                        self._range_check(value, f"{ppath}.target_mode_numeric.{key}", errors)

    def _manual_physical_properties_check(
        self, props: Any, object_path: str, errors: list[str]
    ) -> None:
        path = f"{object_path}.physical_properties"
        if not isinstance(props, dict):
            errors.append(f"{path} must be object")
            return
        required = {
            "stiffness",
            "deformability",
            "surface_friction",
            "slip_tendency",
            "mass_category",
            "density_category",
            "restitution",
            "fragility",
            "press_response_type",
            "tip_force_transmission",
            "load_bearing",
            "failure_mode",
        }
        self._exact_keys(props, required, path, errors)
        label_enums = {
            "stiffness": LOW_MED_HIGH,
            "deformability": LOW_MED_HIGH,
            "surface_friction": LOW_MED_HIGH,
            "slip_tendency": LOW_MED_HIGH,
            "mass_category": MASS_CATEGORY,
            "density_category": DENSITY_CATEGORY,
            "restitution": LOW_MED_HIGH,
            "fragility": LOW_MED_HIGH,
            "press_response_type": PRESS_RESPONSE,
            "tip_force_transmission": LOW_MED_HIGH,
            "load_bearing": LOW_MED_HIGH,
            "failure_mode": FAILURE_MODE,
        }
        for key, enum_set in label_enums.items():
            item = props.get(key)
            ipath = f"{path}.{key}"
            if not isinstance(item, dict):
                errors.append(f"{ipath} must be object")
                continue
            self._exact_keys(item, {"label", "confidence", "evidence"}, ipath, errors)
            self._enum_check(item.get("label"), enum_set, ipath + ".label", errors)
            self._range_check(item.get("confidence"), ipath + ".confidence", errors)

    def _manual_uncertainty_check(self, unc: Any, object_path: str, errors: list[str]) -> None:
        path = f"{object_path}.uncertainty"
        if not isinstance(unc, dict):
            errors.append(f"{path} must be object")
            return
        self._exact_keys(
            unc,
            {"overall", "occlusion", "scale", "material", "mass", "dynamics", "part_structure"},
            path,
            errors,
        )
        for key in ["overall", "occlusion", "scale", "material", "mass", "dynamics", "part_structure"]:
            self._range_check(unc.get(key), f"{path}.{key}", errors)

    @staticmethod
    def _validate_uncertainty_consistency(
<<<<<<< HEAD
        obj: dict[str, Any], index: int, errors: list[str]
    ) -> None:
=======
        obj: dict[str, Any], index: int, errors: list[str], warnings: list[str] | None = None
    ) -> None:
        if warnings is None:
            warnings = []
>>>>>>> origin/subin/module2c-3-pipeline
        uncertainty = obj.get("uncertainty", {})
        components = [
            uncertainty.get("occlusion"),
            uncertainty.get("scale"),
            uncertainty.get("material"),
            uncertainty.get("mass"),
            uncertainty.get("dynamics"),
            uncertainty.get("part_structure"),
        ]
        if any(value is None for value in components):
            errors.append(f"objects[{index}].uncertainty missing one or more components")
            return
        avg = round(sum(float(value) for value in components) / 6.0, 2)
        overall = round(float(uncertainty.get("overall", -1)), 2)
<<<<<<< HEAD
        if avg != overall:
            errors.append(
                f"objects[{index}].uncertainty.overall={overall:.2f} must equal component mean {avg:.2f}"
            )

=======
        if abs(avg - overall) > 0.15:
            warnings.append(
                f"objects[{index}].uncertainty.overall={overall:.2f} must equal component mean {avg:.2f}"
            )
>>>>>>> origin/subin/module2c-3-pipeline
    @staticmethod
    def _validate_score_thresholds(obj: dict[str, Any], index: int, errors: list[str]) -> None:
        usable_parts = obj.get("affordance_card", {}).get("usable_parts", [])
        for part_idx, part in enumerate(usable_parts):
            for field_name in ("affordance_scores", "interaction_primitives"):
                scores = part.get(field_name, {})
                if not isinstance(scores, dict):
                    continue
                for key, value in scores.items():
                    if float(value) < 0.15:
                        errors.append(
                            f"objects[{index}].affordance_card.usable_parts[{part_idx}]."
                            f"{field_name}.{key}={value} < 0.15 (forbidden by spec)"
                        )

    @staticmethod
    def _exact_keys(
        obj: Any, expected: set[str], path: str, errors: list[str]
    ) -> None:
        if not isinstance(obj, dict):
            errors.append(f"{path} must be object")
            return
        keys = set(obj.keys())
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        for key in missing:
            errors.append(f"{path}.{key} missing")
        for key in extra:
            errors.append(f"{path}.{key} unexpected")

    @staticmethod
    def _enum_check(value: Any, allowed: set[str], path: str, errors: list[str]) -> None:
        if value not in allowed:
            errors.append(f"{path} must be one of {sorted(allowed)}")

    @staticmethod
    def _range_check(value: Any, path: str, errors: list[str]) -> None:
        if not isinstance(value, (int, float)):
            errors.append(f"{path} must be number")
            return
        if not (0.0 <= float(value) <= 1.0):
            errors.append(f"{path} must be in [0,1]")

    @staticmethod
    def _score_object_check(value: Any, path: str, errors: list[str]) -> None:
        if not isinstance(value, dict):
            errors.append(f"{path} must be object")
            return
        for key, score in value.items():
            if not isinstance(key, str) or not key or not key[0].islower():
                errors.append(f"{path}.{key} must be lower_snake_case key")
            if not isinstance(score, (int, float)):
                errors.append(f"{path}.{key} must be numeric")
            elif not (0.0 <= float(score) <= 1.0):
                errors.append(f"{path}.{key} must be in [0,1]")
