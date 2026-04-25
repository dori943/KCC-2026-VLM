"""Validators for Module 2-B input bundle and strict env-only output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.utils import project_root
from app.validators.schema_validator import validate_with_schema


@dataclass(slots=True)
class ValidationReport:
    """Validation result object with errors, warnings, and check metadata."""

    valid: bool
    errors: list[str]
    warnings: list[str]
    checks: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize report to plain dictionary."""
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "checks": self.checks,
        }


class Module2BInputValidator:
    """Validate module2_common_input + module2a_output bundle for Module 2-B."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or project_root()

    def validate(self, bundle: dict[str, Any]) -> ValidationReport:
        """Validate schema + referential integrity + emptiness warnings."""
        errors: list[str] = []
        warnings: list[str] = []
        checks: dict[str, Any] = {}

        bundle_schema_errors = validate_with_schema(
            payload=bundle,
            schema_path=self.root / "schemas" / "module2b_input_bundle.schema.json",
        )
        errors.extend(bundle_schema_errors)
        checks["bundle_schema_error_count"] = len(bundle_schema_errors)

        module2_common_input = bundle.get("module2_common_input", {})
        module2a_output = bundle.get("module2a_output", {})

        common_schema_errors = validate_with_schema(
            payload=module2_common_input,
            schema_path=self.root / "schemas" / "module2_common_input_for_module2b_derived_min.schema.json",
        )
        errors.extend(common_schema_errors)
        checks["module2_common_schema_error_count"] = len(common_schema_errors)

        module2a_schema_errors = validate_with_schema(
            payload=module2a_output,
            schema_path=self.root / "schemas" / "module2a_output.schema.json",
        )
        errors.extend(module2a_schema_errors)
        checks["module2a_schema_error_count"] = len(module2a_schema_errors)

        inventory = (
            module2_common_input.get("scene_resources", {})
            .get("resource_inventory", [])
        )
        if not inventory:
            warnings.append("resource_inventory is empty; env-only constraints may be deferred.")

        object_ids: list[str] = []
        for idx, obj in enumerate(inventory):
            object_id = obj.get("object_id")
            if isinstance(object_id, str):
                object_ids.append(object_id)
            else:
                errors.append(f"$.module2_common_input.scene_resources.resource_inventory[{idx}].object_id missing")

        duplicates = _duplicates(object_ids)
        if duplicates:
<<<<<<< HEAD
            errors.append("Duplicate object_id values: " + ", ".join(sorted(duplicates)))
=======
            warnings.append("Duplicate object_id values: " + ", ".join(sorted(duplicates)))
>>>>>>> origin/subin/module2c-3-pipeline

        object_id_set = set(object_ids)
        for idx, obj in enumerate(inventory):
            relations = obj.get("scene_relations", []) if isinstance(obj, dict) else []
            for ridx, relation in enumerate(relations):
                object_ref = relation.get("object_ref") if isinstance(relation, dict) else None
                if object_ref is None:
                    continue
                if not isinstance(object_ref, str) or object_ref not in object_id_set:
                    errors.append(
                        "$.module2_common_input.scene_resources.resource_inventory"
                        f"[{idx}].scene_relations[{ridx}].object_ref invalid reference"
                    )

        subgoals = module2a_output.get("subgoals", [])
        subgoal_ids = [
            str(subgoal.get("subgoal_id"))
            for subgoal in subgoals
            if isinstance(subgoal, dict)
        ]
        duplicate_subgoals = _duplicates(subgoal_ids)
        if duplicate_subgoals:
            errors.append("Duplicate subgoal_id values: " + ", ".join(sorted(duplicate_subgoals)))
        subgoal_id_set = set(subgoal_ids)

        for idx, subgoal in enumerate(subgoals):
            if not isinstance(subgoal, dict):
                continue
            depends_on = subgoal.get("depends_on", [])
            for dep in depends_on:
                if dep not in subgoal_id_set:
                    errors.append(
                        f"$.module2a_output.subgoals[{idx}].depends_on contains unknown subgoal_id: {dep}"
                    )

        checks["inventory_count"] = len(inventory)
        checks["subgoal_count"] = len(subgoals)
        checks["warning_count"] = len(warnings)

        return ValidationReport(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            checks=checks,
        )


class Module2BOutputValidator:
    """Strict validator for Module 2-B env-only output + cross-reference checks."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or project_root()

    def validate(
        self,
        payload: dict[str, Any],
        inventory_ids: list[str],
        subgoal_ids: list[str],
    ) -> ValidationReport:
        """Validate output schema and logical consistency constraints."""
        errors: list[str] = []
        warnings: list[str] = []
        checks: dict[str, Any] = {}

        schema_errors = validate_with_schema(
            payload=payload,
            schema_path=self.root / "schemas" / "module2b_output_env_only.schema.json",
        )
        errors.extend(schema_errors)
        checks["schema_error_count"] = len(schema_errors)

        inventory_id_set = set(inventory_ids)
        subgoal_id_set = set(subgoal_ids)

        target_binding = payload.get("target_binding", {})
        target_binding_id = target_binding.get("binding_id")
        primary_targets = target_binding.get("primary_targets", [])
        candidate_ids_ranked = target_binding.get("candidate_ids_ranked", [])

        for idx, target in enumerate(primary_targets):
            object_id = target.get("object_id")
            if object_id not in inventory_id_set:
                errors.append(f"$.target_binding.primary_targets[{idx}].object_id not found in inventory")
        for idx, object_id in enumerate(candidate_ids_ranked):
            if object_id not in inventory_id_set:
                errors.append(f"$.target_binding.candidate_ids_ranked[{idx}] not found in inventory")

        env_context = payload.get("environment_context", {})
        topology_tags = env_context.get("topology_tags", [])
        relevant_structures = env_context.get("relevant_structures", [])
        numeric_estimates = env_context.get("numeric_estimates", [])

        tag_ids = {tag.get("tag_id") for tag in topology_tags if isinstance(tag, dict)}
        structure_ids = {
            item.get("environment_structure_id")
            for item in relevant_structures
            if isinstance(item, dict)
        }

        for idx, structure in enumerate(relevant_structures):
            if not isinstance(structure, dict):
                continue
            for ridx, object_id in enumerate(structure.get("related_object_ids", [])):
                if object_id not in inventory_id_set:
                    errors.append(
                        "$.environment_context.relevant_structures"
                        f"[{idx}].related_object_ids[{ridx}] not found in inventory"
                    )

        measurement_ids: set[str] = set()
        for idx, measure in enumerate(numeric_estimates):
            if not isinstance(measure, dict):
                continue
            measurement_id = measure.get("measurement_id")
            if isinstance(measurement_id, str):
                measurement_ids.add(measurement_id)

            lower_value = measure.get("lower_value")
            upper_value = measure.get("upper_value")
            bound_type = measure.get("bound_type")

            if lower_value is not None and upper_value is not None and float(lower_value) > float(upper_value):
                errors.append(
                    "$.environment_context.numeric_estimates"
                    f"[{idx}] lower_value must be <= upper_value"
                )

            if bound_type == "range" and (lower_value is None or upper_value is None):
                errors.append(
                    "$.environment_context.numeric_estimates"
                    f"[{idx}] range bound_type requires both lower_value and upper_value"
                )
            if bound_type == "upper_bound" and upper_value is None:
                errors.append(
                    "$.environment_context.numeric_estimates"
                    f"[{idx}] upper_bound requires upper_value"
                )
            if bound_type == "lower_bound" and lower_value is None:
                errors.append(
                    "$.environment_context.numeric_estimates"
                    f"[{idx}] lower_bound requires lower_value"
                )

            parameter_name = measure.get("parameter_name")
            unit = measure.get("unit")
            if not _is_parameter_unit_compatible(parameter_name=parameter_name, unit=unit):
                errors.append(
                    "$.environment_context.numeric_estimates"
                    f"[{idx}] parameter_name/unit incompatible: {parameter_name}/{unit}"
                )

            for sid in measure.get("related_environment_structure_ids", []):
                if sid not in structure_ids:
                    errors.append(
                        "$.environment_context.numeric_estimates"
                        f"[{idx}].related_environment_structure_ids has unknown id: {sid}"
                    )

        derived = payload.get("derived_constraints", {})
        constraints = derived.get("constraint_catalog", [])
        global_constraint_ids = derived.get("global_constraint_ids", [])
        subgoal_bindings = derived.get("subgoal_bindings", [])

        constraint_ids = [
            str(constraint.get("constraint_id"))
            for constraint in constraints
            if isinstance(constraint, dict)
        ]
        duplicate_constraints = _duplicates(constraint_ids)
        if duplicate_constraints:
            errors.append("Duplicate constraint_id values: " + ", ".join(sorted(duplicate_constraints)))
        constraint_id_set = set(constraint_ids)

        for cid in global_constraint_ids:
            if cid not in constraint_id_set:
                errors.append(f"$.derived_constraints.global_constraint_ids contains unknown id: {cid}")

        for idx, constraint in enumerate(constraints):
            if not isinstance(constraint, dict):
                continue
            for subgoal_id in constraint.get("subgoal_ids", []):
                if subgoal_id not in subgoal_id_set:
                    errors.append(
                        f"$.derived_constraints.constraint_catalog[{idx}].subgoal_ids has unknown id: {subgoal_id}"
                    )
            for measurement_id in constraint.get("measurement_ids", []):
                if measurement_id not in measurement_ids:
                    errors.append(
                        f"$.derived_constraints.constraint_catalog[{idx}].measurement_ids unknown: {measurement_id}"
                    )
            for binding_id in constraint.get("target_binding_ids", []):
                if binding_id != target_binding_id:
                    errors.append(
                        f"$.derived_constraints.constraint_catalog[{idx}].target_binding_ids unknown: {binding_id}"
                    )

        subgoal_binding_ids_in_order = []
        for idx, binding in enumerate(subgoal_bindings):
            if not isinstance(binding, dict):
                continue
            subgoal_id = binding.get("subgoal_id")
            if subgoal_id not in subgoal_id_set:
                errors.append(f"$.derived_constraints.subgoal_bindings[{idx}].subgoal_id unknown: {subgoal_id}")
            subgoal_binding_ids_in_order.append(subgoal_id)
            for cid in binding.get("constraint_ids", []):
                if cid not in constraint_id_set:
                    errors.append(
                        f"$.derived_constraints.subgoal_bindings[{idx}].constraint_ids unknown: {cid}"
                    )

        expected_subgoal_order = list(subgoal_ids)
        if subgoal_binding_ids_in_order != expected_subgoal_order:
            errors.append("subgoal_bindings order must match module2a_output.subgoals order")

        module3_handoff = payload.get("module3_handoff", {})
        for cid in module3_handoff.get("handoff_constraint_ids", []):
            if cid not in constraint_id_set:
                errors.append(f"$.module3_handoff.handoff_constraint_ids unknown id: {cid}")

        known_refs: set[str] = set(inventory_ids)
        if isinstance(target_binding_id, str):
            known_refs.add(target_binding_id)
        known_refs |= {item for item in tag_ids if isinstance(item, str)}
        known_refs |= {item for item in structure_ids if isinstance(item, str)}
        known_refs |= measurement_ids
        known_refs |= constraint_id_set
        known_refs |= set(subgoal_ids)

        _validate_source_refs(payload=payload, known_refs=known_refs, errors=errors)

        checks["inventory_count"] = len(inventory_ids)
        checks["structure_count"] = len(structure_ids)
        checks["measurement_count"] = len(measurement_ids)
        checks["constraint_count"] = len(constraint_id_set)

        return ValidationReport(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            checks=checks,
        )


def _validate_source_refs(payload: dict[str, Any], known_refs: set[str], errors: list[str]) -> None:
    env_context = payload.get("environment_context", {})

    for idx, tag in enumerate(env_context.get("topology_tags", [])):
        _check_source_ref_array(
            refs=tag.get("source_refs", []) if isinstance(tag, dict) else [],
            known_refs=known_refs,
            path=f"$.environment_context.topology_tags[{idx}].source_refs",
            errors=errors,
        )
    for idx, structure in enumerate(env_context.get("relevant_structures", [])):
        _check_source_ref_array(
            refs=structure.get("source_refs", []) if isinstance(structure, dict) else [],
            known_refs=known_refs,
            path=f"$.environment_context.relevant_structures[{idx}].source_refs",
            errors=errors,
        )
    for idx, measure in enumerate(env_context.get("numeric_estimates", [])):
        _check_source_ref_array(
            refs=measure.get("source_refs", []) if isinstance(measure, dict) else [],
            known_refs=known_refs,
            path=f"$.environment_context.numeric_estimates[{idx}].source_refs",
            errors=errors,
        )

    for idx, constraint in enumerate(
        payload.get("derived_constraints", {}).get("constraint_catalog", [])
    ):
        _check_source_ref_array(
            refs=constraint.get("source_refs", []) if isinstance(constraint, dict) else [],
            known_refs=known_refs,
            path=f"$.derived_constraints.constraint_catalog[{idx}].source_refs",
            errors=errors,
        )

    deferred_items = payload.get("deferred_items", {})
    for key in (
        "unresolved_target_ambiguities",
        "unresolved_environment_ambiguities",
        "not_numericized_items",
        "needs_user_or_scale_anchor",
    ):
        items = deferred_items.get(key, []) if isinstance(deferred_items, dict) else []
        for idx, item in enumerate(items):
            refs = item.get("source_refs", []) if isinstance(item, dict) else []
            _check_source_ref_array(
                refs=refs,
                known_refs=known_refs,
                path=f"$.deferred_items.{key}[{idx}].source_refs",
                errors=errors,
            )


def _check_source_ref_array(
    refs: list[str],
    known_refs: set[str],
    path: str,
    errors: list[str],
) -> None:
    if not refs:
        return
    for idx, ref in enumerate(refs):
        if ref not in known_refs:
            errors.append(f"{path}[{idx}] unknown source_ref: {ref}")


def _duplicates(items: list[str]) -> set[str]:
    seen: set[str] = set()
    dupes: set[str] = set()
    for item in items:
        if item in seen:
            dupes.add(item)
        seen.add(item)
    return dupes


def _is_parameter_unit_compatible(parameter_name: Any, unit: Any) -> bool:
    if not isinstance(parameter_name, str) or not isinstance(unit, str):
        return False
    compatible: dict[str, set[str]] = {
        "opening_width": {"m", "level_1_to_5"},
        "opening_height": {"m", "level_1_to_5"},
        "neck_inner_diameter": {"m"},
        "recess_depth": {"m", "level_1_to_5"},
        "reachable_depth": {"m", "level_1_to_5"},
        "lateral_clearance": {"m", "level_1_to_5"},
        "vertical_clearance": {"m", "level_1_to_5"},
        "available_entry_angle_deg": {"deg"},
        "target_exposed_edge_length": {"m", "level_1_to_5"},
        "support_surface_span": {"m", "level_1_to_5"},
    }
    return unit in compatible.get(parameter_name, set())
