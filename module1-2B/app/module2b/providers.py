"""Provider interfaces and implementations for Module 2-B input bundles."""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.bridges.module1_to_module2a import build_module2_bridge_package
from app.models.module1_normalizer import normalize_module1_raw
from app.module2a.reasoner import generate_module2a_output
from app.providers.vision_provider import VisionProvider
from app.utils import load_json, load_yaml, project_root
from app.validators.module1_validator import Module1Validator
from app.validators.schema_validator import validate_with_schema


@dataclass(slots=True)
class Module2BBundleResult:
    """Resolved Module 2-B input bundle and metadata."""

    bundle: dict[str, Any]
    metadata: dict[str, Any]


class Module2BBundleProvider(Protocol):
    """Contract for providers that return module2_common_input + module2a_output bundle."""

    def get_bundle(
        self,
        bundle_path: Path | None = None,
        module2_common_path: Path | None = None,
        module2a_output_path: Path | None = None,
        case_id: str | None = None,
        image_path: Path | None = None,
        module1_output_path: Path | None = None,
        user_goal: str | None = None,
        success_criteria: list[str] | None = None,
        task_notes: list[str] | None = None,
    ) -> Module2BBundleResult:
        """Return a resolved bundle dictionary."""


class FileBundleProvider:
    """Load Module 2-B bundle from file(s)."""

    def get_bundle(
        self,
        bundle_path: Path | None = None,
        module2_common_path: Path | None = None,
        module2a_output_path: Path | None = None,
        case_id: str | None = None,
        image_path: Path | None = None,
        module1_output_path: Path | None = None,
        user_goal: str | None = None,
        success_criteria: list[str] | None = None,
        task_notes: list[str] | None = None,
    ) -> Module2BBundleResult:
        """Load bundle from one file or two split files."""
        if bundle_path is not None:
            payload = load_json(bundle_path)
            if isinstance(payload, dict) and isinstance(payload.get("module2_common_input"), dict):
                payload = dict(payload)
                payload["module2_common_input"] = _prepare_module2_common_for_module2b(
                    payload["module2_common_input"],
                    fallback_tag=case_id or bundle_path.stem,
                )
            return Module2BBundleResult(
                bundle=payload,
                metadata={
                    "provider": "file",
                    "mode": "bundle",
                    "bundle_path": str(bundle_path),
                    "case_id": case_id,
                },
            )

        if module2_common_path is None or module2a_output_path is None:
            raise ValueError(
                "FileBundleProvider requires --bundle or both --module2-common and --module2a-output."
            )

        module2_common_input = _prepare_module2_common_for_module2b(
            load_json(module2_common_path),
            fallback_tag=case_id or module2_common_path.stem,
        )
        module2a_output = load_json(module2a_output_path)
        return Module2BBundleResult(
            bundle={
                "module2_common_input": module2_common_input,
                "module2a_output": module2a_output,
            },
            metadata={
                "provider": "file",
                "mode": "split",
                "module2_common_path": str(module2_common_path),
                "module2a_output_path": str(module2a_output_path),
                "case_id": case_id,
            },
        )


class MockBundleProvider:
    """Fixture-backed Module 2-B provider with deterministic case resolution."""

    def __init__(self, fixtures_root: Path | None = None) -> None:
        self.fixtures_root = fixtures_root or (project_root() / "fixtures")
        self.cases_root = self.fixtures_root / "module2b_cases"

    def get_bundle(
        self,
        bundle_path: Path | None = None,
        module2_common_path: Path | None = None,
        module2a_output_path: Path | None = None,
        case_id: str | None = None,
        image_path: Path | None = None,
        module1_output_path: Path | None = None,
        user_goal: str | None = None,
        success_criteria: list[str] | None = None,
        task_notes: list[str] | None = None,
    ) -> Module2BBundleResult:
        """Load bundle from fixture case id."""
        if case_id is None:
            raise ValueError("MockBundleProvider requires --case-id.")

        case_dir = self.cases_root / case_id
        if not case_dir.exists():
            raise FileNotFoundError(f"Unknown Module 2-B fixture case_id: {case_id}")

        bundle = load_json(case_dir / "bundle.json")
        return Module2BBundleResult(
            bundle=bundle,
            metadata={
                "provider": "mock",
                "mode": "fixture_bundle",
                "case_id": case_id,
                "bundle_path": str(case_dir / "bundle.json"),
            },
        )


class VisionBundleProvider:
    """Build Module 2-B bundle from image via Module 1 vision + Module 2-A."""

    def get_bundle(
        self,
        bundle_path: Path | None = None,
        module2_common_path: Path | None = None,
        module2a_output_path: Path | None = None,
        case_id: str | None = None,
        image_path: Path | None = None,
        module1_output_path: Path | None = None,
        user_goal: str | None = None,
        success_criteria: list[str] | None = None,
        task_notes: list[str] | None = None,
    ) -> Module2BBundleResult:
        """Generate Module 2-B bundle from an input image."""
        if image_path is None:
            raise ValueError("VisionBundleProvider requires --image.")
        if bundle_path is not None or module2_common_path is not None or module2a_output_path is not None:
            raise ValueError(
                "provider=vision for Module 2-B does not use --bundle/--module2-common/--module2a-output."
            )
        if module1_output_path is not None:
            raise ValueError("provider=vision for Module 2-B does not use --module1-output.")

        root = project_root()
        vocab_registry = load_json(root / "configs" / "vocab_registry.json")
        atom_cfg = load_yaml(root / "configs" / "module1_to_module2a_atom_rules.yaml")
        provider_result = VisionProvider().get_module1_output(
            image_path=image_path,
            case_id=case_id,
            module1_output_path=None,
        )
        raw_module1_output = provider_result.raw_output
        validation = Module1Validator().validate(raw_module1_output)
        if not validation.valid:
            raise ValueError(
                "Module 1 validation failed before Module 2-B generation: "
                + " | ".join(validation.errors[:3])
                + (" ..." if len(validation.errors) > 3 else "")
            )

        normalized = normalize_module1_raw(raw_module1_output)
        bridge_artifacts = build_module2_bridge_package(
            normalized=normalized,
            rule_cfg=atom_cfg,
            vocab_registry=vocab_registry,
        )
        module2_common_input_template = bridge_artifacts["module2_common_input_template"]
        _apply_task_overrides(
            module2_common_input=module2_common_input_template,
            user_goal=user_goal,
            success_criteria=success_criteria,
            task_notes=task_notes,
        )
        module2a_output = generate_module2a_output(
            module2_common_input=module2_common_input_template,
            vocab_registry=vocab_registry,
        )
        module2a_schema_errors = validate_with_schema(
            payload=module2a_output,
            schema_path=root / "schemas" / "module2a_output.schema.json",
        )
        if module2a_schema_errors:
            raise ValueError(
                "Module 2-A validation failed before Module 2-B generation: "
                + " | ".join(module2a_schema_errors[:3])
                + (" ..." if len(module2a_schema_errors) > 3 else "")
            )

        resolved_case_id = case_id or image_path.stem
        module2_common_input = _prepare_module2_common_for_module2b(
            module2_common_input_template,
            fallback_tag=resolved_case_id,
        )

        return Module2BBundleResult(
            bundle={
                "module2_common_input": module2_common_input,
                "module2a_output": module2a_output,
            },
            metadata={
                "provider": "vision",
                "mode": "image_to_bundle",
                "case_id": resolved_case_id,
                "image_path": str(image_path),
                "module1_provider_metadata": provider_result.metadata,
                "module1_warning_count": len(validation.warnings) + len(normalized.warnings),
                "bridge_rule_version": atom_cfg.get("bridge_rule_version"),
            },
        )


def _apply_task_overrides(
    module2_common_input: dict[str, Any],
    user_goal: str | None,
    success_criteria: list[str] | None,
    task_notes: list[str] | None,
) -> None:
    task_brief = module2_common_input.setdefault("task_brief", {})
    task_brief.setdefault("user_goal", None)
    task_brief.setdefault("success_criteria", [])
    task_brief.setdefault("task_notes", [])

    if user_goal is not None:
        task_brief["user_goal"] = user_goal
    if success_criteria is not None:
        task_brief["success_criteria"] = [item for item in success_criteria if item]
    if task_notes is not None:
        task_brief["task_notes"] = [item for item in task_notes if item]


def _prepare_module2_common_for_module2b(
    module2_common_input: dict[str, Any],
    fallback_tag: str | None = None,
) -> dict[str, Any]:
    """Coerce bridge/template-style module2_common_input to Module 2-B derived-min format."""
    payload = deepcopy(module2_common_input) if isinstance(module2_common_input, dict) else {}

    payload["schema_name"] = "module2_common_input_for_module2b_derived_min"
    payload["schema_version"] = "0.1"

    task_brief_raw = payload.get("task_brief")
    if not isinstance(task_brief_raw, dict):
        task_brief_raw = {}
    payload["task_brief"] = {
        "user_goal": task_brief_raw.get("user_goal"),
        "success_criteria": [
            str(item) for item in task_brief_raw.get("success_criteria", []) if item is not None
        ],
        "task_notes": [str(item) for item in task_brief_raw.get("task_notes", []) if item is not None],
    }
    payload["task_id"] = _resolve_task_id(
        raw_task_id=payload.get("task_id"),
        fallback_tag=fallback_tag,
        task_brief=payload["task_brief"],
    )

    scene_resources_raw = payload.get("scene_resources")
    scene_resources = deepcopy(scene_resources_raw) if isinstance(scene_resources_raw, dict) else {}
    summary_raw = scene_resources.get("resource_summary")
    summary = deepcopy(summary_raw) if isinstance(summary_raw, dict) else {}
    inventory_raw = scene_resources.get("resource_inventory")
    inventory_items = inventory_raw if isinstance(inventory_raw, list) else []

    summary["affordance_histogram"] = _coerce_histogram(
        raw=summary.get("affordance_histogram"),
        fallback_counts=_derive_histogram_from_inventory(
            inventory_items=inventory_items,
            item_key="atom",
        ),
    )
    summary["risk_histogram"] = _coerce_histogram(raw=summary.get("risk_histogram"))
    summary["primitive_histogram"] = _coerce_histogram(raw=summary.get("primitive_histogram"))

    scene_resources["resource_summary"] = summary
    scene_resources["resource_inventory"] = _coerce_inventory_for_module2b(inventory_items)
    payload["scene_resources"] = scene_resources
    return payload


def _resolve_task_id(
    raw_task_id: Any,
    fallback_tag: str | None,
    task_brief: dict[str, Any],
) -> str:
    if isinstance(raw_task_id, str) and raw_task_id.strip():
        return raw_task_id.strip()

    user_goal = task_brief.get("user_goal")
    seed = fallback_tag or (str(user_goal).strip() if user_goal else "") or "ad_hoc"
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", seed).strip("_").lower()
    if not normalized:
        normalized = "ad_hoc"
    return f"task_{normalized[:48]}"


def _coerce_histogram(
    raw: Any,
    fallback_counts: dict[str, int] | None = None,
) -> dict[str, int]:
    source = raw if isinstance(raw, dict) else {}
    result: dict[str, int] = {}
    for key, value in source.items():
        if not isinstance(key, str):
            continue
        try:
            result[key] = max(0, int(value))
        except (TypeError, ValueError):
            continue
    if result:
        return result
    return dict(fallback_counts or {})


def _derive_histogram_from_inventory(
    inventory_items: list[Any],
    item_key: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in inventory_items:
        if not isinstance(item, dict):
            continue
        label = item.get(item_key)
        if not isinstance(label, str) or not label.strip():
            continue
        key = label.strip()
        counts[key] = counts.get(key, 0) + 1
    return counts


def _coerce_inventory_for_module2b(inventory_items: list[Any]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    next_auto_idx = 1

    for item in inventory_items:
        if not isinstance(item, dict):
            continue

        object_id_raw = item.get("object_id")
        object_id = str(object_id_raw).strip() if isinstance(object_id_raw, str) else ""
        if not object_id:
            while f"obj_{next_auto_idx:02d}" in grouped:
                next_auto_idx += 1
            object_id = f"obj_{next_auto_idx:02d}"
            next_auto_idx += 1

        current = grouped.setdefault(object_id, _default_inventory_object(object_id))
        _merge_inventory_entry(current=current, item=item)

    return list(grouped.values())


def _default_inventory_object(object_id: str) -> dict[str, Any]:
    return {
        "object_id": object_id,
        "object_name": object_id,
        "object_type_canonical": "unknown",
        "coarse_location_hint": "unknown",
        "visibility": "partial",
        "accessibility": "partial",
        "state": {
            "pose_class": "unknown",
            "support_context": "unknown",
        },
        "scene_relations": [],
        "geometry_cues": {
            "shape_class": "unknown",
            "aspect_ratio_hint": "unknown",
            "size_relative": "unknown",
            "thickness_class": "unknown",
            "primary_contact_profile": "unknown",
            "has_pointed_or_thin_end": False,
            "has_flat_contact_face": False,
            "has_open_cavity": False,
            "roll_risk_source": "none",
        },
        "functional_parts": [],
        "target_mode_numeric_summary": {
            "exposure_ratio": None,
            "clearance_ratio": None,
            "usable_span_m": None,
            "local_thickness_m": None,
            "tip_radius_m": None,
            "flat_patch_m2": None,
            "approach_directions_count": None,
        },
        "uncertainty": {
            "overall": 0.5,
        },
    }


def _merge_inventory_entry(current: dict[str, Any], item: dict[str, Any]) -> None:
    object_name = item.get("object_name")
    if (
        isinstance(object_name, str)
        and object_name.strip()
        and current["object_name"] == current["object_id"]
    ):
        current["object_name"] = object_name.strip()

    object_type = item.get("object_type_canonical")
    if (
        isinstance(object_type, str)
        and object_type.strip()
        and current["object_type_canonical"] == "unknown"
    ):
        current["object_type_canonical"] = object_type.strip()

    coarse_location = item.get("coarse_location_hint")
    if (
        isinstance(coarse_location, str)
        and coarse_location.strip()
        and current["coarse_location_hint"] == "unknown"
    ):
        current["coarse_location_hint"] = coarse_location.strip()

    visibility = item.get("visibility")
    if visibility in {"full", "partial", "heavily_occluded"}:
        current["visibility"] = visibility

    accessibility = item.get("accessibility")
    if accessibility in {"clear", "partial", "occluded", "nested", "entangled"}:
        current["accessibility"] = accessibility

    state = item.get("state")
    if isinstance(state, dict):
        pose_class = state.get("pose_class")
        support_context = state.get("support_context")
        if isinstance(pose_class, str) and pose_class.strip():
            current["state"]["pose_class"] = pose_class.strip()
        if isinstance(support_context, str) and support_context.strip():
            current["state"]["support_context"] = support_context.strip()

    geometry = item.get("geometry_cues")
    if isinstance(geometry, dict):
        for key in (
            "shape_class",
            "aspect_ratio_hint",
            "size_relative",
            "thickness_class",
            "primary_contact_profile",
            "roll_risk_source",
        ):
            value = geometry.get(key)
            if (
                isinstance(value, str)
                and value.strip()
                and current["geometry_cues"].get(key) in {"unknown", "none"}
            ):
                current["geometry_cues"][key] = value.strip()
        for key in ("has_pointed_or_thin_end", "has_flat_contact_face", "has_open_cavity"):
            value = geometry.get(key)
            if isinstance(value, bool):
                current["geometry_cues"][key] = value

    target_summary = item.get("target_mode_numeric_summary")
    if isinstance(target_summary, dict):
        for key in (
            "exposure_ratio",
            "clearance_ratio",
            "usable_span_m",
            "local_thickness_m",
            "tip_radius_m",
            "flat_patch_m2",
            "approach_directions_count",
        ):
            if target_summary.get(key) is not None:
                current["target_mode_numeric_summary"][key] = target_summary.get(key)

    uncertainty_score = _extract_uncertainty_score(item)
    current["uncertainty"]["overall"] = max(current["uncertainty"]["overall"], uncertainty_score)

    _merge_functional_parts(current=current, item=item)
    _merge_scene_relations(current=current, item=item)


def _extract_uncertainty_score(item: dict[str, Any]) -> float:
    candidate = None
    uncertainty = item.get("uncertainty")
    if isinstance(uncertainty, dict):
        candidate = uncertainty.get("overall")
    if candidate is None:
        candidate = item.get("uncertainty_overall")
    try:
        value = float(candidate)
    except (TypeError, ValueError):
        value = 0.5
    return max(0.0, min(1.0, value))


def _merge_functional_parts(current: dict[str, Any], item: dict[str, Any]) -> None:
    parts = current["functional_parts"]
    part_map = {part.get("part_name"): part for part in parts if isinstance(part, dict)}

    raw_parts = item.get("functional_parts")
    if isinstance(raw_parts, list):
        for raw_part in raw_parts:
            if not isinstance(raw_part, dict):
                continue
            part_name = str(raw_part.get("part_name", "")).strip()
            if not part_name:
                continue
            merged = part_map.get(part_name) or {
                "part_name": part_name,
                "role_canonical": "unknown",
                "contact_profile": "unknown",
                "local_property_tags": [],
            }
            role = raw_part.get("role_canonical")
            if isinstance(role, str) and role.strip() and merged["role_canonical"] == "unknown":
                merged["role_canonical"] = role.strip()
            profile = raw_part.get("contact_profile")
            if isinstance(profile, str) and profile.strip() and merged["contact_profile"] == "unknown":
                merged["contact_profile"] = profile.strip()
            tags = raw_part.get("local_property_tags")
            if isinstance(tags, list):
                merged["local_property_tags"] = sorted(
                    {
                        *[
                            str(tag)
                            for tag in merged.get("local_property_tags", [])
                            if str(tag).strip()
                        ],
                        *[str(tag) for tag in tags if str(tag).strip()],
                    }
                )
            part_map[part_name] = merged

    part_name = item.get("part_name")
    if isinstance(part_name, str) and part_name.strip():
        evidence = item.get("evidence")
        role = "unknown"
        profile = "unknown"
        if isinstance(evidence, dict):
            role_raw = evidence.get("role_canonical")
            profile_raw = evidence.get("contact_profile")
            if isinstance(role_raw, str) and role_raw.strip():
                role = role_raw.strip()
            if isinstance(profile_raw, str) and profile_raw.strip():
                profile = profile_raw.strip()
        merged = part_map.get(part_name.strip()) or {
            "part_name": part_name.strip(),
            "role_canonical": "unknown",
            "contact_profile": "unknown",
            "local_property_tags": [],
        }
        if merged["role_canonical"] == "unknown":
            merged["role_canonical"] = role
        if merged["contact_profile"] == "unknown":
            merged["contact_profile"] = profile
        part_map[part_name.strip()] = merged

    current["functional_parts"] = list(part_map.values())


def _merge_scene_relations(current: dict[str, Any], item: dict[str, Any]) -> None:
    existing = {
        (
            str(rel.get("relation", "")).strip(),
            rel.get("object_ref"),
            str(rel.get("relation_note", "")).strip(),
        )
        for rel in current["scene_relations"]
        if isinstance(rel, dict)
    }

    raw_relations = item.get("scene_relations")
    if not isinstance(raw_relations, list):
        return
    for rel in raw_relations:
        if not isinstance(rel, dict):
            continue
        relation = str(rel.get("relation", "unknown")).strip() or "unknown"
        object_ref_raw = rel.get("object_ref")
        object_ref = (
            object_ref_raw.strip()
            if isinstance(object_ref_raw, str) and object_ref_raw.strip()
            else None
        )
        relation_note = str(rel.get("relation_note", "")).strip()
        key = (relation, object_ref, relation_note)
        if key in existing:
            continue
        current["scene_relations"].append(
            {
                "relation": relation,
                "object_ref": object_ref,
                "relation_note": relation_note,
            }
        )
        existing.add(key)


def list_module2b_case_ids(fixtures_root: Path | None = None) -> list[str]:
    """Return declared Module 2-B fixture case IDs."""
    root = fixtures_root or (project_root() / "fixtures")
    index = load_json(root / "module2b_cases" / "index.json")
    return list(index.get("cases", []))
