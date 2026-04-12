"""Provider interfaces for Module 2-C input bundles (Material Reasoner 통합)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.utils import load_json, project_root


@dataclass(slots=True)
class Module2CInputResult:
    bundle: dict[str, Any]
    metadata: dict[str, Any]


class Module2CInputProvider(Protocol):
    def get_bundle(
        self,
        bundle_path: Path | None = None,
        case_id: str | None = None,
    ) -> Module2CInputResult: ...


class FileInputProvider:
    def get_bundle(
        self,
        bundle_path: Path | None = None,
        case_id: str | None = None,
    ) -> Module2CInputResult:
        if bundle_path is None:
            raise ValueError("FileInputProvider requires --bundle path.")
        payload = load_json(bundle_path)
        return Module2CInputResult(
            bundle=payload,
            metadata={"provider": "file", "bundle_path": str(bundle_path), "case_id": case_id},
        )


class MockInputProvider:
    def __init__(self, fixtures_root: Path | None = None) -> None:
        self.cases_root = (fixtures_root or project_root() / "fixtures") / "module2c_cases"

    def get_bundle(
        self,
        bundle_path: Path | None = None,
        case_id: str | None = None,
    ) -> Module2CInputResult:
        if case_id is None:
            raise ValueError("MockInputProvider requires --case-id.")
        case_dir = self.cases_root / case_id
        if not case_dir.exists():
            raise FileNotFoundError(f"Unknown Module 2-C fixture case_id: {case_id}")
        bundle = load_json(case_dir / "bundle.json")
        return Module2CInputResult(
            bundle=bundle,
            metadata={"provider": "mock", "case_id": case_id,
                      "bundle_path": str(case_dir / "bundle.json")},
        )


class Module2BOutputProvider:
    """Module 2-B run_dir을 읽어 Module 2-C input으로 변환.
    
    image_path + target_name + task를 주면 Material Reasoner도 함께 호출함.
    """

    def __init__(
        self,
        image_path: Path | None = None,
        target_name: str = "business card",
        task: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.image_path = Path(image_path) if image_path else None
        self.target_name = target_name
        self.task = task
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def get_bundle(
        self,
        bundle_path: Path | None = None,
        case_id: str | None = None,
    ) -> Module2CInputResult:
        if bundle_path is None:
            raise ValueError("Module2BOutputProvider requires --bundle (module2b run_dir).")

        p = Path(bundle_path)
        if p.is_dir():
            module2b_output = load_json(p / "module2b_output.json")
            nc_path = p / "normalized_context.json"
            normalized_context = load_json(nc_path) if nc_path.exists() else {}
            module1_normalized = _find_module1_normalized(p)
            if module1_normalized:
                normalized_context = _merge_module1_into_context(
                    normalized_context, module1_normalized
                )
        else:
            module2b_output = load_json(p)
            normalized_context = load_json(p.parent / "normalized_context.json")

        # Material Reasoner 호출
        material_result = None
        if self.image_path and self.image_path.exists():
            try:
                from app.module2c.material_reasoner import run_material_reasoner
                task_text = self.task or _extract_task(normalized_context)
                material_result = run_material_reasoner(
                    image_path=self.image_path,
                    target_name=self.target_name,
                    task=task_text,
                    api_key=self.api_key,
                )
                print(f"[Material Reasoner] 완료: difficulty={material_result.get('manipulation_analysis', {}).get('difficulty')}")
            except Exception as e:
                print(f"[Material Reasoner] 경고: {e}")

        bundle = _convert_module2b_to_2c_input(
            module2b_output, normalized_context, material_result
        )

        return Module2CInputResult(
            bundle=bundle,
            metadata={
                "provider": "module2b_output",
                "bundle_path": str(bundle_path),
                "case_id": case_id,
                "material_reasoner_used": material_result is not None,
            },
        )


# ─── Module 1 탐색 및 병합 ─────────────────────────────────────

def _find_module1_normalized(module2b_dir: Path) -> dict[str, Any] | None:
    search_roots = [
        module2b_dir.parent,
        module2b_dir.parent.parent / "module1-2B" / "outputs",
        module2b_dir.parent.parent / "outputs",
    ]
    for root in search_roots:
        if not root.exists():
            continue
        candidates = sorted(
            [d for d in root.iterdir() if d.is_dir() and d.name.startswith("run_")],
            reverse=True,
        )
        for candidate in candidates:
            path = candidate / "normalized_module1_output.json"
            if path.exists():
                try:
                    return load_json(path)
                except Exception:
                    continue
    return None


def _merge_module1_into_context(
    normalized_context: dict[str, Any],
    module1_normalized: dict[str, Any],
) -> dict[str, Any]:
    import copy
    ctx = copy.deepcopy(normalized_context)
    m1_objects: dict[str, dict[str, Any]] = {}
    for obj in module1_normalized.get("objects", []):
        oid = obj.get("raw_object_id") or obj.get("object_id", "")
        if oid:
            m1_objects[oid] = obj
    for inv_obj in ctx.get("inventory", []):
        oid = inv_obj.get("object_id", "")
        m1_obj = m1_objects.get(oid)
        if not m1_obj:
            continue
        if not inv_obj.get("object_name"):
            inv_obj["object_name"] = m1_obj.get("object_name", "")
        if not inv_obj.get("object_type_canonical"):
            inv_obj["object_type_canonical"] = m1_obj.get("object_type_canonical", "")
        m1_geo = m1_obj.get("geometry_cues", {})
        inv_geo = inv_obj.get("geometry_cues", {})
        for key, val in m1_geo.items():
            if not inv_geo.get(key) or inv_geo.get(key) in ("unknown", "none", False, 0):
                inv_geo[key] = val
        if not inv_obj.get("functional_parts") and m1_obj.get("functional_parts"):
            inv_obj["functional_parts"] = m1_obj["functional_parts"]
        if m1_obj.get("physical_properties"):
            inv_obj["physical_properties"] = m1_obj["physical_properties"]
    return ctx


# ─── 변환 함수 ────────────────────────────────────────────────

def _convert_module2b_to_2c_input(
    module2b_output: dict[str, Any],
    normalized_context: dict[str, Any],
    material_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "task": _extract_task(normalized_context),
        "tool_constraints": _extract_tool_constraints(
            module2b_output, normalized_context, material_result
        ),
        "scene_objects": _extract_scene_objects(normalized_context),
        "object_physical_properties": _extract_physical_properties(normalized_context),
    }


def _extract_task(normalized_context: dict[str, Any]) -> str:
    brief = normalized_context.get("task_brief", {})
    if isinstance(brief, dict):
        for key in ("user_goal", "task_text", "text", "description", "brief"):
            val = brief.get(key)
            if val and isinstance(val, str):
                return val
    corpus = normalized_context.get("task_text_corpus", [])
    if corpus:
        return " ".join(corpus)
    task_model = normalized_context.get("task_model", {})
    if isinstance(task_model, dict):
        for key in ("user_goal", "task_text", "text", "description"):
            val = task_model.get(key)
            if val and isinstance(val, str):
                return val
    return normalized_context.get("task_id", "unknown_task")


def _extract_tool_constraints(
    module2b_output: dict[str, Any],
    normalized_context: dict[str, Any],
    material_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    derived = module2b_output.get("derived_constraints", {})
    env_ctx = module2b_output.get("environment_context", {})
    subgoals = normalized_context.get("subgoals", [])

    global_constraints: list[str] = []
    global_ids = set(derived.get("global_constraint_ids", []))
    for c in derived.get("constraint_catalog", []):
        if c.get("constraint_id") in global_ids:
            desc = _constraint_to_text(c)
            if desc:
                global_constraints.append(desc)

    subgoal_bindings = {
        sb["subgoal_id"]: sb.get("constraint_ids", [])
        for sb in derived.get("subgoal_bindings", [])
    }
    catalog_by_id = {
        c["constraint_id"]: c
        for c in derived.get("constraint_catalog", [])
    }
    subgoal_constraints: list[dict[str, Any]] = []
    for sg in subgoals:
        sg_id = sg["subgoal_id"]
        bound_ids = subgoal_bindings.get(sg_id, [])
        bound = [catalog_by_id[cid] for cid in bound_ids if cid in catalog_by_id]
        subgoal_constraints.append({
            "subgoal_id": sg_id,
            "objective": sg.get("objective", ""),
            "required_atoms": [c["parameter_name"] for c in bound if c.get("hardness") == "hard"],
            "preferred_atoms": [c["parameter_name"] for c in bound if c.get("hardness") == "soft" and c.get("priority") == "medium"],
            "risk_atoms_to_avoid": [c["parameter_name"] for c in bound if c.get("priority") == "high" and c.get("hardness") == "soft"],
        })

    numeric_estimates: dict[str, Any] = {}
    for m in env_ctx.get("numeric_estimates", []):
        param = m.get("parameter_name")
        if param:
            numeric_estimates[param] = m.get("upper_value") if m.get("upper_value") is not None else m.get("lower_value")

    derived_summary: dict[str, str] = {}
    for c in derived.get("constraint_catalog", []):
        param = c.get("parameter_name")
        if param and c.get("bound_type"):
            bound_val = c.get("upper_value") or c.get("lower_value")
            if bound_val is not None:
                derived_summary[param] = f"{c['bound_type']}:{bound_val}{c.get('unit','')}"

    topology_tags = env_ctx.get("topology_tags", [])
    scene_capability_bias = [tag.get("label") for tag in topology_tags if tag.get("label")]

    access = env_ctx.get("access_path_profile", {})
    constraint_context = (
        f"entry_mode={access.get('entry_mode','unknown')}, "
        f"confinement_level={access.get('confinement_level','unknown')}, "
        f"requires_deep_reach={access.get('requires_deep_reach', False)}"
    )

    result: dict[str, Any] = {
        "global_constraints": global_constraints,
        "subgoal_constraints": subgoal_constraints,
        "numeric_estimates": numeric_estimates,
        "derived_constraints": derived_summary,
        "scene_capability_bias": scene_capability_bias,
        "constraint_context": constraint_context,
    }

    # Material Reasoner output 병합
    if material_result:
        result["target_material_constraints"] = {
            "target_name": material_result.get("target_name", ""),
            "physical_properties": material_result.get("physical_properties", {}),
            "manipulation_analysis": material_result.get("manipulation_analysis", {}),
            "material_hypotheses": material_result.get("material_hypotheses", []),
            "uncertainty": material_result.get("uncertainty", []),
        }

    return result


def _extract_scene_objects(normalized_context: dict[str, Any]) -> list[dict[str, Any]]:
    seen_ids: set[str] = set()
    scene_objects = []
    for obj in normalized_context.get("inventory", []):
        object_id = obj.get("object_id") or obj.get("raw_object_id", "")
        if not object_id or object_id in seen_ids:
            continue
        seen_ids.add(object_id)
        geo = obj.get("geometry_cues", {})
        graspable_regions = []
        functional_regions = []
        for part in obj.get("functional_parts", []):
            role = part.get("role_canonical", "")
            part_name = part.get("part_name", "")
            if any(k in role for k in ["grasp", "handle", "grip"]):
                graspable_regions.append(part_name)
            else:
                functional_regions.append(part_name)
        name = obj.get("object_name") or obj.get("name") or object_id
        scene_objects.append({
            "object_id": object_id,
            "name": name,
            "center_world": geo.get("center_world", [0.0, 0.0, 0.0]),
            "aabb_min": geo.get("aabb_min", [0.0, 0.0, 0.0]),
            "aabb_max": geo.get("aabb_max", [0.0, 0.0, 0.0]),
            "principal_axis_hint": geo.get("principal_axis_hint", "z_axis"),
            "graspable_regions": graspable_regions if graspable_regions else ["body"],
            "functional_regions": functional_regions if functional_regions else ["surface"],
        })
    return scene_objects


def _extract_physical_properties(normalized_context: dict[str, Any]) -> list[dict[str, Any]]:
    seen_ids: set[str] = set()
    properties = []
    for obj in normalized_context.get("inventory", []):
        object_id = obj.get("object_id") or obj.get("raw_object_id", "")
        if not object_id or object_id in seen_ids:
            continue
        seen_ids.add(object_id)
        geo = obj.get("geometry_cues", {})
        physical = obj.get("physical_properties", {})
        inferred_functions = list({
            part.get("role_canonical", "")
            for part in obj.get("functional_parts", [])
            if part.get("role_canonical")
        })
        surface_friction = "medium"
        if isinstance(physical, dict):
            fl = physical.get("surface_friction", {})
            if isinstance(fl, dict):
                surface_friction = fl.get("label", "medium")
            elif isinstance(fl, str):
                surface_friction = fl
        rigidity = "rigid"
        if isinstance(physical, dict):
            dl = physical.get("deformability", {})
            if isinstance(dl, dict):
                deform = dl.get("label", "low")
                if deform == "high":
                    rigidity = "flexible"
                elif deform == "medium":
                    rigidity = "semi-rigid"
        for part in obj.get("functional_parts", []):
            tags = part.get("local_property_tags", [])
            if any(t in tags for t in ["high_friction", "rubber", "grip"]):
                surface_friction = "high"
            elif any(t in tags for t in ["low_friction", "smooth", "slippery"]):
                surface_friction = "low"
            if "flexible" in tags or "deformable" in tags:
                rigidity = "flexible"
            elif "semi_rigid" in tags:
                rigidity = "semi-rigid"
        estimated_mass = 0.1
        if isinstance(physical, dict):
            ml = physical.get("mass_category", {})
            if isinstance(ml, dict):
                label = ml.get("label", "light")
                mass_map = {"very_light": 0.05, "light": 0.1, "medium": 0.3, "heavy": 1.0}
                estimated_mass = mass_map.get(label, 0.1)
        properties.append({
            "object_id": object_id,
            "shape_category": geo.get("shape_category", "unknown"),
            "estimated_mass_kg": estimated_mass,
            "surface_friction": surface_friction,
            "rigidity": rigidity,
            "inferred_functions": inferred_functions,
        })
    return properties


def _constraint_to_text(c: dict[str, Any]) -> str:
    param = c.get("parameter_name", "")
    bound = c.get("bound_type", "")
    val = c.get("upper_value") or c.get("lower_value")
    unit = c.get("unit", "")
    if not param:
        return ""
    if val is not None:
        return f"{param} {bound} {val}{unit}"
    return param
