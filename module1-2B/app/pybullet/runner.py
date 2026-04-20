"""PyBullet experiment runner for Module 1 surrogate validation scenarios."""

from __future__ import annotations

import random
from typing import Any

from app.pybullet.scenarios.drop_test import run_drop_test
from app.pybullet.scenarios.force_response_test import run_force_response_test
from app.pybullet.scenarios.slide_test import run_slide_test


def run_pybullet_experiments(
    proxy_spec: dict[str, Any],
    surrogate_spec: dict[str, Any],
    mapping_cfg: dict[str, Any],
    scenarios: list[str] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run object-wise experiments in DIRECT mode with fixed timestep."""
    try:
        import pybullet as p  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("pybullet is required for experiments.") from exc

    if seed is None:
        seed = int(mapping_cfg.get("scenario_defaults", {}).get("seed", 123))
    random.seed(seed)

    available_scenarios = ["drop_test", "slide_test", "force_response_test"]
    selected = available_scenarios if not scenarios or "all" in scenarios else scenarios

    timestep = float(mapping_cfg["scenario_defaults"]["fixed_timestep_s"])
    gravity = float(mapping_cfg["scenario_defaults"]["gravity_mps2"])
    proxy_by_object = {item["object_id"]: item for item in proxy_spec["objects"]}
    surrogate_by_object = {item["object_id"]: item for item in surrogate_spec["objects"]}

    metrics: dict[str, dict[str, Any]] = {}
    trajectory_rows: list[dict[str, Any]] = []
    applied_dynamics_rows: list[dict[str, Any]] = []

    for object_id, obj_proxy in proxy_by_object.items():
        if object_id not in surrogate_by_object:
            continue
        surrogate_entry = surrogate_by_object[object_id]
        metrics[object_id] = {}

        for scenario_name in selected:
            sim_result = _run_single_scenario(
                p=p,
                object_id=object_id,
                proxy_entry=obj_proxy,
                surrogate_entry=surrogate_entry,
                mapping_cfg=mapping_cfg,
                scenario_name=scenario_name,
                timestep=timestep,
                gravity=gravity,
            )
            scenario_metrics = sim_result["metrics"]
            metrics[object_id][scenario_name] = scenario_metrics
            for row in sim_result["trajectory"]:
                row_with_meta = dict(row)
                row_with_meta["object_id"] = object_id
                row_with_meta["scenario"] = scenario_name
                trajectory_rows.append(row_with_meta)
            applied_dynamics_rows.append(sim_result["applied_dynamics"])

    return {
        "seed": seed,
        "fixed_timestep_s": timestep,
        "scenarios": selected,
        "metrics": metrics,
        "trajectory_rows": trajectory_rows,
        "applied_dynamics": {
            "schema_name": "pybullet_applied_dynamics",
            "schema_version": "0.1",
            "rows": applied_dynamics_rows,
        },
    }


def _run_single_scenario(
    p: Any,
    object_id: str,
    proxy_entry: dict[str, Any],
    surrogate_entry: dict[str, Any],
    mapping_cfg: dict[str, Any],
    scenario_name: str,
    timestep: float,
    gravity: float,
) -> dict[str, Any]:
    client_id = p.connect(p.DIRECT)
    try:
        p.resetSimulation(physicsClientId=client_id)
        p.setGravity(0, 0, gravity, physicsClientId=client_id)
        p.setTimeStep(timestep, physicsClientId=client_id)
        p.setRealTimeSimulation(0, physicsClientId=client_id)
        p.setPhysicsEngineParameter(
            fixedTimeStep=timestep,
            deterministicOverlappingPairs=1,
            physicsClientId=client_id,
        )

        plane_shape = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=client_id)
        plane_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=plane_shape,
            basePosition=[0.0, 0.0, 0.0],
            physicsClientId=client_id,
        )
        p.changeDynamics(plane_id, -1, lateralFriction=0.9, physicsClientId=client_id)

        scenario_cfg = mapping_cfg["scenario_defaults"][scenario_name]
        body_id = _create_object_body(
            p=p,
            client_id=client_id,
            proxy_entry=proxy_entry,
            surrogate_entry=surrogate_entry,
            start_height=float(scenario_cfg["start_height_m"]),
        )
        surrogate = surrogate_entry["surrogate_parameters"]
        p.changeDynamics(
            body_id,
            -1,
            lateralFriction=float(surrogate["lateral_friction"]),
            restitution=float(surrogate["restitution"]),
            linearDamping=float(surrogate["linear_damping"]),
            angularDamping=float(surrogate["angular_damping"]),
            physicsClientId=client_id,
        )
        dynamics_info = p.getDynamicsInfo(body_id, -1, physicsClientId=client_id)

        if scenario_name == "drop_test":
            metrics, trajectory = run_drop_test(
                p=p,
                body_id=body_id,
                plane_id=plane_id,
                timestep=timestep,
                config=scenario_cfg,
            )
        elif scenario_name == "slide_test":
            metrics, trajectory = run_slide_test(
                p=p, body_id=body_id, timestep=timestep, config=scenario_cfg
            )
        elif scenario_name == "force_response_test":
            metrics, trajectory = run_force_response_test(
                p=p, body_id=body_id, timestep=timestep, config=scenario_cfg
            )
        else:
            raise ValueError(f"Unsupported scenario: {scenario_name}")

        applied_dynamics = {
            "object_id": object_id,
            "scenario": scenario_name,
            "mass_kg": surrogate["mass_kg"],
            "lateral_friction": surrogate["lateral_friction"],
            "restitution": surrogate["restitution"],
            "linear_damping": surrogate["linear_damping"],
            "angular_damping": surrogate["angular_damping"],
            "requested_dynamics": {
                "mass_kg": surrogate["mass_kg"],
                "lateral_friction": surrogate["lateral_friction"],
                "restitution": surrogate["restitution"],
                "linear_damping": surrogate["linear_damping"],
                "angular_damping": surrogate["angular_damping"],
            },
            "actual_dynamics": _extract_actual_dynamics(dynamics_info),
            "realization_error": _compute_realization_error(
                requested={
                    "mass_kg": float(surrogate["mass_kg"]),
                    "lateral_friction": float(surrogate["lateral_friction"]),
                    "restitution": float(surrogate["restitution"]),
                },
                actual=_extract_actual_dynamics(dynamics_info),
            ),
            "primitive": proxy_entry["primitive"],
            "dimensions": proxy_entry["dimensions"],
            "start_height_m": scenario_cfg["start_height_m"],
            "fixed_timestep_s": timestep,
        }
        return {
            "metrics": metrics,
            "trajectory": trajectory,
            "applied_dynamics": applied_dynamics,
        }
    finally:
        p.disconnect(physicsClientId=client_id)


def _create_object_body(
    p: Any,
    client_id: int,
    proxy_entry: dict[str, Any],
    surrogate_entry: dict[str, Any],
    start_height: float,
) -> int:
    primitive = proxy_entry["primitive"]
    dims = proxy_entry["dimensions"]
    collision_id: int
    visual_id: int
    if primitive in {"box", "thin_box"}:
        half_extents = dims["half_extents_m"]
        collision_id = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=half_extents, physicsClientId=client_id
        )
        visual_id = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.7, 0.7, 0.7, 1.0], physicsClientId=client_id
        )
    elif primitive == "sphere":
        radius = float(dims["radius_m"])
        collision_id = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=client_id)
        visual_id = p.createVisualShape(
            p.GEOM_SPHERE, radius=radius, rgbaColor=[0.7, 0.2, 0.2, 1.0], physicsClientId=client_id
        )
    elif primitive == "cylinder":
        radius = float(dims["radius_m"])
        height = float(dims["height_m"])
        collision_id = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=radius, height=height, physicsClientId=client_id
        )
        visual_id = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=[0.2, 0.6, 0.2, 1.0],
            physicsClientId=client_id,
        )
    elif primitive == "capsule":
        radius = float(dims["radius_m"])
        height = float(dims["height_m"])
        collision_id = p.createCollisionShape(
            p.GEOM_CAPSULE, radius=radius, height=height, physicsClientId=client_id
        )
        visual_id = p.createVisualShape(
            p.GEOM_CAPSULE,
            radius=radius,
            length=height,
            rgbaColor=[0.2, 0.2, 0.8, 1.0],
            physicsClientId=client_id,
        )
    else:
        # conservative fallback to box
        collision_id = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.03], physicsClientId=client_id
        )
        visual_id = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.03], physicsClientId=client_id
        )

    body_id = p.createMultiBody(
        baseMass=float(surrogate_entry["surrogate_parameters"]["mass_kg"]),
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
        basePosition=[0.0, 0.0, start_height],
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
        physicsClientId=client_id,
    )
    return body_id


def _extract_actual_dynamics(dynamics_info: Any) -> dict[str, float | None]:
    """Extract selected fields from pybullet.getDynamicsInfo tuple."""
    if not isinstance(dynamics_info, (tuple, list)):
        return {
            "mass_kg": None,
            "lateral_friction": None,
            "restitution": None,
        }
    return {
        "mass_kg": float(dynamics_info[0]) if len(dynamics_info) > 0 else None,
        "lateral_friction": float(dynamics_info[1]) if len(dynamics_info) > 1 else None,
        "restitution": float(dynamics_info[5]) if len(dynamics_info) > 5 else None,
    }


def _compute_realization_error(
    requested: dict[str, float], actual: dict[str, float | None]
) -> dict[str, float | None]:
    return {
        "mass_abs_error": (
            abs(requested["mass_kg"] - actual["mass_kg"])
            if actual.get("mass_kg") is not None
            else None
        ),
        "lateral_friction_abs_error": (
            abs(requested["lateral_friction"] - actual["lateral_friction"])
            if actual.get("lateral_friction") is not None
            else None
        ),
        "restitution_abs_error": (
            abs(requested["restitution"] - actual["restitution"])
            if actual.get("restitution") is not None
            else None
        ),
    }
