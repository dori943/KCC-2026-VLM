"""Slide test scenario for friction surrogate checks."""

from __future__ import annotations

from typing import Any


def run_slide_test(
    p: Any,
    body_id: int,
    timestep: float,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run sliding simulation and return metrics + trajectory rows."""
    max_steps = int(config["max_steps"])
    stop_vel_threshold = float(config["stop_velocity_threshold_mps"])
    initial_velocity = float(config["initial_velocity_mps"])
    p.resetBaseVelocity(body_id, linearVelocity=[initial_velocity, 0.0, 0.0])

    pos0, _ = p.getBasePositionAndOrientation(body_id)
    x0 = float(pos0[0])
    stop_step: int | None = None
    trajectory: list[dict[str, Any]] = []

    for step in range(max_steps):
        p.stepSimulation()
        pos, _ = p.getBasePositionAndOrientation(body_id)
        vel_lin, _ = p.getBaseVelocity(body_id)
        speed_xy = (vel_lin[0] ** 2 + vel_lin[1] ** 2) ** 0.5
        if stop_step is None and speed_xy <= stop_vel_threshold:
            stop_step = step
        trajectory.append(
            {
                "step": step,
                "time_s": round(step * timestep, 6),
                "x_m": round(float(pos[0]), 6),
                "y_m": round(float(pos[1]), 6),
                "z_m": round(float(pos[2]), 6),
                "vx_mps": round(float(vel_lin[0]), 6),
                "vy_mps": round(float(vel_lin[1]), 6),
                "vz_mps": round(float(vel_lin[2]), 6),
            }
        )

    pos_end, _ = p.getBasePositionAndOrientation(body_id)
    stopping_distance = abs(float(pos_end[0]) - x0)
    stopping_time = (stop_step * timestep) if stop_step is not None else (max_steps * timestep)
    avg_decel = initial_velocity / stopping_time if stopping_time > 1e-9 else 0.0
    metrics = {
        "stopping_distance_m": round(stopping_distance, 6),
        "stopping_time_s": round(stopping_time, 6),
        "average_deceleration_mps2": round(avg_decel, 6),
    }
    return metrics, trajectory
