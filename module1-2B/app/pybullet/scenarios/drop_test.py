"""Drop test scenario for restitution surrogate checks."""

from __future__ import annotations

from typing import Any


def run_drop_test(
    p: Any,
    body_id: int,
    plane_id: int,
    timestep: float,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run drop simulation and return metrics + trajectory rows."""
    max_steps = int(config["max_steps"])
    settle_vel_threshold = float(config["settle_velocity_threshold_mps"])
    settle_window_steps = int(config["settle_window_steps"])

    pos0, _ = p.getBasePositionAndOrientation(body_id)
    start_height = float(pos0[2])
    first_contact_step: int | None = None
    max_after_contact = 0.0
    bounce_count = 0
    was_in_contact = False
    settle_counter = 0
    settle_time: float | None = None
    trajectory: list[dict[str, Any]] = []

    for step in range(max_steps):
        p.stepSimulation()
        pos, _ = p.getBasePositionAndOrientation(body_id)
        vel_lin, _ = p.getBaseVelocity(body_id)
        speed = (vel_lin[0] ** 2 + vel_lin[1] ** 2 + vel_lin[2] ** 2) ** 0.5
        in_contact = len(p.getContactPoints(bodyA=body_id, bodyB=plane_id)) > 0

        if in_contact and first_contact_step is None:
            first_contact_step = step

        if first_contact_step is not None:
            max_after_contact = max(max_after_contact, float(pos[2]))
            if was_in_contact and not in_contact:
                bounce_count += 1

            if speed <= settle_vel_threshold:
                settle_counter += 1
                if settle_counter >= settle_window_steps and settle_time is None:
                    settle_time = (step - settle_window_steps + 1) * timestep
            else:
                settle_counter = 0

        was_in_contact = in_contact
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

    rebound_height = max(0.0, max_after_contact)
    rebound_height_ratio = rebound_height / start_height if start_height > 1e-9 else 0.0
    metrics = {
        "rebound_height_ratio": round(rebound_height_ratio, 6),
        "bounce_count": int(bounce_count),
        "settle_time_s": round(settle_time if settle_time is not None else (max_steps * timestep), 6),
    }
    return metrics, trajectory
