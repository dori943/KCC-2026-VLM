"""Force response scenario for mass surrogate checks."""

from __future__ import annotations

from typing import Any


def run_force_response_test(
    p: Any,
    body_id: int,
    timestep: float,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run force response simulation and return metrics + trajectory rows."""
    max_steps = int(config["max_steps"])
    force_steps = int(config["force_steps"])
    force_newton = float(config["force_newton"])

    pos0, _ = p.getBasePositionAndOrientation(body_id)
    x0 = float(pos0[0])
    prev_vx: float | None = None
    initial_accel: float | None = None
    peak_velocity = 0.0
    trajectory: list[dict[str, Any]] = []

    for step in range(max_steps):
        if step < force_steps:
            p.applyExternalForce(
                objectUniqueId=body_id,
                linkIndex=-1,
                forceObj=[force_newton, 0.0, 0.0],
                posObj=[0.0, 0.0, 0.0],
                flags=p.LINK_FRAME,
            )
        p.stepSimulation()
        pos, _ = p.getBasePositionAndOrientation(body_id)
        vel_lin, _ = p.getBaseVelocity(body_id)
        vmag = (vel_lin[0] ** 2 + vel_lin[1] ** 2 + vel_lin[2] ** 2) ** 0.5
        peak_velocity = max(peak_velocity, vmag)
        if prev_vx is not None and initial_accel is None:
            initial_accel = (float(vel_lin[0]) - prev_vx) / timestep
        prev_vx = float(vel_lin[0])
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
    displacement = abs(float(pos_end[0]) - x0)
    metrics = {
        "initial_acceleration_mps2": round(initial_accel if initial_accel is not None else 0.0, 6),
        "peak_velocity_mps": round(peak_velocity, 6),
        "displacement_at_horizon_m": round(displacement, 6),
    }
    return metrics, trajectory
