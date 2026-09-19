from __future__ import annotations

import math
from typing import Any

from .kinematics import vec_derivative
from .model import MotionClip, Vec2


def analyze_system_dynamics(
    clip: MotionClip,
    body_report: dict[str, Any],
    weapon_report: dict[str, Any],
) -> dict[str, Any]:
    """Analyze the person + weapon as one mechanical system.

    Internal hand/sword forces cancel at system level. The resulting system COM
    and required ground reaction are therefore better balance diagnostics than
    body-only COM when a weapon has meaningful mass and is far from the torso.
    """
    body_profile = body_report.get("profile", {})
    weapon_profile = weapon_report.get("profile", {})
    body_mass = float(body_profile.get("mass_kg", 75.0))
    weapon_mass = float(weapon_profile.get("mass_kg", 1.3))
    total_mass = body_mass + weapon_mass
    ppm = max(float(body_profile.get("pixels_per_meter", 40.0)), 1e-9)
    gravity = float(body_profile.get("gravity_m_s2", 9.81))
    mu = float(body_profile.get("friction_coefficient", 0.8))

    body_frames = body_report.get("frames", [])
    weapon_frames = weapon_report.get("frames", [])
    if len(body_frames) != len(weapon_frames):
        return {
            "profile": {
                "body_mass_kg": body_mass,
                "weapon_mass_kg": weapon_mass,
                "total_mass_kg": total_mass,
            },
            "frames": [],
            "warnings": [
                {
                    "code": "system_frame_mismatch",
                    "frame": -1,
                    "message": "Body and weapon reports have different frame counts.",
                }
            ],
        }

    system_com_px: list[Vec2] = []
    for body_frame, weapon_frame in zip(body_frames, weapon_frames):
        body_com = Vec2.from_any(body_frame["com_px"])
        weapon_com = Vec2.from_any(weapon_frame["weapon_com_px"])
        system_com_px.append(
            Vec2(
                (body_com.x * body_mass + weapon_com.x * weapon_mass) / total_mass,
                (body_com.y * body_mass + weapon_com.y * weapon_mass) / total_mass,
            )
        )

    system_com_m = [
        Vec2(point.x / ppm, point.y / ppm)
        for point in system_com_px
    ]
    dt = 1.0 / clip.fps
    velocity = vec_derivative(system_com_m, dt)
    acceleration = vec_derivative(velocity, dt)
    jerk = vec_derivative(acceleration, dt)

    warnings: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []

    for index, frame in enumerate(clip.frames):
        support = body_frames[index].get("support")
        stability_margin = None
        if support:
            stability_margin = min(
                system_com_px[index].x - float(support["min_x"]),
                float(support["max_x"]) - system_com_px[index].x,
            )
            if stability_margin < 0:
                warnings.append(
                    {
                        "code": "system_com_outside_support",
                        "frame": frame.frame,
                        "message": (
                            f"Person+weapon COM x={system_com_px[index].x:.2f}px lies "
                            f"outside support [{support['min_x']:.2f}, "
                            f"{support['max_x']:.2f}]px."
                        ),
                    }
                )

        net_force = acceleration[index] * total_mass
        ground_reaction = None
        friction_ratio = None
        if support:
            grf_x = net_force.x
            grf_up = total_mass * (gravity - acceleration[index].y)
            normal = max(grf_up, 0.0)
            friction_ratio = abs(grf_x) / max(normal, 1e-9)
            ground_reaction = {
                "horizontal_n": grf_x,
                "vertical_up_n": grf_up,
                "magnitude_n": math.hypot(grf_x, grf_up),
                "required_friction_ratio": friction_ratio,
            }
            if normal > 1e-6 and friction_ratio > mu:
                warnings.append(
                    {
                        "code": "system_friction_limit_exceeded",
                        "frame": frame.frame,
                        "message": (
                            f"Person+weapon friction ratio {friction_ratio:.2f} "
                            f"exceeds configured coefficient {mu:.2f}."
                        ),
                    }
                )

        frames.append(
            {
                "frame": frame.frame,
                "label": frame.label,
                "system_com_px": system_com_px[index].as_list(),
                "system_com_velocity_m_s": velocity[index].as_list(),
                "system_com_speed_m_s": velocity[index].length(),
                "system_com_acceleration_m_s2": acceleration[index].as_list(),
                "system_com_jerk_m_s3": jerk[index].as_list(),
                "linear_momentum_kg_m_s": velocity[index].length() * total_mass,
                "translational_kinetic_energy_j": (
                    0.5 * total_mass * velocity[index].length() ** 2
                ),
                "estimated_net_force_n": net_force.length(),
                "estimated_net_force_vector_n": net_force.as_list(),
                "support": support,
                "stability_margin_px": stability_margin,
                "ground_reaction_force": ground_reaction,
            }
        )

    return {
        "profile": {
            "body_mass_kg": body_mass,
            "weapon_mass_kg": weapon_mass,
            "total_mass_kg": total_mass,
            "pixels_per_meter": ppm,
            "gravity_m_s2": gravity,
            "friction_coefficient": mu,
        },
        "frames": frames,
        "warnings": warnings,
    }
