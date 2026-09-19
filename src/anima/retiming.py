from __future__ import annotations

import math
from typing import Any

from .model import MotionClip


def _weapon_scale(weapon: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    profile = weapon.get("profile", {})
    inertia = float(profile.get("inertia_kg_m2", 0.0))
    damping = float(profile.get("damping_nm_per_rad_s", 0.0))
    max_torque = float(profile.get("max_braking_torque_nm", 0.0))
    if max_torque <= 1e-9:
        return 1.0, []

    required = 1.0
    reasons: list[dict[str, Any]] = []

    for item in weapon.get("frames", []):
        omega = math.radians(float(item.get("angular_velocity_deg_s", 0.0)))
        alpha = math.radians(float(item.get("angular_acceleration_deg_s2", 0.0)))

        def demand(scale: float) -> float:
            return abs(
                inertia * alpha / (scale * scale)
                + damping * omega / scale
            )

        if demand(1.0) <= max_torque:
            continue

        low = 1.0
        high = 2.0
        while demand(high) > max_torque and high < 128.0:
            high *= 2.0
        for _ in range(50):
            mid = (low + high) * 0.5
            if demand(mid) > max_torque:
                low = mid
            else:
                high = mid

        required = max(required, high)
        reasons.append(
            {
                "domain": "weapon",
                "frame": item.get("frame"),
                "code": "torque_retime",
                "time_scale": high,
                "message": (
                    f"Weapon torque needs about {high:.2f}x more time at the "
                    f"configured {max_torque:.1f} Nm torque limit."
                ),
            }
        )

    return required, reasons


def _body_scale(body: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    profile = body.get("profile", {})
    max_accel = float(profile.get("max_com_accel_m_s2", 0.0))
    max_jerk = float(profile.get("max_com_jerk_m_s3", 0.0))
    mu = float(profile.get("friction_coefficient", 0.0))
    gravity = float(profile.get("gravity_m_s2", 9.81))

    required = 1.0
    reasons: list[dict[str, Any]] = []

    for item in body.get("frames", []):
        accel = item.get("com_acceleration_m_s2", [0.0, 0.0])
        ax = float(accel[0])
        ay = float(accel[1])
        accel_mag = math.hypot(ax, ay)

        if max_accel > 1e-9 and accel_mag > max_accel:
            scale = math.sqrt(accel_mag / max_accel)
            required = max(required, scale)
            reasons.append(
                {
                    "domain": "body",
                    "frame": item.get("frame"),
                    "code": "acceleration_retime",
                    "time_scale": scale,
                    "message": (
                        f"COM acceleration requires about {scale:.2f}x more time."
                    ),
                }
            )

        jerk = item.get("com_jerk_m_s3", [0.0, 0.0])
        jerk_mag = math.hypot(float(jerk[0]), float(jerk[1]))
        if max_jerk > 1e-9 and jerk_mag > max_jerk:
            scale = (jerk_mag / max_jerk) ** (1.0 / 3.0)
            required = max(required, scale)
            reasons.append(
                {
                    "domain": "body",
                    "frame": item.get("frame"),
                    "code": "jerk_retime",
                    "time_scale": scale,
                    "message": (
                        f"COM jerk requires about {scale:.2f}x more time."
                    ),
                }
            )

        # With time scaling s: ax -> ax/s^2 and ay -> ay/s^2.
        # Required no-slip condition:
        # |ax|/s^2 <= mu * (g - ay/s^2)
        # => s^2 >= (|ax|/mu + ay) / g
        if (
            item.get("ground_reaction_force") is not None
            and mu > 1e-9
            and gravity > 1e-9
        ):
            numerator = abs(ax) / mu + ay
            if numerator > gravity:
                scale = math.sqrt(max(numerator / gravity, 1.0))
                required = max(required, scale)
                reasons.append(
                    {
                        "domain": "body",
                        "frame": item.get("frame"),
                        "code": "friction_retime",
                        "time_scale": scale,
                        "message": (
                            f"Ground friction demand requires about {scale:.2f}x more time."
                        ),
                    }
                )

    return required, reasons


def _system_scale(
    system: dict[str, Any],
    body: dict[str, Any],
) -> tuple[float, list[dict[str, Any]]]:
    body_profile = body.get("profile", {})
    max_accel = float(body_profile.get("max_com_accel_m_s2", 0.0))
    max_jerk = float(body_profile.get("max_com_jerk_m_s3", 0.0))
    mu = float(system.get("profile", {}).get("friction_coefficient", 0.0))
    gravity = float(system.get("profile", {}).get("gravity_m_s2", 9.81))

    required = 1.0
    reasons: list[dict[str, Any]] = []

    for item in system.get("frames", []):
        accel = item.get("system_com_acceleration_m_s2", [0.0, 0.0])
        ax = float(accel[0])
        ay = float(accel[1])
        accel_mag = math.hypot(ax, ay)

        if max_accel > 1e-9 and accel_mag > max_accel:
            scale = math.sqrt(accel_mag / max_accel)
            required = max(required, scale)
            reasons.append(
                {
                    "domain": "system",
                    "frame": item.get("frame"),
                    "code": "system_acceleration_retime",
                    "time_scale": scale,
                    "message": (
                        f"Person+weapon COM acceleration needs about "
                        f"{scale:.2f}x more time."
                    ),
                }
            )

        jerk = item.get("system_com_jerk_m_s3", [0.0, 0.0])
        jerk_mag = math.hypot(float(jerk[0]), float(jerk[1]))
        if max_jerk > 1e-9 and jerk_mag > max_jerk:
            scale = (jerk_mag / max_jerk) ** (1.0 / 3.0)
            required = max(required, scale)
            reasons.append(
                {
                    "domain": "system",
                    "frame": item.get("frame"),
                    "code": "system_jerk_retime",
                    "time_scale": scale,
                    "message": (
                        f"Person+weapon COM jerk needs about {scale:.2f}x more time."
                    ),
                }
            )

        if (
            item.get("ground_reaction_force") is not None
            and mu > 1e-9
            and gravity > 1e-9
        ):
            numerator = abs(ax) / mu + ay
            if numerator > gravity:
                scale = math.sqrt(max(numerator / gravity, 1.0))
                required = max(required, scale)
                reasons.append(
                    {
                        "domain": "system",
                        "frame": item.get("frame"),
                        "code": "system_friction_retime",
                        "time_scale": scale,
                        "message": (
                            f"Person+weapon friction demand needs about "
                            f"{scale:.2f}x more time."
                        ),
                    }
                )

    return required, reasons


def recommend_timing(
    clip: MotionClip,
    dynamics_report: dict[str, Any],
) -> dict[str, Any]:
    """Recommend a global time scale that reduces dynamic-limit violations.

    This is a conservative first pass. It does not change pose geometry. A
    future optimizer can distribute extra time only to the phases that need it.
    """
    weapon_scale, weapon_reasons = _weapon_scale(dynamics_report.get("weapon", {}))
    body_scale, body_reasons = _body_scale(dynamics_report.get("body", {}))
    system_scale, system_reasons = _system_scale(
        dynamics_report.get("system", {}),
        dynamics_report.get("body", {}),
    )
    scale = max(1.0, weapon_scale, body_scale, system_scale)

    frame_count = len(clip.frames)
    duration_s = (frame_count - 1) / clip.fps if frame_count > 1 else 0.0
    recommended_duration = duration_s * scale
    recommended_fps = clip.fps / scale if scale > 0 else clip.fps
    recommended_frames_same_fps = (
        math.ceil((frame_count - 1) * scale) + 1
        if frame_count > 1
        else frame_count
    )

    return {
        "current": {
            "fps": clip.fps,
            "frame_count": frame_count,
            "duration_s": duration_s,
        },
        "recommended": {
            "global_time_scale": scale,
            "fps_if_same_frames": recommended_fps,
            "duration_s": recommended_duration,
            "frame_count_if_same_fps": recommended_frames_same_fps,
        },
        "components": {
            "weapon_time_scale": weapon_scale,
            "body_time_scale": body_scale,
            "system_time_scale": system_scale,
        },
        "reasons": weapon_reasons + body_reasons + system_reasons,
    }
