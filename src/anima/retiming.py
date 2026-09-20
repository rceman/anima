from __future__ import annotations

import math
from typing import Any

from .model import MotionClip


def _weapon_scale(weapon: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    profile = weapon.get("profile", {})
    inertia = float(profile.get("inertia_kg_m2", 0.0))
    damping = float(profile.get("damping_nm_per_rad_s", 0.0))
    max_torque = float(
        profile.get(
            "max_drive_torque_nm",
            profile.get("max_braking_torque_nm", 0.0),
        )
    )
    if max_torque <= 1e-9:
        return 1.0, []

    required = 1.0
    reasons: list[dict[str, Any]] = []

    for item in weapon.get("frames", []):
        omega = math.radians(float(item.get("angular_velocity_deg_s", 0.0)))
        alpha = math.radians(float(item.get("angular_acceleration_deg_s2", 0.0)))

        inertial = float(
            item.get(
                "inertial_torque_nm",
                inertia * alpha,
            )
        )
        damping_torque = float(
            item.get(
                "damping_torque_nm",
                damping * omega,
            )
        )
        gravity_torque = float(
            item.get("gravity_torque_nm", 0.0)
        )

        def demand(scale: float) -> float:
            # Inertial load scales with 1/s^2 and viscous damping with 1/s.
            # Gravity does not get easier merely because the animation is
            # slowed down.
            return abs(
                inertial / (scale * scale)
                - gravity_torque
                + damping_torque / scale
            )

        if demand(1.0) <= max_torque:
            continue

        # Search for the first slower timing that actually satisfies the
        # torque budget. A fixed gravity term means demand is not guaranteed to
        # be perfectly monotonic, so a small multiplicative scan is safer than
        # assuming a simple 1/s^2 relationship.
        low = 1.0
        high = 1.0
        found = False
        while high < 128.0:
            candidate = min(128.0, high * 1.10)
            if demand(candidate) <= max_torque:
                low = high
                high = candidate
                found = True
                break
            high = candidate

        if not found:
            reasons.append(
                {
                    "domain": "weapon",
                    "frame": item.get("frame"),
                    "code": "torque_not_retimeable",
                    "time_scale": 1.0,
                    "retime_possible": False,
                    "message": (
                        f"Weapon torque remains above the configured "
                        f"{max_torque:.1f} Nm limit even when motion-dependent "
                        "loads are heavily slowed; change pose, weapon, or "
                        "strength/torque budget instead of retiming."
                    ),
                }
            )
            continue

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
                "retime_possible": True,
                "message": (
                    f"Weapon torque needs about {high:.2f}x more time at the "
                    f"configured {max_torque:.1f} Nm drive-torque limit."
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



def _segment_recommendations(
    clip: MotionClip,
    reasons: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    anchors = [
        frame
        for frame in clip.frames
        if frame.label
    ]
    if len(anchors) < 2:
        return []

    times = clip.times_s()
    time_by_frame = {
        frame.frame: times[index]
        for index, frame in enumerate(clip.frames)
    }

    segments: list[dict[str, Any]] = []
    for left, right in zip(anchors, anchors[1:]):
        local_reasons = [
            reason
            for reason in reasons
            if reason.get("frame") is not None
            and left.frame <= int(reason["frame"]) <= right.frame
        ]
        scale = max(
            [1.0] + [float(reason.get("time_scale", 1.0)) for reason in local_reasons]
        )
        span = right.frame - left.frame
        duration = time_by_frame[right.frame] - time_by_frame[left.frame]
        segments.append(
            {
                "from_frame": left.frame,
                "to_frame": right.frame,
                "from_label": left.label,
                "to_label": right.label,
                "current_frame_span": span,
                "current_duration_s": duration,
                "recommended_time_scale": scale,
                "recommended_frame_span_same_fps": max(
                    span,
                    math.ceil(span * scale),
                ),
                "recommended_duration_s": duration * scale,
                "reasons": local_reasons,
            }
        )
    return segments


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
    times = clip.times_s()
    duration_s = (times[-1] - times[0]) if frame_count > 1 else 0.0
    recommended_duration = duration_s * scale
    recommended_fps = clip.fps / scale if scale > 0 else clip.fps
    recommended_frames_same_fps = (
        math.ceil((frame_count - 1) * scale) + 1
        if frame_count > 1
        else frame_count
    )

    reasons = weapon_reasons + body_reasons + system_reasons

    return {
        "current": {
            "fps": clip.fps,
            "frame_count": frame_count,
            "duration_s": duration_s,
            "uses_explicit_timestamps": any(
                getattr(frame, "time_s", None) is not None
                for frame in clip.frames
            ),
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
        "reasons": reasons,
        "segments": _segment_recommendations(clip, reasons),
    }
