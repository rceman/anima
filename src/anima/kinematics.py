from __future__ import annotations

import math
from typing import Any

from .model import MotionClip, Vec2
from .rig import HUMANOID_BONES


def vec_derivative(values: list[Vec2], dt: float) -> list[Vec2]:
    if not values:
        return []
    out = [Vec2(0.0, 0.0)]
    for index in range(1, len(values)):
        out.append((values[index] - values[index - 1]) * (1.0 / dt))
    return out


def scalar_derivative(values: list[float], dt: float) -> list[float]:
    if not values:
        return []
    out = [0.0]
    for index in range(1, len(values)):
        out.append((values[index] - values[index - 1]) / dt)
    return out


def unwrap_angles(values: list[float]) -> list[float]:
    if not values:
        return []

    out = [values[0]]
    for value in values[1:]:
        previous = out[-1]
        while value - previous > math.pi:
            value -= 2.0 * math.pi
        while value - previous < -math.pi:
            value += 2.0 * math.pi
        out.append(value)
    return out


def _angle(a: Vec2, b: Vec2) -> float:
    delta = b - a
    return math.atan2(-delta.y, delta.x)


def analyze_joint_kinematics(clip: MotionClip, pixels_per_meter: float) -> dict[str, Any]:
    """Compute position/velocity/acceleration/jerk for every common joint."""
    if not clip.frames:
        return {"joints": {}, "bones": {}}

    dt = 1.0 / clip.fps
    ppm = max(pixels_per_meter, 1e-9)
    common = set(clip.frames[0].joints)
    for frame in clip.frames[1:]:
        common &= set(frame.joints)

    joints_report: dict[str, Any] = {}
    for name in sorted(common):
        pos_px = [frame.joints[name] for frame in clip.frames]
        pos_m = [Vec2(point.x / ppm, point.y / ppm) for point in pos_px]
        velocity = vec_derivative(pos_m, dt)
        acceleration = vec_derivative(velocity, dt)
        jerk = vec_derivative(acceleration, dt)

        frames = []
        for index, frame in enumerate(clip.frames):
            frames.append(
                {
                    "frame": frame.frame,
                    "position_px": pos_px[index].as_list(),
                    "velocity_m_s": velocity[index].as_list(),
                    "speed_m_s": velocity[index].length(),
                    "acceleration_m_s2": acceleration[index].as_list(),
                    "acceleration_mag_m_s2": acceleration[index].length(),
                    "jerk_m_s3": jerk[index].as_list(),
                    "jerk_mag_m_s3": jerk[index].length(),
                }
            )

        joints_report[name] = {
            "frames": frames,
            "max_speed_m_s": max(item["speed_m_s"] for item in frames),
            "max_acceleration_m_s2": max(item["acceleration_mag_m_s2"] for item in frames),
            "max_jerk_m_s3": max(item["jerk_mag_m_s3"] for item in frames),
        }

    bones_report: dict[str, Any] = {}
    for bone in HUMANOID_BONES:
        if bone.parent not in common or bone.child not in common:
            continue

        angles = unwrap_angles(
            [_angle(frame.joints[bone.parent], frame.joints[bone.child]) for frame in clip.frames]
        )
        omega = scalar_derivative(angles, dt)
        alpha = scalar_derivative(omega, dt)
        angular_jerk = scalar_derivative(alpha, dt)

        frames = []
        for index, frame in enumerate(clip.frames):
            frames.append(
                {
                    "frame": frame.frame,
                    "angle_deg": math.degrees(angles[index]),
                    "angular_velocity_deg_s": math.degrees(omega[index]),
                    "angular_acceleration_deg_s2": math.degrees(alpha[index]),
                    "angular_jerk_deg_s3": math.degrees(angular_jerk[index]),
                }
            )

        bones_report[bone.name] = {
            "parent": bone.parent,
            "child": bone.child,
            "frames": frames,
            "max_angular_velocity_deg_s": max(
                abs(item["angular_velocity_deg_s"]) for item in frames
            ),
            "max_angular_acceleration_deg_s2": max(
                abs(item["angular_acceleration_deg_s2"]) for item in frames
            ),
            "max_angular_jerk_deg_s3": max(
                abs(item["angular_jerk_deg_s3"]) for item in frames
            ),
        }

    return {"joints": joints_report, "bones": bones_report}
