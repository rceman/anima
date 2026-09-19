from __future__ import annotations

import math
from typing import Any

from .model import MotionClip, Vec2
from .rig import HUMANOID_BONES


def _validate_times(count: int, times: list[float]) -> None:
    if len(times) != count:
        raise ValueError("Value/time arrays must have the same length")
    for previous, current in zip(times, times[1:]):
        if current <= previous:
            raise ValueError("Derivative timestamps must be strictly increasing")


def vec_derivative_times(
    values: list[Vec2],
    times: list[float],
    zero_indices: set[int] | None = None,
) -> list[Vec2]:
    """Finite-difference derivative for non-uniform timestamps.

    Endpoints use one-sided secants. Internal samples use the secant spanning
    their immediate neighbors. Explicit stop indices override the inferred
    derivative so a held guard/rest pose has exactly zero velocity.
    """
    count = len(values)
    _validate_times(count, times)
    if count == 0:
        return []
    if count == 1:
        return [Vec2(0.0, 0.0)]

    out = [
        (values[1] - values[0]) * (1.0 / (times[1] - times[0]))
    ]
    for index in range(1, count - 1):
        dt = times[index + 1] - times[index - 1]
        out.append((values[index + 1] - values[index - 1]) * (1.0 / dt))
    out.append(
        (values[-1] - values[-2]) * (1.0 / (times[-1] - times[-2]))
    )

    for index in zero_indices or set():
        if 0 <= index < len(out):
            out[index] = Vec2(0.0, 0.0)
    return out


def scalar_derivative_times(
    values: list[float],
    times: list[float],
    zero_indices: set[int] | None = None,
) -> list[float]:
    """Finite-difference scalar derivative for non-uniform timestamps."""
    count = len(values)
    _validate_times(count, times)
    if count == 0:
        return []
    if count == 1:
        return [0.0]

    out = [
        (values[1] - values[0]) / (times[1] - times[0])
    ]
    for index in range(1, count - 1):
        out.append(
            (values[index + 1] - values[index - 1])
            / (times[index + 1] - times[index - 1])
        )
    out.append(
        (values[-1] - values[-2]) / (times[-1] - times[-2])
    )

    for index in zero_indices or set():
        if 0 <= index < len(out):
            out[index] = 0.0
    return out


def vec_derivative(values: list[Vec2], dt: float) -> list[Vec2]:
    """Finite-difference derivative for uniformly sampled values."""
    times = [index * dt for index in range(len(values))]
    return vec_derivative_times(values, times)


def scalar_derivative(values: list[float], dt: float) -> list[float]:
    """Finite-difference scalar derivative for uniformly sampled values."""
    times = [index * dt for index in range(len(values))]
    return scalar_derivative_times(values, times)


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


def stop_indices(clip: MotionClip) -> set[int]:
    return {
        index
        for index, frame in enumerate(clip.frames)
        if frame.kinematic_stop
    }


def analyze_joint_kinematics(
    clip: MotionClip,
    pixels_per_meter: float,
) -> dict[str, Any]:
    """Compute position/velocity/acceleration/jerk for every common joint."""
    if not clip.frames:
        return {"joints": {}, "bones": {}}

    times = clip.times_s()
    stops = stop_indices(clip)
    ppm = max(pixels_per_meter, 1e-9)
    common = set(clip.frames[0].joints)
    for frame in clip.frames[1:]:
        common &= set(frame.joints)

    joints_report: dict[str, Any] = {}
    for name in sorted(common):
        pos_px = [frame.joints[name] for frame in clip.frames]
        pos_m = [Vec2(point.x / ppm, point.y / ppm) for point in pos_px]
        velocity = vec_derivative_times(pos_m, times, stops)
        acceleration = vec_derivative_times(velocity, times)
        jerk = vec_derivative_times(acceleration, times)

        frames = []
        for index, frame in enumerate(clip.frames):
            frames.append(
                {
                    "frame": frame.frame,
                    "time_s": times[index],
                    "kinematic_stop": frame.kinematic_stop,
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
            "max_acceleration_m_s2": max(
                item["acceleration_mag_m_s2"] for item in frames
            ),
            "max_jerk_m_s3": max(item["jerk_mag_m_s3"] for item in frames),
        }

    bones_report: dict[str, Any] = {}
    for bone in HUMANOID_BONES:
        if bone.parent not in common or bone.child not in common:
            continue

        angles = unwrap_angles(
            [
                _angle(
                    frame.joints[bone.parent],
                    frame.joints[bone.child],
                )
                for frame in clip.frames
            ]
        )
        omega = scalar_derivative_times(angles, times, stops)
        alpha = scalar_derivative_times(omega, times)
        angular_jerk = scalar_derivative_times(alpha, times)

        frames = []
        for index, frame in enumerate(clip.frames):
            frames.append(
                {
                    "frame": frame.frame,
                    "time_s": times[index],
                    "kinematic_stop": frame.kinematic_stop,
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
