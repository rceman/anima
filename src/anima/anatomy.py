from __future__ import annotations

import math
from typing import Any

from .kinematics import scalar_derivative
from .model import MotionClip, Vec2


DEFAULT_LIMITS_DEG: dict[str, tuple[float, float]] = {
    "elbow_l": (5.0, 178.0),
    "elbow_r": (5.0, 178.0),
    "knee_l": (5.0, 178.0),
    "knee_r": (5.0, 178.0),
}

CHAINS: dict[str, tuple[str, str, str]] = {
    "elbow_l": ("shoulder_l", "elbow_l", "hand_l"),
    "elbow_r": ("shoulder_r", "elbow_r", "hand_r"),
    "knee_l": ("hip_l", "knee_l", "foot_l"),
    "knee_r": ("hip_r", "knee_r", "foot_r"),
}


def _internal_angle(a: Vec2, joint: Vec2, b: Vec2) -> float:
    va = a - joint
    vb = b - joint
    denom = max(va.length() * vb.length(), 1e-9)
    cosine = (va.x * vb.x + va.y * vb.y) / denom
    cosine = min(1.0, max(-1.0, cosine))
    return math.acos(cosine)


def _limits(clip: MotionClip) -> dict[str, tuple[float, float]]:
    raw = clip.dynamics.get("body", {}).get("joint_limits_deg", {})
    result = dict(DEFAULT_LIMITS_DEG)
    for name, value in raw.items():
        if name not in CHAINS:
            continue
        result[name] = (float(value[0]), float(value[1]))
    return result


def analyze_joint_limits(clip: MotionClip) -> dict[str, Any]:
    limits = _limits(clip)
    dt = 1.0 / clip.fps
    warnings: list[dict[str, Any]] = []
    joints: dict[str, Any] = {}

    for name, chain in CHAINS.items():
        parent_name, joint_name, child_name = chain
        if not all(
            all(part in frame.joints for part in chain)
            for frame in clip.frames
        ):
            continue

        angles = [
            _internal_angle(
                frame.joints[parent_name],
                frame.joints[joint_name],
                frame.joints[child_name],
            )
            for frame in clip.frames
        ]
        omega = scalar_derivative(angles, dt)
        alpha = scalar_derivative(omega, dt)
        minimum, maximum = limits[name]

        frames = []
        for index, frame in enumerate(clip.frames):
            angle_deg = math.degrees(angles[index])
            frames.append(
                {
                    "frame": frame.frame,
                    "angle_deg": angle_deg,
                    "angular_velocity_deg_s": math.degrees(omega[index]),
                    "angular_acceleration_deg_s2": math.degrees(alpha[index]),
                }
            )
            if angle_deg < minimum - 1e-6 or angle_deg > maximum + 1e-6:
                warnings.append(
                    {
                        "code": "joint_limit_violation",
                        "frame": frame.frame,
                        "joint": name,
                        "message": (
                            f"{name} angle {angle_deg:.1f} deg is outside "
                            f"[{minimum:.1f}, {maximum:.1f}] deg."
                        ),
                    }
                )

        joints[name] = {
            "limits_deg": [minimum, maximum],
            "frames": frames,
        }

    return {
        "joints": joints,
        "warnings": warnings,
    }
