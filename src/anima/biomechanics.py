from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

from .model import FramePose, MotionClip, Vec2


@dataclass(frozen=True)
class BodyDynamicsProfile:
    mass_kg: float = 75.0
    pixels_per_meter: float = 40.0
    foot_half_width_px: float = 4.0
    max_planted_slip_px: float = 0.25
    max_com_accel_m_s2: float = 35.0
    max_com_jerk_m_s3: float = 350.0

    @classmethod
    def from_clip(cls, clip: MotionClip) -> "BodyDynamicsProfile":
        raw = clip.dynamics.get("body", {})
        return cls(
            mass_kg=float(raw.get("mass_kg", 75.0)),
            pixels_per_meter=float(raw.get("pixels_per_meter", 40.0)),
            foot_half_width_px=float(raw.get("foot_half_width_px", 4.0)),
            max_planted_slip_px=float(raw.get("max_planted_slip_px", 0.25)),
            max_com_accel_m_s2=float(raw.get("max_com_accel_m_s2", 35.0)),
            max_com_jerk_m_s3=float(raw.get("max_com_jerk_m_s3", 350.0)),
        )


def _mid(a: Vec2, b: Vec2) -> Vec2:
    return Vec2((a.x + b.x) * 0.5, (a.y + b.y) * 0.5)


def _weighted_center(parts: list[tuple[Vec2, float]]) -> Vec2:
    total = sum(weight for _, weight in parts)
    if total <= 1e-9:
        return Vec2(0.0, 0.0)
    return Vec2(
        sum(point.x * weight for point, weight in parts) / total,
        sum(point.y * weight for point, weight in parts) / total,
    )


def estimate_com(frame: FramePose) -> Vec2:
    """Estimate 2D center of mass from a simple human segment mass model.

    Fractions sum to 1.0:
    head 8%, torso 50%, upper arms 6%, forearms 4%,
    thighs 20%, shins/feet 12%.
    """
    j = frame.joints

    required = {
        "head",
        "chest",
        "shoulder_l",
        "shoulder_r",
        "elbow_l",
        "elbow_r",
        "hand_l",
        "hand_r",
        "hip_l",
        "hip_r",
        "knee_l",
        "knee_r",
        "foot_l",
        "foot_r",
    }
    if not required.issubset(j):
        return frame.root

    parts = [
        (j["head"], 0.08),
        (_mid(j["chest"], frame.root), 0.50),
        (_mid(j["shoulder_l"], j["elbow_l"]), 0.03),
        (_mid(j["shoulder_r"], j["elbow_r"]), 0.03),
        (_mid(j["elbow_l"], j["hand_l"]), 0.02),
        (_mid(j["elbow_r"], j["hand_r"]), 0.02),
        (_mid(j["hip_l"], j["knee_l"]), 0.10),
        (_mid(j["hip_r"], j["knee_r"]), 0.10),
        (_mid(j["knee_l"], j["foot_l"]), 0.06),
        (_mid(j["knee_r"], j["foot_r"]), 0.06),
    ]
    return _weighted_center(parts)


def _vec_diff(values: list[Vec2], dt: float) -> list[Vec2]:
    if not values:
        return []
    out = [Vec2(0.0, 0.0)]
    for index in range(1, len(values)):
        out.append((values[index] - values[index - 1]) * (1.0 / dt))
    return out


def _torso_angle(frame: FramePose) -> float:
    j = frame.joints
    if not {"shoulder_l", "shoulder_r", "hip_l", "hip_r"}.issubset(j):
        return 0.0
    shoulder_mid = _mid(j["shoulder_l"], j["shoulder_r"])
    hip_mid = _mid(j["hip_l"], j["hip_r"])
    delta = shoulder_mid - hip_mid
    return math.atan2(-delta.y, delta.x)


def _unwrap(values: list[float]) -> list[float]:
    if not values:
        return []
    out = [values[0]]
    for value in values[1:]:
        prev = out[-1]
        while value - prev > math.pi:
            value -= 2.0 * math.pi
        while value - prev < -math.pi:
            value += 2.0 * math.pi
        out.append(value)
    return out


def _scalar_diff(values: list[float], dt: float) -> list[float]:
    if not values:
        return []
    out = [0.0]
    for index in range(1, len(values)):
        out.append((values[index] - values[index - 1]) / dt)
    return out


def analyze_body_kinematics(clip: MotionClip) -> dict[str, Any]:
    profile = BodyDynamicsProfile.from_clip(clip)
    dt = 1.0 / clip.fps
    ppm = max(profile.pixels_per_meter, 1e-9)

    com_px = [estimate_com(frame) for frame in clip.frames]
    com_m = [Vec2(point.x / ppm, point.y / ppm) for point in com_px]
    velocity = _vec_diff(com_m, dt)
    acceleration = _vec_diff(velocity, dt)
    jerk = _vec_diff(acceleration, dt)

    root_m = [
        Vec2(frame.root.x / ppm, frame.root.y / ppm)
        for frame in clip.frames
    ]
    root_velocity = _vec_diff(root_m, dt)
    root_acceleration = _vec_diff(root_velocity, dt)

    torso_angles = _unwrap([_torso_angle(frame) for frame in clip.frames])
    torso_omega = _scalar_diff(torso_angles, dt)
    torso_alpha = _scalar_diff(torso_omega, dt)

    warnings: list[dict[str, Any]] = []
    frame_reports: list[dict[str, Any]] = []

    previous = None
    for index, frame in enumerate(clip.frames):
        support_x = [
            frame.joints[name].x
            for name in ("foot_l", "foot_r")
            if frame.contacts.get(name) and name in frame.joints
        ]
        support = None
        if support_x:
            support = {
                "min_x": min(support_x) - profile.foot_half_width_px,
                "max_x": max(support_x) + profile.foot_half_width_px,
            }
            if not (support["min_x"] <= com_px[index].x <= support["max_x"]):
                warnings.append(
                    {
                        "code": "com_outside_support",
                        "frame": frame.frame,
                        "message": (
                            f"COM x={com_px[index].x:.2f}px lies outside planted-foot "
                            f"support [{support['min_x']:.2f}, {support['max_x']:.2f}]px."
                        ),
                    }
                )

        if previous is not None:
            for foot in ("foot_l", "foot_r"):
                if (
                    previous.contacts.get(foot)
                    and frame.contacts.get(foot)
                    and foot in previous.joints
                    and foot in frame.joints
                ):
                    slip = frame.joints[foot].distance_to(previous.joints[foot])
                    if slip > profile.max_planted_slip_px:
                        warnings.append(
                            {
                                "code": "planted_foot_slip",
                                "frame": frame.frame,
                                "message": (
                                    f"{foot} moved {slip:.3f}px while continuously planted."
                                ),
                            }
                        )

        accel_mag = acceleration[index].length()
        jerk_mag = jerk[index].length()
        if accel_mag > profile.max_com_accel_m_s2:
            warnings.append(
                {
                    "code": "com_acceleration_high",
                    "frame": frame.frame,
                    "message": (
                        f"COM acceleration {accel_mag:.2f}m/s^2 exceeds configured "
                        f"{profile.max_com_accel_m_s2:.2f}m/s^2."
                    ),
                }
            )
        if jerk_mag > profile.max_com_jerk_m_s3:
            warnings.append(
                {
                    "code": "com_jerk_high",
                    "frame": frame.frame,
                    "message": (
                        f"COM jerk {jerk_mag:.2f}m/s^3 exceeds configured "
                        f"{profile.max_com_jerk_m_s3:.2f}m/s^3."
                    ),
                }
            )

        speed = velocity[index].length()
        momentum = speed * profile.mass_kg
        kinetic = 0.5 * profile.mass_kg * speed * speed
        force = acceleration[index].length() * profile.mass_kg

        frame_reports.append(
            {
                "frame": frame.frame,
                "label": frame.label,
                "com_px": com_px[index].as_list(),
                "com_velocity_m_s": velocity[index].as_list(),
                "com_speed_m_s": speed,
                "com_acceleration_m_s2": acceleration[index].as_list(),
                "com_jerk_m_s3": jerk[index].as_list(),
                "linear_momentum_kg_m_s": momentum,
                "translational_kinetic_energy_j": kinetic,
                "estimated_net_force_n": force,
                "root_velocity_m_s": root_velocity[index].as_list(),
                "root_acceleration_m_s2": root_acceleration[index].as_list(),
                "torso_angle_deg": math.degrees(torso_angles[index]),
                "torso_angular_velocity_deg_s": math.degrees(torso_omega[index]),
                "torso_angular_acceleration_deg_s2": math.degrees(torso_alpha[index]),
                "support": support,
            }
        )
        previous = frame

    return {
        "profile": asdict(profile),
        "frames": frame_reports,
        "warnings": warnings,
    }
