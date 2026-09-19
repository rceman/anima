from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

from .contacts import contact_mode, is_ground_contact, is_planted
from .kinematics import analyze_joint_kinematics, scalar_derivative, unwrap_angles, vec_derivative
from .model import FramePose, MotionClip, Vec2


@dataclass(frozen=True)
class BodyDynamicsProfile:
    mass_kg: float = 75.0
    pixels_per_meter: float = 40.0
    foot_half_width_px: float = 4.0
    gravity_m_s2: float = 9.81
    friction_coefficient: float = 0.8
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
            gravity_m_s2=float(raw.get("gravity_m_s2", 9.81)),
            friction_coefficient=float(raw.get("friction_coefficient", 0.8)),
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
    """Estimate 2D center of mass using coarse human segment mass fractions."""
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

    # Fractions sum to 1.0.
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


def _torso_angle(frame: FramePose) -> float:
    j = frame.joints
    if not {"shoulder_l", "shoulder_r", "hip_l", "hip_r"}.issubset(j):
        return 0.0
    shoulder_mid = _mid(j["shoulder_l"], j["shoulder_r"])
    hip_mid = _mid(j["hip_l"], j["hip_r"])
    delta = shoulder_mid - hip_mid
    return math.atan2(-delta.y, delta.x)


def analyze_body_kinematics(clip: MotionClip) -> dict[str, Any]:
    """Analyze whole-body motion, balance, contacts and approximate forces.

    The model is intentionally 2D and diagnostic rather than a full rigid-body
    biomechanics simulation. It still enforces the important real-world
    relationships: velocity -> acceleration -> force, planted contacts,
    support polygon, friction demand, and momentum continuity.
    """
    profile = BodyDynamicsProfile.from_clip(clip)
    dt = 1.0 / clip.fps
    ppm = max(profile.pixels_per_meter, 1e-9)

    com_px = [estimate_com(frame) for frame in clip.frames]
    com_m = [Vec2(point.x / ppm, point.y / ppm) for point in com_px]
    velocity = vec_derivative(com_m, dt)
    acceleration = vec_derivative(velocity, dt)
    jerk = vec_derivative(acceleration, dt)

    root_m = [Vec2(frame.root.x / ppm, frame.root.y / ppm) for frame in clip.frames]
    root_velocity = vec_derivative(root_m, dt)
    root_acceleration = vec_derivative(root_velocity, dt)

    torso_angles = unwrap_angles([_torso_angle(frame) for frame in clip.frames])
    torso_omega = scalar_derivative(torso_angles, dt)
    torso_alpha = scalar_derivative(torso_omega, dt)
    torso_jerk = scalar_derivative(torso_alpha, dt)

    warnings: list[dict[str, Any]] = []
    frame_reports: list[dict[str, Any]] = []

    previous: FramePose | None = None
    for index, frame in enumerate(clip.frames):
        contact_names = [
            name
            for name in ("foot_l", "foot_r")
            if name in frame.joints and is_ground_contact(frame.contacts.get(name))
        ]
        support_x = [frame.joints[name].x for name in contact_names]

        support = None
        stability_margin = None
        load_share = None
        if support_x:
            support_min = min(support_x) - profile.foot_half_width_px
            support_max = max(support_x) + profile.foot_half_width_px
            stability_margin = min(
                com_px[index].x - support_min,
                support_max - com_px[index].x,
            )
            support = {
                "min_x": support_min,
                "max_x": support_max,
                "stability_margin_px": stability_margin,
                "contacts": {
                    name: contact_mode(frame.contacts.get(name))
                    for name in contact_names
                },
            }
            if stability_margin < 0:
                warnings.append(
                    {
                        "code": "com_outside_support",
                        "frame": frame.frame,
                        "message": (
                            f"COM x={com_px[index].x:.2f}px lies outside support "
                            f"[{support_min:.2f}, {support_max:.2f}]px."
                        ),
                    }
                )

            if len(contact_names) == 2:
                ordered = sorted(contact_names, key=lambda name: frame.joints[name].x)
                left_name, right_name = ordered
                left_x = frame.joints[left_name].x
                right_x = frame.joints[right_name].x
                span = max(right_x - left_x, 1e-9)
                right_fraction = min(1.0, max(0.0, (com_px[index].x - left_x) / span))
                load_share = {
                    left_name: 1.0 - right_fraction,
                    right_name: right_fraction,
                }

        if previous is not None:
            for foot in ("foot_l", "foot_r"):
                if (
                    foot in previous.joints
                    and foot in frame.joints
                    and is_planted(previous.contacts.get(foot))
                    and is_planted(frame.contacts.get(foot))
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
        net_force = acceleration[index] * profile.mass_kg

        # Screen Y grows downward. Gravity is therefore +Y. If feet are on the
        # ground, the required ground reaction force is what remains after
        # subtracting gravity from m*a.
        ground_reaction = None
        friction_ratio = None
        if contact_names:
            grf_x = net_force.x
            grf_up = profile.mass_kg * (profile.gravity_m_s2 - acceleration[index].y)
            normal_force = max(grf_up, 0.0)
            friction_ratio = abs(grf_x) / max(normal_force, 1e-9)
            ground_reaction = {
                "horizontal_n": grf_x,
                "vertical_up_n": grf_up,
                "magnitude_n": math.hypot(grf_x, grf_up),
                "required_friction_ratio": friction_ratio,
            }
            if normal_force > 1e-6 and friction_ratio > profile.friction_coefficient:
                warnings.append(
                    {
                        "code": "friction_limit_exceeded",
                        "frame": frame.frame,
                        "message": (
                            f"Required friction ratio {friction_ratio:.2f} exceeds "
                            f"configured coefficient {profile.friction_coefficient:.2f}."
                        ),
                    }
                )

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
                "estimated_net_force_n": net_force.as_list(),
                "root_velocity_m_s": root_velocity[index].as_list(),
                "root_acceleration_m_s2": root_acceleration[index].as_list(),
                "torso_angle_deg": math.degrees(torso_angles[index]),
                "torso_angular_velocity_deg_s": math.degrees(torso_omega[index]),
                "torso_angular_acceleration_deg_s2": math.degrees(torso_alpha[index]),
                "torso_angular_jerk_deg_s3": math.degrees(torso_jerk[index]),
                "support": support,
                "load_share": load_share,
                "ground_reaction_force": ground_reaction,
            }
        )
        previous = frame

    return {
        "profile": asdict(profile),
        "frames": frame_reports,
        "joint_kinematics": analyze_joint_kinematics(clip, profile.pixels_per_meter),
        "warnings": warnings,
    }
