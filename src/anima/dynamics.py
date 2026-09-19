from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any

from .kinematics import scalar_derivative, vec_derivative
from .model import FramePose, MotionClip, Vec2


@dataclass(frozen=True)
class WeaponDynamicsProfile:
    mass_kg: float = 1.3
    effective_length_m: float = 0.9
    inertia_factor: float = 0.33
    center_of_mass_fraction: float = 0.45
    damping_nm_per_rad_s: float = 0.8
    max_braking_torque_nm: float = 24.0
    max_handle_force_n: float = 1200.0
    impact_frame: int | None = None
    collision: bool = False

    @property
    def inertia_kg_m2(self) -> float:
        return (
            self.mass_kg
            * self.effective_length_m
            * self.effective_length_m
            * self.inertia_factor
        )

    @classmethod
    def from_clip(cls, clip: MotionClip) -> "WeaponDynamicsProfile":
        raw = clip.dynamics.get("weapon", {})
        return cls(
            mass_kg=float(raw.get("mass_kg", 1.3)),
            effective_length_m=float(raw.get("effective_length_m", 0.9)),
            inertia_factor=float(raw.get("inertia_factor", 0.33)),
            center_of_mass_fraction=float(raw.get("center_of_mass_fraction", 0.45)),
            damping_nm_per_rad_s=float(raw.get("damping_nm_per_rad_s", 0.8)),
            max_braking_torque_nm=float(raw.get("max_braking_torque_nm", 24.0)),
            max_handle_force_n=float(raw.get("max_handle_force_n", 1200.0)),
            impact_frame=(
                int(raw["impact_frame"])
                if raw.get("impact_frame") is not None
                else None
            ),
            collision=bool(raw.get("collision", False)),
        )


def sword_angle(frame: FramePose) -> float:
    """Return sword angle in mathematical coordinates (radians)."""
    delta = frame.weapon.tip - frame.weapon.grip_main
    return math.atan2(-delta.y, delta.x)


def unwrap_angles(values: list[float]) -> list[float]:
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


def _weapon_com_px(frame: FramePose, fraction: float) -> Vec2:
    return frame.weapon.grip_main + (
        frame.weapon.tip - frame.weapon.grip_main
    ) * fraction


def analyze_weapon_dynamics(clip: MotionClip) -> dict[str, Any]:
    """Analyze rigid-weapon rotation, translation, force, power and inertia.

    This is a planar rigid-body approximation. It intentionally captures the
    quantities that make animation read as physical: angular velocity,
    acceleration, torque, translational acceleration, handle reaction force,
    energy and follow-through.
    """
    profile = WeaponDynamicsProfile.from_clip(clip)
    dt = 1.0 / clip.fps
    angles = unwrap_angles([sword_angle(frame) for frame in clip.frames])

    omega = scalar_derivative(angles, dt)
    alpha = scalar_derivative(omega, dt)
    inertia = profile.inertia_kg_m2
    torque = [
        inertia * alpha_i + profile.damping_nm_per_rad_s * omega_i
        for alpha_i, omega_i in zip(alpha, omega)
    ]
    rotational_energy = [
        0.5 * inertia * omega_i * omega_i
        for omega_i in omega
    ]

    body_raw = clip.dynamics.get("body", {})
    ppm = max(float(body_raw.get("pixels_per_meter", 40.0)), 1e-9)
    gravity = float(body_raw.get("gravity_m_s2", 9.81))

    com_px = [
        _weapon_com_px(frame, profile.center_of_mass_fraction)
        for frame in clip.frames
    ]
    com_m = [Vec2(point.x / ppm, point.y / ppm) for point in com_px]
    com_velocity = vec_derivative(com_m, dt)
    com_acceleration = vec_derivative(com_velocity, dt)
    com_jerk = vec_derivative(com_acceleration, dt)

    handle_force: list[Vec2] = []
    reaction_force: list[Vec2] = []
    translational_energy: list[float] = []
    total_energy: list[float] = []
    mechanical_power: list[float] = []

    for i in range(len(clip.frames)):
        # Screen Y grows downward. Gravity is +Y. Force that the hands must
        # apply to the weapon: m*a - m*g.
        force = Vec2(
            profile.mass_kg * com_acceleration[i].x,
            profile.mass_kg * (com_acceleration[i].y - gravity),
        )
        reaction = force * -1.0
        handle_force.append(force)
        reaction_force.append(reaction)

        speed = com_velocity[i].length()
        translational = 0.5 * profile.mass_kg * speed * speed
        translational_energy.append(translational)
        total_energy.append(translational + rotational_energy[i])

        translational_power = (
            force.x * com_velocity[i].x
            + force.y * com_velocity[i].y
        )
        rotational_power = torque[i] * omega[i]
        mechanical_power.append(translational_power + rotational_power)

    warnings: list[dict[str, Any]] = []

    for i, frame in enumerate(clip.frames):
        force_mag = handle_force[i].length()
        if force_mag > profile.max_handle_force_n:
            warnings.append(
                {
                    "code": "handle_force_exceeded",
                    "frame": frame.frame,
                    "message": (
                        f"Estimated handle force {force_mag:.1f} N exceeds "
                        f"configured {profile.max_handle_force_n:.1f} N."
                    ),
                }
            )

    impact_index: int | None = None
    if profile.impact_frame is not None:
        for i, frame in enumerate(clip.frames):
            if frame.frame == profile.impact_frame:
                impact_index = i
                break

    follow_through: dict[str, Any] | None = None
    if impact_index is not None and impact_index > 0:
        impact_omega = omega[impact_index]
        braking_alpha = (
            profile.max_braking_torque_nm / inertia
            if inertia > 1e-9
            else float("inf")
        )
        min_stop_time = (
            abs(impact_omega) / braking_alpha
            if braking_alpha > 1e-9
            else float("inf")
        )
        min_stop_angle = (
            impact_omega * impact_omega / (2.0 * braking_alpha)
            if braking_alpha > 1e-9
            else float("inf")
        )

        follow_through = {
            "impact_frame": profile.impact_frame,
            "impact_angular_velocity_rad_s": impact_omega,
            "impact_angular_velocity_deg_s": math.degrees(impact_omega),
            "impact_rotational_energy_j": rotational_energy[impact_index],
            "impact_total_kinetic_energy_j": total_energy[impact_index],
            "minimum_stop_time_s_at_max_braking": min_stop_time,
            "minimum_follow_through_deg_at_max_braking": math.degrees(min_stop_angle),
        }

        if not profile.collision and impact_index + 1 < len(angles):
            post_delta = angles[impact_index + 1] - angles[impact_index]

            if abs(impact_omega) > 1e-6:
                same_direction = (
                    post_delta == 0.0
                    or math.copysign(1.0, post_delta) == math.copysign(1.0, impact_omega)
                )
                if not same_direction:
                    warnings.append(
                        {
                            "code": "weapon_direction_reversal",
                            "frame": clip.frames[impact_index + 1].frame,
                            "message": (
                                "Sword reverses direction immediately after impact "
                                "without a collision event."
                            ),
                        }
                    )

            if abs(torque[impact_index + 1]) > profile.max_braking_torque_nm:
                warnings.append(
                    {
                        "code": "implausible_braking_torque",
                        "frame": clip.frames[impact_index + 1].frame,
                        "message": (
                            f"Required braking torque {abs(torque[impact_index + 1]):.2f} Nm "
                            f"exceeds configured limit {profile.max_braking_torque_nm:.2f} Nm."
                        ),
                    }
                )

            actual_follow = abs(post_delta)
            if actual_follow + 1e-9 < min_stop_angle:
                warnings.append(
                    {
                        "code": "insufficient_follow_through",
                        "frame": clip.frames[impact_index + 1].frame,
                        "message": (
                            f"Authored follow-through is {math.degrees(actual_follow):.1f} deg; "
                            f"configured weapon/profile implies at least "
                            f"{math.degrees(min_stop_angle):.1f} deg to stop at max braking."
                        ),
                    }
                )

    frames = []
    for i, frame in enumerate(clip.frames):
        frames.append(
            {
                "frame": frame.frame,
                "label": frame.label,
                "angle_deg": math.degrees(angles[i]),
                "angular_velocity_deg_s": math.degrees(omega[i]),
                "angular_acceleration_deg_s2": math.degrees(alpha[i]),
                "estimated_torque_nm": torque[i],
                "rotational_energy_j": rotational_energy[i],
                "weapon_com_px": com_px[i].as_list(),
                "com_velocity_m_s": com_velocity[i].as_list(),
                "com_acceleration_m_s2": com_acceleration[i].as_list(),
                "com_jerk_m_s3": com_jerk[i].as_list(),
                "handle_force_on_weapon_n": handle_force[i].as_list(),
                "handle_force_magnitude_n": handle_force[i].length(),
                "reaction_force_on_body_n": reaction_force[i].as_list(),
                "translational_energy_j": translational_energy[i],
                "total_kinetic_energy_j": total_energy[i],
                "mechanical_power_w": mechanical_power[i],
            }
        )

    return {
        "profile": {
            **asdict(profile),
            "inertia_kg_m2": inertia,
        },
        "frames": frames,
        "follow_through": follow_through,
        "warnings": warnings,
    }
