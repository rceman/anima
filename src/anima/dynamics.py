from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any

from .model import FramePose, MotionClip


@dataclass(frozen=True)
class WeaponDynamicsProfile:
    mass_kg: float = 1.3
    effective_length_m: float = 0.9
    inertia_factor: float = 0.33
    damping_nm_per_rad_s: float = 0.8
    max_braking_torque_nm: float = 24.0
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
            damping_nm_per_rad_s=float(raw.get("damping_nm_per_rad_s", 0.8)),
            max_braking_torque_nm=float(raw.get("max_braking_torque_nm", 24.0)),
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


def analyze_weapon_dynamics(clip: MotionClip) -> dict[str, Any]:
    """Produce a physically-inspired diagnostic report.

    This does not claim full rigid-body simulation. It provides consistent
    kinematic/torque diagnostics so an authored swing cannot arbitrarily stop
    a heavy weapon without that discontinuity becoming visible.
    """
    profile = WeaponDynamicsProfile.from_clip(clip)
    dt = 1.0 / clip.fps
    angles = unwrap_angles([sword_angle(frame) for frame in clip.frames])

    omega: list[float] = [0.0] * len(angles)
    alpha: list[float] = [0.0] * len(angles)
    torque: list[float] = [0.0] * len(angles)
    energy: list[float] = [0.0] * len(angles)

    for i in range(1, len(angles)):
        omega[i] = (angles[i] - angles[i - 1]) / dt

    inertia = profile.inertia_kg_m2
    for i in range(1, len(angles)):
        alpha[i] = (omega[i] - omega[i - 1]) / dt
        # Torque required to both change angular speed and overcome damping.
        torque[i] = inertia * alpha[i] + profile.damping_nm_per_rad_s * omega[i]
        energy[i] = 0.5 * inertia * omega[i] * omega[i]

    warnings: list[dict[str, Any]] = []

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
            "impact_rotational_energy_j": energy[impact_index],
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
                "rotational_energy_j": energy[i],
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
