from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

from .kinematics import scalar_derivative_times
from .model import MotionClip, Vec2


@dataclass(frozen=True)
class AngularMomentumProfile:
    """Coarse planar rotational model for person + weapon balance.

    body_radius_of_gyration_m controls the effective whole-body inertia around
    the body's COM. This is deliberately configurable because a single 2D rig
    cannot infer a subject's true 3D mass distribution.
    """

    body_radius_of_gyration_m: float = 0.30
    body_spin_scale: float = 1.0
    weapon_spin_scale: float = 1.0
    orbital_scale: float = 1.0

    @classmethod
    def from_clip(cls, clip: MotionClip) -> "AngularMomentumProfile":
        raw = clip.dynamics.get("body", {}).get("angular_momentum", {})
        return cls(
            body_radius_of_gyration_m=float(
                raw.get("body_radius_of_gyration_m", 0.30)
            ),
            body_spin_scale=float(raw.get("body_spin_scale", 1.0)),
            weapon_spin_scale=float(raw.get("weapon_spin_scale", 1.0)),
            orbital_scale=float(raw.get("orbital_scale", 1.0)),
        )


def _world_position(
    point_px: Vec2,
    ground_y: float,
    pixels_per_meter: float,
) -> Vec2:
    """Convert screen x/down-y coordinates to x/up-z meters."""
    return Vec2(
        point_px.x / pixels_per_meter,
        (ground_y - point_px.y) / pixels_per_meter,
    )


def _world_velocity(screen_velocity_m_s: Vec2) -> Vec2:
    return Vec2(screen_velocity_m_s.x, -screen_velocity_m_s.y)


def _cross(a: Vec2, b: Vec2) -> float:
    return a.x * b.y - a.y * b.x


def analyze_angular_momentum(
    clip: MotionClip,
    body_report: dict[str, Any],
    weapon_report: dict[str, Any],
) -> dict[str, Any]:
    """Estimate planar angular momentum of the combined body + weapon system.

    The model contains:
      - whole-body spin around body COM, using torso angular velocity;
      - weapon spin around weapon COM;
      - orbital angular momentum of body and weapon COM around system COM.

    It is intentionally diagnostic rather than a full multibody solver, but it
    gives the balance model a physically meaningful angular-momentum-rate term
    instead of assuming all rotational momentum is zero.
    """
    profile = AngularMomentumProfile.from_clip(clip)
    body_profile = body_report.get("profile", {})
    weapon_profile = weapon_report.get("profile", {})

    body_mass = float(body_profile.get("mass_kg", 75.0))
    weapon_mass = float(weapon_profile.get("mass_kg", 1.3))
    total_mass = body_mass + weapon_mass
    ppm = max(float(body_profile.get("pixels_per_meter", 40.0)), 1e-9)

    body_inertia = (
        body_mass
        * profile.body_radius_of_gyration_m
        * profile.body_radius_of_gyration_m
    )

    weapon_inertia_grip = float(
        weapon_profile.get("inertia_kg_m2", 0.0)
    )
    weapon_length = float(
        weapon_profile.get("effective_length_m", 0.0)
    )
    weapon_com_fraction = float(
        weapon_profile.get("center_of_mass_fraction", 0.45)
    )
    weapon_com_radius = weapon_length * weapon_com_fraction

    # The weapon dynamics profile stores an inertia around the grip. Convert
    # that to COM inertia via the parallel-axis theorem for the spin term.
    weapon_inertia_com = max(
        0.0,
        weapon_inertia_grip
        - weapon_mass * weapon_com_radius * weapon_com_radius,
    )

    body_frames = body_report.get("frames", [])
    weapon_frames = weapon_report.get("frames", [])
    count = min(len(clip.frames), len(body_frames), len(weapon_frames))
    times = clip.times_s()[:count]

    total_h: list[float] = []
    raw_frames: list[dict[str, Any]] = []

    for index in range(count):
        body_frame = body_frames[index]
        weapon_frame = weapon_frames[index]

        body_pos = _world_position(
            Vec2.from_any(body_frame["com_px"]),
            clip.ground_y,
            ppm,
        )
        weapon_pos = _world_position(
            Vec2.from_any(weapon_frame["weapon_com_px"]),
            clip.ground_y,
            ppm,
        )
        system_pos = Vec2(
            (
                body_pos.x * body_mass
                + weapon_pos.x * weapon_mass
            )
            / max(total_mass, 1e-9),
            (
                body_pos.y * body_mass
                + weapon_pos.y * weapon_mass
            )
            / max(total_mass, 1e-9),
        )

        body_velocity = _world_velocity(
            Vec2.from_any(
                body_frame.get("com_velocity_m_s", [0.0, 0.0])
            )
        )
        weapon_velocity = _world_velocity(
            Vec2.from_any(
                weapon_frame.get("com_velocity_m_s", [0.0, 0.0])
            )
        )
        system_velocity = Vec2(
            (
                body_velocity.x * body_mass
                + weapon_velocity.x * weapon_mass
            )
            / max(total_mass, 1e-9),
            (
                body_velocity.y * body_mass
                + weapon_velocity.y * weapon_mass
            )
            / max(total_mass, 1e-9),
        )

        torso_omega = math.radians(
            float(body_frame.get("torso_angular_velocity_deg_s", 0.0))
        )
        weapon_omega = math.radians(
            float(weapon_frame.get("angular_velocity_deg_s", 0.0))
        )

        body_spin = (
            body_inertia
            * torso_omega
            * profile.body_spin_scale
        )
        weapon_spin = (
            weapon_inertia_com
            * weapon_omega
            * profile.weapon_spin_scale
        )

        body_orbital = _cross(
            body_pos - system_pos,
            (body_velocity - system_velocity) * body_mass,
        )
        weapon_orbital = _cross(
            weapon_pos - system_pos,
            (weapon_velocity - system_velocity) * weapon_mass,
        )
        orbital = (
            body_orbital + weapon_orbital
        ) * profile.orbital_scale

        angular_momentum = body_spin + weapon_spin + orbital
        total_h.append(angular_momentum)
        raw_frames.append(
            {
                "frame": clip.frames[index].frame,
                "time_s": times[index],
                "label": clip.frames[index].label,
                "body_spin_angular_momentum_kg_m2_s": body_spin,
                "weapon_spin_angular_momentum_kg_m2_s": weapon_spin,
                "orbital_angular_momentum_kg_m2_s": orbital,
                "total_angular_momentum_kg_m2_s": angular_momentum,
            }
        )

    hdot = scalar_derivative_times(total_h, times) if count else []
    for index, item in enumerate(raw_frames):
        item["angular_momentum_rate_nm"] = hdot[index]

    return {
        "profile": {
            **asdict(profile),
            "body_inertia_kg_m2": body_inertia,
            "weapon_inertia_about_com_kg_m2": weapon_inertia_com,
        },
        "frames": raw_frames,
    }
