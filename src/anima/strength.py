from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .model import MotionClip, Vec2


@dataclass(frozen=True)
class ArmStrengthProfile:
    primary_force_fraction: float = 0.60
    shoulder_limit_nm: float = 150.0
    elbow_limit_nm: float = 100.0
    enforce_limits: bool = False

    @classmethod
    def from_clip(cls, clip: MotionClip) -> "ArmStrengthProfile":
        raw = clip.dynamics.get("body", {}).get("strength", {})
        return cls(
            primary_force_fraction=float(raw.get("primary_force_fraction", 0.60)),
            shoulder_limit_nm=float(raw.get("shoulder_limit_nm", 150.0)),
            elbow_limit_nm=float(raw.get("elbow_limit_nm", 100.0)),
            enforce_limits=bool(raw.get("enforce_limits", False)),
        )


def _cross_moment(r_m: Vec2, force_n: Vec2) -> float:
    return r_m.x * force_n.y - r_m.y * force_n.x


def _side_from_hand(hand_name: str) -> str:
    if hand_name.endswith("_l"):
        return "l"
    if hand_name.endswith("_r"):
        return "r"
    raise ValueError(f"Unsupported hand joint name: {hand_name}")


def analyze_arm_strength(
    clip: MotionClip,
    weapon_report: dict[str, Any],
) -> dict[str, Any]:
    """Approximate shoulder/elbow loads induced by the weapon.

    This is a planar inverse-load estimate, not a muscle simulator. The weapon's
    reaction force and reaction torque are distributed across the two hands and
    converted into moments about each elbow and shoulder.
    """
    profile = ArmStrengthProfile.from_clip(clip)
    body_raw = clip.dynamics.get("body", {})
    ppm = max(float(body_raw.get("pixels_per_meter", 40.0)), 1e-9)

    primary_side = _side_from_hand(clip.primary_hand)
    secondary_side = _side_from_hand(clip.secondary_hand)

    p_fraction = min(1.0, max(0.0, profile.primary_force_fraction))
    fractions = {
        primary_side: p_fraction,
        secondary_side: 1.0 - p_fraction,
    }

    weapon_frames = weapon_report.get("frames", [])
    warnings: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []

    for index, frame in enumerate(clip.frames):
        if index >= len(weapon_frames):
            break
        w = weapon_frames[index]
        reaction = Vec2.from_any(w.get("reaction_force_on_body_n", [0.0, 0.0]))
        # estimated_torque_nm is the torque applied to the weapon; the body
        # receives the equal and opposite reaction.
        reaction_torque = -float(w.get("estimated_torque_nm", 0.0))

        side_reports: dict[str, Any] = {}
        for side in ("l", "r"):
            hand_name = f"hand_{side}"
            elbow_name = f"elbow_{side}"
            shoulder_name = f"shoulder_{side}"
            if not {hand_name, elbow_name, shoulder_name}.issubset(frame.joints):
                continue

            fraction = fractions.get(side, 0.5)
            force = reaction * fraction
            torque_share = reaction_torque * fraction

            hand = frame.joints[hand_name]
            elbow = frame.joints[elbow_name]
            shoulder = frame.joints[shoulder_name]

            r_shoulder = (hand - shoulder) * (1.0 / ppm)
            r_elbow = (hand - elbow) * (1.0 / ppm)

            shoulder_moment = _cross_moment(r_shoulder, force) + torque_share
            elbow_moment = _cross_moment(r_elbow, force) + torque_share

            side_reports[side] = {
                "force_fraction": fraction,
                "hand_force_n": force.as_list(),
                "hand_force_magnitude_n": force.length(),
                "shoulder_torque_nm": shoulder_moment,
                "elbow_torque_nm": elbow_moment,
            }

            if profile.enforce_limits:
                if abs(shoulder_moment) > profile.shoulder_limit_nm:
                    warnings.append(
                        {
                            "code": "joint_torque_limit_exceeded",
                            "frame": frame.frame,
                            "joint": shoulder_name,
                            "message": (
                                f"{shoulder_name} estimated torque "
                                f"{abs(shoulder_moment):.1f} Nm exceeds "
                                f"{profile.shoulder_limit_nm:.1f} Nm."
                            ),
                        }
                    )
                if abs(elbow_moment) > profile.elbow_limit_nm:
                    warnings.append(
                        {
                            "code": "joint_torque_limit_exceeded",
                            "frame": frame.frame,
                            "joint": elbow_name,
                            "message": (
                                f"{elbow_name} estimated torque "
                                f"{abs(elbow_moment):.1f} Nm exceeds "
                                f"{profile.elbow_limit_nm:.1f} Nm."
                            ),
                        }
                    )

        frames.append(
            {
                "frame": frame.frame,
                "label": frame.label,
                "sides": side_reports,
            }
        )

    return {
        "profile": asdict(profile),
        "frames": frames,
        "warnings": warnings,
    }
