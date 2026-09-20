from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .kinematics import stop_indices, vec_derivative_times
from .model import MotionClip, Vec2
from .rig import RestGeometry


@dataclass(frozen=True)
class CoordinationProfile:
    near_lock_extension_ratio: float = 0.985
    minimum_useful_extension_ratio: float = 0.20
    peak_order_tolerance_s: float = 0.05
    warn_near_lock: bool = True
    warn_kinetic_chain_order: bool = True
    warn_ik_branch_flip: bool = True
    branch_flip_min_offset_px: float = 1.5

    @classmethod
    def from_clip(cls, clip: MotionClip) -> "CoordinationProfile":
        raw = clip.dynamics.get("body", {}).get("coordination", {})
        return cls(
            near_lock_extension_ratio=float(
                raw.get("near_lock_extension_ratio", 0.985)
            ),
            minimum_useful_extension_ratio=float(
                raw.get("minimum_useful_extension_ratio", 0.20)
            ),
            peak_order_tolerance_s=float(
                raw.get("peak_order_tolerance_s", 0.05)
            ),
            warn_near_lock=bool(raw.get("warn_near_lock", True)),
            warn_kinetic_chain_order=bool(
                raw.get("warn_kinetic_chain_order", True)
            ),
            warn_ik_branch_flip=bool(
                raw.get("warn_ik_branch_flip", True)
            ),
            branch_flip_min_offset_px=float(
                raw.get("branch_flip_min_offset_px", 1.5)
            ),
        )


def _mid(a: Vec2, b: Vec2) -> Vec2:
    return Vec2((a.x + b.x) * 0.5, (a.y + b.y) * 0.5)




def _bend_side_offset(
    start: Vec2,
    mid: Vec2,
    end: Vec2,
) -> float:
    axis = end - start
    length = axis.length()
    if length <= 1e-9:
        return 0.0
    rel = mid - start
    cross = axis.x * rel.y - axis.y * rel.x
    return cross / length


def _peak_time(
    values: list[float],
    times: list[float],
) -> tuple[int, float, float] | None:
    if not values:
        return None
    index = max(range(len(values)), key=lambda i: values[i])
    return index, times[index], values[index]


def analyze_coordination(
    clip: MotionClip,
    body_report: dict[str, Any],
    weapon_report: dict[str, Any],
) -> dict[str, Any]:
    """Diagnose reach comfort and proximal-to-distal motion sequencing.

    This is not a prescriptive martial-arts model. It highlights common
    animation defects: locked elbows, collapsed arms, and a weapon that reaches
    peak speed before the hands/body have meaningfully accelerated it.
    """
    if not clip.frames:
        return {
            "profile": asdict(CoordinationProfile.from_clip(clip)),
            "frames": [],
            "peaks": {},
            "warnings": [],
        }

    profile = CoordinationProfile.from_clip(clip)
    rest = RestGeometry.from_frame(clip.frames[0])
    times = clip.times_s()
    stops = stop_indices(clip)
    warnings: list[dict[str, Any]] = []
    frame_reports: list[dict[str, Any]] = []

    hand_midpoints: list[Vec2] = []
    sword_tips: list[Vec2] = []
    previous_bend_offsets: dict[str, float] = {}

    for index, frame in enumerate(clip.frames):
        per_side: dict[str, Any] = {}
        for side in ("l", "r"):
            shoulder_name = f"shoulder_{side}"
            hand_name = f"hand_{side}"
            if shoulder_name not in frame.joints or hand_name not in frame.joints:
                continue

            reach = frame.joints[shoulder_name].distance_to(frame.joints[hand_name])
            max_reach = (
                rest.bone_lengths[f"upper_arm_{side}"]
                + rest.bone_lengths[f"forearm_{side}"]
            )
            ratio = reach / max(max_reach, 1e-9)
            per_side[side] = {
                "reach_px": reach,
                "max_reach_px": max_reach,
                "extension_ratio": ratio,
            }

            if profile.warn_near_lock and ratio >= profile.near_lock_extension_ratio:
                warnings.append(
                    {
                        "code": "arm_near_full_extension",
                        "frame": frame.frame,
                        "side": side,
                        "message": (
                            f"{side} arm extension ratio {ratio:.3f} is near "
                            f"the configured lock threshold "
                            f"{profile.near_lock_extension_ratio:.3f}."
                        ),
                    }
                )

            if ratio <= profile.minimum_useful_extension_ratio:
                warnings.append(
                    {
                        "code": "arm_collapsed",
                        "frame": frame.frame,
                        "side": side,
                        "message": (
                            f"{side} arm extension ratio {ratio:.3f} is unusually "
                            f"compressed for a two-handed sword action."
                        ),
                    }
                )

        left = frame.joints.get("hand_l")
        right = frame.joints.get("hand_r")
        if left is not None and right is not None:
            hand_midpoints.append(_mid(left, right))
        else:
            hand_midpoints.append(frame.weapon.grip_main)

        sword_tips.append(frame.weapon.tip)

        bend_offsets: dict[str, float] = {}
        for chain, start_name, mid_name, end_name in (
            ("elbow_l", "shoulder_l", "elbow_l", "hand_l"),
            ("elbow_r", "shoulder_r", "elbow_r", "hand_r"),
            ("knee_l", "hip_l", "knee_l", "foot_l"),
            ("knee_r", "hip_r", "knee_r", "foot_r"),
        ):
            if not {
                start_name,
                mid_name,
                end_name,
            }.issubset(frame.joints):
                continue
            offset = _bend_side_offset(
                frame.joints[start_name],
                frame.joints[mid_name],
                frame.joints[end_name],
            )
            bend_offsets[chain] = offset

            previous_offset = previous_bend_offsets.get(chain)
            if (
                profile.warn_ik_branch_flip
                and previous_offset is not None
                and previous_offset * offset < 0.0
                and abs(previous_offset)
                >= profile.branch_flip_min_offset_px
                and abs(offset)
                >= profile.branch_flip_min_offset_px
            ):
                warnings.append(
                    {
                        "code": "ik_branch_flip",
                        "frame": frame.frame,
                        "joint": chain,
                        "message": (
                            f"{chain} changes bend side abruptly "
                            f"({previous_offset:.2f}px -> {offset:.2f}px) "
                            "without passing near a straight chain."
                        ),
                    }
                )
            previous_bend_offsets[chain] = offset

        frame_reports.append(
            {
                "frame": frame.frame,
                "time_s": times[index],
                "label": frame.label,
                "arms": per_side,
                "bend_offsets_px": bend_offsets,
            }
        )

    body_ppm = max(
        float(body_report.get("profile", {}).get("pixels_per_meter", 40.0)),
        1e-9,
    )
    hand_m = [Vec2(p.x / body_ppm, p.y / body_ppm) for p in hand_midpoints]
    tip_m = [Vec2(p.x / body_ppm, p.y / body_ppm) for p in sword_tips]
    hand_velocity = vec_derivative_times(hand_m, times, stops)
    tip_velocity = vec_derivative_times(tip_m, times, stops)

    root_speeds = [
        Vec2.from_any(item.get("root_velocity_m_s", [0.0, 0.0])).length()
        for item in body_report.get("frames", [])
    ]
    hand_speeds = [v.length() for v in hand_velocity]
    tip_speeds = [v.length() for v in tip_velocity]
    sword_omega = [
        abs(float(item.get("angular_velocity_deg_s", 0.0)))
        for item in weapon_report.get("frames", [])
    ]

    peaks: dict[str, Any] = {}
    for name, values in (
        ("root_speed", root_speeds),
        ("hands_speed", hand_speeds),
        ("sword_tip_speed", tip_speeds),
        ("sword_angular_speed", sword_omega),
    ):
        peak = _peak_time(values, times)
        if peak is None:
            continue
        index, time_s, value = peak
        peaks[name] = {
            "frame": clip.frames[index].frame,
            "time_s": time_s,
            "value": value,
        }

    # A common power-transfer pattern is body/root motion -> hands -> weapon
    # tip. We keep this diagnostic soft because technique and camera plane can
    # legitimately change the ordering.
    if profile.warn_kinetic_chain_order:
        hands_peak = peaks.get("hands_speed")
        tip_peak = peaks.get("sword_tip_speed")
        if (
            hands_peak
            and tip_peak
            and tip_peak["time_s"]
            < hands_peak["time_s"] - profile.peak_order_tolerance_s
        ):
            warnings.append(
                {
                    "code": "kinetic_chain_order",
                    "frame": tip_peak["frame"],
                    "message": (
                        "Sword-tip speed peaks materially before hand speed; "
                        "check whether the weapon is being accelerated by the "
                        "body/hands or is visually leading them."
                    ),
                }
            )

    return {
        "profile": asdict(profile),
        "frames": frame_reports,
        "peaks": peaks,
        "warnings": warnings,
    }
