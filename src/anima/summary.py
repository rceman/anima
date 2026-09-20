from __future__ import annotations

from typing import Any

from .constraints import ValidationReport
from .model import MotionClip


def _max_abs(items: list[dict[str, Any]], key: str) -> float:
    values = [abs(float(item.get(key, 0.0))) for item in items]
    return max(values, default=0.0)


def _max(items: list[dict[str, Any]], key: str) -> float:
    values = [float(item.get(key, 0.0)) for item in items]
    return max(values, default=0.0)


def _min_optional(items: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(item[key])
        for item in items
        if item.get(key) is not None
    ]
    return min(values) if values else None


def build_diagnostics_summary(
    clip: MotionClip,
    geometry: ValidationReport,
    analysis: dict[str, Any],
) -> dict[str, Any]:
    dynamics = analysis["dynamics"]
    physics = analysis["physics_validation"]
    occlusion = analysis.get("occlusion", {"warnings": []})
    timing = analysis["timing_recommendation"]

    weapon_frames = dynamics.get("weapon", {}).get("frames", [])
    body_frames = dynamics.get("body", {}).get("frames", [])
    system_frames = dynamics.get("system", {}).get("frames", [])

    times = clip.times_s()
    duration = times[-1] - times[0] if len(times) > 1 else 0.0

    friction_values = [
        float(frame["ground_reaction_force"]["required_friction_ratio"])
        for frame in system_frames
        if frame.get("ground_reaction_force")
    ]

    return {
        "clip": {
            "rig": clip.rig,
            "frame_count": len(clip.frames),
            "canvas": [clip.width, clip.height],
            "ground_y": clip.ground_y,
            "fps_nominal": clip.fps,
            "duration_s": duration,
            "explicit_timing": any(
                getattr(frame, "time_s", None) is not None
                for frame in clip.frames
            ),
        },
        "geometry": {
            "ok": geometry.ok,
            "issue_count": len(geometry.issues),
            "issue_codes": sorted({issue.code for issue in geometry.issues}),
        },
        "physics": {
            "ok": physics["ok"],
            "hard_issue_count": physics["counts"]["hard"],
            "warning_count": physics["counts"]["warnings"],
            "hard_issue_codes": sorted(
                {item["code"] for item in physics["hard_issues"]}
            ),
            "warning_codes": sorted(
                {item["code"] for item in physics["warnings"]}
            ),
        },
        "occlusion": {
            "warning_count": len(occlusion.get("warnings", [])),
            "warning_codes": sorted(
                {
                    item["code"]
                    for item in occlusion.get("warnings", [])
                }
            ),
        },
        "timing": {
            "recommended_global_time_scale": float(
                timing["recommended"]["global_time_scale"]
            ),
            "recommended_duration_s": float(
                timing["recommended"]["duration_s"]
            ),
        },
        "weapon": {
            "max_angular_velocity_deg_s": _max_abs(
                weapon_frames,
                "angular_velocity_deg_s",
            ),
            "max_angular_acceleration_deg_s2": _max_abs(
                weapon_frames,
                "angular_acceleration_deg_s2",
            ),
            "max_abs_torque_nm": _max_abs(
                weapon_frames,
                "estimated_torque_nm",
            ),
            "max_handle_force_n": _max(
                weapon_frames,
                "handle_force_magnitude_n",
            ),
            "max_total_kinetic_energy_j": _max(
                weapon_frames,
                "total_kinetic_energy_j",
            ),
        },
        "body": {
            "max_com_speed_m_s": _max(body_frames, "com_speed_m_s"),
            "max_net_force_n": _max(body_frames, "estimated_net_force_n"),
            "min_stability_margin_px": _min_optional(
                [
                    {
                        "stability_margin_px": (
                            frame.get("support", {}) or {}
                        ).get("stability_margin_px")
                    }
                    for frame in body_frames
                ],
                "stability_margin_px",
            ),
        },
        "system": {
            "max_com_speed_m_s": _max(
                system_frames,
                "system_com_speed_m_s",
            ),
            "max_net_force_n": _max(
                system_frames,
                "estimated_net_force_n",
            ),
            "min_stability_margin_px": _min_optional(
                system_frames,
                "stability_margin_px",
            ),
            "max_required_friction_ratio": max(
                friction_values,
                default=0.0,
            ),
        },
    }
