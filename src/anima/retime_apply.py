from __future__ import annotations

from dataclasses import replace
from typing import Any

from .analysis import analyze_motion
from .model import MotionClip


def _interval_scale(
    left_frame: int,
    right_frame: int,
    timing_report: dict[str, Any],
) -> float:
    for segment in timing_report.get("segments", []):
        start = int(segment["from_frame"])
        end = int(segment["to_frame"])
        if left_frame >= start and right_frame <= end:
            return max(1.0, float(segment.get("recommended_time_scale", 1.0)))

    return max(
        1.0,
        float(
            timing_report.get("recommended", {}).get(
                "global_time_scale",
                1.0,
            )
        ),
    )


def apply_timing_recommendation(
    clip: MotionClip,
    timing_report: dict[str, Any],
    max_interval_scale: float = 4.0,
) -> MotionClip:
    """Apply physics timing recommendations without changing pose geometry.

    The output uses explicit timestamps. Each interval receives the local
    labeled-phase scale when available, otherwise the global recommendation.
    """
    if len(clip.frames) < 2:
        return clip

    old_times = clip.times_s()
    new_times = [old_times[0]]

    for index, (left, right) in enumerate(zip(clip.frames, clip.frames[1:])):
        scale = _interval_scale(left.frame, right.frame, timing_report)
        scale = min(max_interval_scale, max(1.0, scale))
        old_dt = old_times[index + 1] - old_times[index]
        new_times.append(new_times[-1] + old_dt * scale)

    frames = [
        replace(frame, time_s=new_times[index])
        for index, frame in enumerate(clip.frames)
    ]
    return replace(clip, frames=frames)


def auto_retime(
    clip: MotionClip,
    iterations: int = 3,
    tolerance: float = 1.01,
    max_interval_scale: float = 4.0,
) -> tuple[MotionClip, list[dict[str, Any]]]:
    """Iteratively slow physically over-demanding phases.

    Geometry is never changed. This is intentionally monotonic: automatic
    retiming may add time but never speeds authored motion up.
    """
    current = clip
    history: list[dict[str, Any]] = []

    for iteration in range(max(0, iterations)):
        analysis = analyze_motion(current)
        timing = analysis["timing_recommendation"]
        scale = float(
            timing.get("recommended", {}).get("global_time_scale", 1.0)
        )
        history.append(
            {
                "iteration": iteration,
                "duration_s": (
                    current.times_s()[-1] - current.times_s()[0]
                    if len(current.frames) > 1
                    else 0.0
                ),
                "global_time_scale": scale,
                "physics_ok": bool(
                    analysis["physics_validation"]["ok"]
                ),
                "hard_issue_count": int(
                    analysis["physics_validation"]["counts"]["hard"]
                ),
            }
        )
        if scale <= tolerance:
            break

        current = apply_timing_recommendation(
            current,
            timing,
            max_interval_scale=max_interval_scale,
        )

    final_analysis = analyze_motion(current)
    history.append(
        {
            "iteration": "final",
            "duration_s": (
                current.times_s()[-1] - current.times_s()[0]
                if len(current.frames) > 1
                else 0.0
            ),
            "global_time_scale": float(
                final_analysis["timing_recommendation"]
                .get("recommended", {})
                .get("global_time_scale", 1.0)
            ),
            "physics_ok": bool(final_analysis["physics_validation"]["ok"]),
            "hard_issue_count": int(
                final_analysis["physics_validation"]["counts"]["hard"]
            ),
        }
    )
    return current, history
