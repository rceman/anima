from __future__ import annotations

from typing import Any

from .contacts import contact_mode
from .model import MotionClip


def build_animation_manifest(
    clip: MotionClip,
    columns: int = 4,
) -> dict[str, Any]:
    """Build a runtime-friendly manifest for the canonical spritesheet.

    The manifest deliberately separates *pose index* from *duration*. A sprite
    sheet can keep one 128x128 cell per authored pose while the runtime honors
    non-uniform physical timing.
    """
    if columns < 1:
        raise ValueError("columns must be >= 1")

    times = clip.times_s()
    if not clip.frames:
        durations_ms: list[int] = []
    elif len(clip.frames) == 1:
        durations_ms = [max(1, round(1000.0 / clip.fps))]
    else:
        durations_ms = [
            max(1, round((right - left) * 1000.0))
            for left, right in zip(times, times[1:])
        ]
        durations_ms.append(durations_ms[-1])

    impact_frame = (
        clip.dynamics.get("weapon", {}).get("impact_frame")
        if clip.dynamics
        else None
    )

    frames: list[dict[str, Any]] = []
    for index, frame in enumerate(clip.frames):
        col = index % columns
        row = index // columns
        frames.append(
            {
                "index": index,
                "source_frame": frame.frame,
                "label": frame.label,
                "time_s": times[index],
                "duration_ms": durations_ms[index],
                "cell": {
                    "x": col * clip.width,
                    "y": row * clip.height,
                    "width": clip.width,
                    "height": clip.height,
                },
                "root": frame.root.as_list(),
                "ground_y": clip.ground_y,
                "contacts": {
                    name: contact_mode(value)
                    for name, value in frame.contacts.items()
                },
                "events": (
                    ["impact"]
                    if impact_frame is not None
                    and int(impact_frame) == frame.frame
                    else []
                ),
            }
        )

    rows = (len(clip.frames) + columns - 1) // columns
    return {
        "version": 1,
        "rig": clip.rig,
        "sheet": {
            "columns": columns,
            "rows": rows,
            "cell_width": clip.width,
            "cell_height": clip.height,
            "width": columns * clip.width,
            "height": rows * clip.height,
        },
        "timing": {
            "nominal_fps": clip.fps,
            "explicit_timestamps": any(
                getattr(frame, "time_s", None) is not None
                for frame in clip.frames
            ),
            "duration_s": (
                times[-1] - times[0]
                if len(times) > 1
                else 0.0
            ),
        },
        "frames": frames,
    }
