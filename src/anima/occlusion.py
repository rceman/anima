from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

from .layers import resolve_layer_order
from .model import FramePose, MotionClip, Vec2


@dataclass(frozen=True)
class OcclusionProfile:
    arm_crossing_distance_px: float = 3.0
    require_explicit_arm_crossing_order: bool = True
    require_explicit_arm_torso_order: bool = True

    @classmethod
    def from_clip(cls, clip: MotionClip) -> "OcclusionProfile":
        raw = clip.metadata.get("occlusion", {})
        return cls(
            arm_crossing_distance_px=float(
                raw.get("arm_crossing_distance_px", 3.0)
            ),
            require_explicit_arm_crossing_order=bool(
                raw.get("require_explicit_arm_crossing_order", True)
            ),
            require_explicit_arm_torso_order=bool(
                raw.get("require_explicit_arm_torso_order", True)
            ),
        )


def _cross(a: Vec2, b: Vec2) -> float:
    return a.x * b.y - a.y * b.x


def _orientation(a: Vec2, b: Vec2, c: Vec2) -> float:
    return _cross(b - a, c - a)


def _on_segment(a: Vec2, b: Vec2, p: Vec2, eps: float = 1e-9) -> bool:
    return (
        min(a.x, b.x) - eps <= p.x <= max(a.x, b.x) + eps
        and min(a.y, b.y) - eps <= p.y <= max(a.y, b.y) + eps
        and abs(_orientation(a, b, p)) <= eps
    )


def _segments_intersect(
    a: Vec2,
    b: Vec2,
    c: Vec2,
    d: Vec2,
) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)

    if o1 * o2 < 0.0 and o3 * o4 < 0.0:
        return True

    return (
        _on_segment(a, b, c)
        or _on_segment(a, b, d)
        or _on_segment(c, d, a)
        or _on_segment(c, d, b)
    )


def _point_segment_distance(p: Vec2, a: Vec2, b: Vec2) -> float:
    ab = b - a
    length_sq = ab.x * ab.x + ab.y * ab.y
    if length_sq <= 1e-12:
        return p.distance_to(a)

    ap = p - a
    t = (ap.x * ab.x + ap.y * ab.y) / length_sq
    t = min(1.0, max(0.0, t))
    projection = a + ab * t
    return p.distance_to(projection)


def _segment_distance(
    a: Vec2,
    b: Vec2,
    c: Vec2,
    d: Vec2,
) -> float:
    if _segments_intersect(a, b, c, d):
        return 0.0
    return min(
        _point_segment_distance(a, c, d),
        _point_segment_distance(b, c, d),
        _point_segment_distance(c, a, b),
        _point_segment_distance(d, a, b),
    )


def _arm_segments(frame: FramePose, side: str) -> list[tuple[Vec2, Vec2]]:
    names = (
        (f"shoulder_{side}", f"elbow_{side}"),
        (f"elbow_{side}", f"hand_{side}"),
    )
    if not all(
        a in frame.joints and b in frame.joints
        for a, b in names
    ):
        return []
    return [
        (frame.joints[a], frame.joints[b])
        for a, b in names
    ]


def _point_in_polygon(point: Vec2, polygon: list[Vec2]) -> bool:
    inside = False
    count = len(polygon)
    if count < 3:
        return False

    j = count - 1
    for i in range(count):
        a = polygon[i]
        b = polygon[j]
        if (
            (a.y > point.y) != (b.y > point.y)
            and point.x
            < (b.x - a.x)
            * (point.y - a.y)
            / max(b.y - a.y, 1e-12)
            + a.x
        ):
            inside = not inside
        j = i
    return inside


def _torso_polygon(frame: FramePose) -> list[Vec2]:
    names = ("shoulder_l", "shoulder_r", "hip_r", "hip_l")
    if not all(name in frame.joints for name in names):
        return []
    return [frame.joints[name] for name in names]


def _explicit_pair(frame: FramePose, a: str, b: str) -> bool:
    return a in frame.layer_order and b in frame.layer_order


def _resolved_relation(
    frame: FramePose,
    a: str,
    b: str,
) -> dict[str, str]:
    order = resolve_layer_order(frame.layer_order)
    if order.index(a) < order.index(b):
        return {"behind": a, "front": b}
    return {"behind": b, "front": a}


def analyze_occlusion(clip: MotionClip) -> dict[str, Any]:
    profile = OcclusionProfile.from_clip(clip)
    warnings: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []

    for frame in clip.frames:
        requirements: list[dict[str, Any]] = []

        left_segments = _arm_segments(frame, "l")
        right_segments = _arm_segments(frame, "r")
        if left_segments and right_segments:
            minimum = min(
                _segment_distance(la, lb, ra, rb)
                for la, lb in left_segments
                for ra, rb in right_segments
            )
            if minimum <= profile.arm_crossing_distance_px:
                explicit = _explicit_pair(
                    frame,
                    "left_arm",
                    "right_arm",
                )
                requirement = {
                    "layers": ["left_arm", "right_arm"],
                    "reason": "arm_crossing",
                    "minimum_distance_px": minimum,
                    "explicit": explicit,
                    **_resolved_relation(
                        frame,
                        "left_arm",
                        "right_arm",
                    ),
                }
                requirements.append(requirement)
                if (
                    profile.require_explicit_arm_crossing_order
                    and not explicit
                ):
                    warnings.append(
                        {
                            "code": "ambiguous_arm_occlusion",
                            "frame": frame.frame,
                            "message": (
                                "Left/right arm projections cross or nearly "
                                "touch, but their z-order is not explicitly "
                                "authored for this frame."
                            ),
                        }
                    )

        torso = _torso_polygon(frame)
        if torso:
            for side, layer in (
                ("l", "left_arm"),
                ("r", "right_arm"),
            ):
                key_points = [
                    frame.joints.get(f"elbow_{side}"),
                    frame.joints.get(f"hand_{side}"),
                ]
                penetrates = any(
                    point is not None and _point_in_polygon(point, torso)
                    for point in key_points
                )
                if not penetrates:
                    continue

                explicit = _explicit_pair(frame, layer, "torso")
                requirement = {
                    "layers": [layer, "torso"],
                    "reason": "arm_torso_overlap",
                    "explicit": explicit,
                    **_resolved_relation(frame, layer, "torso"),
                }
                requirements.append(requirement)

                if (
                    profile.require_explicit_arm_torso_order
                    and not explicit
                ):
                    warnings.append(
                        {
                            "code": "ambiguous_arm_torso_occlusion",
                            "frame": frame.frame,
                            "side": side,
                            "message": (
                                f"{layer} overlaps the torso projection, "
                                "but their z-order is not explicitly authored."
                            ),
                        }
                    )

        frames.append(
            {
                "frame": frame.frame,
                "label": frame.label,
                "layer_order": list(
                    resolve_layer_order(frame.layer_order)
                ),
                "requirements": requirements,
            }
        )

    return {
        "profile": asdict(profile),
        "frames": frames,
        "warnings": warnings,
    }
