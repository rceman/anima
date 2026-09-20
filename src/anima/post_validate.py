from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter

from .contacts import is_ground_contact
from .model import FramePose, MotionClip, Vec2
from .render import render_control_frame


@dataclass(frozen=True)
class RenderValidationProfile:
    background_distance_threshold: float = 24.0
    bbox_center_tolerance_px: float = 8.0
    bbox_height_ratio_min: float = 0.70
    bbox_height_ratio_max: float = 1.40
    anchor_radius_px: int = 7
    sword_tip_radius_px: int = 9
    foot_radius_px: int = 7
    ground_tolerance_px: int = 2
    pose_mask_radius_px: int = 6
    min_control_mask_coverage: float = 0.70
    min_rendered_near_control: float = 0.55
    max_background_drift: float = 12.0
    structural_anchor_radius_px: int = 6
    sword_line_radius_px: int = 3
    min_sword_line_coverage: float = 0.72


def _color_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    return math.sqrt(
        (a[0] - b[0]) ** 2
        + (a[1] - b[1]) ** 2
        + (a[2] - b[2]) ** 2
    )


def _foreground_points(
    image: Image.Image,
    background: tuple[int, int, int],
    threshold: float,
    max_y: int | None = None,
) -> list[tuple[int, int]]:
    pixels = image.load()
    width, height = image.size
    if max_y is None:
        max_y = height - 1
    max_y = min(height - 1, max_y)

    points: list[tuple[int, int]] = []
    for y in range(max_y + 1):
        for x in range(width):
            color = pixels[x, y]
            if _color_distance(color, background) > threshold:
                points.append((x, y))
    return points




def _mask_from_points(
    size: tuple[int, int],
    points: list[tuple[int, int]],
) -> Image.Image:
    mask = Image.new("L", size, 0)
    pixels = mask.load()
    for x, y in points:
        pixels[x, y] = 255
    return mask


def _mask_overlap_metrics(
    rendered: Image.Image,
    frame: FramePose,
    clip: MotionClip,
    background: tuple[int, int, int],
    profile: RenderValidationProfile,
) -> dict[str, float | int]:
    max_y = max(0, round(clip.ground_y) - 2)
    rendered_points = _foreground_points(
        rendered,
        background,
        profile.background_distance_threshold,
        max_y=max_y,
    )
    rendered_mask = _mask_from_points(
        rendered.size,
        rendered_points,
    )

    control = render_control_frame(
        clip,
        frame,
        scale=1,
    ).convert("RGB")
    control_background = tuple(
        int(value)
        for value in control.getpixel((0, 0))[:3]
    )
    control_points = _foreground_points(
        control,
        control_background,
        profile.background_distance_threshold,
        max_y=max_y,
    )
    control_mask = _mask_from_points(
        control.size,
        control_points,
    )

    radius = max(0, int(profile.pose_mask_radius_px))
    if radius > 0:
        kernel = radius * 2 + 1
        rendered_dilated = rendered_mask.filter(
            ImageFilter.MaxFilter(kernel)
        )
        control_dilated = control_mask.filter(
            ImageFilter.MaxFilter(kernel)
        )
    else:
        rendered_dilated = rendered_mask
        control_dilated = control_mask

    rendered_pixels = rendered_mask.load()
    rendered_near = control_dilated.load()
    control_pixels = control_mask.load()
    control_near = rendered_dilated.load()

    rendered_count = 0
    rendered_near_count = 0
    control_count = 0
    control_covered_count = 0

    width, height = rendered.size
    for y in range(height):
        for x in range(width):
            if rendered_pixels[x, y]:
                rendered_count += 1
                if rendered_near[x, y]:
                    rendered_near_count += 1
            if control_pixels[x, y]:
                control_count += 1
                if control_near[x, y]:
                    control_covered_count += 1

    control_coverage = (
        control_covered_count / control_count
        if control_count
        else 0.0
    )
    rendered_near_control = (
        rendered_near_count / rendered_count
        if rendered_count
        else 0.0
    )
    return {
        "radius_px": radius,
        "control_pixels": control_count,
        "rendered_pixels": rendered_count,
        "control_coverage": control_coverage,
        "rendered_near_control": rendered_near_control,
    }


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _expected_bbox(frame: FramePose, ground_y: float) -> tuple[float, float, float, float]:
    points = list(frame.joints.values()) + [
        frame.weapon.grip_main,
        frame.weapon.grip_off,
        frame.weapon.tip,
        frame.root,
    ]
    usable = [point for point in points if point.y <= ground_y + 1]
    xs = [point.x for point in usable]
    ys = [point.y for point in usable]

    # Account for actual sprite thickness around the mathematical rig.
    margin = 6.0
    return (
        min(xs) - margin,
        min(ys) - margin,
        max(xs) + margin,
        min(ground_y, max(ys) + margin),
    )


def _patch_has_foreground(
    image: Image.Image,
    point: Vec2,
    radius: int,
    background: tuple[int, int, int],
    threshold: float,
) -> bool:
    pixels = image.load()
    width, height = image.size
    cx, cy = round(point.x), round(point.y)

    for y in range(max(0, cy - radius), min(height, cy + radius + 1)):
        for x in range(max(0, cx - radius), min(width, cx + radius + 1)):
            if (x - cx) ** 2 + (y - cy) ** 2 > radius * radius:
                continue
            if _color_distance(pixels[x, y], background) > threshold:
                return True
    return False




def _segment_foreground_coverage(
    image: Image.Image,
    start: Vec2,
    end: Vec2,
    samples: int,
    radius: int,
    background: tuple[int, int, int],
    threshold: float,
) -> float:
    samples = max(2, samples)
    hits = 0
    for index in range(samples):
        t = index / (samples - 1)
        point = Vec2(
            start.x + (end.x - start.x) * t,
            start.y + (end.y - start.y) * t,
        )
        if _patch_has_foreground(
            image,
            point,
            radius,
            background,
            threshold,
        ):
            hits += 1
    return hits / samples


def _ground_contact_exists(
    image: Image.Image,
    foot: Vec2,
    radius: int,
    ground_y: float,
    y_tolerance: int,
    background: tuple[int, int, int],
    threshold: float,
) -> bool:
    pixels = image.load()
    width, height = image.size
    cx = round(foot.x)
    gy = round(ground_y)

    for y in range(max(0, gy - y_tolerance), min(height, gy + y_tolerance + 1)):
        for x in range(max(0, cx - radius), min(width, cx + radius + 1)):
            if _color_distance(pixels[x, y], background) > threshold:
                return True
    return False


def _frame_report(
    image: Image.Image,
    frame: FramePose,
    clip: MotionClip,
    profile: RenderValidationProfile,
) -> dict[str, Any]:
    background = image.getpixel((0, 0))
    if not isinstance(background, tuple) or len(background) < 3:
        raise ValueError("Rendered frame must be RGB/RGBA")
    background_rgb = tuple(int(value) for value in background[:3])

    # Ignore the ground-line area for silhouette bbox. A full-width guide line
    # must never be interpreted as character width.
    foreground = _foreground_points(
        image.convert("RGB"),
        background_rgb,
        profile.background_distance_threshold,
        max_y=max(0, round(clip.ground_y) - 2),
    )
    rendered_bbox = _bbox(foreground)
    expected_bbox = _expected_bbox(frame, clip.ground_y)

    issues: list[dict[str, Any]] = []

    bbox_metrics = None
    if rendered_bbox is None:
        issues.append(
            {
                "code": "missing_foreground",
                "message": "No foreground character pixels detected.",
            }
        )
    else:
        rx0, ry0, rx1, ry1 = rendered_bbox
        ex0, ey0, ex1, ey1 = expected_bbox
        rendered_center = Vec2((rx0 + rx1) * 0.5, (ry0 + ry1) * 0.5)
        expected_center = Vec2((ex0 + ex1) * 0.5, (ey0 + ey1) * 0.5)
        center_delta = rendered_center.distance_to(expected_center)

        rendered_height = max(1.0, float(ry1 - ry0 + 1))
        expected_height = max(1.0, float(ey1 - ey0 + 1))
        height_ratio = rendered_height / expected_height

        bbox_metrics = {
            "rendered": list(rendered_bbox),
            "expected": list(expected_bbox),
            "center_delta_px": center_delta,
            "height_ratio": height_ratio,
        }

        if center_delta > profile.bbox_center_tolerance_px:
            issues.append(
                {
                    "code": "character_center_drift",
                    "message": (
                        f"Rendered silhouette center is {center_delta:.2f}px "
                        f"from expected motion geometry."
                    ),
                }
            )

        if not (
            profile.bbox_height_ratio_min
            <= height_ratio
            <= profile.bbox_height_ratio_max
        ):
            issues.append(
                {
                    "code": "character_scale_drift",
                    "message": (
                        f"Rendered silhouette height ratio {height_ratio:.2f} "
                        f"is outside [{profile.bbox_height_ratio_min:.2f}, "
                        f"{profile.bbox_height_ratio_max:.2f}]."
                    ),
                }
            )

    pose_mask = _mask_overlap_metrics(
        image.convert("RGB"),
        frame,
        clip,
        background_rgb,
        profile,
    )
    if (
        float(pose_mask["control_coverage"])
        < profile.min_control_mask_coverage
    ):
        issues.append(
            {
                "code": "pose_mask_miss",
                "message": (
                    "Rendered sprite does not cover enough of the expected "
                    f"control pose: {float(pose_mask['control_coverage']):.2f} "
                    f"< {profile.min_control_mask_coverage:.2f}."
                ),
            }
        )
    if (
        float(pose_mask["rendered_near_control"])
        < profile.min_rendered_near_control
    ):
        issues.append(
            {
                "code": "pose_mask_excess",
                "message": (
                    "Too much rendered foreground lies away from the expected "
                    f"control pose: {float(pose_mask['rendered_near_control']):.2f} "
                    f"< {profile.min_rendered_near_control:.2f}."
                ),
            }
        )

    anchor_checks: dict[str, bool] = {}

    important = {
        "head": frame.joints.get("head"),
        "elbow_l": frame.joints.get("elbow_l"),
        "elbow_r": frame.joints.get("elbow_r"),
        "knee_l": frame.joints.get("knee_l"),
        "knee_r": frame.joints.get("knee_r"),
        "primary_hand": frame.joints.get(clip.primary_hand),
        "secondary_hand": frame.joints.get(clip.secondary_hand),
        "sword_tip": frame.weapon.tip,
    }
    for name, point in important.items():
        if point is None:
            continue
        if name == "sword_tip":
            radius = profile.sword_tip_radius_px
        elif name.startswith(("elbow_", "knee_")):
            radius = profile.structural_anchor_radius_px
        else:
            radius = profile.anchor_radius_px
        found = _patch_has_foreground(
            image.convert("RGB"),
            point,
            radius,
            background_rgb,
            profile.background_distance_threshold,
        )
        anchor_checks[name] = found
        if not found:
            code = "sword_tip_drift" if name == "sword_tip" else "anchor_missing"
            issues.append(
                {
                    "code": code,
                    "anchor": name,
                    "message": f"No foreground detected near expected {name}.",
                }
            )

    sword_line_coverage = _segment_foreground_coverage(
        image.convert("RGB"),
        frame.weapon.grip_main,
        frame.weapon.tip,
        samples=max(
            8,
            round(
                frame.weapon.grip_main.distance_to(
                    frame.weapon.tip
                )
                / 3.0
            ),
        ),
        radius=profile.sword_line_radius_px,
        background=background_rgb,
        threshold=profile.background_distance_threshold,
    )
    if sword_line_coverage < profile.min_sword_line_coverage:
        issues.append(
            {
                "code": "sword_path_missing",
                "message": (
                    f"Only {sword_line_coverage:.2f} of the expected sword "
                    f"blade path has nearby rendered foreground; required "
                    f"{profile.min_sword_line_coverage:.2f}."
                ),
            }
        )

    contacts: dict[str, bool] = {}
    for foot_name in ("foot_l", "foot_r"):
        if (
            foot_name not in frame.joints
            or not is_ground_contact(frame.contacts.get(foot_name))
        ):
            continue
        found = _ground_contact_exists(
            image.convert("RGB"),
            frame.joints[foot_name],
            profile.foot_radius_px,
            clip.ground_y,
            profile.ground_tolerance_px,
            background_rgb,
            profile.background_distance_threshold,
        )
        contacts[foot_name] = found
        if not found:
            issues.append(
                {
                    "code": "ground_contact_missing",
                    "foot": foot_name,
                    "message": (
                        f"No foreground foot contact found near {foot_name} "
                        f"at ground y={clip.ground_y:.1f}."
                    ),
                }
            )

    return {
        "frame": frame.frame,
        "label": frame.label,
        "background_rgb": list(background_rgb),
        "bbox": bbox_metrics,
        "pose_mask": pose_mask,
        "anchors": anchor_checks,
        "sword_line_coverage": sword_line_coverage,
        "contacts": contacts,
        "issues": issues,
    }


def validate_rendered_sheet(
    path: str | Path,
    clip: MotionClip,
    columns: int = 4,
    profile: RenderValidationProfile | None = None,
) -> dict[str, Any]:
    profile = profile or RenderValidationProfile()
    image = Image.open(path).convert("RGB")

    rows = math.ceil(len(clip.frames) / columns)
    expected_size = (columns * clip.width, rows * clip.height)
    if image.size != expected_size:
        return {
            "ok": False,
            "profile": asdict(profile),
            "sheet": {
                "actual_size": list(image.size),
                "expected_size": list(expected_size),
                "columns": columns,
                "rows": rows,
            },
            "frames": [],
            "issues": [
                {
                    "code": "sheet_size_mismatch",
                    "message": (
                        f"Rendered sheet size {image.size} does not match "
                        f"expected {expected_size}."
                    ),
                }
            ],
        }

    frame_reports: list[dict[str, Any]] = []
    for index, frame in enumerate(clip.frames):
        col = index % columns
        row = index // columns
        crop = image.crop(
            (
                col * clip.width,
                row * clip.height,
                (col + 1) * clip.width,
                (row + 1) * clip.height,
            )
        )
        frame_reports.append(
            _frame_report(crop, frame, clip, profile)
        )

    if frame_reports:
        canonical_background = tuple(
            int(value)
            for value in frame_reports[0]["background_rgb"]
        )
        for frame_report in frame_reports[1:]:
            background = tuple(
                int(value)
                for value in frame_report["background_rgb"]
            )
            drift = _color_distance(
                background,
                canonical_background,
            )
            frame_report["background_drift"] = drift
            if drift > profile.max_background_drift:
                frame_report["issues"].append(
                    {
                        "code": "background_drift",
                        "message": (
                            f"Frame background differs from frame 0 by "
                            f"{drift:.1f} RGB-distance units; allowed "
                            f"{profile.max_background_drift:.1f}."
                        ),
                    }
                )
        frame_reports[0]["background_drift"] = 0.0

    all_issues = [
        {
            **issue,
            "frame": frame_report["frame"],
            "label": frame_report["label"],
        }
        for frame_report in frame_reports
        for issue in frame_report["issues"]
    ]

    return {
        "ok": not all_issues,
        "profile": asdict(profile),
        "sheet": {
            "actual_size": list(image.size),
            "expected_size": list(expected_size),
            "columns": columns,
            "rows": rows,
        },
        "frames": frame_reports,
        "issues": all_issues,
        "summary": {
            "issue_count": len(all_issues),
            "issue_codes": sorted(
                {issue["code"] for issue in all_issues}
            ),
        },
    }
