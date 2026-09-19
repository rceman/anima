from __future__ import annotations

from pathlib import Path
import math
from PIL import Image, ImageDraw

from .biomechanics import analyze_body_kinematics
from .dynamics import analyze_weapon_dynamics
from .model import FramePose, MotionClip, Vec2

# Colorblind-safe debug palette. Marker shapes are also different so color is
# never the only left/right cue.
DEBUG = {
    "background": "#1E252A",
    "ground": "#AEB8C2",
    "torso": "#F0E6C5",
    "left": "#56B4E9",
    "right": "#E69F00",
    "weapon": "#9B8CFF",
    "joint": "#FFFFFF",
    "velocity": "#00E5FF",
    "acceleration": "#FFD166",
    "support": "#8A98A6",
}
CONTROL = {
    "background": "#1E252A",
    "body": "#DAD3B5",
    "ground": "#626E78",
}


def _xy(p: Vec2) -> tuple[int, int]:
    return round(p.x), round(p.y)


def _line(
    draw: ImageDraw.ImageDraw,
    a: Vec2,
    b: Vec2,
    fill: str,
    width: int,
) -> None:
    draw.line([_xy(a), _xy(b)], fill=fill, width=width)


def _arrow(
    draw: ImageDraw.ImageDraw,
    start: Vec2,
    delta: Vec2,
    fill: str,
    width: int = 1,
) -> None:
    end = start + delta
    draw.line([_xy(start), _xy(end)], fill=fill, width=width)
    length = delta.length()
    if length <= 1e-6:
        return
    direction = delta * (1.0 / length)
    normal = Vec2(-direction.y, direction.x)
    back = end - direction * 4.0
    left = back + normal * 2.0
    right = back - normal * 2.0
    draw.polygon([_xy(end), _xy(left), _xy(right)], fill=fill)


def render_debug_frame(
    clip: MotionClip,
    frame: FramePose,
    scale: int = 4,
    body_diag: dict | None = None,
    weapon_diag: dict | None = None,
) -> Image.Image:
    image = Image.new("RGB", (clip.width, clip.height), DEBUG["background"])
    draw = ImageDraw.Draw(image)
    draw.line(
        [(0, round(clip.ground_y)), (clip.width - 1, round(clip.ground_y))],
        fill=DEBUG["ground"],
        width=2,
    )

    caption = f"F{frame.frame:02d}"
    if frame.label:
        caption += f" {frame.label}"
    draw.text((3, 3), caption, fill=DEBUG["joint"])

    if "chest" in frame.joints:
        _line(draw, frame.root, frame.joints["chest"], DEBUG["torso"], 3)
    if {"chest", "head"}.issubset(frame.joints):
        _line(draw, frame.joints["chest"], frame.joints["head"], DEBUG["torso"], 2)
    if {"shoulder_l", "shoulder_r"}.issubset(frame.joints):
        _line(
            draw,
            frame.joints["shoulder_l"],
            frame.joints["shoulder_r"],
            DEBUG["torso"],
            3,
        )
    if {"hip_l", "hip_r"}.issubset(frame.joints):
        _line(
            draw,
            frame.joints["hip_l"],
            frame.joints["hip_r"],
            DEBUG["torso"],
            3,
        )

    chains = (
        (("shoulder_l", "elbow_l", "hand_l"), DEBUG["left"]),
        (("hip_l", "knee_l", "foot_l"), DEBUG["left"]),
        (("shoulder_r", "elbow_r", "hand_r"), DEBUG["right"]),
        (("hip_r", "knee_r", "foot_r"), DEBUG["right"]),
    )
    for names, color in chains:
        if all(name in frame.joints for name in names):
            _line(draw, frame.joints[names[0]], frame.joints[names[1]], color, 3)
            _line(draw, frame.joints[names[1]], frame.joints[names[2]], color, 3)

    _line(draw, frame.weapon.grip_off, frame.weapon.tip, DEBUG["weapon"], 3)

    for name, point in frame.joints.items():
        x, y = _xy(point)
        radius = 3 if name.startswith(("hand", "foot")) else 2
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=DEBUG["joint"],
        )

    # Physics overlay. Color is redundant with labels/shapes so the diagnostics
    # remain readable with red/green color-vision deficiencies.
    if body_diag:
        com_data = body_diag.get("com_px")
        if com_data:
            com = Vec2(float(com_data[0]), float(com_data[1]))
            cx, cy = _xy(com)
            draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], outline=DEBUG["joint"], width=1)
            draw.line([(cx - 5, cy), (cx + 5, cy)], fill=DEBUG["joint"], width=1)
            draw.line([(cx, cy - 5), (cx, cy + 5)], fill=DEBUG["joint"], width=1)
            draw.text((cx + 5, cy - 8), "COM", fill=DEBUG["joint"])

            velocity = body_diag.get("com_velocity_m_s", [0.0, 0.0])
            acceleration = body_diag.get("com_acceleration_m_s2", [0.0, 0.0])
            _arrow(
                draw,
                com,
                Vec2(float(velocity[0]) * 5.0, float(velocity[1]) * 5.0),
                DEBUG["velocity"],
                2,
            )
            _arrow(
                draw,
                com,
                Vec2(float(acceleration[0]) * 0.25, float(acceleration[1]) * 0.25),
                DEBUG["acceleration"],
                1,
            )

        support = body_diag.get("support")
        if support:
            y = round(clip.ground_y) + 2
            draw.line(
                [(round(support["min_x"]), y), (round(support["max_x"]), y)],
                fill=DEBUG["support"],
                width=2,
            )

        speed = float(body_diag.get("com_speed_m_s", 0.0))
        accel = body_diag.get("com_acceleration_m_s2", [0.0, 0.0])
        accel_mag = math.hypot(float(accel[0]), float(accel[1]))
        draw.text((3, 13), f"V {speed:.2f}m/s  A {accel_mag:.1f}m/s2", fill=DEBUG["velocity"])

    if weapon_diag:
        omega = float(weapon_diag.get("angular_velocity_deg_s", 0.0))
        torque = float(weapon_diag.get("estimated_torque_nm", 0.0))
        draw.text((3, 23), f"SWORD w {omega:.0f}deg/s  T {torque:.1f}Nm", fill=DEBUG["weapon"])

    # Root/pelvis anchor: diamond. This makes root drift immediately visible.
    rx, ry = _xy(frame.root)
    draw.polygon(
        [(rx, ry - 4), (rx + 4, ry), (rx, ry + 4), (rx - 4, ry)],
        outline=DEBUG["torso"],
    )

    # Head outline makes torso/head motion easier to inspect.
    if "head" in frame.joints:
        hx, hy = _xy(frame.joints["head"])
        draw.ellipse([hx - 5, hy - 6, hx + 5, hy + 5], outline=DEBUG["torso"], width=2)

    # Redundant marker shapes: left=box, right=circle.
    for name in ("hand_l", "foot_l"):
        if name in frame.joints:
            x, y = _xy(frame.joints[name])
            draw.rectangle([x - 2, y - 2, x + 2, y + 2], outline=DEBUG["left"])
    for name in ("hand_r", "foot_r"):
        if name in frame.joints:
            x, y = _xy(frame.joints[name])
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], outline=DEBUG["right"])

    if scale != 1:
        image = image.resize(
            (clip.width * scale, clip.height * scale),
            Image.Resampling.NEAREST,
        )
    return image


def render_control_frame(
    clip: MotionClip,
    frame: FramePose,
    scale: int = 4,
) -> Image.Image:
    """Render a simple mannequin control image for downstream image generation."""
    image = Image.new("RGB", (clip.width, clip.height), CONTROL["background"])
    draw = ImageDraw.Draw(image)
    draw.line(
        [(0, round(clip.ground_y)), (clip.width - 1, round(clip.ground_y))],
        fill=CONTROL["ground"],
        width=1,
    )
    body = CONTROL["body"]

    chains = (
        ("shoulder_l", "elbow_l", "hand_l"),
        ("shoulder_r", "elbow_r", "hand_r"),
        ("hip_l", "knee_l", "foot_l"),
        ("hip_r", "knee_r", "foot_r"),
    )
    for names in chains:
        if all(name in frame.joints for name in names):
            for a, b in zip(names, names[1:]):
                _line(draw, frame.joints[a], frame.joints[b], body, 5)
            for name in names[1:]:
                x, y = _xy(frame.joints[name])
                draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=body)

    if all(
        name in frame.joints
        for name in ("shoulder_l", "shoulder_r", "hip_r", "hip_l")
    ):
        draw.polygon(
            [
                _xy(frame.joints["shoulder_l"]),
                _xy(frame.joints["shoulder_r"]),
                _xy(frame.joints["hip_r"]),
                _xy(frame.joints["hip_l"]),
            ],
            fill=body,
        )

    head = frame.joints.get("head")
    if head:
        x, y = _xy(head)
        draw.ellipse([x - 5, y - 6, x + 5, y + 5], fill=body)

    _line(draw, frame.weapon.grip_off, frame.weapon.tip, body, 3)
    gx, gy = _xy(frame.weapon.grip_main)
    draw.line([(gx - 4, gy - 1), (gx + 4, gy + 1)], fill=body, width=2)

    if scale != 1:
        image = image.resize(
            (clip.width * scale, clip.height * scale),
            Image.Resampling.NEAREST,
        )
    return image


def make_sheet(
    frames: list[Image.Image],
    columns: int = 4,
    gap: int = 0,
) -> Image.Image:
    if not frames:
        raise ValueError("No frames")

    width, height = frames[0].size
    rows = (len(frames) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (
            columns * width + (columns - 1) * gap,
            rows * height + (rows - 1) * gap,
        ),
        frames[0].getpixel((0, 0)),
    )
    for index, frame in enumerate(frames):
        x = (index % columns) * (width + gap)
        y = (index // columns) * (height + gap)
        sheet.paste(frame, (x, y))
    return sheet


def _scaled(image: Image.Image, scale: int) -> Image.Image:
    if scale == 1:
        return image
    return image.resize(
        (image.width * scale, image.height * scale),
        Image.Resampling.NEAREST,
    )


def export_render_set(
    clip: MotionClip,
    output: str | Path,
    scale: int = 4,
    columns: int = 4,
) -> None:
    """Export canonical 1x assets plus nearest-neighbor previews.

    Canonical frames always retain the clip canvas size (for example 128x128).
    The scale argument affects preview assets only.
    """
    output = Path(output)
    debug_dir = output / "debug_frames"
    control_dir = output / "control_frames"
    debug_dir.mkdir(parents=True, exist_ok=True)
    control_dir.mkdir(parents=True, exist_ok=True)

    body_report = analyze_body_kinematics(clip)
    weapon_report = analyze_weapon_dynamics(clip)
    body_by_frame = {item["frame"]: item for item in body_report["frames"]}
    weapon_by_frame = {item["frame"]: item for item in weapon_report["frames"]}

    debug: list[Image.Image] = []
    control: list[Image.Image] = []
    for index, frame in enumerate(clip.frames):
        debug_frame = render_debug_frame(
            clip,
            frame,
            scale=1,
            body_diag=body_by_frame.get(frame.frame),
            weapon_diag=weapon_by_frame.get(frame.frame),
        )
        control_frame = render_control_frame(clip, frame, scale=1)
        debug_frame.save(debug_dir / f"frame_{index:02d}.png")
        control_frame.save(control_dir / f"frame_{index:02d}.png")
        debug.append(debug_frame)
        control.append(control_frame)

    debug_sheet = make_sheet(debug, columns=columns)
    control_sheet = make_sheet(control, columns=columns)
    debug_sheet.save(output / "debug_sheet.png")
    control_sheet.save(output / "control_sheet.png")

    duration = max(1, round(1000.0 / clip.fps))

    # Debug is intentionally larger than the control preview: it is a
    # diagnostic instrument, not an art asset. 128px source frames become
    # at least 1024px so elbow flips, foot drift, and root motion are obvious.
    debug_scale = max(8, scale * 2)
    control_scale = max(4, scale)
    debug_preview = [_scaled(frame, debug_scale) for frame in debug]
    control_preview = [_scaled(frame, control_scale) for frame in control]

    debug_preview[0].save(
        output / "debug_preview.gif",
        save_all=True,
        append_images=debug_preview[1:],
        duration=duration,
        loop=0,
        disposal=2,
    )
    control_preview[0].save(
        output / "control_preview.gif",
        save_all=True,
        append_images=control_preview[1:],
        duration=duration,
        loop=0,
        disposal=2,
    )

    _scaled(debug_sheet, debug_scale).save(output / "debug_sheet.preview.png")
    _scaled(control_sheet, control_scale).save(output / "control_sheet.preview.png")

    # Side-by-side review GIF at the larger debug scale. Useful for checking
    # that the mannequin still matches the solved skeleton.
    control_review = [_scaled(frame, debug_scale) for frame in control]
    review_frames: list[Image.Image] = []
    for debug_frame, control_frame in zip(debug_preview, control_review):
        review = Image.new(
            "RGB",
            (debug_frame.width + control_frame.width, debug_frame.height),
            DEBUG["background"],
        )
        review.paste(debug_frame, (0, 0))
        review.paste(control_frame, (debug_frame.width, 0))
        review_frames.append(review)

    review_frames[0].save(
        output / "review_preview.gif",
        save_all=True,
        append_images=review_frames[1:],
        duration=duration,
        loop=0,
        disposal=2,
    )
