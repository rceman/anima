from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw

from .model import FramePose, MotionClip, Vec2

# Colorblind-safe debug palette. Marker shapes are also different so color is
# never the only left/right cue.
DEBUG = {
    "background": "#1E252A",
    "ground": "#AEB8C2",
    "torso": "#F0E6C5",
    "left": "#56B4E9",
    "right": "#E69F00",
    "weapon": "#CC79A7",
    "joint": "#FFFFFF",
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


def render_debug_frame(
    clip: MotionClip,
    frame: FramePose,
    scale: int = 4,
) -> Image.Image:
    image = Image.new("RGB", (clip.width, clip.height), DEBUG["background"])
    draw = ImageDraw.Draw(image)
    draw.line(
        [(0, round(clip.ground_y)), (clip.width - 1, round(clip.ground_y))],
        fill=DEBUG["ground"],
        width=1,
    )

    if "chest" in frame.joints:
        _line(draw, frame.root, frame.joints["chest"], DEBUG["torso"], 2)
    if {"shoulder_l", "shoulder_r"}.issubset(frame.joints):
        _line(
            draw,
            frame.joints["shoulder_l"],
            frame.joints["shoulder_r"],
            DEBUG["torso"],
            2,
        )
    if {"hip_l", "hip_r"}.issubset(frame.joints):
        _line(
            draw,
            frame.joints["hip_l"],
            frame.joints["hip_r"],
            DEBUG["torso"],
            2,
        )

    chains = (
        (("shoulder_l", "elbow_l", "hand_l"), DEBUG["left"]),
        (("hip_l", "knee_l", "foot_l"), DEBUG["left"]),
        (("shoulder_r", "elbow_r", "hand_r"), DEBUG["right"]),
        (("hip_r", "knee_r", "foot_r"), DEBUG["right"]),
    )
    for names, color in chains:
        if all(name in frame.joints for name in names):
            _line(draw, frame.joints[names[0]], frame.joints[names[1]], color, 2)
            _line(draw, frame.joints[names[1]], frame.joints[names[2]], color, 2)

    _line(draw, frame.weapon.grip_off, frame.weapon.tip, DEBUG["weapon"], 2)

    for name, point in frame.joints.items():
        x, y = _xy(point)
        radius = 2 if name.startswith(("hand", "foot")) else 1
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=DEBUG["joint"],
        )

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

    debug: list[Image.Image] = []
    control: list[Image.Image] = []
    for index, frame in enumerate(clip.frames):
        debug_frame = render_debug_frame(clip, frame, scale=1)
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
    debug_preview = [_scaled(frame, scale) for frame in debug]
    control_preview = [_scaled(frame, scale) for frame in control]

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

    _scaled(debug_sheet, scale).save(output / "debug_sheet.preview.png")
    _scaled(control_sheet, scale).save(output / "control_sheet.preview.png")
