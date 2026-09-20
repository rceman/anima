from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .layers import resolve_layer_order
from .model import MotionClip, Vec2
from .transforms import build_transform_manifest


BACKGROUND = "#1E252A"
GROUND = "#626E78"
CUTOUT = {
    "head": "#F0E6C5",
    "torso": "#DAD3B5",
    "left_arm": "#56B4E9",
    "right_arm": "#E69F00",
    "left_leg": "#7AA6FF",
    "right_leg": "#F0C05A",
    "weapon": "#9B8CFF",
}

SEMANTIC_PIECES: dict[str, tuple[str, ...]] = {
    "left_leg": ("thigh_l", "shin_l"),
    "right_leg": ("thigh_r", "shin_r"),
    "torso": ("torso",),
    "left_arm": ("upper_arm_l", "forearm_l"),
    "right_arm": ("upper_arm_r", "forearm_r"),
    "head": ("head",),
    "weapon": ("weapon",),
}

PIECE_ORDER: tuple[str, ...] = tuple(
    piece
    for semantic in (
        "left_leg",
        "right_leg",
        "torso",
        "left_arm",
        "right_arm",
        "head",
        "weapon",
    )
    for piece in SEMANTIC_PIECES[semantic]
)


def _xy(point: Vec2) -> tuple[int, int]:
    return round(point.x), round(point.y)


def _blank(size: tuple[int, int]) -> Image.Image:
    return Image.new("RGBA", size, (0, 0, 0, 0))


def _segment_piece(
    size: tuple[int, int],
    start: Vec2,
    end: Vec2,
    width: int,
) -> Image.Image:
    image = _blank(size)
    draw = ImageDraw.Draw(image)
    fill = (255, 255, 255, 255)
    draw.line(
        [_xy(start), _xy(end)],
        fill=fill,
        width=width,
    )
    radius = max(1, width // 2)
    for point in (start, end):
        x, y = _xy(point)
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=fill,
        )
    return image


def build_bind_piece_masks(
    clip: MotionClip,
) -> dict[str, Image.Image]:
    """Draw each bind-pose cutout piece exactly once."""
    if not clip.frames:
        return {}

    rest = clip.frames[0]
    size = (clip.width, clip.height)
    masks: dict[str, Image.Image] = {}

    bone_specs = (
        ("upper_arm_l", "shoulder_l", "elbow_l", 5),
        ("forearm_l", "elbow_l", "hand_l", 5),
        ("upper_arm_r", "shoulder_r", "elbow_r", 5),
        ("forearm_r", "elbow_r", "hand_r", 5),
        ("thigh_l", "hip_l", "knee_l", 6),
        ("shin_l", "knee_l", "foot_l", 6),
        ("thigh_r", "hip_r", "knee_r", 6),
        ("shin_r", "knee_r", "foot_r", 6),
    )
    for piece, parent, child, width in bone_specs:
        if parent not in rest.joints or child not in rest.joints:
            continue
        masks[piece] = _segment_piece(
            size,
            rest.joints[parent],
            rest.joints[child],
            width,
        )

    torso = _blank(size)
    torso_draw = ImageDraw.Draw(torso)
    if all(
        name in rest.joints
        for name in (
            "shoulder_l",
            "shoulder_r",
            "hip_r",
            "hip_l",
        )
    ):
        torso_draw.polygon(
            [
                _xy(rest.joints["shoulder_l"]),
                _xy(rest.joints["shoulder_r"]),
                _xy(rest.joints["hip_r"]),
                _xy(rest.joints["hip_l"]),
            ],
            fill=(255, 255, 255, 255),
        )
    masks["torso"] = torso

    head = _blank(size)
    if "head" in rest.joints:
        draw = ImageDraw.Draw(head)
        x, y = _xy(rest.joints["head"])
        draw.ellipse(
            [x - 5, y - 6, x + 5, y + 5],
            fill=(255, 255, 255, 255),
        )
    masks["head"] = head

    weapon = _blank(size)
    draw = ImageDraw.Draw(weapon)
    draw.line(
        [
            _xy(rest.weapon.grip_off),
            _xy(rest.weapon.tip),
        ],
        fill=(255, 255, 255, 255),
        width=3,
    )
    gx, gy = _xy(rest.weapon.grip_main)
    draw.ellipse(
        [gx - 2, gy - 2, gx + 2, gy + 2],
        fill=(255, 255, 255, 255),
    )
    masks["weapon"] = weapon
    return masks


def build_piece_manifest(
    clip: MotionClip,
) -> dict[str, Any]:
    """Describe the bind-pose art pieces expected by the cutout renderer.

    Every piece is a full-canvas transparent PNG authored in bind-pose
    coordinates. This makes the contract deliberately simple: an artist or
    ImageGen can paint each piece once, and Anima applies the same deterministic
    affine transforms used by the debug rig.
    """
    if not clip.frames:
        return {
            "version": 1,
            "canvas": [clip.width, clip.height],
            "bind_frame": None,
            "pieces": {},
        }

    transforms = build_transform_manifest(clip)
    bind = transforms["frames"][0]
    semantic_by_piece = {
        piece: semantic
        for semantic, pieces in SEMANTIC_PIECES.items()
        for piece in pieces
    }

    pieces: dict[str, Any] = {}
    for piece in PIECE_ORDER:
        transform = _piece_transform(bind, piece)
        if transform is None:
            continue
        pieces[piece] = {
            "file": f"{piece}.png",
            "semantic": semantic_by_piece[piece],
            "canvas": [clip.width, clip.height],
            "rest_start": transform.get("rest_start"),
            "rest_end": transform.get("rest_end"),
            "pivot": transform.get("rest_start"),
            "transparent_background": True,
        }

    return {
        "version": 1,
        "rig": clip.rig,
        "canvas": [clip.width, clip.height],
        "bind_frame": clip.frames[0].frame,
        "coordinate_system": {
            "origin": "top_left",
            "x": "right",
            "y": "down",
        },
        "contract": (
            "Each PNG is a full-canvas RGBA image painted in bind-pose "
            "coordinates. Keep all pixels belonging to that named body piece "
            "on its own transparent layer."
        ),
        "piece_sheet_order": [
            name
            for name in PIECE_ORDER
            if name in pieces
        ],
        "pieces": pieces,
    }


def load_bind_piece_pack(
    clip: MotionClip,
    directory: str | Path,
) -> dict[str, Image.Image]:
    """Load a painted bind-pose piece pack with strict geometry checks."""
    root = Path(directory)
    manifest = build_piece_manifest(clip)
    expected = manifest["pieces"]
    loaded: dict[str, Image.Image] = {}
    missing: list[str] = []

    for piece, info in expected.items():
        path = root / str(info["file"])
        if not path.exists():
            missing.append(str(path))
            continue

        image = Image.open(path).convert("RGBA")
        if image.size != (clip.width, clip.height):
            raise ValueError(
                f"{piece} has size {image.size}; expected "
                f"{(clip.width, clip.height)}"
            )
        if image.getbbox() is None:
            raise ValueError(f"{piece} is completely transparent")
        loaded[piece] = image

    if missing:
        raise FileNotFoundError(
            "Bind piece pack is incomplete; missing: "
            + ", ".join(missing)
        )
    return loaded


def _inverse_affine(
    matrix: list[float],
) -> tuple[float, float, float, float, float, float]:
    a, b, c, d, tx, ty = (float(value) for value in matrix)
    determinant = a * d - b * c
    if abs(determinant) <= 1e-12:
        raise ValueError("Non-invertible cutout transform")

    ia = d / determinant
    ib = -b / determinant
    id_ = -c / determinant
    ie = a / determinant
    ic = -(ia * tx + ib * ty)
    iff = -(id_ * tx + ie * ty)
    return ia, ib, ic, id_, ie, iff


def _transform_piece(
    image: Image.Image,
    matrix: list[float],
) -> Image.Image:
    return image.transform(
        image.size,
        Image.Transform.AFFINE,
        _inverse_affine(matrix),
        resample=Image.Resampling.NEAREST,
        fillcolor=(0, 0, 0, 0),
    )


def _piece_transform(
    frame_transforms: dict[str, Any],
    piece: str,
) -> dict[str, Any] | None:
    if piece == "torso":
        return frame_transforms.get("torso")
    if piece == "head":
        return frame_transforms.get("head")
    if piece == "weapon":
        return frame_transforms.get("weapon")
    return frame_transforms.get("bones", {}).get(piece)


def render_cutout_frame(
    clip: MotionClip,
    frame_index: int,
    bind_masks: dict[str, Image.Image] | None = None,
    transform_manifest: dict[str, Any] | None = None,
    preserve_piece_colors: bool = False,
) -> Image.Image:
    """Render a pose by transforming bind-pose raster pieces, not redrawing them."""
    if not 0 <= frame_index < len(clip.frames):
        raise IndexError("frame_index outside motion clip")

    masks = bind_masks or build_bind_piece_masks(clip)
    transforms = transform_manifest or build_transform_manifest(clip)
    frame = clip.frames[frame_index]
    frame_transforms = transforms["frames"][frame_index]

    output = Image.new(
        "RGB",
        (clip.width, clip.height),
        BACKGROUND,
    )
    ground = ImageDraw.Draw(output)
    ground.line(
        [
            (0, round(clip.ground_y)),
            (clip.width - 1, round(clip.ground_y)),
        ],
        fill=GROUND,
        width=1,
    )

    for semantic in resolve_layer_order(frame.layer_order):
        color = CUTOUT[semantic]
        color_layer = Image.new(
            "RGB",
            output.size,
            color,
        )
        for piece in SEMANTIC_PIECES[semantic]:
            mask = masks.get(piece)
            transform = _piece_transform(
                frame_transforms,
                piece,
            )
            if mask is None or transform is None:
                continue
            transformed = _transform_piece(
                mask,
                transform["matrix"],
            )
            if preserve_piece_colors:
                output.paste(
                    transformed.convert("RGB"),
                    (0, 0),
                    transformed.getchannel("A"),
                )
            else:
                output.paste(
                    color_layer,
                    (0, 0),
                    transformed.getchannel("A"),
                )

    return output




def _piece_contact_sheet(
    pieces: dict[str, Image.Image],
    columns: int = 4,
    preserve_piece_colors: bool = False,
) -> Image.Image:
    ordered = [
        name
        for name in PIECE_ORDER
        if name in pieces
    ]
    if not ordered:
        raise ValueError("No bind pieces")

    width, height = next(iter(pieces.values())).size
    rows = math.ceil(len(ordered) / columns)
    sheet = Image.new(
        "RGB",
        (columns * width, rows * height),
        BACKGROUND,
    )
    semantic_by_piece = {
        piece: semantic
        for semantic, names in SEMANTIC_PIECES.items()
        for piece in names
    }

    for index, name in enumerate(ordered):
        piece = pieces[name].convert("RGBA")
        cell = Image.new("RGB", (width, height), BACKGROUND)
        if preserve_piece_colors:
            cell.paste(
                piece.convert("RGB"),
                (0, 0),
                piece.getchannel("A"),
            )
        else:
            color = CUTOUT[semantic_by_piece[name]]
            color_layer = Image.new("RGB", (width, height), color)
            cell.paste(
                color_layer,
                (0, 0),
                piece.getchannel("A"),
            )
        sheet.paste(
            cell,
            (
                (index % columns) * width,
                (index // columns) * height,
            ),
        )
    return sheet


def _review_frame(
    debug: Image.Image,
    control: Image.Image,
    cutout: Image.Image,
) -> Image.Image:
    debug = debug.convert("RGB")
    control = control.convert("RGB")
    cutout = cutout.convert("RGB")
    width = debug.width + control.width + cutout.width
    height = max(debug.height, control.height, cutout.height)
    review = Image.new("RGB", (width, height), BACKGROUND)
    x = 0
    for panel in (debug, control, cutout):
        review.paste(panel, (x, 0))
        x += panel.width
    return review


def _sheet(
    frames: list[Image.Image],
    columns: int,
) -> Image.Image:
    if not frames:
        raise ValueError("No cutout frames")
    width, height = frames[0].size
    rows = math.ceil(len(frames) / columns)
    sheet = Image.new(
        "RGB",
        (columns * width, rows * height),
        BACKGROUND,
    )
    for index, frame in enumerate(frames):
        sheet.paste(
            frame,
            (
                (index % columns) * width,
                (index // columns) * height,
            ),
        )
    return sheet


def _durations_ms(clip: MotionClip) -> list[int]:
    times = clip.times_s()
    if len(times) <= 1:
        return [max(1, round(1000.0 / clip.fps))]
    durations = [
        max(1, round((right - left) * 1000.0))
        for left, right in zip(times, times[1:])
    ]
    durations.append(durations[-1])
    return durations


def export_cutout_set(
    clip: MotionClip,
    output_dir: str | Path,
    columns: int = 4,
    preview_scale: int = 4,
    piece_dir: str | Path | None = None,
) -> None:
    output = Path(output_dir)
    frame_dir = output / "cutout_frames"
    bind_dir = output / "bind_pieces"
    frame_dir.mkdir(parents=True, exist_ok=True)
    bind_dir.mkdir(parents=True, exist_ok=True)

    use_painted_pieces = piece_dir is not None
    masks = (
        load_bind_piece_pack(clip, piece_dir)
        if piece_dir is not None
        else build_bind_piece_masks(clip)
    )
    transforms = build_transform_manifest(clip)
    for name, mask in masks.items():
        mask.save(bind_dir / f"{name}.png")

    _piece_contact_sheet(
        masks,
        columns=4,
        preserve_piece_colors=use_painted_pieces,
    ).save(output / "bind_piece_sheet.png")

    (output / "piece_manifest.json").write_text(
        json.dumps(
            {
                **build_piece_manifest(clip),
                "source": (
                    "painted_piece_pack"
                    if use_painted_pieces
                    else "procedural_masks"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    frames = [
        render_cutout_frame(
            clip,
            index,
            bind_masks=masks,
            transform_manifest=transforms,
            preserve_piece_colors=use_painted_pieces,
        )
        for index in range(len(clip.frames))
    ]
    for index, frame in enumerate(frames):
        frame.save(frame_dir / f"frame_{index:02d}.png")

    sheet = _sheet(frames, columns)
    sheet.save(output / "cutout_sheet.png")

    previews = [
        frame.resize(
            (
                frame.width * preview_scale,
                frame.height * preview_scale,
            ),
            Image.Resampling.NEAREST,
        )
        for frame in frames
    ]
    previews[0].save(
        output / "cutout_preview.gif",
        save_all=True,
        append_images=previews[1:],
        duration=_durations_ms(clip),
        loop=0,
        disposal=2,
    )


    debug_dir = output / "debug_frames"
    control_dir = output / "control_frames"
    if debug_dir.exists() and control_dir.exists():
        review_frames: list[Image.Image] = []
        for index, cutout in enumerate(frames):
            debug_path = debug_dir / f"frame_{index:02d}.png"
            control_path = control_dir / f"frame_{index:02d}.png"
            if not debug_path.exists() or not control_path.exists():
                review_frames = []
                break
            review = _review_frame(
                Image.open(debug_path),
                Image.open(control_path),
                cutout,
            )
            review_frames.append(
                review.resize(
                    (
                        review.width * preview_scale,
                        review.height * preview_scale,
                    ),
                    Image.Resampling.NEAREST,
                )
            )

        if review_frames:
            review_frames[0].save(
                output / "rig_review_preview.gif",
                save_all=True,
                append_images=review_frames[1:],
                duration=_durations_ms(clip),
                loop=0,
                disposal=2,
            )
