from __future__ import annotations

import math
from typing import Any

from .model import FramePose, MotionClip, Vec2
from .rig import HUMANOID_BONES


def _angle(a: Vec2, b: Vec2) -> float:
    delta = b - a
    return math.atan2(delta.y, delta.x)


def _affine_from_segment(
    rest_start: Vec2,
    rest_end: Vec2,
    current_start: Vec2,
    current_end: Vec2,
) -> dict[str, Any]:
    """Return the affine transform mapping one rest segment to current pose.

    Matrix convention:

        x' = a*x + b*y + tx
        y' = c*x + d*y + ty

    Coordinates are screen-space pixels. The transform is suitable for rigid
    cutout pieces, debug geometry, external renderers, or conversion to another
    animation system.
    """
    rest_vector = rest_end - rest_start
    current_vector = current_end - current_start
    rest_length = rest_vector.length()
    current_length = current_vector.length()

    if rest_length <= 1e-9:
        scale = 1.0
        rest_angle = 0.0
    else:
        scale = current_length / rest_length
        rest_angle = _angle(rest_start, rest_end)

    current_angle = (
        _angle(current_start, current_end)
        if current_length > 1e-9
        else rest_angle
    )
    rotation = current_angle - rest_angle
    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)

    a = scale * cos_r
    b = -scale * sin_r
    c = scale * sin_r
    d = scale * cos_r
    tx = current_start.x - a * rest_start.x - b * rest_start.y
    ty = current_start.y - c * rest_start.x - d * rest_start.y

    return {
        "rest_start": rest_start.as_list(),
        "rest_end": rest_end.as_list(),
        "start": current_start.as_list(),
        "end": current_end.as_list(),
        "rest_length_px": rest_length,
        "length_px": current_length,
        "scale": scale,
        "rotation_deg": math.degrees(rotation),
        "matrix": [a, b, c, d, tx, ty],
    }


def _joint_segment(
    rest: FramePose,
    frame: FramePose,
    parent: str,
    child: str,
) -> dict[str, Any] | None:
    if not {
        parent,
        child,
    }.issubset(rest.joints) or not {
        parent,
        child,
    }.issubset(frame.joints):
        return None
    return _affine_from_segment(
        rest.joints[parent],
        rest.joints[child],
        frame.joints[parent],
        frame.joints[child],
    )


def _weapon_transform(
    rest: FramePose,
    frame: FramePose,
) -> dict[str, Any]:
    transform = _affine_from_segment(
        rest.weapon.grip_main,
        rest.weapon.tip,
        frame.weapon.grip_main,
        frame.weapon.tip,
    )
    transform["grip_main"] = frame.weapon.grip_main.as_list()
    transform["grip_off"] = frame.weapon.grip_off.as_list()
    transform["tip"] = frame.weapon.tip.as_list()
    return transform


def build_transform_manifest(
    clip: MotionClip,
) -> dict[str, Any]:
    """Export a deterministic cutout/bone transform representation.

    The first normalized pose is the bind/rest pose. Every later frame contains
    an affine transform for each rigid bone segment plus the rigid sword. This
    gives Anima both workflows discussed during design:

    1. a piece/cutout renderer can draw a body part once and transform it;
    2. a frame renderer/ImageGen can instead redraw each pose independently,
       while using the exact same solved skeleton as geometry authority.
    """
    if not clip.frames:
        return {
            "version": 1,
            "rig": clip.rig,
            "bind_frame": None,
            "frames": [],
        }

    rest = clip.frames[0]
    times = clip.times_s()
    frames: list[dict[str, Any]] = []

    for index, frame in enumerate(clip.frames):
        bones: dict[str, Any] = {}
        for bone in HUMANOID_BONES:
            transform = _joint_segment(
                rest,
                frame,
                bone.parent,
                bone.child,
            )
            if transform is not None:
                bones[bone.name] = transform

        torso = None
        if (
            "chest" in rest.joints
            and "chest" in frame.joints
        ):
            torso = _affine_from_segment(
                rest.root,
                rest.joints["chest"],
                frame.root,
                frame.joints["chest"],
            )

        head = None
        if (
            "head" in rest.joints
            and "head" in frame.joints
            and "chest" in rest.joints
            and "chest" in frame.joints
        ):
            head = _affine_from_segment(
                rest.joints["chest"],
                rest.joints["head"],
                frame.joints["chest"],
                frame.joints["head"],
            )

        frames.append(
            {
                "frame": frame.frame,
                "time_s": times[index],
                "label": frame.label,
                "root": frame.root.as_list(),
                "layer_order": list(frame.layer_order),
                "bones": bones,
                "torso": torso,
                "head": head,
                "weapon": _weapon_transform(rest, frame),
            }
        )

    return {
        "version": 1,
        "rig": clip.rig,
        "bind_frame": rest.frame,
        "coordinate_system": {
            "origin": "top_left",
            "x": "right",
            "y": "down",
            "matrix": "[a,b,c,d,tx,ty]",
        },
        "frames": frames,
    }
