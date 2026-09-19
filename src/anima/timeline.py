from __future__ import annotations

from dataclasses import replace

from .model import FramePose, MotionClip, Vec2, WeaponPose


def _mix(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _mix_vec(a: Vec2, b: Vec2, t: float) -> Vec2:
    return Vec2(_mix(a.x, b.x, t), _mix(a.y, b.y, t))


def _ease(t: float, mode: str) -> float:
    if mode == "linear":
        return t
    if mode == "smoothstep":
        return t * t * (3.0 - 2.0 * t)
    raise ValueError(f"Unsupported easing: {mode}")


def _interpolate_pose(
    a: FramePose,
    b: FramePose,
    frame_no: int,
    t: float,
) -> FramePose:
    names = a.joints.keys() & b.joints.keys()
    joints = {
        name: _mix_vec(a.joints[name], b.joints[name], t)
        for name in names
    }

    # Contacts are semantic/discrete. Pick the nearest keyframe.
    contact_source = a if t < 0.5 else b

    return FramePose(
        frame=frame_no,
        root=_mix_vec(a.root, b.root, t),
        joints=joints,
        weapon=WeaponPose(
            grip_main=_mix_vec(a.weapon.grip_main, b.weapon.grip_main, t),
            grip_off=_mix_vec(a.weapon.grip_off, b.weapon.grip_off, t),
            tip=_mix_vec(a.weapon.tip, b.weapon.tip, t),
        ),
        contacts=dict(contact_source.contacts),
        label=None,
    )


def densify_clip(
    clip: MotionClip,
    easing: str = "smoothstep",
) -> MotionClip:
    """Fill missing integer frames between sparse keyframes."""
    if len(clip.frames) < 2:
        return clip

    keyframes = sorted(clip.frames, key=lambda frame: frame.frame)
    if len({frame.frame for frame in keyframes}) != len(keyframes):
        raise ValueError("Duplicate keyframe numbers are not allowed")

    output: list[FramePose] = []
    for index, left in enumerate(keyframes[:-1]):
        right = keyframes[index + 1]
        if right.frame <= left.frame:
            raise ValueError("Keyframes must be strictly increasing")

        if not output:
            output.append(left)

        gap = right.frame - left.frame
        for frame_no in range(left.frame + 1, right.frame):
            raw_t = (frame_no - left.frame) / gap
            output.append(
                _interpolate_pose(
                    left,
                    right,
                    frame_no,
                    _ease(raw_t, easing),
                )
            )
        output.append(right)

    return replace(clip, frames=output)
