from anima.cutout import (
    build_bind_piece_masks,
    render_cutout_frame,
)
from anima.model import FramePose, MotionClip, Vec2, WeaponPose


def pose(frame_no: int, elbow_r: Vec2, hand_r: Vec2) -> FramePose:
    joints = {
        "head": Vec2(64, 35),
        "chest": Vec2(64, 52),
        "shoulder_l": Vec2(56, 56),
        "shoulder_r": Vec2(72, 56),
        "elbow_l": Vec2(54, 68),
        "elbow_r": elbow_r,
        "hand_l": Vec2(65, 70),
        "hand_r": hand_r,
        "hip_l": Vec2(60, 78),
        "hip_r": Vec2(68, 78),
        "knee_l": Vec2(56, 92),
        "knee_r": Vec2(72, 92),
        "foot_l": Vec2(52, 108),
        "foot_r": Vec2(76, 108),
    }
    return FramePose(
        frame=frame_no,
        root=Vec2(64, 78),
        joints=joints,
        weapon=WeaponPose(
            grip_main=hand_r,
            grip_off=Vec2(hand_r.x - 5, hand_r.y),
            tip=Vec2(hand_r.x + 25, hand_r.y),
        ),
        time_s=frame_no * 0.1,
    )


def test_cutout_reuses_bind_piece_masks_and_moves_rigid_segments():
    rest = pose(
        0,
        elbow_r=Vec2(78, 66),
        hand_r=Vec2(84, 76),
    )
    moved = pose(
        1,
        elbow_r=Vec2(82, 56),
        hand_r=Vec2(92, 56),
    )
    clip = MotionClip(
        128,
        128,
        108,
        10,
        "test",
        [rest, moved],
    )

    masks = build_bind_piece_masks(clip)
    first = render_cutout_frame(
        clip,
        0,
        bind_masks=masks,
    )
    second = render_cutout_frame(
        clip,
        1,
        bind_masks=masks,
    )

    assert "upper_arm_r" in masks
    assert masks["upper_arm_r"].mode == "RGBA"
    assert first.size == (128, 128)
    assert second.size == (128, 128)
    assert first.tobytes() != second.tobytes()


def test_bind_piece_masks_have_transparent_background():
    rest = pose(
        0,
        elbow_r=Vec2(78, 66),
        hand_r=Vec2(84, 76),
    )
    clip = MotionClip(
        128,
        128,
        108,
        10,
        "test",
        [rest],
    )

    masks = build_bind_piece_masks(clip)

    assert masks["weapon"].getpixel((0, 0))[3] == 0
    assert masks["torso"].getpixel((0, 0))[3] == 0
