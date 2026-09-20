import pytest

from anima.model import FramePose, MotionClip, Vec2, WeaponPose
from anima.transforms import build_transform_manifest


def frame(
    number: int,
    shoulder: Vec2,
    elbow: Vec2,
    hand: Vec2,
) -> FramePose:
    joints = {
        "shoulder_l": Vec2(40, 40),
        "elbow_l": Vec2(45, 50),
        "hand_l": Vec2(50, 60),
        "shoulder_r": shoulder,
        "elbow_r": elbow,
        "hand_r": hand,
        "hip_l": Vec2(45, 70),
        "knee_l": Vec2(43, 85),
        "foot_l": Vec2(40, 100),
        "hip_r": Vec2(55, 70),
        "knee_r": Vec2(57, 85),
        "foot_r": Vec2(60, 100),
        "chest": Vec2(50, 50),
        "head": Vec2(50, 30),
    }
    return FramePose(
        frame=number,
        root=Vec2(50, 70),
        joints=joints,
        weapon=WeaponPose(
            grip_main=hand,
            grip_off=Vec2(hand.x - 5, hand.y),
            tip=Vec2(hand.x + 25, hand.y),
        ),
        time_s=number * 0.1,
    )


def test_bind_pose_has_identity_bone_transform():
    rest = frame(
        0,
        Vec2(60, 40),
        Vec2(65, 50),
        Vec2(70, 60),
    )
    clip = MotionClip(
        128,
        128,
        108,
        10,
        "test",
        [rest],
    )

    manifest = build_transform_manifest(clip)
    upper = manifest["frames"][0]["bones"]["upper_arm_r"]

    assert upper["rotation_deg"] == pytest.approx(0.0)
    assert upper["scale"] == pytest.approx(1.0)
    assert upper["matrix"][:4] == pytest.approx([1.0, 0.0, 0.0, 1.0])


def test_rotated_bone_exports_rotation_and_translation():
    rest = frame(
        0,
        Vec2(60, 40),
        Vec2(70, 40),
        Vec2(80, 40),
    )
    posed = frame(
        1,
        Vec2(62, 42),
        Vec2(62, 52),
        Vec2(62, 62),
    )
    clip = MotionClip(
        128,
        128,
        108,
        10,
        "test",
        [rest, posed],
    )

    manifest = build_transform_manifest(clip)
    upper = manifest["frames"][1]["bones"]["upper_arm_r"]

    assert upper["rotation_deg"] == pytest.approx(90.0)
    assert upper["scale"] == pytest.approx(1.0)
    assert upper["start"] == [62, 42]
