from anima.constraints import normalize_clip, validate_clip
from anima.model import FramePose, MotionClip, Vec2, WeaponPose


def make_clip() -> MotionClip:
    joints = {
        "head": Vec2(64, 42),
        "chest": Vec2(64, 58),
        "shoulder_l": Vec2(59, 57),
        "shoulder_r": Vec2(69, 57),
        "elbow_l": Vec2(63, 66),
        "elbow_r": Vec2(70, 68),
        "hand_l": Vec2(73, 69),
        "hand_r": Vec2(78, 64),
        "hip_l": Vec2(60, 78),
        "hip_r": Vec2(68, 78),
        "knee_l": Vec2(54, 92),
        "knee_r": Vec2(73, 92),
        "foot_l": Vec2(49, 108),
        "foot_r": Vec2(79, 108),
    }
    weapon = WeaponPose(Vec2(78, 64), Vec2(73, 69), Vec2(99, 42))
    frame0 = FramePose(
        0,
        Vec2(64, 78),
        joints,
        weapon,
        {"foot_l": True, "foot_r": True},
    )

    # Deliberately shift the pose up and distort the sword. The compiler
    # must restore ground contact and rigid weapon geometry.
    shifted = frame0.shifted(Vec2(0, -6))
    shifted.frame = 1
    shifted.weapon = WeaponPose(
        shifted.weapon.grip_main,
        shifted.weapon.grip_off,
        Vec2(108, 30),
    )
    return MotionClip(
        128,
        128,
        108,
        12,
        "humanoid_twohand_sword_v1",
        [frame0, shifted],
    )


def test_normalization_locks_ground_grips_and_sword():
    clip = normalize_clip(make_clip())

    assert clip.frames[1].joints["foot_l"].y == 108
    assert clip.frames[1].joints["foot_r"].y == 108
    assert (
        clip.frames[1]
        .joints["hand_r"]
        .distance_to(clip.frames[1].weapon.grip_main)
        < 1e-6
    )
    assert (
        clip.frames[1]
        .joints["hand_l"]
        .distance_to(clip.frames[1].weapon.grip_off)
        < 1e-6
    )
    assert abs(
        clip.frames[1].weapon.grip_main.distance_to(clip.frames[1].weapon.tip)
        - clip.frames[0].weapon.grip_main.distance_to(clip.frames[0].weapon.tip)
    ) < 1e-6


def test_normalized_clip_validates():
    report = validate_clip(normalize_clip(make_clip()))
    assert report.ok, report.issues


def test_grounded_unreachable_foot_is_clamped_along_ground():
    clip = make_clip()
    clip.frames[1].joints["foot_r"] = Vec2(120, 102)

    normalized = normalize_clip(clip)

    assert normalized.frames[1].joints["foot_r"].y == 108
    assert normalized.frames[1].joints["foot_r"].x < 120
