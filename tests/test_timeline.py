from anima.model import FramePose, MotionClip, Vec2, WeaponPose
from anima.timeline import densify_clip


def pose(frame: int, x: float) -> FramePose:
    joints = {
        "head": Vec2(x, 10),
        "shoulder_l": Vec2(x, 20),
        "shoulder_r": Vec2(x + 4, 20),
        "elbow_l": Vec2(x, 25),
        "elbow_r": Vec2(x + 4, 25),
        "hand_l": Vec2(x, 30),
        "hand_r": Vec2(x + 4, 30),
        "hip_l": Vec2(x, 40),
        "hip_r": Vec2(x + 4, 40),
        "knee_l": Vec2(x, 50),
        "knee_r": Vec2(x + 4, 50),
        "foot_l": Vec2(x, 60),
        "foot_r": Vec2(x + 4, 60),
    }
    return FramePose(
        frame,
        Vec2(x, 40),
        joints,
        WeaponPose(
            Vec2(x + 4, 30),
            Vec2(x, 30),
            Vec2(x + 20, 20),
        ),
        {"foot_l": True, "foot_r": True},
    )


def test_sparse_keyframes_are_densified():
    clip = MotionClip(
        128,
        128,
        60,
        12,
        "test",
        [pose(0, 10), pose(3, 40)],
    )
    dense = densify_clip(clip, easing="linear")

    assert [frame.frame for frame in dense.frames] == [0, 1, 2, 3]
    assert dense.frames[1].root.x == 20
    assert dense.frames[2].root.x == 30


def test_weapon_interpolation_follows_arc_not_tip_chord():
    a = pose(0, 10)
    b = pose(2, 10)
    a.weapon = WeaponPose(
        Vec2(20, 20),
        Vec2(18, 20),
        Vec2(30, 20),
    )
    b.weapon = WeaponPose(
        Vec2(20, 20),
        Vec2(20, 18),
        Vec2(20, 30),
    )
    clip = MotionClip(128, 128, 60, 12, "test", [a, b])

    dense = densify_clip(clip, easing="linear")
    mid = dense.frames[1]

    blade = mid.weapon.tip - mid.weapon.grip_main
    assert abs(blade.length() - 10.0) < 1e-6
    assert abs(blade.x - blade.y) < 1e-6
    assert blade.x > 7.0
