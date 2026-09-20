from anima.model import FramePose, MotionClip, Vec2, WeaponPose
from anima.occlusion import analyze_occlusion


def crossing_frame(layer_order=None) -> FramePose:
    joints = {
        "head": Vec2(64, 35),
        "chest": Vec2(64, 50),
        "shoulder_l": Vec2(50, 55),
        "shoulder_r": Vec2(78, 55),
        "elbow_l": Vec2(62, 62),
        "elbow_r": Vec2(66, 62),
        "hand_l": Vec2(78, 68),
        "hand_r": Vec2(50, 68),
        "hip_l": Vec2(58, 78),
        "hip_r": Vec2(70, 78),
    }
    return FramePose(
        frame=0,
        root=Vec2(64, 78),
        joints=joints,
        weapon=WeaponPose(
            grip_main=Vec2(50, 68),
            grip_off=Vec2(56, 68),
            tip=Vec2(90, 50),
        ),
        layer_order=list(layer_order or []),
    )


def test_crossing_arms_without_explicit_order_are_reported():
    clip = MotionClip(
        128,
        128,
        108,
        12,
        "test",
        [crossing_frame()],
    )

    report = analyze_occlusion(clip)
    codes = {warning["code"] for warning in report["warnings"]}

    assert "ambiguous_arm_occlusion" in codes


def test_crossing_arms_with_explicit_order_are_unambiguous():
    clip = MotionClip(
        128,
        128,
        108,
        12,
        "test",
        [
            crossing_frame(
                ["left_arm", "torso", "right_arm"]
            )
        ],
    )

    report = analyze_occlusion(clip)
    arm_warnings = [
        warning
        for warning in report["warnings"]
        if warning["code"] == "ambiguous_arm_occlusion"
    ]

    assert not arm_warnings
    requirement = next(
        item
        for item in report["frames"][0]["requirements"]
        if item["reason"] == "arm_crossing"
    )
    assert requirement["behind"] == "left_arm"
    assert requirement["front"] == "right_arm"
