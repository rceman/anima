from anima.anatomy import analyze_joint_limits
from anima.model import FramePose, MotionClip, Vec2, WeaponPose


def make_clip(hand: Vec2) -> MotionClip:
    joints = {
        "head": Vec2(0, 0),
        "chest": Vec2(0, 10),
        "shoulder_l": Vec2(0, 20),
        "shoulder_r": Vec2(10, 20),
        "elbow_l": Vec2(5, 20),
        "elbow_r": Vec2(15, 20),
        "hand_l": hand,
        "hand_r": Vec2(20, 20),
        "hip_l": Vec2(0, 40),
        "hip_r": Vec2(10, 40),
        "knee_l": Vec2(0, 50),
        "knee_r": Vec2(10, 50),
        "foot_l": Vec2(0, 60),
        "foot_r": Vec2(10, 60),
    }
    frame = FramePose(
        0,
        Vec2(5, 40),
        joints,
        WeaponPose(Vec2(20, 20), Vec2(15, 20), Vec2(30, 20)),
    )
    return MotionClip(
        128,
        128,
        60,
        12,
        "test",
        [frame],
        dynamics={
            "body": {
                "joint_limits_deg": {
                    "elbow_l": [20, 170],
                }
            }
        },
    )


def test_joint_limit_violation_is_reported():
    # shoulder -> elbow -> hand points in the same direction: internal angle 0 deg
    report = analyze_joint_limits(make_clip(Vec2(10, 20)))
    codes = {warning["code"] for warning in report["warnings"]}
    assert "joint_limit_violation" in codes


def test_joint_limit_valid_pose_has_no_warning():
    # shoulder -> elbow -> hand forms a right angle
    report = analyze_joint_limits(make_clip(Vec2(5, 25)))
    elbow_l_warnings = [
        warning
        for warning in report["warnings"]
        if warning.get("joint") == "elbow_l"
    ]
    assert not elbow_l_warnings
