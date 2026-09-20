from anima.model import FramePose, MotionClip, Vec2, WeaponPose
from anima.strength import analyze_arm_strength


def make_clip(enforce: bool) -> MotionClip:
    joints = {
        "shoulder_l": Vec2(50, 50),
        "elbow_l": Vec2(55, 55),
        "hand_l": Vec2(60, 60),
        "shoulder_r": Vec2(70, 50),
        "elbow_r": Vec2(75, 55),
        "hand_r": Vec2(80, 60),
    }
    frame = FramePose(
        0,
        Vec2(64, 80),
        joints,
        WeaponPose(Vec2(80, 60), Vec2(60, 60), Vec2(100, 40)),
    )
    return MotionClip(
        128,
        128,
        108,
        10,
        "test",
        [frame],
        dynamics={
            "body": {
                "pixels_per_meter": 40.0,
                "strength": {
                    "primary_force_fraction": 0.6,
                    "shoulder_limit_nm": 1.0,
                    "elbow_limit_nm": 1.0,
                    "enforce_limits": enforce,
                },
            }
        },
    )


def weapon_report() -> dict:
    return {
        "frames": [
            {
                "reaction_force_on_body_n": [200.0, -100.0],
                "estimated_torque_nm": 20.0,
            }
        ]
    }


def test_arm_strength_reports_two_hand_load_split():
    report = analyze_arm_strength(make_clip(False), weapon_report())
    sides = report["frames"][0]["sides"]

    assert set(sides) == {"l", "r"}
    assert sides["r"]["force_fraction"] == 0.6
    assert sides["l"]["force_fraction"] == 0.4
    assert sides["r"]["hand_force_magnitude_n"] > sides["l"]["hand_force_magnitude_n"]


def test_enforced_strength_limits_emit_joint_torque_warning():
    report = analyze_arm_strength(make_clip(True), weapon_report())
    codes = {warning["code"] for warning in report["warnings"]}

    assert "joint_torque_limit_exceeded" in codes
