from anima.coordination import analyze_coordination
from anima.model import FramePose, MotionClip, Vec2, WeaponPose


def frame(number: int, hand_l: Vec2, hand_r: Vec2) -> FramePose:
    joints = {
        "head": Vec2(50, 20),
        "chest": Vec2(50, 35),
        "shoulder_l": Vec2(45, 40),
        "shoulder_r": Vec2(55, 40),
        "elbow_l": Vec2(40, 50),
        "elbow_r": Vec2(60, 50),
        "hand_l": hand_l,
        "hand_r": hand_r,
        "hip_l": Vec2(47, 65),
        "hip_r": Vec2(53, 65),
        "knee_l": Vec2(45, 80),
        "knee_r": Vec2(55, 80),
        "foot_l": Vec2(43, 95),
        "foot_r": Vec2(57, 95),
    }
    return FramePose(
        frame=number,
        root=Vec2(50, 65),
        joints=joints,
        weapon=WeaponPose(
            grip_main=hand_r,
            grip_off=hand_l,
            tip=Vec2(hand_r.x + 20, hand_r.y),
        ),
        time_s=number * 0.1,
    )


def test_near_locked_arm_is_reported():
    first = frame(0, Vec2(40, 55), Vec2(65, 55))
    second = frame(1, Vec2(24, 40), Vec2(76, 40))
    clip = MotionClip(
        128,
        128,
        100,
        10,
        "test",
        [first, second],
        dynamics={"body": {"pixels_per_meter": 40}},
    )
    body = {
        "profile": {"pixels_per_meter": 40},
        "frames": [
            {"root_velocity_m_s": [0, 0]},
            {"root_velocity_m_s": [0, 0]},
        ],
    }
    weapon = {
        "frames": [
            {"angular_velocity_deg_s": 0},
            {"angular_velocity_deg_s": 100},
        ]
    }

    report = analyze_coordination(clip, body, weapon)
    codes = {warning["code"] for warning in report["warnings"]}

    assert "arm_near_full_extension" in codes
    assert "sword_tip_speed" in report["peaks"]
