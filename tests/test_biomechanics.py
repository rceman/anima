from anima.biomechanics import analyze_body_kinematics
from anima.model import FramePose, MotionClip, Vec2, WeaponPose


def make_frame(frame_no: int, root_x: float, foot_x: float) -> FramePose:
    joints = {
        "head": Vec2(root_x, 40),
        "chest": Vec2(root_x, 58),
        "shoulder_l": Vec2(root_x - 5, 58),
        "shoulder_r": Vec2(root_x + 5, 58),
        "elbow_l": Vec2(root_x - 2, 66),
        "elbow_r": Vec2(root_x + 7, 66),
        "hand_l": Vec2(root_x + 5, 68),
        "hand_r": Vec2(root_x + 10, 64),
        "hip_l": Vec2(root_x - 4, 78),
        "hip_r": Vec2(root_x + 4, 78),
        "knee_l": Vec2(root_x - 8, 92),
        "knee_r": Vec2(root_x + 8, 92),
        "foot_l": Vec2(foot_x, 108),
        "foot_r": Vec2(foot_x + 30, 108),
    }
    return FramePose(
        frame=frame_no,
        root=Vec2(root_x, 78),
        joints=joints,
        weapon=WeaponPose(
            grip_main=Vec2(root_x + 10, 64),
            grip_off=Vec2(root_x + 5, 68),
            tip=Vec2(root_x + 30, 45),
        ),
        contacts={"foot_l": True, "foot_r": True},
    )


def test_body_report_contains_velocity_acceleration_and_force():
    clip = MotionClip(
        128,
        128,
        108,
        10,
        "test",
        [
            make_frame(0, 60, 45),
            make_frame(1, 62, 45),
            make_frame(2, 65, 45),
        ],
        dynamics={
            "body": {
                "mass_kg": 80,
                "pixels_per_meter": 40,
            }
        },
    )

    report = analyze_body_kinematics(clip)

    assert len(report["frames"]) == 3
    assert report["frames"][1]["com_speed_m_s"] > 0
    assert report["frames"][2]["estimated_net_force_n"] > 0
    assert "torso_angular_velocity_deg_s" in report["frames"][2]


def test_continuously_planted_foot_slip_is_reported():
    clip = MotionClip(
        128,
        128,
        108,
        10,
        "test",
        [
            make_frame(0, 60, 45),
            make_frame(1, 60, 47),
        ],
        dynamics={
            "body": {
                "max_planted_slip_px": 0.25,
            }
        },
    )

    report = analyze_body_kinematics(clip)
    codes = {warning["code"] for warning in report["warnings"]}

    assert "planted_foot_slip" in codes
