from anima.dynamics import analyze_weapon_dynamics
from anima.model import FramePose, MotionClip, Vec2, WeaponPose


def frame(number: int, tip: Vec2) -> FramePose:
    joints = {
        "hand_r": Vec2(64, 64),
        "hand_l": Vec2(60, 64),
    }
    return FramePose(
        frame=number,
        root=Vec2(64, 80),
        joints=joints,
        weapon=WeaponPose(
            grip_main=Vec2(64, 64),
            grip_off=Vec2(60, 64),
            tip=tip,
        ),
    )


def clip_for_mass(mass: float) -> MotionClip:
    return MotionClip(
        width=128,
        height=128,
        ground_y=108,
        fps=10,
        rig="test",
        frames=[
            frame(0, Vec2(64, 30)),
            frame(1, Vec2(88, 40)),
            frame(2, Vec2(98, 64)),
        ],
        dynamics={
            "weapon": {
                "mass_kg": mass,
                "effective_length_m": 0.9,
                "inertia_factor": 0.33,
                "damping_nm_per_rad_s": 0.0,
                "max_braking_torque_nm": 20.0,
                "impact_frame": 1,
                "collision": False,
            }
        },
    )


def test_heavier_weapon_requires_more_follow_through():
    light = analyze_weapon_dynamics(clip_for_mass(0.8))
    heavy = analyze_weapon_dynamics(clip_for_mass(1.6))

    assert heavy["profile"]["inertia_kg_m2"] > light["profile"]["inertia_kg_m2"]
    assert (
        heavy["follow_through"]["minimum_follow_through_deg_at_max_braking"]
        > light["follow_through"]["minimum_follow_through_deg_at_max_braking"]
    )


def test_report_contains_per_frame_kinematics():
    report = analyze_weapon_dynamics(clip_for_mass(1.3))

    assert len(report["frames"]) == 3
    assert "angular_velocity_deg_s" in report["frames"][1]
    assert "estimated_torque_nm" in report["frames"][2]
    assert report["follow_through"]["impact_frame"] == 1
