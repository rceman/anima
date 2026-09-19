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


def test_enforced_drive_torque_limit_is_reported():
    clip = clip_for_mass(1.3)
    clip.dynamics["weapon"]["damping_nm_per_rad_s"] = 1.0
    clip.dynamics["weapon"]["max_drive_torque_nm"] = 0.1
    clip.dynamics["weapon"]["enforce_drive_torque_limit"] = True

    report = analyze_weapon_dynamics(clip)
    codes = {warning["code"] for warning in report["warnings"]}

    assert "drive_torque_limit_exceeded" in codes


def test_collision_model_reports_expected_post_impact_response():
    clip = clip_for_mass(1.3)
    clip.dynamics["weapon"].update(
        {
            "collision": True,
            "impact_frame": 1,
            "target_effective_mass_kg": 5.0,
            "coefficient_of_restitution": 0.1,
            "impact_radius_fraction": 0.9,
        }
    )

    report = analyze_weapon_dynamics(clip)
    collision = report["collision_response"]

    assert collision is not None
    assert collision["impact_frame"] == 1
    assert collision["impulse_ns"] >= 0
    assert "expected_omega_after_deg_s" in collision


def test_enforced_collision_mismatch_is_reported():
    clip = clip_for_mass(1.3)
    clip.dynamics["weapon"].update(
        {
            "collision": True,
            "impact_frame": 1,
            "target_effective_mass_kg": 1000.0,
            "coefficient_of_restitution": 0.0,
            "collision_velocity_tolerance_fraction": 0.01,
            "enforce_collision_response": True,
        }
    )

    report = analyze_weapon_dynamics(clip)
    codes = {warning["code"] for warning in report["warnings"]}

    assert "collision_response_mismatch" in codes
