from types import SimpleNamespace

from anima.angular_momentum import analyze_angular_momentum
from anima.model import MotionClip


def test_zero_motion_has_zero_angular_momentum_rate():
    clip = MotionClip(
        width=128,
        height=128,
        ground_y=108,
        fps=10,
        rig="test",
        frames=[
            SimpleNamespace(frame=0, label=None, time_s=0.0, kinematic_stop=True),
            SimpleNamespace(frame=1, label=None, time_s=0.1, kinematic_stop=True),
        ],
        dynamics={
            "body": {
                "angular_momentum": {
                    "body_radius_of_gyration_m": 0.3,
                }
            }
        },
    )
    body = {
        "profile": {"mass_kg": 80.0, "pixels_per_meter": 40.0},
        "frames": [
            {
                "com_px": [64.0, 70.0],
                "com_velocity_m_s": [0.0, 0.0],
                "torso_angular_velocity_deg_s": 0.0,
            },
            {
                "com_px": [64.0, 70.0],
                "com_velocity_m_s": [0.0, 0.0],
                "torso_angular_velocity_deg_s": 0.0,
            },
        ],
    }
    weapon = {
        "profile": {
            "mass_kg": 2.0,
            "inertia_kg_m2": 0.6,
            "effective_length_m": 1.0,
            "center_of_mass_fraction": 0.5,
        },
        "frames": [
            {
                "weapon_com_px": [80.0, 60.0],
                "com_velocity_m_s": [0.0, 0.0],
                "angular_velocity_deg_s": 0.0,
            },
            {
                "weapon_com_px": [80.0, 60.0],
                "com_velocity_m_s": [0.0, 0.0],
                "angular_velocity_deg_s": 0.0,
            },
        ],
    }

    report = analyze_angular_momentum(clip, body, weapon)

    assert report["frames"][0]["total_angular_momentum_kg_m2_s"] == 0.0
    assert report["frames"][1]["total_angular_momentum_kg_m2_s"] == 0.0
    assert report["frames"][0]["angular_momentum_rate_nm"] == 0.0
    assert report["frames"][1]["angular_momentum_rate_nm"] == 0.0


def test_weapon_spin_creates_angular_momentum_and_rate():
    clip = MotionClip(
        width=128,
        height=128,
        ground_y=108,
        fps=10,
        rig="test",
        frames=[
            SimpleNamespace(frame=0, label=None, time_s=0.0, kinematic_stop=False),
            SimpleNamespace(frame=1, label=None, time_s=0.1, kinematic_stop=False),
            SimpleNamespace(frame=2, label=None, time_s=0.2, kinematic_stop=False),
        ],
    )
    body = {
        "profile": {"mass_kg": 80.0, "pixels_per_meter": 40.0},
        "frames": [
            {
                "com_px": [64.0, 70.0],
                "com_velocity_m_s": [0.0, 0.0],
                "torso_angular_velocity_deg_s": 0.0,
            }
            for _ in range(3)
        ],
    }
    weapon = {
        "profile": {
            "mass_kg": 2.0,
            "inertia_kg_m2": 0.7,
            "effective_length_m": 1.0,
            "center_of_mass_fraction": 0.5,
        },
        "frames": [
            {
                "weapon_com_px": [80.0, 60.0],
                "com_velocity_m_s": [0.0, 0.0],
                "angular_velocity_deg_s": omega,
            }
            for omega in (0.0, 90.0, 180.0)
        ],
    }

    report = analyze_angular_momentum(clip, body, weapon)

    assert report["profile"]["weapon_inertia_about_com_kg_m2"] > 0.0
    assert report["frames"][1]["weapon_spin_angular_momentum_kg_m2_s"] > 0.0
    assert report["frames"][1]["angular_momentum_rate_nm"] > 0.0
