from anima.model import MotionClip
from anima.system_dynamics import analyze_system_dynamics


def test_system_com_is_mass_weighted_between_body_and_weapon():
    clip = MotionClip(
        width=128,
        height=128,
        ground_y=108,
        fps=10,
        rig="test",
        frames=[object(), object()],
    )

    body = {
        "profile": {
            "mass_kg": 80.0,
            "pixels_per_meter": 40.0,
            "gravity_m_s2": 9.81,
            "friction_coefficient": 0.8,
        },
        "frames": [
            {"com_px": [60.0, 70.0], "support": {"min_x": 40.0, "max_x": 90.0}},
            {"com_px": [60.0, 70.0], "support": {"min_x": 40.0, "max_x": 90.0}},
        ],
    }
    weapon = {
        "profile": {"mass_kg": 2.0},
        "frames": [
            {"weapon_com_px": [100.0, 60.0]},
            {"weapon_com_px": [100.0, 60.0]},
        ],
    }

    report = analyze_system_dynamics(clip, body, weapon)
    x = report["frames"][0]["system_com_px"][0]

    assert 60.0 < x < 61.0
    assert not report["warnings"]


def test_system_com_outside_support_is_reported():
    clip = MotionClip(
        width=128,
        height=128,
        ground_y=108,
        fps=10,
        rig="test",
        frames=[object()],
    )
    body = {
        "profile": {
            "mass_kg": 10.0,
            "pixels_per_meter": 40.0,
            "gravity_m_s2": 9.81,
            "friction_coefficient": 0.8,
        },
        "frames": [
            {"com_px": [50.0, 70.0], "support": {"min_x": 40.0, "max_x": 60.0}},
        ],
    }
    weapon = {
        "profile": {"mass_kg": 10.0},
        "frames": [
            {"weapon_com_px": [100.0, 70.0]},
        ],
    }

    report = analyze_system_dynamics(clip, body, weapon)
    codes = {warning["code"] for warning in report["warnings"]}

    assert "system_com_outside_support" in codes
