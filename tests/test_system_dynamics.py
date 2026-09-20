import pytest
from types import SimpleNamespace

from anima.model import MotionClip
from anima.system_dynamics import analyze_system_dynamics


def test_system_com_is_mass_weighted_between_body_and_weapon():
    clip = MotionClip(
        width=128,
        height=128,
        ground_y=108,
        fps=10,
        rig="test",
        frames=[
            SimpleNamespace(frame=0, label=None),
            SimpleNamespace(frame=1, label=None),
        ],
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
        frames=[SimpleNamespace(frame=0, label=None)],
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


def test_dynamic_center_of_pressure_moves_against_horizontal_acceleration():
    clip = MotionClip(
        width=128,
        height=128,
        ground_y=108,
        fps=10,
        rig="test",
        frames=[
            SimpleNamespace(frame=0, label=None, kinematic_stop=False),
            SimpleNamespace(frame=1, label=None, kinematic_stop=False),
            SimpleNamespace(frame=2, label=None, kinematic_stop=False),
        ],
    )
    body = {
        "profile": {
            "mass_kg": 80.0,
            "pixels_per_meter": 40.0,
            "gravity_m_s2": 9.81,
            "friction_coefficient": 10.0,
        },
        "frames": [
            {"com_px": [60.0, 68.0], "support": {"min_x": 30.0, "max_x": 95.0}},
            {"com_px": [62.0, 68.0], "support": {"min_x": 30.0, "max_x": 95.0}},
            {"com_px": [68.0, 68.0], "support": {"min_x": 30.0, "max_x": 95.0}},
        ],
    }
    weapon = {
        "profile": {"mass_kg": 0.0},
        "frames": [
            {"weapon_com_px": [60.0, 68.0]},
            {"weapon_com_px": [62.0, 68.0]},
            {"weapon_com_px": [68.0, 68.0]},
        ],
    }

    report = analyze_system_dynamics(clip, body, weapon)
    middle = report["frames"][1]

    assert middle["system_com_acceleration_m_s2"][0] > 0
    assert middle["center_of_pressure_x_px"] < middle["system_com_px"][0]
    assert middle["dynamic_balance_margin_px"] is not None


def test_dynamic_balance_outside_support_is_reported():
    clip = MotionClip(
        width=128,
        height=128,
        ground_y=108,
        fps=10,
        rig="test",
        frames=[
            SimpleNamespace(frame=0, label=None, kinematic_stop=False),
            SimpleNamespace(frame=1, label=None, kinematic_stop=False),
            SimpleNamespace(frame=2, label=None, kinematic_stop=False),
        ],
    )
    body = {
        "profile": {
            "mass_kg": 80.0,
            "pixels_per_meter": 40.0,
            "gravity_m_s2": 9.81,
            "friction_coefficient": 10.0,
            "dynamic_balance_tolerance_px": 0.0,
        },
        "frames": [
            {"com_px": [50.0, 60.0], "support": {"min_x": 45.0, "max_x": 55.0}},
            {"com_px": [55.0, 60.0], "support": {"min_x": 45.0, "max_x": 55.0}},
            {"com_px": [75.0, 60.0], "support": {"min_x": 45.0, "max_x": 55.0}},
        ],
    }
    weapon = {
        "profile": {"mass_kg": 0.0},
        "frames": [
            {"weapon_com_px": [50.0, 60.0]},
            {"weapon_com_px": [55.0, 60.0]},
            {"weapon_com_px": [75.0, 60.0]},
        ],
    }

    report = analyze_system_dynamics(clip, body, weapon)
    codes = {warning["code"] for warning in report["warnings"]}

    assert "dynamic_balance_outside_support" in codes


def test_positive_angular_momentum_rate_shifts_cop_right():
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
        dynamics={
            "body": {
                "angular_momentum": {
                    "body_radius_of_gyration_m": 0.3,
                }
            }
        },
    )
    body = {
        "profile": {
            "mass_kg": 80.0,
            "pixels_per_meter": 40.0,
            "gravity_m_s2": 9.81,
            "friction_coefficient": 10.0,
        },
        "frames": [
            {
                "com_px": [64.0, 68.0],
                "com_velocity_m_s": [0.0, 0.0],
                "torso_angular_velocity_deg_s": omega,
                "support": {"min_x": 30.0, "max_x": 100.0},
            }
            for omega in (0.0, 60.0, 120.0)
        ],
    }
    weapon = {
        "profile": {
            "mass_kg": 0.0,
            "inertia_kg_m2": 0.0,
            "effective_length_m": 0.0,
            "center_of_mass_fraction": 0.0,
        },
        "frames": [
            {
                "weapon_com_px": [64.0, 68.0],
                "com_velocity_m_s": [0.0, 0.0],
                "angular_velocity_deg_s": 0.0,
            }
            for _ in range(3)
        ],
    }

    report = analyze_system_dynamics(clip, body, weapon)
    middle = report["frames"][1]

    assert middle["system_com_acceleration_m_s2"][0] == 0.0
    assert middle["angular_momentum"]["angular_momentum_rate_nm"] > 0.0
    assert middle["center_of_pressure_x_px"] > middle["system_com_px"][0]


def test_dynamic_cop_drives_two_foot_load_distribution():
    clip = MotionClip(
        width=128,
        height=128,
        ground_y=108,
        fps=10,
        rig="test",
        frames=[
            SimpleNamespace(
                frame=index,
                label=None,
                time_s=index * 0.1,
                kinematic_stop=False,
                joints={
                    "foot_l": Vec2(40.0, 108.0),
                    "foot_r": Vec2(80.0, 108.0),
                },
            )
            for index in range(3)
        ],
    )
    body = {
        "profile": {
            "mass_kg": 80.0,
            "pixels_per_meter": 40.0,
            "gravity_m_s2": 9.81,
            "friction_coefficient": 10.0,
        },
        "frames": [
            {
                "com_px": [60.0, 68.0],
                "com_velocity_m_s": [0.0, 0.0],
                "torso_angular_velocity_deg_s": 0.0,
                "support": {
                    "min_x": 36.0,
                    "max_x": 84.0,
                    "contacts": {
                        "foot_l": "planted",
                        "foot_r": "planted",
                    },
                },
            }
            for _ in range(3)
        ],
    }
    weapon = {
        "profile": {
            "mass_kg": 0.0,
            "inertia_kg_m2": 0.0,
            "effective_length_m": 0.0,
            "center_of_mass_fraction": 0.0,
        },
        "frames": [
            {
                "weapon_com_px": [60.0, 68.0],
                "com_velocity_m_s": [0.0, 0.0],
                "angular_velocity_deg_s": 0.0,
            }
            for _ in range(3)
        ],
    }

    report = analyze_system_dynamics(clip, body, weapon)
    middle = report["frames"][1]

    assert middle["contact_load_share"] == {
        "foot_l": 0.5,
        "foot_r": 0.5,
    }
    assert sum(
        middle["contact_vertical_force_n"].values()
    ) == pytest.approx(80.0 * 9.81)
