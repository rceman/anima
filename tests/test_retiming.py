from types import SimpleNamespace

from anima.model import MotionClip
from anima.retiming import recommend_timing


def test_timing_recommendation_slows_excessive_dynamics():
    clip = MotionClip(
        width=128,
        height=128,
        ground_y=108,
        fps=10,
        rig="test",
        frames=[],
    )
    clip.frames = [
        SimpleNamespace(frame=0, label="ready"),
        SimpleNamespace(frame=1, label="impact"),
        SimpleNamespace(frame=2, label="follow"),
    ]

    report = {
        "weapon": {
            "profile": {
                "inertia_kg_m2": 0.5,
                "damping_nm_per_rad_s": 0.0,
                "max_braking_torque_nm": 10.0,
            },
            "frames": [
                {
                    "frame": 1,
                    "angular_velocity_deg_s": 360.0,
                    "angular_acceleration_deg_s2": 3600.0,
                }
            ],
        },
        "body": {
            "profile": {
                "max_com_accel_m_s2": 20.0,
                "max_com_jerk_m_s3": 200.0,
                "friction_coefficient": 0.8,
                "gravity_m_s2": 9.81,
            },
            "frames": [
                {
                    "frame": 1,
                    "com_acceleration_m_s2": [40.0, 0.0],
                    "com_jerk_m_s3": [800.0, 0.0],
                    "ground_reaction_force": {
                        "horizontal_n": 1.0,
                        "vertical_up_n": 1.0,
                    },
                }
            ],
        },
    }

    timing = recommend_timing(clip, report)

    assert timing["recommended"]["global_time_scale"] > 1.0
    assert timing["recommended"]["fps_if_same_frames"] < 10.0
    assert timing["recommended"]["frame_count_if_same_fps"] > 3


def test_segment_recommendation_maps_reasons_to_phase():
    clip = MotionClip(
        width=128,
        height=128,
        ground_y=108,
        fps=10,
        rig="test",
        frames=[
            SimpleNamespace(frame=0, label="ready"),
            SimpleNamespace(frame=2, label="impact"),
            SimpleNamespace(frame=4, label="follow"),
        ],
    )
    report = {
        "weapon": {
            "profile": {
                "inertia_kg_m2": 1.0,
                "damping_nm_per_rad_s": 0.0,
                "max_braking_torque_nm": 1.0,
            },
            "frames": [
                {
                    "frame": 3,
                    "angular_velocity_deg_s": 300.0,
                    "angular_acceleration_deg_s2": 3000.0,
                }
            ],
        },
        "body": {"profile": {}, "frames": []},
        "system": {"profile": {}, "frames": []},
    }

    timing = recommend_timing(clip, report)
    follow_segment = timing["segments"][1]

    assert follow_segment["from_label"] == "impact"
    assert follow_segment["to_label"] == "follow"
    assert follow_segment["recommended_frame_span_same_fps"] > 2


def test_sampled_violation_affects_both_adjacent_keyframe_intervals():
    clip = MotionClip(
        width=128,
        height=128,
        ground_y=108,
        fps=10,
        rig="test",
        frames=[
            SimpleNamespace(frame=0, label="a"),
            SimpleNamespace(frame=1, label="b"),
            SimpleNamespace(frame=2, label="c"),
        ],
    )
    report = {
        "weapon": {
            "profile": {
                "inertia_kg_m2": 1.0,
                "damping_nm_per_rad_s": 0.0,
                "max_drive_torque_nm": 1.0,
            },
            "frames": [
                {
                    "frame": 1,
                    "angular_velocity_deg_s": 0.0,
                    "angular_acceleration_deg_s2": 1000.0,
                }
            ],
        },
        "body": {"profile": {}, "frames": []},
        "system": {"profile": {}, "frames": []},
    }

    timing = recommend_timing(clip, report)
    assert timing["segments"][0]["recommended_time_scale"] > 1.0
    assert timing["segments"][1]["recommended_time_scale"] > 1.0
