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
    clip.frames = [object(), object(), object()]  # frame count only for this unit test

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
