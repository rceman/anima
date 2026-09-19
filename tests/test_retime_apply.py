from anima.model import FramePose, MotionClip, Vec2, WeaponPose
from anima.retime_apply import apply_timing_recommendation


def pose(frame_no: int, x: float, label: str) -> FramePose:
    return FramePose(
        frame=frame_no,
        root=Vec2(x, 80),
        joints={
            "hand_r": Vec2(x + 10, 60),
            "hand_l": Vec2(x + 5, 60),
        },
        weapon=WeaponPose(
            grip_main=Vec2(x + 10, 60),
            grip_off=Vec2(x + 5, 60),
            tip=Vec2(x + 30, 40),
        ),
        label=label,
    )


def test_apply_timing_changes_only_time_not_geometry():
    clip = MotionClip(
        width=128,
        height=128,
        ground_y=108,
        fps=10,
        rig="test",
        frames=[
            pose(0, 10, "ready"),
            pose(1, 20, "impact"),
            pose(2, 30, "follow"),
        ],
    )
    report = {
        "recommended": {"global_time_scale": 2.0},
        "segments": [
            {
                "from_frame": 0,
                "to_frame": 1,
                "recommended_time_scale": 1.0,
            },
            {
                "from_frame": 1,
                "to_frame": 2,
                "recommended_time_scale": 2.0,
            },
        ],
    }

    retimed = apply_timing_recommendation(clip, report)

    assert retimed.times_s() == [0.0, 0.1, 0.3]
    assert retimed.frames[2].root == clip.frames[2].root
    assert retimed.frames[2].weapon.tip == clip.frames[2].weapon.tip
