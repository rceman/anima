from anima.model import FramePose, MotionClip, Vec2, WeaponPose
from anima.resample import resample_clip


def pose(
    frame_no: int,
    time_s: float,
    x: float,
    angle_tip: Vec2,
    label: str | None = None,
    stop: bool = False,
) -> FramePose:
    root = Vec2(x, 80)
    grip_main = Vec2(x + 10, 60)
    return FramePose(
        frame=frame_no,
        root=root,
        joints={
            "hand_r": grip_main,
            "hand_l": Vec2(x + 5, 60),
        },
        weapon=WeaponPose(
            grip_main=grip_main,
            grip_off=Vec2(x + 5, 60),
            tip=angle_tip,
        ),
        label=label,
        time_s=time_s,
        kinematic_stop=stop,
        layer_order=["left_arm", "right_arm"],
    )


def test_resample_uses_real_time_grid_and_exact_endpoint():
    clip = MotionClip(
        width=128,
        height=128,
        ground_y=108,
        fps=10,
        rig="test",
        frames=[
            pose(0, 0.0, 10, Vec2(40, 30), "ready", True),
            pose(1, 0.18, 20, Vec2(50, 40), "impact"),
            pose(2, 0.43, 30, Vec2(60, 50), "return", True),
        ],
    )

    sampled = resample_clip(clip, fps=10)

    assert sampled.times_s() == [0.0, 0.1, 0.2, 0.3, 0.4, 0.43]
    assert [frame.frame for frame in sampled.frames] == list(range(6))
    assert sampled.frames[0].label == "ready"
    assert sampled.frames[-1].label == "return"
    assert sampled.frames[0].kinematic_stop is True
    assert sampled.frames[-1].kinematic_stop is True


def test_resample_intermediate_pose_has_no_fake_stop_or_label():
    clip = MotionClip(
        width=128,
        height=128,
        ground_y=108,
        fps=10,
        rig="test",
        frames=[
            pose(0, 0.0, 10, Vec2(40, 30), "ready", True),
            pose(1, 0.5, 30, Vec2(70, 60), "impact"),
        ],
    )

    sampled = resample_clip(clip, fps=4)

    assert sampled.frames[1].time_s == 0.25
    assert sampled.frames[1].label is None
    assert sampled.frames[1].kinematic_stop is False
    assert sampled.frames[1].root.x > 10
    assert sampled.frames[1].root.x < 30
