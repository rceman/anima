from anima.manifest import build_animation_manifest
from anima.model import FramePose, MotionClip, Vec2, WeaponPose


def pose(frame_no: int, time_s: float, label: str) -> FramePose:
    return FramePose(
        frame=frame_no,
        root=Vec2(64, 80),
        joints={},
        weapon=WeaponPose(
            grip_main=Vec2(70, 60),
            grip_off=Vec2(65, 60),
            tip=Vec2(90, 40),
        ),
        contacts={"foot_l": "planted"},
        label=label,
        time_s=time_s,
    )


def test_manifest_preserves_nonuniform_timing_and_impact_event():
    clip = MotionClip(
        width=128,
        height=128,
        ground_y=108,
        fps=12,
        rig="test",
        frames=[
            pose(0, 0.0, "ready"),
            pose(1, 0.1, "impact"),
            pose(2, 0.35, "follow"),
        ],
        dynamics={"weapon": {"impact_frame": 1}},
    )

    manifest = build_animation_manifest(clip, columns=2)

    assert manifest["sheet"]["width"] == 256
    assert manifest["sheet"]["height"] == 256
    assert [frame["duration_ms"] for frame in manifest["frames"]] == [100, 250, 250]
    assert manifest["frames"][1]["events"] == ["impact"]
    assert manifest["frames"][0]["contacts"]["foot_l"] == "planted"
