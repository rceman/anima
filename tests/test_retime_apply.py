import pytest

from anima.model import FramePose, MotionClip, Vec2, WeaponPose
from anima.retime_apply import apply_timing_recommendation, auto_retime


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

    assert retimed.times_s() == pytest.approx([0.0, 0.1, 0.3])
    assert retimed.frames[2].root == clip.frames[2].root
    assert retimed.frames[2].weapon.tip == clip.frames[2].weapon.tip


def test_auto_retime_does_not_ignore_small_scale_when_physics_is_hard(monkeypatch):
    clip = MotionClip(
        width=128,
        height=128,
        ground_y=108,
        fps=10,
        rig="test",
        frames=[
            pose(0, 10, "ready"),
            pose(1, 20, "impact"),
        ],
    )

    calls = {"count": 0}

    def fake_analyze(current):
        calls["count"] += 1
        # First analysis is a hard failure needing only a 0.5% timing change,
        # below the default 1% numerical tolerance. After retiming it passes.
        hard = calls["count"] == 1
        return {
            "physics_validation": {
                "ok": not hard,
                "counts": {"hard": 1 if hard else 0},
            },
            "timing_recommendation": {
                "recommended": {"global_time_scale": 1.005 if hard else 1.0},
                "segments": [
                    {
                        "from_frame": 0,
                        "to_frame": 1,
                        "recommended_time_scale": 1.005 if hard else 1.0,
                    }
                ],
            },
        }

    monkeypatch.setattr("anima.retime_apply.analyze_motion", fake_analyze)

    retimed, history = auto_retime(
        clip,
        iterations=2,
        tolerance=1.01,
    )

    assert retimed.times_s()[-1] > clip.times_s()[-1]
    assert history[0]["physics_ok"] is False


def test_auto_retime_stops_when_failure_is_not_retimeable(monkeypatch):
    clip = MotionClip(
        width=128,
        height=128,
        ground_y=108,
        fps=10,
        rig="test",
        frames=[
            pose(0, 10, "ready"),
            pose(1, 20, "hold"),
        ],
    )

    def fake_analyze(current):
        return {
            "physics_validation": {
                "ok": False,
                "counts": {"hard": 1},
            },
            "timing_recommendation": {
                "recommended": {"global_time_scale": 1.0},
                "segments": [
                    {
                        "from_frame": 0,
                        "to_frame": 1,
                        "recommended_time_scale": 1.0,
                    }
                ],
                "reasons": [
                    {
                        "code": "torque_not_retimeable",
                        "retime_possible": False,
                        "time_scale": 1.0,
                    }
                ],
            },
        }

    monkeypatch.setattr(
        "anima.retime_apply.analyze_motion",
        fake_analyze,
    )

    retimed, history = auto_retime(
        clip,
        iterations=5,
    )

    assert retimed.times_s() == clip.times_s()
    assert history[0]["retime_blocked"] is True
    assert len(history) == 2  # first attempt + final analysis
