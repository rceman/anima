from anima.analysis import analyze_motion
from anima.constraints import normalize_clip, validate_clip
from anima.model import MotionClip
from anima.timeline import densify_clip


def solved_clip() -> MotionClip:
    clip = MotionClip.load("examples/twohand_sword_slash/motion.json")
    return normalize_clip(densify_clip(clip))


def test_canonical_swing_preserves_two_handed_geometry():
    clip = solved_clip()
    report = validate_clip(clip)

    assert report.ok, report.issues

    for frame in clip.frames:
        assert frame.joints[clip.primary_hand].distance_to(
            frame.weapon.grip_main
        ) < 0.75
        assert frame.joints[clip.secondary_hand].distance_to(
            frame.weapon.grip_off
        ) < 0.75


def test_canonical_impact_carries_weapon_speed_through_contact():
    clip = solved_clip()
    analysis = analyze_motion(clip)
    weapon_frames = {
        item["label"]: item
        for item in analysis["dynamics"]["weapon"]["frames"]
        if item["label"]
    }

    attack_start = abs(
        weapon_frames["attack_start"]["angular_velocity_deg_s"]
    )
    impact = abs(
        weapon_frames["impact"]["angular_velocity_deg_s"]
    )
    follow = abs(
        weapon_frames["follow_through"]["angular_velocity_deg_s"]
    )

    # The sword must not have already peaked and visibly "arrived" before the
    # impact pose. Impact carries at least as much angular speed as attack_start.
    assert impact >= attack_start * 0.98

    # Momentum continues after impact, but the free swing is already braking.
    assert follow > 0
    assert follow < impact


def test_canonical_ready_and_return_are_true_rest_states():
    clip = solved_clip()
    analysis = analyze_motion(clip)
    weapon_frames = analysis["dynamics"]["weapon"]["frames"]

    assert weapon_frames[0]["angular_velocity_deg_s"] == 0.0
    assert weapon_frames[-1]["angular_velocity_deg_s"] == 0.0
