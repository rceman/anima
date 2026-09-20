from anima.authoring import ParametricAuthor, PoseRecipe
from anima.constraints import normalize_clip, validate_clip
from anima.model import FramePose, Vec2, WeaponPose


def rest_frame() -> FramePose:
    joints = {
        "head": Vec2(64, 38),
        "chest": Vec2(64, 56),
        "shoulder_l": Vec2(58, 58),
        "shoulder_r": Vec2(70, 58),
        "elbow_l": Vec2(55, 68),
        "elbow_r": Vec2(73, 68),
        "hand_l": Vec2(66, 70),
        "hand_r": Vec2(72, 66),
        "hip_l": Vec2(60, 78),
        "hip_r": Vec2(68, 78),
        "knee_l": Vec2(56, 92),
        "knee_r": Vec2(72, 92),
        "foot_l": Vec2(52, 108),
        "foot_r": Vec2(76, 108),
    }
    return FramePose(
        frame=0,
        root=Vec2(64, 78),
        joints=joints,
        weapon=WeaponPose(
            grip_main=Vec2(72, 66),
            grip_off=Vec2(66, 70),
            tip=Vec2(98, 46),
        ),
        contacts={"foot_l": "planted", "foot_r": "planted"},
    )


def test_parametric_recipe_builds_full_pose_without_authored_elbows():
    author = ParametricAuthor(rest_frame())
    recipe = PoseRecipe(
        frame=1,
        root=Vec2(66, 78),
        main_grip=Vec2(77, 60),
        sword_angle_deg=-45,
        foot_l=Vec2(52, 108),
        foot_r=Vec2(76, 108),
        torso_lean_deg=8,
        shoulder_line_deg=6,
        contacts={"foot_l": "planted", "foot_r": "planted"},
        label="windup",
        time_s=0.2,
    )

    pose = author.frame(recipe)

    assert pose.joints["hand_r"] == pose.weapon.grip_main
    assert pose.joints["hand_l"] == pose.weapon.grip_off
    assert "elbow_l" in pose.ik_poles
    assert "elbow_r" in pose.ik_poles
    assert pose.time_s == 0.2


def test_parametric_clip_normalizes_to_valid_fixed_geometry():
    rest = rest_frame()
    author = ParametricAuthor(rest)
    recipes = [
        PoseRecipe(
            frame=0,
            root=rest.root,
            main_grip=rest.weapon.grip_main,
            sword_angle_deg=-38,
            foot_l=rest.joints["foot_l"],
            foot_r=rest.joints["foot_r"],
            contacts={"foot_l": "planted", "foot_r": "planted"},
            label="ready",
            time_s=0.0,
            kinematic_stop=True,
        ),
        PoseRecipe(
            frame=1,
            root=Vec2(66, 78),
            main_grip=Vec2(76, 60),
            sword_angle_deg=-10,
            foot_l=rest.joints["foot_l"],
            foot_r=rest.joints["foot_r"],
            torso_lean_deg=6,
            shoulder_line_deg=5,
            contacts={"foot_l": "planted", "foot_r": "planted"},
            label="move",
            time_s=0.2,
        ),
    ]
    clip = author.clip(
        recipes,
        width=128,
        height=128,
        ground_y=108,
        fps=12,
        rig="humanoid_twohand_sword_v1",
        dynamics={
            "body": {
                "shoulder_girdle_max_shift_px": 4.0,
            }
        },
    )

    normalized = normalize_clip(clip)
    report = validate_clip(normalized)

    assert report.ok, report.issues
    assert normalized.frames[1].joints["hand_r"].distance_to(
        normalized.frames[1].weapon.grip_main
    ) < 0.75
    assert normalized.frames[1].joints["hand_l"].distance_to(
        normalized.frames[1].weapon.grip_off
    ) < 0.75
