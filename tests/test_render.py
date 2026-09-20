from anima.model import FramePose, MotionClip, Vec2, WeaponPose
from anima.render import (
    PARTS,
    make_sheet,
    render_control_frame,
    render_parts_frame,
)


def test_control_render_and_sheet_dimensions():
    joints = {
        "head": Vec2(64, 42),
        "chest": Vec2(64, 58),
        "shoulder_l": Vec2(59, 57),
        "shoulder_r": Vec2(69, 57),
        "elbow_l": Vec2(63, 66),
        "elbow_r": Vec2(70, 68),
        "hand_l": Vec2(73, 69),
        "hand_r": Vec2(78, 64),
        "hip_l": Vec2(60, 78),
        "hip_r": Vec2(68, 78),
        "knee_l": Vec2(54, 92),
        "knee_r": Vec2(73, 92),
        "foot_l": Vec2(49, 108),
        "foot_r": Vec2(79, 108),
    }
    frame = FramePose(
        0,
        Vec2(64, 78),
        joints,
        WeaponPose(Vec2(78, 64), Vec2(73, 69), Vec2(99, 42)),
        {"foot_l": True, "foot_r": True},
    )
    clip = MotionClip(128, 128, 108, 12, "test", [frame])

    image = render_control_frame(clip, frame, scale=1)
    assert image.size == (128, 128)

    sheet = make_sheet([image] * 8, columns=4)
    assert sheet.size == (512, 256)


def test_parts_render_respects_explicit_arm_z_order():
    # Deliberately place both arms on exactly the same pixels. The top semantic
    # layer must therefore determine the visible color at the overlap.
    joints = {
        "head": Vec2(64, 35),
        "chest": Vec2(64, 50),
        "shoulder_l": Vec2(50, 55),
        "shoulder_r": Vec2(50, 55),
        "elbow_l": Vec2(60, 60),
        "elbow_r": Vec2(60, 60),
        "hand_l": Vec2(70, 65),
        "hand_r": Vec2(70, 65),
        "hip_l": Vec2(58, 75),
        "hip_r": Vec2(70, 75),
        "knee_l": Vec2(55, 90),
        "knee_r": Vec2(73, 90),
        "foot_l": Vec2(52, 108),
        "foot_r": Vec2(76, 108),
    }
    frame = FramePose(
        frame=0,
        root=Vec2(64, 75),
        joints=joints,
        weapon=WeaponPose(
            Vec2(70, 65),
            Vec2(65, 65),
            Vec2(95, 65),
        ),
        layer_order=["right_arm", "left_arm"],
    )
    clip = MotionClip(128, 128, 108, 12, "test", [frame])

    image = render_parts_frame(clip, frame, scale=1)

    # Explicit order means right arm behind, left arm in front.
    assert image.getpixel((60, 60)) == tuple(
        __import__("PIL.ImageColor", fromlist=["getrgb"]).getrgb(
            PARTS["left_arm"]
        )
    )
