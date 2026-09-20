import json
from pathlib import Path

from PIL import Image

from anima.cutout import (
    CUTOUT,
    build_bind_piece_masks,
    build_piece_manifest,
    export_cutout_set,
    load_bind_piece_pack,
    render_cutout_frame,
)
from anima.model import FramePose, MotionClip, Vec2, WeaponPose


def pose(frame_no: int, elbow_r: Vec2, hand_r: Vec2) -> FramePose:
    joints = {
        "head": Vec2(64, 35),
        "chest": Vec2(64, 52),
        "shoulder_l": Vec2(56, 56),
        "shoulder_r": Vec2(72, 56),
        "elbow_l": Vec2(54, 68),
        "elbow_r": elbow_r,
        "hand_l": Vec2(65, 70),
        "hand_r": hand_r,
        "hip_l": Vec2(60, 78),
        "hip_r": Vec2(68, 78),
        "knee_l": Vec2(56, 92),
        "knee_r": Vec2(72, 92),
        "foot_l": Vec2(52, 108),
        "foot_r": Vec2(76, 108),
    }
    return FramePose(
        frame=frame_no,
        root=Vec2(64, 78),
        joints=joints,
        weapon=WeaponPose(
            grip_main=hand_r,
            grip_off=Vec2(hand_r.x - 5, hand_r.y),
            tip=Vec2(hand_r.x + 25, hand_r.y),
        ),
        time_s=frame_no * 0.1,
    )


def test_cutout_reuses_bind_piece_masks_and_moves_rigid_segments():
    rest = pose(
        0,
        elbow_r=Vec2(78, 66),
        hand_r=Vec2(84, 76),
    )
    moved = pose(
        1,
        elbow_r=Vec2(82, 56),
        hand_r=Vec2(92, 56),
    )
    clip = MotionClip(
        128,
        128,
        108,
        10,
        "test",
        [rest, moved],
    )

    masks = build_bind_piece_masks(clip)
    first = render_cutout_frame(
        clip,
        0,
        bind_masks=masks,
    )
    second = render_cutout_frame(
        clip,
        1,
        bind_masks=masks,
    )

    assert "upper_arm_r" in masks
    assert masks["upper_arm_r"].mode == "RGBA"
    assert first.size == (128, 128)
    assert second.size == (128, 128)
    assert first.tobytes() != second.tobytes()


def test_bind_piece_masks_have_transparent_background():
    rest = pose(
        0,
        elbow_r=Vec2(78, 66),
        hand_r=Vec2(84, 76),
    )
    clip = MotionClip(
        128,
        128,
        108,
        10,
        "test",
        [rest],
    )

    masks = build_bind_piece_masks(clip)

    assert masks["weapon"].getpixel((0, 0))[3] == 0
    assert masks["torso"].getpixel((0, 0))[3] == 0


def test_piece_manifest_lists_full_canvas_bind_contract():
    rest = pose(
        0,
        elbow_r=Vec2(78, 66),
        hand_r=Vec2(84, 76),
    )
    clip = MotionClip(128, 128, 108, 10, "test", [rest])

    manifest = build_piece_manifest(clip)

    assert manifest["canvas"] == [128, 128]
    assert manifest["pieces"]["upper_arm_r"]["file"] == "upper_arm_r.png"
    assert manifest["pieces"]["upper_arm_r"]["pivot"] == [72, 56]
    assert manifest["pieces"]["weapon"]["semantic"] == "weapon"


def test_painted_piece_pack_is_loaded_and_preserved(tmp_path: Path):
    rest = pose(
        0,
        elbow_r=Vec2(78, 66),
        hand_r=Vec2(84, 76),
    )
    moved = pose(
        1,
        elbow_r=Vec2(82, 56),
        hand_r=Vec2(92, 56),
    )
    clip = MotionClip(128, 128, 108, 10, "test", [rest, moved])

    source = build_bind_piece_masks(clip)
    pack = tmp_path / "pack"
    pack.mkdir()
    painted_color = (210, 80, 220, 255)

    for name, mask in source.items():
        rgba = Image.new("RGBA", mask.size, (0, 0, 0, 0))
        alpha = mask.getchannel("A")
        fill = Image.new(
            "RGBA",
            mask.size,
            (
                painted_color
                if name == "upper_arm_r"
                else (220, 215, 190, 255)
            ),
        )
        rgba.paste(fill, (0, 0), alpha)
        rgba.save(pack / f"{name}.png")

    loaded = load_bind_piece_pack(clip, pack)
    assert loaded["upper_arm_r"].getbbox() is not None

    output = tmp_path / "out"
    export_cutout_set(
        clip,
        output,
        columns=2,
        preview_scale=1,
        piece_dir=pack,
    )

    manifest = json.loads(
        (output / "piece_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["source"] == "painted_piece_pack"

    rendered = Image.open(output / "cutout_frames" / "frame_00.png").convert("RGB")
    assert painted_color[:3] in set(rendered.getdata())
