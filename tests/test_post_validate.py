from pathlib import Path

from PIL import Image, ImageDraw

from anima.constraints import normalize_clip
from anima.model import MotionClip
from anima.post_validate import validate_rendered_sheet
from anima.render import export_render_set
from anima.timeline import densify_clip


def test_control_sheet_validates_against_solved_motion(tmp_path: Path):
    clip = MotionClip.load("examples/twohand_sword_slash/motion.json")
    clip = normalize_clip(densify_clip(clip))
    export_render_set(clip, tmp_path, scale=1)

    report = validate_rendered_sheet(
        tmp_path / "control_sheet.png",
        clip,
        columns=4,
    )

    assert report["ok"], report["issues"]
    assert report["frames"][0]["pose_mask"]["control_coverage"] == 1.0
    assert report["frames"][0]["pose_mask"]["rendered_near_control"] == 1.0


def test_wrong_sheet_size_is_rejected(tmp_path: Path):
    clip = MotionClip.load("examples/twohand_sword_slash/motion.json")
    clip = normalize_clip(densify_clip(clip))
    path = tmp_path / "wrong.png"
    Image.new("RGB", (64, 64), (30, 37, 42)).save(path)

    report = validate_rendered_sheet(path, clip, columns=4)

    assert not report["ok"]
    assert report["issues"][0]["code"] == "sheet_size_mismatch"


def test_shifted_render_is_rejected_by_pose_mask(tmp_path: Path):
    clip = MotionClip.load("examples/twohand_sword_slash/motion.json")
    clip = normalize_clip(densify_clip(clip))
    export_render_set(clip, tmp_path, scale=1)

    source = Image.open(tmp_path / "control_sheet.png").convert("RGB")
    shifted_sheet = source.copy()
    first = source.crop((0, 0, 128, 128))
    background = first.getpixel((0, 0))
    shifted = Image.new("RGB", (128, 128), background)
    shifted.paste(first, (20, 0))
    shifted_sheet.paste(shifted, (0, 0))

    path = tmp_path / "shifted_sheet.png"
    shifted_sheet.save(path)

    report = validate_rendered_sheet(path, clip, columns=4)
    first_codes = {
        issue["code"]
        for issue in report["frames"][0]["issues"]
    }

    assert not report["ok"]
    assert "pose_mask_miss" in first_codes


def test_per_frame_background_drift_is_rejected(tmp_path: Path):
    clip = MotionClip.load("examples/twohand_sword_slash/motion.json")
    clip = normalize_clip(densify_clip(clip))
    export_render_set(clip, tmp_path, scale=1)

    source = Image.open(tmp_path / "control_sheet.png").convert("RGB")
    changed = source.copy()
    frame = source.crop((128, 0, 256, 128))
    pixels = frame.load()
    old_bg = frame.getpixel((0, 0))
    new_bg = (
        min(255, old_bg[0] + 30),
        min(255, old_bg[1] + 30),
        min(255, old_bg[2] + 30),
    )
    for y in range(128):
        for x in range(128):
            if pixels[x, y] == old_bg:
                pixels[x, y] = new_bg
    changed.paste(frame, (128, 0))

    path = tmp_path / "background_drift.png"
    changed.save(path)
    report = validate_rendered_sheet(path, clip, columns=4)

    frame1_codes = {
        issue["code"]
        for issue in report["frames"][1]["issues"]
    }
    assert "background_drift" in frame1_codes


def test_missing_sword_blade_path_is_rejected(tmp_path: Path):
    clip = MotionClip.load("examples/twohand_sword_slash/motion.json")
    clip = normalize_clip(densify_clip(clip))
    export_render_set(clip, tmp_path, scale=1)

    source = Image.open(tmp_path / "control_sheet.png").convert("RGB")
    damaged = source.copy()
    first = damaged.crop((0, 0, 128, 128))
    draw = ImageDraw.Draw(first)
    frame = clip.frames[0]
    background = first.getpixel((0, 0))
    draw.line(
        [
            (
                round(frame.weapon.grip_main.x),
                round(frame.weapon.grip_main.y),
            ),
            (
                round(frame.weapon.tip.x),
                round(frame.weapon.tip.y),
            ),
        ],
        fill=background,
        width=7,
    )
    damaged.paste(first, (0, 0))

    path = tmp_path / "missing_sword.png"
    damaged.save(path)
    report = validate_rendered_sheet(path, clip, columns=4)

    first_codes = {
        issue["code"]
        for issue in report["frames"][0]["issues"]
    }
    assert "sword_path_missing" in first_codes
