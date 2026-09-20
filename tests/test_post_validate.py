from pathlib import Path

from PIL import Image

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
