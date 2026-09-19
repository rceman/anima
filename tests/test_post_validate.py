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


def test_wrong_sheet_size_is_rejected(tmp_path: Path):
    clip = MotionClip.load("examples/twohand_sword_slash/motion.json")
    clip = normalize_clip(densify_clip(clip))
    path = tmp_path / "wrong.png"
    Image.new("RGB", (64, 64), (30, 37, 42)).save(path)

    report = validate_rendered_sheet(path, clip, columns=4)

    assert not report["ok"]
    assert report["issues"][0]["code"] == "sheet_size_mismatch"
