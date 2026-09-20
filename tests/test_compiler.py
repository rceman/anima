import json
from pathlib import Path

from PIL import Image

from anima.compiler import compile_motion


def test_example_exports_canonical_128_frames(tmp_path: Path):
    source = Path("examples/twohand_sword_slash/motion.json")
    ok = compile_motion(
        source,
        tmp_path,
        scale=3,
        auto_retime_iterations=1,
    )
    assert ok, (tmp_path / "validation.json").read_text(encoding="utf-8")

    frame = Image.open(tmp_path / "control_frames" / "frame_00.png")
    assert frame.size == (128, 128)

    sheet = Image.open(tmp_path / "control_sheet.png")
    assert sheet.size == (512, 256)

    preview = Image.open(tmp_path / "control_sheet.preview.png")
    assert preview.size == (2048, 1024)

    parts_sheet = Image.open(tmp_path / "parts_sheet.png")
    assert parts_sheet.size == (512, 256)

    cutout_sheet = Image.open(tmp_path / "cutout_sheet.png")
    assert cutout_sheet.size == (512, 256)

    parts_frame = Image.open(tmp_path / "parts_frames" / "frame_00.png")
    assert parts_frame.size == (128, 128)

    left_arm_layer = Image.open(
        tmp_path / "layers" / "left_arm" / "frame_00.png"
    )
    assert left_arm_layer.mode == "RGBA"
    assert left_arm_layer.size == (128, 128)

    weapon_layer_sheet = Image.open(
        tmp_path / "layers" / "weapon_sheet.png"
    )
    assert weapon_layer_sheet.mode == "RGBA"
    assert weapon_layer_sheet.size == (512, 256)

    debug_preview = Image.open(tmp_path / "debug_preview.gif")
    assert debug_preview.size == (1024, 1024)

    control_preview = Image.open(tmp_path / "control_preview.gif")
    assert control_preview.size == (512, 512)

    parts_preview = Image.open(tmp_path / "parts_preview.gif")
    assert parts_preview.size == (512, 512)

    cutout_preview = Image.open(tmp_path / "cutout_preview.gif")
    assert cutout_preview.size == (512, 512)

    review_preview = Image.open(tmp_path / "review_preview.gif")
    assert review_preview.size == (2048, 1024)

    inspection_preview = Image.open(tmp_path / "inspection_preview.gif")
    assert inspection_preview.size == (1024, 512)

    assert (tmp_path / "dynamics.json").exists()
    assert (tmp_path / "occlusion.json").exists()
    assert (tmp_path / "retime_history.json").exists()
    assert (tmp_path / "diagnostics_summary.json").exists()
    assert (tmp_path / "animation_manifest.json").exists()
    assert (tmp_path / "transforms.json").exists()
    assert (tmp_path / "reference_map.json").exists()
    assert (tmp_path / "piece_manifest.json").exists()
    assert (tmp_path / "bind_pieces" / "weapon.png").exists()
    assert (tmp_path / "cutout_frames" / "frame_00.png").exists()

    prompt = (tmp_path / "imagegen_prompt.txt").read_text(encoding="utf-8")
    assert "two-handed grip" in prompt


def test_compile_can_time_resample_reference_motion(tmp_path: Path):
    source = Path("examples/twohand_sword_slash/motion.json")
    ok = compile_motion(
        source,
        tmp_path,
        scale=1,
        sample_fps=16,
        auto_retime_iterations=0,
    )
    assert ok, (tmp_path / "validation.json").read_text(encoding="utf-8")

    manifest = json.loads(
        (tmp_path / "animation_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    normalized = json.loads(
        (tmp_path / "motion.normalized.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(manifest["frames"]) > 8
    assert normalized["metadata"]["resampling"]["target_fps"] == 16
    impact_frames = [
        frame
        for frame in manifest["frames"]
        if "impact" in frame["events"]
    ]
    assert len(impact_frames) == 1

    rows = (len(manifest["frames"]) + 3) // 4
    sheet = Image.open(tmp_path / "control_sheet.png")
    assert sheet.size == (512, rows * 128)
