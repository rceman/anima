from pathlib import Path

from PIL import Image

from anima.compiler import compile_motion


def test_example_exports_canonical_128_frames(tmp_path: Path):
    source = Path("examples/twohand_sword_slash/motion.json")
    assert compile_motion(
        source,
        tmp_path,
        scale=3,
        auto_retime_iterations=1,
    )

    frame = Image.open(tmp_path / "control_frames" / "frame_00.png")
    assert frame.size == (128, 128)

    sheet = Image.open(tmp_path / "control_sheet.png")
    assert sheet.size == (512, 256)

    preview = Image.open(tmp_path / "control_sheet.preview.png")
    assert preview.size == (2048, 1024)

    debug_preview = Image.open(tmp_path / "debug_preview.gif")
    assert debug_preview.size == (1024, 1024)

    control_preview = Image.open(tmp_path / "control_preview.gif")
    assert control_preview.size == (512, 512)

    review_preview = Image.open(tmp_path / "review_preview.gif")
    assert review_preview.size == (2048, 1024)

    assert (tmp_path / "dynamics.json").exists()
    assert (tmp_path / "retime_history.json").exists()

    prompt = (tmp_path / "imagegen_prompt.txt").read_text(encoding="utf-8")
    assert "two-handed grip" in prompt
