from __future__ import annotations

import json
from pathlib import Path

from .constraints import normalize_clip, validate_clip
from .model import MotionClip
from .render import export_render_set


def compile_motion(
    input_path: str | Path,
    output_dir: str | Path,
    scale: int = 4,
) -> bool:
    source = MotionClip.load(input_path)
    normalized = normalize_clip(source)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    normalized.save(output / "motion.normalized.json")

    report = validate_clip(normalized)
    report_json = {
        "ok": report.ok,
        "issues": [issue.__dict__ for issue in report.issues],
    }
    (output / "validation.json").write_text(
        json.dumps(report_json, indent=2) + "\n",
        encoding="utf-8",
    )

    export_render_set(normalized, output, scale=scale)
    return report.ok
