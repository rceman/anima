from __future__ import annotations

import json
from pathlib import Path

from .constraints import normalize_clip, validate_clip
from .handoff import imagegen_prompt
from .model import MotionClip
from .render import export_render_set
from .timeline import densify_clip


def compile_motion(
    input_path: str | Path,
    output_dir: str | Path,
    scale: int = 4,
) -> bool:
    source = MotionClip.load(input_path)
    dense = densify_clip(source)
    normalized = normalize_clip(dense)

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
    (output / "imagegen_prompt.txt").write_text(
        imagegen_prompt(normalized),
        encoding="utf-8",
    )
    return report.ok
