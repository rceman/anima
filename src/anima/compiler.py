from __future__ import annotations

import json
from pathlib import Path

from .analysis import analyze_motion
from .constraints import normalize_clip, validate_clip
from .handoff import imagegen_prompt
from .model import MotionClip
from .render import export_render_set
from .retime_apply import auto_retime
from .timeline import densify_clip


def compile_motion(
    input_path: str | Path,
    output_dir: str | Path,
    scale: int = 4,
    strict_physics: bool = False,
    auto_retime_iterations: int = 0,
) -> bool:
    source = MotionClip.load(input_path)
    dense = densify_clip(source)
    normalized = normalize_clip(dense)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    retime_history = None
    if auto_retime_iterations > 0:
        normalized, retime_history = auto_retime(
            normalized,
            iterations=auto_retime_iterations,
        )
        (output / "retime_history.json").write_text(
            json.dumps({"history": retime_history}, indent=2) + "\n",
            encoding="utf-8",
        )

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

    analysis = analyze_motion(normalized)
    dynamics_report = analysis["dynamics"]
    physics_report = analysis["physics_validation"]
    timing_report = analysis["timing_recommendation"]

    (output / "dynamics.json").write_text(
        json.dumps(dynamics_report, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "physics_validation.json").write_text(
        json.dumps(physics_report, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "timing_recommendation.json").write_text(
        json.dumps(timing_report, indent=2) + "\n",
        encoding="utf-8",
    )

    export_render_set(normalized, output, scale=scale)
    (output / "imagegen_prompt.txt").write_text(
        imagegen_prompt(normalized),
        encoding="utf-8",
    )
    return report.ok and (physics_report["ok"] or not strict_physics)
