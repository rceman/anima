from __future__ import annotations

import json
from pathlib import Path

from .analysis import analyze_motion
from .constraints import normalize_clip, validate_clip
from .cutout import export_cutout_set
from .handoff import imagegen_prompt, piecegen_prompt
from .manifest import build_animation_manifest
from .model import MotionClip
from .render import export_render_set
from .resample import resample_clip
from .retime_apply import auto_retime
from .summary import build_diagnostics_summary
from .timeline import densify_clip
from .transforms import build_transform_manifest


def compile_motion(
    input_path: str | Path,
    output_dir: str | Path,
    scale: int = 4,
    strict_physics: bool = False,
    auto_retime_iterations: int = 0,
    sample_fps: float | None = None,
    piece_dir: str | Path | None = None,
) -> bool:
    source = MotionClip.load(input_path)
    dense = (
        resample_clip(source, sample_fps)
        if sample_fps is not None
        else densify_clip(source)
    )
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
    (output / "reference_map.json").write_text(
        json.dumps(
            {
                "poses": [
                    {
                        "frame": frame.frame,
                        "time_s": frame.time_s,
                        "label": frame.label,
                        "reference": frame.reference,
                    }
                    for frame in normalized.frames
                    if frame.reference
                ]
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

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
    occlusion_report = analysis["occlusion"]
    physics_report = analysis["physics_validation"]
    timing_report = analysis["timing_recommendation"]

    (output / "dynamics.json").write_text(
        json.dumps(dynamics_report, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "occlusion.json").write_text(
        json.dumps(occlusion_report, indent=2) + "\n",
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

    summary = build_diagnostics_summary(
        normalized,
        report,
        analysis,
    )
    (output / "diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    export_render_set(normalized, output, scale=scale)
    export_cutout_set(
        normalized,
        output,
        columns=4,
        preview_scale=max(1, scale),
        piece_dir=piece_dir,
    )
    (output / "transforms.json").write_text(
        json.dumps(
            build_transform_manifest(normalized),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    (output / "animation_manifest.json").write_text(
        json.dumps(
            build_animation_manifest(normalized, columns=4),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    (output / "imagegen_prompt.txt").write_text(
        imagegen_prompt(normalized),
        encoding="utf-8",
    )
    (output / "piecegen_prompt.txt").write_text(
        piecegen_prompt(normalized),
        encoding="utf-8",
    )
    return report.ok and (physics_report["ok"] or not strict_physics)
