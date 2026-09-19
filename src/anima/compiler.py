from __future__ import annotations

import json
from pathlib import Path

from .biomechanics import analyze_body_kinematics
from .constraints import normalize_clip, validate_clip
from .dynamics import analyze_weapon_dynamics
from .handoff import imagegen_prompt
from .model import MotionClip
from .physics_policy import evaluate_physics
from .render import export_render_set
from .timeline import densify_clip


def compile_motion(
    input_path: str | Path,
    output_dir: str | Path,
    scale: int = 4,
    strict_physics: bool = False,
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

    dynamics_report = {
        "weapon": analyze_weapon_dynamics(normalized),
        "body": analyze_body_kinematics(normalized),
    }
    (output / "dynamics.json").write_text(
        json.dumps(dynamics_report, indent=2) + "\n",
        encoding="utf-8",
    )

    physics_report = evaluate_physics(dynamics_report)
    (output / "physics_validation.json").write_text(
        json.dumps(physics_report, indent=2) + "\n",
        encoding="utf-8",
    )

    export_render_set(normalized, output, scale=scale)
    (output / "imagegen_prompt.txt").write_text(
        imagegen_prompt(normalized),
        encoding="utf-8",
    )
    return report.ok and (physics_report["ok"] or not strict_physics)
