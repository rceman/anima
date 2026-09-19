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
from .retiming import recommend_timing
from .strength import analyze_arm_strength
from .system_dynamics import analyze_system_dynamics
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

    body_report = analyze_body_kinematics(normalized)
    weapon_report = analyze_weapon_dynamics(normalized)
    dynamics_report = {
        "weapon": weapon_report,
        "body": body_report,
        "strength": analyze_arm_strength(normalized, weapon_report),
        "system": analyze_system_dynamics(
            normalized,
            body_report,
            weapon_report,
        ),
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

    timing_report = recommend_timing(normalized, dynamics_report)
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
