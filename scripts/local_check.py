#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anima.authoring import build_clip_from_recipe_files  # noqa: E402
from anima.compiler import compile_motion  # noqa: E402
from anima.constraints import normalize_clip, validate_clip  # noqa: E402
from anima.model import MotionClip  # noqa: E402
from anima.post_validate import validate_rendered_sheet  # noqa: E402


@dataclass
class Check:
    name: str
    ok: bool
    details: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_tests() -> Check:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return Check(
        name="pytest",
        ok=completed.returncode == 0,
        details={
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        },
    )


def _assert_image_size(path: Path, expected: tuple[int, int]) -> dict[str, Any]:
    with Image.open(path) as image:
        actual = image.size
        mode = image.mode
    return {
        "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "actual": list(actual),
        "expected": list(expected),
        "mode": mode,
        "ok": actual == expected,
    }


def _artifact_checks(output: Path, frame_count: int) -> Check:
    columns = 4
    rows = (frame_count + columns - 1) // columns
    cell = (128, 128)
    sheet = (columns * cell[0], rows * cell[1])

    required = [
        output / "motion.normalized.json",
        output / "validation.json",
        output / "dynamics.json",
        output / "occlusion.json",
        output / "physics_validation.json",
        output / "timing_recommendation.json",
        output / "diagnostics_summary.json",
        output / "animation_manifest.json",
        output / "transforms.json",
        output / "reference_map.json",
        output / "piece_manifest.json",
        output / "imagegen_prompt.txt",
        output / "piecegen_prompt.txt",
        output / "control_sheet.png",
        output / "parts_sheet.png",
        output / "cutout_sheet.png",
        output / "bind_piece_sheet.png",
        output / "debug_sheet.png",
        output / "control_preview.gif",
        output / "parts_preview.gif",
        output / "cutout_preview.gif",
        output / "rig_review_preview.gif",
        output / "inspection_preview.gif",
    ]
    missing = [str(path) for path in required if not path.exists()]

    image_checks = []
    if not missing:
        image_checks = [
            _assert_image_size(output / "control_sheet.png", sheet),
            _assert_image_size(output / "parts_sheet.png", sheet),
            _assert_image_size(output / "cutout_sheet.png", sheet),
            _assert_image_size(
                output / "bind_piece_sheet.png",
                (512, 384),
            ),
        ]
        for index in range(frame_count):
            image_checks.append(
                _assert_image_size(
                    output / "control_frames" / f"frame_{index:02d}.png",
                    cell,
                )
            )
            image_checks.append(
                _assert_image_size(
                    output / "parts_frames" / f"frame_{index:02d}.png",
                    cell,
                )
            )
            image_checks.append(
                _assert_image_size(
                    output / "cutout_frames" / f"frame_{index:02d}.png",
                    cell,
                )
            )

    ok = not missing and all(item["ok"] for item in image_checks)
    return Check(
        name="artifacts",
        ok=ok,
        details={
            "missing": missing,
            "images": image_checks,
        },
    )


def _compile_canonical(
    *,
    output: Path,
    auto_retime: int,
    sample_fps: float | None,
) -> tuple[Check, MotionClip]:
    canonical = ROOT / "examples" / "twohand_sword_slash"
    rest_motion = canonical / "motion.json"
    recipe = canonical / "recipe.json"
    authored_path = output / "motion.authored.json"

    authored = build_clip_from_recipe_files(rest_motion, recipe)
    normalized_authored = normalize_clip(authored)
    authored_report = validate_clip(normalized_authored)
    normalized_authored.save(authored_path)

    compile_ok = compile_motion(
        authored_path,
        output,
        scale=1,
        strict_physics=False,
        auto_retime_iterations=auto_retime,
        sample_fps=sample_fps,
    )

    geometry = _load_json(output / "validation.json")
    physics = _load_json(output / "physics_validation.json")
    occlusion = _load_json(output / "occlusion.json")
    diagnostics = _load_json(output / "diagnostics_summary.json")
    normalized = MotionClip.load(output / "motion.normalized.json")

    return (
        Check(
            name="canonical_compile",
            ok=bool(compile_ok and authored_report.ok and geometry["ok"]),
            details={
                "recipe_geometry_ok": authored_report.ok,
                "recipe_geometry_issues": [
                    issue.__dict__ for issue in authored_report.issues
                ],
                "compiled_geometry_ok": geometry["ok"],
                "compiled_geometry_issues": geometry["issues"],
                "physics_ok": physics["ok"],
                "physics_hard_issues": physics["hard_issues"],
                "physics_warnings": physics["warnings"],
                "occlusion_warnings": occlusion.get("warnings", []),
                "diagnostics": diagnostics,
            },
        ),
        normalized,
    )


def _self_validate_render(output: Path, clip: MotionClip) -> Check:
    report = validate_rendered_sheet(
        output / "control_sheet.png",
        clip,
        columns=4,
    )
    (output / "control_render_validation.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return Check(
        name="control_render_self_validation",
        ok=bool(report["ok"]),
        details=report,
    )


def _golden_motion_checks(output: Path) -> Check:
    dynamics = _load_json(output / "dynamics.json")
    manifest = _load_json(output / "animation_manifest.json")
    physics = _load_json(output / "physics_validation.json")

    weapon_frames = {
        item.get("label"): item
        for item in dynamics["weapon"]["frames"]
        if item.get("label")
    }

    failures: list[str] = []

    ready = weapon_frames.get("ready")
    attack_start = weapon_frames.get("attack_start")
    impact = weapon_frames.get("impact")
    follow = weapon_frames.get("follow_through")
    returned = weapon_frames.get("return")

    if ready is None or returned is None:
        failures.append("ready/return labeled frames missing")
    else:
        if abs(float(ready["angular_velocity_deg_s"])) > 1e-9:
            failures.append("ready is not a true zero-velocity rest")
        if abs(float(returned["angular_velocity_deg_s"])) > 1e-9:
            failures.append("return is not a true zero-velocity rest")

    if attack_start is None or impact is None or follow is None:
        failures.append("attack_start/impact/follow_through labels missing")
    else:
        attack_speed = abs(float(attack_start["angular_velocity_deg_s"]))
        impact_speed = abs(float(impact["angular_velocity_deg_s"]))
        follow_speed = abs(float(follow["angular_velocity_deg_s"]))

        if impact_speed < attack_speed * 0.98:
            failures.append(
                "weapon speed peaks before impact "
                f"(attack={attack_speed:.2f}, impact={impact_speed:.2f})"
            )
        if follow_speed <= 0.0:
            failures.append("follow-through has zero weapon speed")
        if follow_speed >= impact_speed:
            failures.append(
                "follow-through is not braking after impact "
                f"(impact={impact_speed:.2f}, follow={follow_speed:.2f})"
            )

    frames = manifest.get("frames", [])
    if not frames:
        failures.append("animation manifest has no frames")
    else:
        if frames[0].get("kinematic_stop") is not True:
            failures.append("first manifest frame is not marked kinematic_stop")
        if frames[-1].get("kinematic_stop") is not True:
            failures.append("last manifest frame is not marked kinematic_stop")

    return Check(
        name="golden_motion",
        ok=not failures,
        details={
            "failures": failures,
            "physics_ok": physics["ok"],
            "hard_issue_codes": sorted(
                {
                    issue["code"]
                    for issue in physics.get("hard_issues", [])
                }
            ),
        },
    )


def _print_result(check: Check) -> None:
    marker = "PASS" if check.ok else "FAIL"
    print(f"[{marker}] {check.name}")

    if check.name == "pytest" and not check.ok:
        stdout = check.details.get("stdout", "").strip()
        stderr = check.details.get("stderr", "").strip()
        if stdout:
            print(stdout)
        if stderr:
            print(stderr, file=sys.stderr)

    if check.name == "canonical_compile":
        details = check.details
        print(
            "       geometry="
            f"{details['compiled_geometry_ok']} "
            "physics="
            f"{details['physics_ok']} "
            "occlusion_warnings="
            f"{len(details['occlusion_warnings'])}"
        )

    if check.name == "golden_motion":
        for failure in check.details.get("failures", []):
            print(f"       - {failure}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Anima's complete local validation/render loop. "
            "No GitHub Actions or network access is used."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output" / "local-check",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=None,
        help="Optionally time-resample the canonical motion before rendering",
    )
    parser.add_argument(
        "--auto-retime",
        type=int,
        default=3,
        help="Physics retiming passes for canonical compile",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip pytest and run only the canonical local pipeline",
    )
    parser.add_argument(
        "--strict-physics",
        action="store_true",
        help="Fail the local check when hard physics issues remain",
    )
    parser.add_argument(
        "--strict-occlusion",
        action="store_true",
        help="Fail when semantic occlusion warnings remain",
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Do not clear the output directory before the run",
    )
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists() and not args.keep_output:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    checks: list[Check] = []

    if not args.skip_tests:
        checks.append(_run_tests())
        _print_result(checks[-1])

    compile_check, normalized = _compile_canonical(
        output=output,
        auto_retime=args.auto_retime,
        sample_fps=args.sample_fps,
    )
    checks.append(compile_check)
    _print_result(compile_check)

    artifact_check = _artifact_checks(
        output,
        frame_count=len(normalized.frames),
    )
    checks.append(artifact_check)
    _print_result(artifact_check)

    render_check = _self_validate_render(output, normalized)
    checks.append(render_check)
    _print_result(render_check)

    golden_check = _golden_motion_checks(output)
    checks.append(golden_check)
    _print_result(golden_check)

    physics = _load_json(output / "physics_validation.json")
    occlusion = _load_json(output / "occlusion.json")

    policy_failures: list[str] = []
    if args.strict_physics and not physics["ok"]:
        policy_failures.append("hard physics issues remain")
    if args.strict_occlusion and occlusion.get("warnings"):
        policy_failures.append("semantic occlusion warnings remain")

    policy_check = Check(
        name="strict_policy",
        ok=not policy_failures,
        details={"failures": policy_failures},
    )
    checks.append(policy_check)
    _print_result(policy_check)

    payload = {
        "ok": all(check.ok for check in checks),
        "output": str(output),
        "checks": [asdict(check) for check in checks],
    }
    (output / "local_check_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"output: {output}")
    print(
        "result: "
        + ("PASS" if payload["ok"] else "FAIL")
    )
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
