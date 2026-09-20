from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import analyze_motion
from .authoring import build_clip_from_recipe_files
from .compiler import compile_motion
from .constraints import normalize_clip, validate_clip
from .model import MotionClip
from .post_validate import validate_rendered_sheet
from .retime_apply import auto_retime
from .resample import resample_clip
from .timeline import densify_clip



def _cmd_author(args: argparse.Namespace) -> int:
    clip = build_clip_from_recipe_files(
        args.rest_motion,
        args.recipe,
    )
    if args.normalize:
        clip = normalize_clip(clip)

    clip.save(args.output)
    print(f"authored: {args.output}")

    if not args.validate:
        return 0

    report = validate_clip(clip)
    payload = {
        "ok": report.ok,
        "issues": [issue.__dict__ for issue in report.issues],
    }
    print(json.dumps(payload, indent=2))
    return 0 if report.ok else 2


def _cmd_compile(args: argparse.Namespace) -> int:
    ok = compile_motion(
        args.input,
        args.output,
        scale=args.scale,
        strict_physics=args.strict_physics,
        auto_retime_iterations=args.auto_retime,
        sample_fps=args.sample_fps,
    )
    print(f"compiled: {args.output}")
    print(f"validation: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


def _cmd_validate(args: argparse.Namespace) -> int:
    clip = MotionClip.load(args.input)
    if args.normalize:
        clip = normalize_clip(clip)
    report = validate_clip(clip)
    payload = {
        "ok": report.ok,
        "issues": [issue.__dict__ for issue in report.issues],
    }
    print(json.dumps(payload, indent=2))
    return 0 if report.ok else 2



def _cmd_analyze(args: argparse.Namespace) -> int:
    clip = MotionClip.load(args.input)
    clip = densify_clip(clip)
    if args.normalize:
        clip = normalize_clip(clip)
    report = analyze_motion(clip)
    print(json.dumps(report, indent=2))
    return 0 if report["physics_validation"]["ok"] else 2



def _cmd_resample(args: argparse.Namespace) -> int:
    clip = MotionClip.load(args.input)
    sampled = resample_clip(clip, args.fps)
    if args.normalize:
        sampled = normalize_clip(sampled)
    sampled.save(args.output)
    print(f"resampled: {args.output}")
    print(f"frames: {len(sampled.frames)}")
    print(f"duration_s: {sampled.times_s()[-1] - sampled.times_s()[0]:.3f}")
    return 0


def _cmd_retime(args: argparse.Namespace) -> int:
    clip = MotionClip.load(args.input)
    clip = densify_clip(clip)
    if args.normalize:
        clip = normalize_clip(clip)

    retimed, history = auto_retime(
        clip,
        iterations=args.iterations,
        tolerance=args.tolerance,
        max_interval_scale=args.max_interval_scale,
    )
    retimed.save(args.output)

    final = analyze_motion(retimed)
    history_path = args.output.with_suffix(args.output.suffix + ".retime.json")
    history_path.write_text(
        json.dumps(
            {
                "history": history,
                "final_physics_validation": final["physics_validation"],
                "final_timing_recommendation": final["timing_recommendation"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"retimed: {args.output}")
    print(f"report: {history_path}")
    print(json.dumps(history[-1], indent=2))
    if args.strict and not final["physics_validation"]["ok"]:
        return 2
    return 0



def _cmd_verify_render(args: argparse.Namespace) -> int:
    clip = MotionClip.load(args.motion)
    clip = normalize_clip(densify_clip(clip))
    report = validate_rendered_sheet(
        args.sheet,
        clip,
        columns=args.columns,
    )

    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"render validation: {args.output}")
    else:
        print(payload)
    return 0 if report["ok"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anima",
        description="Deterministic 2D motion compiler",
    )
    sub = parser.add_subparsers(dest="command", required=True)


    author_cmd = sub.add_parser(
        "author",
        help="Build a full motion clip from a compact parametric pose recipe",
    )
    author_cmd.add_argument(
        "rest_motion",
        type=Path,
        help="Motion file whose first frame defines canonical rig geometry",
    )
    author_cmd.add_argument("recipe", type=Path)
    author_cmd.add_argument("--output", "-o", type=Path, required=True)
    author_cmd.add_argument(
        "--normalize",
        action="store_true",
        help="Solve IK/contacts and rigid geometry before writing output",
    )
    author_cmd.add_argument(
        "--validate",
        action="store_true",
        help="Validate the authored output and return failure on geometry issues",
    )
    author_cmd.set_defaults(func=_cmd_author)

    compile_cmd = sub.add_parser(
        "compile",
        help="Normalize, validate, and render a motion clip",
    )
    compile_cmd.add_argument("input", type=Path)
    compile_cmd.add_argument("--output", "-o", type=Path, required=True)
    compile_cmd.add_argument(
        "--scale",
        type=int,
        default=4,
        help="Nearest-neighbor preview scale",
    )
    compile_cmd.add_argument(
        "--strict-physics",
        action="store_true",
        help="Fail compilation on hard physics plausibility issues",
    )
    compile_cmd.add_argument(
        "--sample-fps",
        type=float,
        default=None,
        metavar="FPS",
        help="Time-resample authored motion to this pose rate before solving/rendering",
    )
    compile_cmd.add_argument(
        "--auto-retime",
        type=int,
        default=0,
        metavar="N",
        help="Run N physics-informed timing refinement passes before rendering",
    )
    compile_cmd.set_defaults(func=_cmd_compile)

    validate_cmd = sub.add_parser("validate", help="Validate a motion clip")
    validate_cmd.add_argument("input", type=Path)
    validate_cmd.add_argument("--normalize", action="store_true")
    validate_cmd.set_defaults(func=_cmd_validate)

    analyze_cmd = sub.add_parser(
        "analyze",
        help="Print full dynamics, physics validation, and timing analysis",
    )
    analyze_cmd.add_argument("input", type=Path)
    analyze_cmd.add_argument("--normalize", action="store_true")
    analyze_cmd.set_defaults(func=_cmd_analyze)

    resample_cmd = sub.add_parser(
        "resample",
        help="Sample authored motion on a real-time Hermite pose grid",
    )
    resample_cmd.add_argument("input", type=Path)
    resample_cmd.add_argument("--fps", type=float, required=True)
    resample_cmd.add_argument("--output", "-o", type=Path, required=True)
    resample_cmd.add_argument("--normalize", action="store_true")
    resample_cmd.set_defaults(func=_cmd_resample)

    retime_cmd = sub.add_parser(
        "retime",
        help="Apply physics-informed phase timing without changing pose geometry",
    )
    retime_cmd.add_argument("input", type=Path)
    retime_cmd.add_argument("--output", "-o", type=Path, required=True)
    retime_cmd.add_argument("--normalize", action="store_true")
    retime_cmd.add_argument("--iterations", type=int, default=3)
    retime_cmd.add_argument("--tolerance", type=float, default=1.01)
    retime_cmd.add_argument("--max-interval-scale", type=float, default=4.0)
    retime_cmd.add_argument(
        "--strict",
        action="store_true",
        help="Return failure if hard physics issues remain after retiming",
    )
    retime_cmd.set_defaults(func=_cmd_retime)

    verify_cmd = sub.add_parser(
        "verify-render",
        help="Compare a rendered spritesheet against solved motion geometry",
    )
    verify_cmd.add_argument("motion", type=Path)
    verify_cmd.add_argument("sheet", type=Path)
    verify_cmd.add_argument("--columns", type=int, default=4)
    verify_cmd.add_argument("--output", "-o", type=Path)
    verify_cmd.set_defaults(func=_cmd_verify_render)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
