from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compiler import compile_motion
from .constraints import normalize_clip, validate_clip
from .model import MotionClip


def _cmd_compile(args: argparse.Namespace) -> int:
    ok = compile_motion(args.input, args.output, scale=args.scale)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anima",
        description="Deterministic 2D motion compiler",
    )
    sub = parser.add_subparsers(dest="command", required=True)

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
    compile_cmd.set_defaults(func=_cmd_compile)

    validate_cmd = sub.add_parser("validate", help="Validate a motion clip")
    validate_cmd.add_argument("input", type=Path)
    validate_cmd.add_argument("--normalize", action="store_true")
    validate_cmd.set_defaults(func=_cmd_validate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
