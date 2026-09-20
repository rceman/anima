from __future__ import annotations

from collections import defaultdict
from typing import Any


ISSUE_HINTS: dict[str, str] = {
    "sheet_size_mismatch": (
        "Return the exact requested spritesheet dimensions and panel grid."
    ),
    "background_drift": (
        "Use exactly the same flat background color as frame 0."
    ),
    "character_center_drift": (
        "Move the character back onto the supplied control pose; do not recenter "
        "the panel independently."
    ),
    "character_scale_drift": (
        "Restore the character to the same scale/proportions as the control pose."
    ),
    "pose_mask_miss": (
        "Redraw the silhouette so it follows the supplied control mannequin more "
        "closely, especially the missing limbs/torso regions."
    ),
    "pose_mask_excess": (
        "Remove or reposition foreground pixels that do not belong near the "
        "supplied control pose."
    ),
    "anchor_missing": (
        "Restore the missing body joint region at its control position."
    ),
    "sword_tip_drift": (
        "Place the sword tip at the exact control position and preserve blade length."
    ),
    "sword_path_missing": (
        "Restore one continuous rigid sword blade from the primary grip to the "
        "control sword tip."
    ),
    "ground_contact_missing": (
        "Put the indicated foot back onto the exact ground baseline."
    ),
    "missing_foreground": (
        "Restore the character in this panel."
    ),
}


def _hint(issue: dict[str, Any]) -> str:
    code = str(issue.get("code", "unknown"))
    base = ISSUE_HINTS.get(
        code,
        str(issue.get("message", code)),
    )
    anchor = issue.get("anchor")
    foot = issue.get("foot")
    suffix: list[str] = []
    if anchor:
        suffix.append(f"anchor={anchor}")
    if foot:
        suffix.append(f"foot={foot}")
    if suffix:
        return f"{base} ({', '.join(suffix)})"
    return base


def build_render_correction_prompt(
    report: dict[str, Any],
) -> str:
    """Build a minimal ImageGen repair prompt from geometric validation issues."""
    issues = list(report.get("issues", []))
    if not issues:
        return (
            "The spritesheet passes Anima geometric validation. "
            "No correction is required."
        )

    sheet_issues = [
        issue
        for issue in issues
        if issue.get("frame") is None
    ]
    frame_issues: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for issue in issues:
        if issue.get("frame") is None:
            continue
        frame_issues[int(issue["frame"])].append(issue)

    lines = [
        "Correct the existing pixel-art spritesheet using the original control "
        "sheet and semantic guides as geometry authority.",
        "",
        "NON-NEGOTIABLE:",
        "- Do not redesign the character.",
        "- Do not change any panel that is not listed below.",
        "- Preserve the exact grid, panel size, camera, scale, palette/style, "
        "two-handed grip, rigid sword, and ground baseline.",
        "- Do not add VFX, motion blur, anti-aliasing, or new scenery.",
        "",
    ]

    if sheet_issues:
        lines.append("WHOLE-SHEET CORRECTIONS:")
        for issue in sheet_issues:
            lines.append(f"- {_hint(issue)}")
        lines.append("")

    if frame_issues:
        lines.append("FRAME CORRECTIONS:")
        for frame in sorted(frame_issues):
            unique: list[str] = []
            for issue in frame_issues[frame]:
                hint = _hint(issue)
                if hint not in unique:
                    unique.append(hint)
            lines.append(f"- Frame {frame}:")
            for hint in unique:
                lines.append(f"  - {hint}")

    lines.extend(
        [
            "",
            "Return the corrected spritesheet only. Preserve all already-correct "
            "frames pixel-for-pixel where possible.",
        ]
    )
    return "\n".join(lines) + "\n"
