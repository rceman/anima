from __future__ import annotations

from typing import Any


HARD_CODES = {
    "planted_foot_slip",
    "weapon_direction_reversal",
    "implausible_braking_torque",
    "insufficient_follow_through",
    "friction_limit_exceeded",
}

SOFT_CODES = {
    "com_outside_support",
    "com_acceleration_high",
    "com_jerk_high",
}


def evaluate_physics(dynamics_report: dict[str, Any]) -> dict[str, Any]:
    """Classify physics diagnostics into hard failures and soft warnings."""
    body_warnings = list(dynamics_report.get("body", {}).get("warnings", []))
    weapon_warnings = list(dynamics_report.get("weapon", {}).get("warnings", []))

    all_warnings = [
        {**warning, "domain": "body"} for warning in body_warnings
    ] + [
        {**warning, "domain": "weapon"} for warning in weapon_warnings
    ]

    hard = [warning for warning in all_warnings if warning.get("code") in HARD_CODES]
    soft = [
        warning
        for warning in all_warnings
        if warning.get("code") not in HARD_CODES
    ]

    return {
        "ok": not hard,
        "hard_issues": hard,
        "warnings": soft,
        "counts": {
            "hard": len(hard),
            "warnings": len(soft),
            "total": len(all_warnings),
        },
    }
