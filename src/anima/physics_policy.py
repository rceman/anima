from __future__ import annotations

from typing import Any


HARD_CODES = {
    "planted_foot_slip",
    "weapon_direction_reversal",
    "implausible_braking_torque",
    "insufficient_follow_through",
    "friction_limit_exceeded",
    "joint_limit_violation",
    "handle_force_exceeded",
    "drive_torque_limit_exceeded",
    "collision_response_mismatch",
    "system_friction_limit_exceeded",
    "joint_torque_limit_exceeded",
    "dynamic_balance_limit_exceeded",
}

SOFT_CODES = {
    "com_outside_support",
    "com_acceleration_high",
    "com_jerk_high",
    "system_com_outside_support",
    "dynamic_balance_outside_support",
}


def evaluate_physics(dynamics_report: dict[str, Any]) -> dict[str, Any]:
    """Classify physics diagnostics into hard failures and soft warnings."""
    body_warnings = list(dynamics_report.get("body", {}).get("warnings", []))
    weapon_warnings = list(dynamics_report.get("weapon", {}).get("warnings", []))
    system_warnings = list(dynamics_report.get("system", {}).get("warnings", []))
    strength_warnings = list(dynamics_report.get("strength", {}).get("warnings", []))
    coordination_warnings = list(
        dynamics_report.get("coordination", {}).get("warnings", [])
    )

    all_warnings = [
        {**warning, "domain": "body"} for warning in body_warnings
    ] + [
        {**warning, "domain": "weapon"} for warning in weapon_warnings
    ] + [
        {**warning, "domain": "system"} for warning in system_warnings
    ] + [
        {**warning, "domain": "strength"} for warning in strength_warnings
    ] + [
        {**warning, "domain": "coordination"} for warning in coordination_warnings
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
