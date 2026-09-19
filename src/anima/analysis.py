from __future__ import annotations

from typing import Any

from .biomechanics import analyze_body_kinematics
from .dynamics import analyze_weapon_dynamics
from .model import MotionClip
from .physics_policy import evaluate_physics
from .retiming import recommend_timing
from .strength import analyze_arm_strength
from .system_dynamics import analyze_system_dynamics


def analyze_motion(clip: MotionClip) -> dict[str, Any]:
    body = analyze_body_kinematics(clip)
    weapon = analyze_weapon_dynamics(clip)
    strength = analyze_arm_strength(clip, weapon)
    system = analyze_system_dynamics(clip, body, weapon)

    dynamics = {
        "body": body,
        "weapon": weapon,
        "strength": strength,
        "system": system,
    }

    return {
        "dynamics": dynamics,
        "physics_validation": evaluate_physics(dynamics),
        "timing_recommendation": recommend_timing(clip, dynamics),
    }
