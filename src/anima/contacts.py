from __future__ import annotations

from typing import Any

FREE = "free"
GROUNDED = "grounded"
PLANTED = "planted"

VALID_MODES = {FREE, GROUNDED, PLANTED}


def contact_mode(value: Any) -> str:
    """Normalize legacy booleans and explicit contact modes.

    true  -> planted (world-space lock)
    false -> free
    """
    if value is True:
        return PLANTED
    if value is False or value is None:
        return FREE

    mode = str(value).lower()
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported contact mode: {value!r}")
    return mode


def is_ground_contact(value: Any) -> bool:
    return contact_mode(value) in {GROUNDED, PLANTED}


def is_planted(value: Any) -> bool:
    return contact_mode(value) == PLANTED
