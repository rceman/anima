from __future__ import annotations

from typing import Iterable


SEMANTIC_LAYERS: tuple[str, ...] = (
    "left_leg",
    "right_leg",
    "torso",
    "left_arm",
    "right_arm",
    "head",
    "weapon",
)

DEFAULT_LAYER_ORDER: tuple[str, ...] = SEMANTIC_LAYERS


def resolve_layer_order(order: Iterable[str] | None) -> tuple[str, ...]:
    """Resolve a partial per-frame z-order into one complete deterministic order.

    Earlier entries are drawn first (behind); later entries are drawn last
    (in front). Authors only need to name layers whose relative order differs
    from the default. Missing layers are appended in canonical order.
    """
    if order is None:
        return DEFAULT_LAYER_ORDER

    requested = list(order)
    unknown = [name for name in requested if name not in SEMANTIC_LAYERS]
    if unknown:
        raise ValueError(
            "Unknown semantic layer(s): " + ", ".join(sorted(set(unknown)))
        )

    if len(set(requested)) != len(requested):
        raise ValueError("Semantic layer order contains duplicates")

    if not requested:
        return DEFAULT_LAYER_ORDER

    # Treat a partial order as a local permutation of the named layers while
    # every unnamed layer keeps its canonical z-slot. Example:
    # default:   legs, torso, left_arm, right_arm, head, weapon
    # requested: left_arm, torso, right_arm
    # result:    legs, left_arm, torso, right_arm, head, weapon
    #
    # This lets an author express the exact crossing relation they care about
    # without having to repeat all seven semantic layers in every frame.
    result = list(DEFAULT_LAYER_ORDER)
    slots = sorted(
        DEFAULT_LAYER_ORDER.index(name)
        for name in requested
    )
    for slot, name in zip(slots, requested):
        result[slot] = name
    return tuple(result)


def validate_layer_order(order: Iterable[str] | None) -> list[str]:
    """Return human-readable validation errors without raising."""
    try:
        resolve_layer_order(order)
    except ValueError as exc:
        return [str(exc)]
    return []
