import pytest

from anima.layers import DEFAULT_LAYER_ORDER, resolve_layer_order


def test_empty_layer_order_uses_canonical_default():
    assert resolve_layer_order([]) == DEFAULT_LAYER_ORDER


def test_partial_layer_order_reorders_only_named_slots():
    resolved = resolve_layer_order(
        ["left_arm", "torso", "right_arm"]
    )

    assert resolved == (
        "left_leg",
        "right_leg",
        "left_arm",
        "torso",
        "right_arm",
        "head",
        "weapon",
    )


def test_layer_order_rejects_duplicates_and_unknown_parts():
    with pytest.raises(ValueError):
        resolve_layer_order(["left_arm", "left_arm"])

    with pytest.raises(ValueError):
        resolve_layer_order(["cape"])
