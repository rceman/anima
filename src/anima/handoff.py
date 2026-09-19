from __future__ import annotations

from .model import MotionClip


def imagegen_prompt(clip: MotionClip) -> str:
    name = clip.metadata.get("name", "this animation")
    return f"""Render the supplied control spritesheet as one consistent pixel-art character animation.

ANIMATION: {name}
CANVAS: each panel is exactly {clip.width}x{clip.height} pixels.
GROUND: preserve the single ground baseline and all planted foot contacts exactly.

AUTHORITY RULES:
- The control spritesheet is authoritative for pose, position, proportions, hand placement, sword direction, and timing order.
- The separate master-character image is authoritative for appearance, clothing, palette, and pixel-art style.
- Do not invent or reinterpret poses.
- Every panel depicts exactly the same character at exactly the same scale.
- Preserve the two-handed grip in every panel: both hands remain attached to the same sword handle.
- Preserve one rigid sword with constant blade length and constant handle/grip spacing.
- Preserve foot placement and the ground baseline; do not make the character float.
- Preserve head, torso, arm, and leg proportions across every panel.
- Do not add or remove limbs, hands, weapons, clothing, or equipment.
- Do not add motion blur, anti-aliasing, camera movement, perspective changes, lighting changes, particles, elemental effects, or extra scenery.
- Use crisp nearest-neighbor-style pixel edges.
- Keep the background uniform and unchanged across all panels.

OUTPUT:
Return one spritesheet with the exact same panel order and grid geometry as the supplied control spritesheet. Change only the mannequin rendering into the master character's final pixel-art appearance.
"""
