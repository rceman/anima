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
- If a semantic body-parts sheet is supplied, use it to disambiguate which pixels belong to the left/right arms, left/right legs, torso/head, and sword. Its colors are labels, not final character colors.
- Semantic layer order is authoritative for occlusion. When an arm, torso, leg, head, or weapon crosses another part, preserve which part is behind and which part is in front in that panel. Never swap left/right limb depth between frames unless the supplied semantic guides do so.
- If transparent per-part layer masks are supplied, treat them as additional geometry/occlusion constraints rather than as appearance references.
- If a deterministic cutout sheet is supplied, preserve its limb lengths, joint locations, sword transform, root motion, and front/back ordering. A procedural cutout's colors are diagnostic labels, not final appearance.
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
Return one spritesheet with the exact same panel order and grid geometry as the supplied control spritesheet. Change only the mannequin rendering into the master character's final pixel-art appearance. The semantic body-part colors must not appear in the final art.
"""



def piecegen_prompt(clip: MotionClip) -> str:
    """Prompt contract for painting one reusable bind-pose cutout pack."""
    pieces = (
        "upper_arm_l, forearm_l, upper_arm_r, forearm_r, "
        "thigh_l, shin_l, thigh_r, shin_r, torso, head, weapon"
    )
    return f"""Create a reusable pixel-art bind-pose piece pack for the supplied master character.

CANVAS CONTRACT:
- Every output piece is exactly {clip.width}x{clip.height} pixels.
- Transparent RGBA background.
- Keep the master character in the exact supplied bind-pose coordinates.
- Do not crop, center, resize, rotate, or move a piece within its canvas.
- Crisp pixel edges only; no anti-aliasing.

REQUIRED PIECES:
{pieces}

LAYER RULE:
- Each PNG contains exactly one named piece and nothing from neighboring pieces.
- Preserve clothing/armor details that physically belong to that piece.
- Joint overlap pixels may extend slightly past the mathematical bone endpoint so transformed pieces do not open visible gaps.
- The weapon is one rigid piece and must include its complete handle, guard, and blade.
- Left/right identity is anatomical and must never be mirrored or swapped.

GEOMETRY AUTHORITY:
Use the supplied bind-piece masks / piece_manifest.json as exact position and extent guidance. Use the master-character image only for appearance, material, clothing, palette, and pixel style.

OUTPUT CONTRACT:
Produce the named full-canvas transparent PNG pieces so Anima can animate them deterministically using transforms.json. Do not create animation frames; paint the character pieces once.
"""
