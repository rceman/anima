# Anima

Anima is a deterministic **2D motion compiler** for creating consistent control spritesheets that can be passed to an image generator for final pixel-art rendering.

The core principle is:

> **Motion is math and constraints. Appearance is a separate rendering problem.**

Anima intentionally does **not** include video extraction or pose detection. A human, ChatGPT, another model, a 3D tool, or any external process can author motion semantics. Anima normalizes that motion into a stable rig, enforces hard animation invariants, and produces control images for downstream generative rendering.

## Why

Whole-spritesheet image generation tends to introduce frame drift: floating feet, changing proportions, inconsistent sword length, or one-handed/two-handed grip changes. Anima makes those properties explicit and machine-checkable.

The initial `humanoid_twohand_sword_v1` workflow enforces:

- one explicit ground baseline;
- fixed limb lengths derived from the rest frame;
- rigid sword length and grip spacing;
- primary and secondary hands attached to the same weapon every frame;
- two-bone IK for elbows and knees;
- fixed canvas bounds;
- validation before a control sheet is considered usable.

## Pipeline

```text
video / GIF / 3D animation / hand-authored motion / model reasoning
                              |
                              v
                     sparse motion keyframes
                              |
                              v
                   +---------------------+
                   |        Anima        |
                   | interpolate         |
                   | normalize + IK      |
                   | constraints         |
                   | validation          |
                   +----------+----------+
                              |
                +-------------+-------------+
                |                           |
                v                           v
         debug_sheet.png             control_sheet.png
                                            |
                                            +--> imagegen_prompt.txt
                                            |
                                            v
                                      Image generator
                                            |
                                            v
                                  final pixel spritesheet
```

The **debug sheet** exposes joints, left/right limb identity, ground contacts, and weapon geometry. The **control sheet** is a thicker mannequin representation intended to communicate pose and silhouette to an image generator without asking it to invent the motion.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Compile the reference sword slash

```bash
anima compile examples/twohand_sword_slash/motion.json \
  --output /tmp/anima-sword \
  --scale 4
```

Canonical outputs:

```text
motion.normalized.json
validation.json
imagegen_prompt.txt

debug_frames/          # exact canvas size: 128x128 in the example
control_frames/        # exact canvas size: 128x128 in the example

debug_sheet.png        # canonical grid, no scaling
control_sheet.png      # canonical grid, no scaling
```

Preview-only outputs:

```text
debug_preview.gif
control_preview.gif
debug_sheet.preview.png
control_sheet.preview.png
```

The `--scale` argument affects only preview assets. **Canonical frame files never change size.** For the current sword example each cell is always exactly **128x128**, and the 4x2 canonical control sheet is exactly **512x256**.

Validate only:

```bash
anima validate examples/twohand_sword_slash/motion.json --normalize
```

## Motion authoring model

`motion.json` stores semantic geometry rather than rendered body parts:

- fixed canvas and explicit ground Y;
- root and named 2D joints;
- per-frame foot-contact state;
- primary/off-hand weapon grips;
- sword tip;
- sparse keyframe numbers;
- frame labels and timing.

Keyframes do not need to be adjacent. If an author supplies frames `0`, `3`, and `7`, Anima fills frames `1`, `2`, `4`, `5`, and `6` using smoothstep interpolation before constraint solving. The interpolated positions are only proposals: the rig/IK layer still enforces the actual invariants afterward.

The first frame defines the canonical limb lengths, sword length, and grip spacing.

The two-handed sword normalization order is dependency-driven:

```text
sparse keyframes
    -> interpolation
        -> ground/root alignment
            -> rigid sword geometry
                -> both hand targets
                    -> arm IK
                        -> planted foot targets
                            -> leg IK
                                -> validation
```

That order prevents the exact failure modes that motivated the project: floating feet, weapon morphing, and mixed one/two-handed swings.

## Image-generation handoff

Recommended inputs to an image generator:

1. **`control_sheet.png`** — authoritative pose and geometry reference.
2. **one canonical character image** — authoritative appearance/style reference.
3. **`imagegen_prompt.txt`** — generated handoff contract.

Image generation is downstream. It should render the character **onto motion that Anima has already solved**, not invent the animation itself.

The generated prompt explicitly locks:

- panel geometry and order;
- character scale;
- two-handed grip;
- rigid sword geometry;
- ground contact;
- body proportions;
- uniform background;
- no extra VFX or camera changes.

A later validation stage can split the generated sheet and compare silhouette, baseline contact, character scale, and weapon angle against Anima's control frames.

## Canonical first target

`examples/twohand_sword_slash/motion.json` is the first gold motion case:

- right-facing;
- two-handed grip in every frame;
- both feet grounded;
- rigid sword;
- eight poses: ready -> anticipation -> maximum wind-up -> attack start -> impact -> follow-through -> recovery -> return.

## Status

### 0.1.0 foundation

Implemented:

- motion JSON model;
- sparse keyframe densification with smoothstep interpolation;
- canonical rest geometry;
- ground locking;
- rigid sword normalization;
- hard two-handed attachment;
- two-bone arm and leg IK;
- planted-foot reach clamping while preserving the ground line;
- geometry validation;
- colorblind-safe debug rendering with redundant left/right marker shapes;
- mannequin control rendering;
- exact-size canonical PNG frame and spritesheet export;
- separately scaled preview GIF/sheets;
- generated ImageGen handoff prompt;
- CLI;
- tests;
- GitHub Actions test/compile gate;
- canonical two-handed sword example.

Next work should focus on motion quality: per-segment easing, explicit weapon-arc helpers, planted-foot phase changes, root/center-of-mass constraints, and post-ImageGen comparison.
