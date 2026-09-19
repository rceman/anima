# Anima

Anima is a deterministic **2D motion compiler** for creating consistent control spritesheets that can be passed to an image generator for final pixel-art rendering.

The core principle is:

> **Motion is math and constraints. Appearance is a separate rendering problem.**

Anima intentionally does **not** include video extraction or pose detection. A human, ChatGPT, another model, a 3D tool, or any external process can author `motion.json`. Anima normalizes that motion into a stable rig, enforces hard animation invariants, and produces control images for downstream generative rendering.

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
                         motion.json
                              |
                              v
                   +---------------------+
                   |        Anima        |
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

Outputs:

```text
motion.normalized.json
validation.json
debug_frames/
control_frames/
debug_sheet.png
control_sheet.png
debug.gif
control.gif
```

Validate only:

```bash
anima validate examples/twohand_sword_slash/motion.json --normalize
```

## Motion authoring model

`motion.json` stores semantic geometry rather than rendered body parts:

- fixed 128x128 canvas;
- explicit ground Y;
- root and named 2D joints;
- per-frame foot-contact state;
- primary/off-hand weapon grips;
- sword tip;
- frame labels and timing.

The first frame defines the canonical limb lengths, sword length, and grip spacing. During compilation Anima uses those values as invariants.

The two-handed sword normalization order is deliberately dependency-driven:

```text
ground/root
    -> rigid sword geometry
        -> both hand targets
            -> arm IK
                -> planted foot targets
                    -> leg IK
```

That order prevents the exact failure modes that motivated the project: floating feet, weapon morphing, and mixed one/two-handed swings.

## Image-generation handoff

Recommended inputs to an image generator:

1. **`control_sheet.png`** — authoritative pose and geometry reference.
2. **one canonical character image** — authoritative appearance/style reference.
3. a prompt stating that panel geometry, ground contact, grip, sword direction, character scale, and pose must be preserved exactly.

Image generation is downstream. It should render the character **onto motion that Anima has already solved**, not invent the animation itself.

A later validation stage can split the generated sheet and compare silhouette, baseline contact, character scale, and weapon angle against Anima's control frames.

## Canonical first target

`examples/twohand_sword_slash/motion.json` is the gold motion case:

- right-facing;
- two-handed grip in every frame;
- both feet grounded;
- rigid sword;
- eight poses: ready -> anticipation -> maximum wind-up -> attack start -> impact -> follow-through -> recovery -> return.

## Status

### 0.1.0 foundation

Implemented:

- motion JSON model;
- canonical rest geometry;
- ground locking;
- rigid sword normalization;
- hard two-handed attachment;
- two-bone arm and leg IK;
- planted-foot reach clamping while preserving the ground line;
- geometry validation;
- colorblind-safe debug rendering with redundant left/right marker shapes;
- mannequin control rendering;
- PNG frame, spritesheet, and GIF export;
- CLI;
- tests;
- GitHub Actions test/compile gate;
- canonical two-handed sword example.

Next work should focus on motion quality rather than adding video extraction: interpolation/easing, keyframe-only authoring, weapon-arc helpers, root/center-of-mass constraints, and post-ImageGen comparison.
