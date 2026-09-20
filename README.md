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
- explicit IK pole targets and temporal bend continuity to prevent elbow/knee flips;
- physically-inspired weapon inertia diagnostics for follow-through and braking;
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
                |                           |
                |                    parts_sheet.png
                |                           |
                +-------------+-------------+
                              |
                              +--> imagegen_prompt.txt
                              |
                              v
                        Image generator
                                            |
                                            v
                                  final pixel spritesheet
```

The **debug sheet** exposes joints, left/right limb identity, contact state, IK poles, motion trails, center-of-mass motion, support, and weapon dynamics. The **control sheet** is a one-color mannequin pose guide. The **parts sheet** is a semantic layered guide that identifies torso/head, left/right arms, left/right legs, and weapon separately; those colors are labels only and are never final art colors.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Local development loop

Development is intentionally local-only. The repository does **not** use GitHub Actions.

Run the complete deterministic check/render loop from the repository root:

```bash
python scripts/local_check.py
```

This runs the unit tests, rebuilds the canonical motion from its compact recipe,
normalizes and compiles it, generates all debug/control/semantic assets,
self-validates the control sheet, and checks the golden sword-motion invariants.

For the stricter release-quality gate:

```bash
python scripts/local_check.py --strict-physics --strict-occlusion
```

For fast render/physics iteration without pytest:

```bash
python scripts/local_check.py --skip-tests --keep-output
```

All artifacts are written under `output/local-check/` by default, including
`local_check_summary.json`. No network or CI service is required.

## Compile the reference sword slash

```bash
anima compile examples/twohand_sword_slash/motion.json \
  --output /tmp/anima-sword \
  --scale 4 \
  --auto-retime 3 \
  --strict-physics
```

Canonical outputs:

```text
motion.normalized.json
validation.json
dynamics.json
occlusion.json
physics_validation.json
timing_recommendation.json
diagnostics_summary.json
retime_history.json
imagegen_prompt.txt

debug_frames/          # exact canvas size: 128x128 in the example
control_frames/        # exact canvas size: 128x128 in the example
parts_frames/          # semantic body-part guide, same exact canvas size

debug_sheet.png        # canonical grid, no scaling
control_sheet.png      # canonical one-color pose guide
parts_sheet.png        # canonical semantic/layered guide
```

Preview-only outputs:

```text
inspection_preview.gif     # preferred large diagnostic view with metrics sidebar
debug_preview.gif          # dense diagnostic overlay
control_preview.gif
parts_preview.gif
review_preview.gif         # debug skeleton + mannequin side-by-side
debug_sheet.preview.png
control_sheet.preview.png
parts_sheet.preview.png
```

The `--scale` argument affects only preview assets. **Canonical frame files never change size.** For the current sword example each cell is always exactly **128x128**, and the 4x2 canonical control sheet is exactly **512x256**.

Validate only:

```bash
anima validate examples/twohand_sword_slash/motion.json --normalize
```

Full physics analysis:

```bash
anima analyze examples/twohand_sword_slash/motion.json --normalize
```

Treat hard physics violations as a compile failure:

```bash
anima compile examples/twohand_sword_slash/motion.json \
  --output /tmp/anima-sword \
  --strict-physics
```

## Motion authoring model

`motion.json` stores semantic geometry rather than rendered body parts:

- fixed canvas and explicit ground Y;
- root and named 2D joints;
- per-frame foot-contact mode: `planted`, `grounded`, or `free`;
- optional elbow/knee IK pole targets;
- primary/off-hand weapon grips;
- sword tip;
- sparse keyframe numbers;
- frame labels and timing.

Keyframes do not need to be adjacent. If an author supplies frames `0`, `3`, and `7`, Anima fills the missing frames before constraint solving.

The default interpolation is **C1-continuous cubic Hermite motion**, so internal keyframes carry velocity through the pose instead of forcing an artificial stop at every authored point. The sword is interpolated as an **angular arc**, not by dragging its tip along a straight chord.

The interpolated coordinates are only motion proposals: the rig/IK layer still enforces the actual invariants afterward.

The first frame defines the canonical limb lengths, torso length, shoulder width, hip width, head offset, sword length, and grip spacing.

### IK realism

Two-bone IK alone is not enough for believable animation because every two-bone chain has two mathematical solutions. Anima therefore supports per-joint **IK pole targets** and also carries the previous solved elbow/knee forward as a temporal continuity hint. This prevents an arm from suddenly changing which side it bends toward while the hands remain on the sword.

The axial body is normalized separately: torso length, head offset, shoulder width, and hip width remain canonical even while the torso rotates or leans. This prevents the full character from subtly changing proportions while all limb IK constraints still technically pass.

In the debug animation, left limbs and right limbs use separate colorblind-safe colors plus different marker shapes. The debug preview is intentionally much larger than the final 128x128 art cell so elbow flips, foot drift, and root motion are easy to spot.

### Inertia and follow-through

Weapon motion is not treated as a sequence of unrelated angles. A motion clip can include a physical profile such as:

```json
{
  "dynamics": {
    "weapon": {
      "mass_kg": 1.35,
      "effective_length_m": 0.92,
      "inertia_factor": 0.33,
      "damping_nm_per_rad_s": 0.8,
      "max_drive_torque_nm": 32.0,
      "max_braking_torque_nm": 24.0,
      "enforce_drive_torque_limit": true,
      "impact_frame": 4,
      "collision": false
    }
  }
}
```

Anima computes angular velocity, angular acceleration, estimated drive torque, rotational energy, weapon center-of-mass translation, handle force, and mechanical power for every frame. Drive torque and braking torque are intentionally separate budgets: accelerating the sword and arresting its momentum are different physical tasks. At impact it also estimates minimum stopping time and minimum follow-through angle from the configured inertia and braking torque.

This is deliberately **physically inspired rather than a full biomechanics simulator**. The important rule is that a heavy sword cannot change angular velocity arbitrarily. If a non-colliding swing reverses direction immediately, demands excessive braking torque, or stops with too little follow-through, the dynamics report flags it.

Mass, effective length, inertia factor, center-of-mass location, drive torque, braking torque, damping, and handle-force limits are authoring parameters. Different weapons can therefore feel materially different without changing the rig architecture.

### Whole-body physics diagnostics

Anima also evaluates the body and the **person + weapon** system:

- center of mass position, velocity, acceleration, and jerk;
- root velocity and acceleration;
- per-joint linear kinematics;
- per-bone angular velocity, acceleration, and jerk;
- torso angular motion;
- support interval and stability margin;
- planted-foot slip;
- approximate ground reaction force;
- required friction ratio;
- body/system momentum and kinetic energy;
- anatomical elbow/knee angle limits;
- weapon center-of-mass translation;
- handle force and equal/opposite reaction on the body;
- two-handed shoulder/elbow load estimates;
- configurable joint torque limits.

The debug GIF visualizes the skeleton, IK poles, contact modes, body COM, system COM, velocity, acceleration, support interval, sword angular speed/torque, and ground-reaction-force vector.

### Contact semantics

Feet are not represented by one boolean anymore:

- `planted` — fixed world-space X/Y anchor for a contiguous contact phase;
- `grounded` — must remain on the ground line but may slide/pivot horizontally;
- `free` — unconstrained by the ground.

This lets a sword swing keep a front foot planted while the rear foot pivots without ever making the character appear to float.

### Physics-informed timing

`timing_recommendation.json` converts dynamic-limit violations back into animation timing suggestions. It reports:

- global time-scale recommendation;
- equivalent FPS for the existing frame count;
- equivalent frame count at the current FPS;
- separate body, weapon, and combined-system limits;
- phase-local recommendations between labeled keyframes.

Velocity scales with `1/time`, acceleration and inertial torque with approximately `1/time²`, and jerk with approximately `1/time³`. Anima uses those relationships when estimating how much additional time a physically demanding phase needs.

Anima can also apply those recommendations directly without changing pose geometry:

```bash
anima retime examples/twohand_sword_slash/motion.json \
  --output /tmp/sword.retimed.json \
  --normalize \
  --iterations 3
```

Or during compile with `--auto-retime N`. Retiming writes explicit per-pose timestamps; the debug/control GIFs use those real per-frame durations instead of assuming one uniform delay.

The two-handed sword normalization order is dependency-driven:

```text
sparse keyframes
    -> C1 motion + sword arc interpolation
        -> ground/root alignment
            -> axial body normalization
                -> rigid sword geometry
                    -> both hand targets
                        -> arm IK + pole continuity
                            -> contact-mode foot targets
                                -> leg IK
                                    -> geometry validation
                                        -> body/weapon/system physics
                                            -> timing recommendation
```

That order prevents the exact failure modes that motivated the project: floating feet, weapon morphing, and mixed one/two-handed swings.

## Image-generation handoff

Recommended inputs to an image generator:

1. **`control_sheet.png`** — authoritative pose and geometry reference.
2. **`parts_sheet.png`** — optional semantic/layered body-part reference.
3. **one canonical character image** — authoritative appearance/style reference.
4. **`imagegen_prompt.txt`** — generated handoff contract.

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
- sparse keyframe densification with C1 Hermite interpolation;
- true angular sword-arc interpolation;
- canonical limb and axial rest geometry;
- planted/grounded/free contact modes;
- ground locking and planted-foot world-space anchoring;
- rigid sword normalization;
- hard two-handed attachment;
- two-bone arm and leg IK;
- explicit IK poles plus temporal bend-branch continuity;
- planted-foot and grounded-slide reach handling;
- torso/head/shoulder/hip proportion normalization;
- geometry validation;
- colorblind-safe debug rendering with redundant left/right marker shapes;
- mannequin control rendering;
- exact-size canonical PNG frame and spritesheet export;
- separately scaled preview GIF/sheets;
- enlarged color-coded debug GIF with onion skin, joint labels, motion trails, IK poles, contacts and physics overlays;
- side-by-side debug/control review GIF;
- semantic/layered body-part control sheet and GIF;
- weapon inertia / torque / translational force / energy / follow-through diagnostics;
- body COM, momentum, acceleration, jerk, support, friction and GRF diagnostics;
- combined person+weapon COM and ground-force diagnostics;
- anatomical joint-limit diagnostics;
- two-handed arm/shoulder load estimates;
- physics policy with optional strict compile gate;
- global and phase-local physics-informed retiming recommendations;
- automatic phase-local retiming with explicit timestamps;
- compact diagnostics summary for fast local review;
- generated ImageGen handoff prompt;
- CLI;
- tests;
- local-only test/compile/render harness (`scripts/local_check.py`);
- canonical two-handed sword example.

Next work should focus on stronger shoulder/torso biomechanics, collision/impact impulses, reference-motion authoring helpers, and post-ImageGen geometric comparison.
