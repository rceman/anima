# Motion Authoring Protocol

This document defines the intended ChatGPT/Anima workflow for turning a motion reference into a constrained control spritesheet and then into final pixel art.

## Separation of responsibilities

Anima is not a video downloader or pose-estimation product.

Reference acquisition and interpretation happen outside Anima. The author may use an uploaded video, a GIF, a 3D animation, another spritesheet, a sequence of stills, or a motion designed from first principles.

Anima starts at **semantic motion**.

```text
reference media
    -> human/model motion interpretation
    -> sparse semantic keyframes
    -> Anima constraints + physics
    -> control / parts / debug sheets
    -> ImageGen
    -> Anima post-render validation
```

## 1. Inspect the reference

Do not trace every source frame. Identify mechanically meaningful moments:

1. ready / guard;
2. anticipation;
3. maximum wind-up;
4. acceleration begins;
5. maximum weapon speed;
6. impact/crossing point;
7. follow-through;
8. recovery;
9. return only when a loop is required.

Record the actual source timestamps. Timing is part of the motion, not an export detail.

Motion blur, occluded hands, camera movement, perspective scale changes, and detector noise must not be copied literally.

A key pose is not automatically a stop. Anticipation, attack-start, impact, and follow-through normally carry velocity through the pose. Use `kinematic_stop` only when the character is genuinely at rest (for example the initial held guard or the final recovered guard). This prevents finite-difference analysis from inventing motion at a known rest state without falsely freezing every named keyframe.

## 2. Retarget to the Anima rig

For every selected key pose, author:

- root/pelvis;
- head and chest;
- both shoulders, elbows, and hands;
- both hips, knees, and feet;
- primary/off-hand sword grips;
- sword tip;
- foot contact mode;
- elbow/knee IK pole hints;
- source-relative `time_s`;
- `kinematic_stop: true` only for poses that are genuine held/rest states.

Use the reference to infer **intent and mechanics**, then map it to the canonical rig. Bone lengths and sword length are not re-measured independently per frame.

## 2a. Prefer compact parametric recipes

For model-authored or reference-derived motion, do **not** hand-author every elbow
and knee coordinate unless the reference specifically requires it.

The preferred source is a compact `recipe.json` containing, per key pose:

- root/pelvis position;
- torso lean;
- shoulder-line and hip-line angle;
- primary sword grip position;
- sword angle;
- left/right foot position and contact mode;
- optional elbow/knee pole hints;
- real timestamp;
- optional semantic layer order.

Anima reconstructs the full axial rig and weapon, then solves elbows/knees using
the canonical bone lengths and IK constraints.

```bash
anima author examples/twohand_sword_slash/motion.json \
  examples/twohand_sword_slash/recipe.json \
  --output /tmp/authored.json \
  --normalize \
  --validate
```

This is the intended interface for ChatGPT when interpreting a video: extract a
small set of physically meaningful transforms, not dozens of noisy pixel joint
coordinates.

## 3. Contact semantics

Use contact states deliberately:

- `planted`: world-space foot anchor; no skating;
- `grounded`: remains on the ground but may pivot/slide;
- `free`: airborne/unconstrained.

A foot must not be marked `planted` if the root trajectory makes the leg unable to reach it. Fix the stance/root motion instead of allowing the solver to stretch the leg.

## 4. Two-handed weapon semantics

For a two-handed sword:

- one rigid sword exists for the entire clip;
- primary and secondary grips stay on the same handle;
- the hands are weapon targets;
- elbows are solved after the weapon pose;
- IK poles define the desired bend side;
- the previous solved elbow is used as a temporal continuity hint.

When the source sword is motion-blurred or occluded, reconstruct the rigid weapon from the visible hand/arc motion instead of shortening or bending it.

## 5. Physical authoring profile

Configure approximate physical properties appropriate to the intended weapon and character:

- body mass and pixel scale;
- weapon mass;
- effective weapon length;
- inertia factor;
- weapon center-of-mass fraction;
- drive torque budget;
- braking torque budget;
- damping;
- handle-force limit;
- ground friction;
- optional strength limits.

These values are modeling parameters, not claims of laboratory-grade biomechanics.

## 5a. Semantic depth / occlusion

2D skeleton geometry does not tell us which limb is in front when projections
cross. That ambiguity must not be left for the image generator to guess.

Each pose may include a partial `layer_order`, listed back-to-front. Example:

```json
{
  "layer_order": ["left_arm", "torso", "right_arm"]
}
```

Only the named layers are permuted; unmentioned head/legs/weapon retain their
canonical z-slots. Anima detects arm/arm and arm/torso projection overlaps and
writes `occlusion.json`. Ambiguous crossings are warnings until the author
explicitly chooses the depth relation.

The semantic `parts_sheet.png` is composited using this same z-order, and the
resolved order is exported in `animation_manifest.json`.

## 6. Compile and inspect

Recommended development command:

```bash
anima compile motion.json \
  --output output/ \
  --scale 4 \
  --auto-retime 3 \
  --strict-physics
```

Review artifacts in this order:

1. `diagnostics_summary.json`;
2. `inspection_preview.gif` — preferred large debugging view;
3. `debug_preview.gif`;
4. `review_preview.gif`;
5. `control_preview.gif`;
6. `parts_preview.gif`;
7. `occlusion.json`;
8. `physics_validation.json`;
9. `timing_recommendation.json`.

### Debug acceptance

The colored debug skeleton must show:

- no foot skating during planted phases;
- no elbow/knee branch flips;
- no body proportion changes;
- both hands staying on the handle;
- one continuous sword arc;
- plausible center-of-mass transfer;
- no abrupt weapon stop without either braking time or collision;
- no unexplained direction reversal;
- no unresolved arm/arm or arm/torso depth ambiguity at crossings.

Fix motion data before touching final art.

## 7. ImageGen handoff

Provide ImageGen with:

1. `control_sheet.png` — pose/geometry authority;
2. `parts_sheet.png` — semantic body-part authority;
3. one master character image — appearance authority;
4. `imagegen_prompt.txt` — generated rendering contract.

The final art may redraw pixels and silhouettes, but it must not invent motion.

## 8. Validate the generated result

After ImageGen returns a sheet:

```bash
anima verify-render output/motion.normalized.json final_sheet.png \
  --columns 4 \
  --output output/render_validation.json
```

The first validator checks:

- exact sheet/cell dimensions;
- character center drift;
- character scale drift;
- head and hand anchor presence;
- sword-tip presence near the expected tip;
- expected ground contacts.

A failed render is sent back for correction instead of silently becoming the source of truth.

## 9. Canonical quality rule

The semantic motion file is authoritative and durable.

Never fix a motion defect only by repainting the final sprite. If the debug skeleton is wrong, correct the motion/rig/physics first, regenerate the control guides, and only then regenerate the art.
