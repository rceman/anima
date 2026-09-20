from __future__ import annotations

from dataclasses import dataclass
import math

from .contacts import GROUNDED, PLANTED, contact_mode, is_ground_contact
from .model import FramePose, MotionClip, Vec2, WeaponPose
from .rig import HUMANOID_BONES, RestGeometry


@dataclass(frozen=True)
class Issue:
    frame: int
    code: str
    message: str


@dataclass
class ValidationReport:
    issues: list[Issue]

    @property
    def ok(self) -> bool:
        return not self.issues


def _cross(a: Vec2, b: Vec2) -> float:
    return a.x * b.y - a.y * b.x


def solve_two_bone(
    start: Vec2,
    target: Vec2,
    len_a: float,
    len_b: float,
    bend_hint: Vec2,
    previous_mid: Vec2 | None = None,
) -> tuple[Vec2, Vec2, bool]:
    """Solve a 2-bone chain with temporal bend continuity.

    A 2D two-bone chain has two valid elbow/knee solutions. Choosing a branch
    independently per frame causes the classic IK "elbow flip". We generate
    both solutions and select the one nearest the previous solved joint when
    available; the authored bend_hint is used for the first frame.
    """
    delta = target - start
    distance = delta.length()
    min_reach = abs(len_a - len_b) + 1e-6
    max_reach = len_a + len_b - 1e-6
    reachable = min_reach <= distance <= max_reach
    clamped = min(max(distance, min_reach), max_reach)
    direction = delta.normalized(Vec2(1.0, 0.0))
    end = target if reachable else start + direction * clamped

    cos_a = (len_a * len_a + clamped * clamped - len_b * len_b) / (2.0 * len_a * clamped)
    cos_a = min(1.0, max(-1.0, cos_a))
    angle = math.acos(cos_a)
    base = math.atan2(direction.y, direction.x)

    candidates: list[Vec2] = []
    for sign in (1.0, -1.0):
        joint_angle = base + sign * angle
        candidates.append(
            Vec2(
                start.x + math.cos(joint_angle) * len_a,
                start.y + math.sin(joint_angle) * len_a,
            )
        )

    # The authored pole/bend hint remains authoritative on every frame.
    # Temporal continuity is a secondary term, not a replacement for the pole;
    # otherwise changing an IK pole later in the motion would have no effect.
    hint_vector = bend_hint - start
    hint_side = _cross(direction, hint_vector)
    pole_mismatch_penalty = (len_a + len_b) * 4.0

    def candidate_score(point: Vec2) -> float:
        candidate_vector = point - start
        candidate_side = _cross(direction, candidate_vector)

        pole_cost = point.distance_to(bend_hint)
        if (
            abs(hint_side) > 1e-6
            and abs(candidate_side) > 1e-6
            and hint_side * candidate_side < 0.0
        ):
            pole_cost += pole_mismatch_penalty

        temporal_cost = (
            0.35 * point.distance_to(previous_mid)
            if previous_mid is not None
            else 0.0
        )
        return pole_cost + temporal_cost

    mid = min(candidates, key=candidate_score)
    return mid, end, reachable


def clamp_target_to_horizontal_reach(
    start: Vec2,
    target: Vec2,
    len_a: float,
    len_b: float,
) -> Vec2:
    """Keep target.y fixed while clamping x into the reachable annulus."""
    dy = target.y - start.y
    max_reach = max(0.0, len_a + len_b - 1e-6)
    min_reach = abs(len_a - len_b) + 1e-6

    if abs(dy) > max_reach:
        return Vec2(start.x, start.y + math.copysign(max_reach, dy))

    dx = target.x - start.x
    max_dx = math.sqrt(max(0.0, max_reach * max_reach - dy * dy))
    if abs(dx) > max_dx:
        dx = math.copysign(max_dx, dx)

    distance = math.hypot(dx, dy)
    if distance < min_reach:
        min_dx_sq = min_reach * min_reach - dy * dy
        if min_dx_sq > 0:
            min_dx = math.sqrt(min_dx_sq)
            dx = (-1.0 if dx < 0 else 1.0) * min_dx

    return Vec2(start.x + dx, target.y)



def _world_from_local(
    origin: Vec2,
    axis: Vec2,
    local: tuple[float, float],
) -> Vec2:
    perpendicular = Vec2(-axis.y, axis.x)
    return origin + axis * local[0] + perpendicular * local[1]


def _normalize_axial_pose(
    root: Vec2,
    joints: dict[str, Vec2],
    rest: RestGeometry,
) -> dict[str, Vec2]:
    """Lock torso/head/shoulder/hip proportions while preserving authored pose direction."""
    required = {
        "chest",
        "head",
        "shoulder_l",
        "shoulder_r",
        "hip_l",
        "hip_r",
    }
    if not required.issubset(joints):
        return joints

    out = dict(joints)
    authored_axis = (joints["chest"] - root).normalized(Vec2(0.0, -1.0))
    chest = root + authored_axis * rest.torso_length
    out["chest"] = chest

    out["head"] = _world_from_local(
        chest,
        authored_axis,
        rest.head_offset_local,
    )

    shoulder_center = _world_from_local(
        chest,
        authored_axis,
        rest.shoulder_center_offset_local,
    )
    shoulder_direction = (
        joints["shoulder_r"] - joints["shoulder_l"]
    ).normalized(rest.rest_shoulder_direction)
    shoulder_half = rest.shoulder_width * 0.5
    out["shoulder_l"] = shoulder_center - shoulder_direction * shoulder_half
    out["shoulder_r"] = shoulder_center + shoulder_direction * shoulder_half

    hip_center = _world_from_local(
        root,
        authored_axis,
        rest.hip_center_offset_local,
    )
    hip_direction = (
        joints["hip_r"] - joints["hip_l"]
    ).normalized(rest.rest_hip_direction)
    hip_half = rest.hip_width * 0.5
    out["hip_l"] = hip_center - hip_direction * hip_half
    out["hip_r"] = hip_center + hip_direction * hip_half
    return out



def _fit_shoulder_girdle_to_grips(
    joints: dict[str, Vec2],
    rest: RestGeometry,
    primary_hand: str,
    secondary_hand: str,
    main_target: Vec2,
    off_target: Vec2,
    max_shift_px: float,
    reach_margin_px: float = 0.5,
    iterations: int = 6,
) -> dict[str, Vec2]:
    """Translate the shoulder girdle slightly so both sword grips are reachable.

    Human shoulders are not welded to the rib cage: clavicle/scapula motion
    allows a few centimeters of protraction/retraction. Modeling a bounded
    shared shoulder translation is more realistic than letting an unreachable
    IK target detach the hand from the sword or stretch the arm.

    The shoulder width remains fixed; only the pair's center moves. This is a
    deterministic 2D approximation, not a full scapulothoracic model.
    """
    if max_shift_px <= 0.0:
        return joints

    arm_specs = (
        ("shoulder_r", "upper_arm_r", "forearm_r", main_target)
        if primary_hand.endswith("_r")
        else ("shoulder_l", "upper_arm_l", "forearm_l", main_target),
        ("shoulder_l", "upper_arm_l", "forearm_l", off_target)
        if secondary_hand.endswith("_l")
        else ("shoulder_r", "upper_arm_r", "forearm_r", off_target),
    )
    if not all(spec[0] in joints for spec in arm_specs):
        return joints

    out = dict(joints)
    total_shift = Vec2(0.0, 0.0)

    for _ in range(max(1, iterations)):
        correction = Vec2(0.0, 0.0)
        active = 0

        for shoulder_name, upper_name, lower_name, target in arm_specs:
            shoulder = out[shoulder_name]
            delta = target - shoulder
            distance = delta.length()
            if distance <= 1e-9:
                continue

            max_reach = (
                rest.bone_lengths[upper_name]
                + rest.bone_lengths[lower_name]
                - reach_margin_px
            )
            min_reach = (
                abs(
                    rest.bone_lengths[upper_name]
                    - rest.bone_lengths[lower_name]
                )
                + reach_margin_px
            )
            direction = delta.normalized()

            if distance > max_reach:
                correction = correction + direction * (distance - max_reach)
                active += 1
            elif distance < min_reach:
                correction = correction - direction * (min_reach - distance)
                active += 1

        if active == 0:
            break

        step = correction * (1.0 / active)
        remaining = max_shift_px - total_shift.length()
        if remaining <= 1e-9:
            break

        step_length = step.length()
        if step_length > remaining:
            step = step.normalized() * remaining

        total_shift = total_shift + step
        out["shoulder_l"] = out["shoulder_l"] + step
        out["shoulder_r"] = out["shoulder_r"] + step

        if step.length() <= 1e-6:
            break

    return out


def normalize_clip(clip: MotionClip) -> MotionClip:
    if not clip.frames:
        return clip

    rest = RestGeometry.from_frame(clip.frames[0])
    normalized: list[FramePose] = []
    previous_mids: dict[str, Vec2] = {}
    contact_anchors: dict[str, Vec2] = {}

    for source in clip.frames:
        frame = source

        # 1. Lock planted feet to one explicit ground baseline by translating the whole pose.
        contact_feet = [
            name
            for name in ("foot_l", "foot_r")
            if is_ground_contact(frame.contacts.get(name)) and name in frame.joints
        ]
        if contact_feet:
            avg_y = sum(frame.joints[name].y for name in contact_feet) / len(contact_feet)
            frame = frame.shifted(Vec2(0.0, clip.ground_y - avg_y))

        joints = _normalize_axial_pose(
            frame.root,
            dict(frame.joints),
            rest,
        )

        # 2. Rigidize sword from primary grip and requested blade direction.
        direction = (frame.weapon.tip - frame.weapon.grip_main).normalized(Vec2(1.0, 0.0))
        main = frame.weapon.grip_main
        tip = main + direction * rest.blade_length
        off = main + direction * (rest.grip_spacing * rest.off_grip_sign)
        weapon = WeaponPose(main, off, tip)

        # 3. Allow bounded shoulder-girdle translation before arm IK. This
        # represents scapula/clavicle protraction and prevents small reach
        # errors from turning into detached hands or stretched arms.
        body_cfg = clip.dynamics.get("body", {})
        joints = _fit_shoulder_girdle_to_grips(
            joints,
            rest,
            clip.primary_hand,
            clip.secondary_hand,
            main,
            off,
            max_shift_px=float(
                body_cfg.get("shoulder_girdle_max_shift_px", 4.0)
            ),
            reach_margin_px=float(
                body_cfg.get("arm_reach_margin_px", 0.5)
            ),
        )

        # 4. Hard two-handed attachment: hands follow sword, elbows are solved by IK.
        arm_specs = (
            (clip.primary_hand, "elbow_r", "shoulder_r", "upper_arm_r", "forearm_r", main),
            (clip.secondary_hand, "elbow_l", "shoulder_l", "upper_arm_l", "forearm_l", off),
        )
        for hand_name, elbow_name, shoulder_name, upper_name, lower_name, target in arm_specs:
            if {hand_name, elbow_name, shoulder_name}.issubset(joints):
                elbow, hand, _ = solve_two_bone(
                    joints[shoulder_name],
                    target,
                    rest.bone_lengths[upper_name],
                    rest.bone_lengths[lower_name],
                    frame.ik_poles.get(elbow_name, joints[elbow_name]),
                    previous_mid=previous_mids.get(elbow_name),
                )
                joints[elbow_name] = elbow
                joints[hand_name] = hand
                previous_mids[elbow_name] = elbow

        # 5. Ground contacts are exact. Knees are solved while preserving bend direction.
        leg_specs = (
            ("foot_l", "knee_l", "hip_l", "thigh_l", "shin_l"),
            ("foot_r", "knee_r", "hip_r", "thigh_r", "shin_r"),
        )
        for foot_name, knee_name, hip_name, thigh_name, shin_name in leg_specs:
            if not {foot_name, knee_name, hip_name}.issubset(joints):
                continue

            foot = joints[foot_name]
            mode = contact_mode(frame.contacts.get(foot_name))
            if mode == PLANTED:
                # A planted foot is a world-space contact. Keep the same X/Y
                # anchor for the full contiguous planted phase to prevent skating.
                if foot_name not in contact_anchors:
                    contact_anchors[foot_name] = Vec2(foot.x, clip.ground_y)
                foot = contact_anchors[foot_name]
            elif mode == GROUNDED:
                # Grounded permits authored horizontal sliding/pivoting, but
                # never vertical lift. Clamp X only if the leg cannot reach.
                contact_anchors.pop(foot_name, None)
                foot = Vec2(foot.x, clip.ground_y)
                foot = clamp_target_to_horizontal_reach(
                    joints[hip_name],
                    foot,
                    rest.bone_lengths[thigh_name],
                    rest.bone_lengths[shin_name],
                )
            else:
                contact_anchors.pop(foot_name, None)

            knee, solved_foot, reachable = solve_two_bone(
                joints[hip_name],
                foot,
                rest.bone_lengths[thigh_name],
                rest.bone_lengths[shin_name],
                frame.ik_poles.get(knee_name, joints[knee_name]),
                previous_mid=previous_mids.get(knee_name),
            )
            joints[knee_name] = knee
            # Preserve hard world-space contact when reachable. If an authored
            # root/hip position makes the planted foot unreachable, the IK
            # solver returns its closest reachable point and validation exposes
            # the resulting contact error instead of silently moving the foot.
            joints[foot_name] = solved_foot
            previous_mids[knee_name] = knee

        normalized.append(
            FramePose(
                frame=frame.frame,
                root=frame.root,
                joints=joints,
                weapon=weapon,
                contacts=dict(frame.contacts),
                ik_poles=dict(frame.ik_poles),
                label=frame.label,
                time_s=frame.time_s,
                kinematic_stop=frame.kinematic_stop,
            )
        )

    return MotionClip(
        width=clip.width,
        height=clip.height,
        ground_y=clip.ground_y,
        fps=clip.fps,
        rig=clip.rig,
        frames=normalized,
        primary_hand=clip.primary_hand,
        secondary_hand=clip.secondary_hand,
        metadata=dict(clip.metadata),
        dynamics=dict(clip.dynamics),
    )


def validate_clip(clip: MotionClip, tolerance: float = 0.75) -> ValidationReport:
    if not clip.frames:
        return ValidationReport([Issue(-1, "empty", "Motion clip has no frames")])

    rest = RestGeometry.from_frame(clip.frames[0])
    issues: list[Issue] = []

    for frame in clip.frames:
        for foot in ("foot_l", "foot_r"):
            if is_ground_contact(frame.contacts.get(foot)) and foot in frame.joints:
                delta = abs(frame.joints[foot].y - clip.ground_y)
                if delta > tolerance:
                    issues.append(
                        Issue(frame.frame, "ground_contact", f"{foot} is {delta:.2f}px from ground")
                    )

        for bone in HUMANOID_BONES:
            if bone.parent not in frame.joints or bone.child not in frame.joints:
                issues.append(Issue(frame.frame, "missing_joint", f"Missing joint for {bone.name}"))
                continue

            length = frame.joints[bone.parent].distance_to(frame.joints[bone.child])
            expected = rest.bone_lengths[bone.name]
            if abs(length - expected) > tolerance:
                issues.append(
                    Issue(
                        frame.frame,
                        "bone_length",
                        f"{bone.name}: {length:.2f}px != {expected:.2f}px",
                    )
                )


        # Axial/body proportions are hard invariants, not just limb lengths.
        if {"chest", "head", "shoulder_l", "shoulder_r", "hip_l", "hip_r"}.issubset(frame.joints):
            torso_length = frame.root.distance_to(frame.joints["chest"])
            shoulder_width = frame.joints["shoulder_l"].distance_to(frame.joints["shoulder_r"])
            hip_width = frame.joints["hip_l"].distance_to(frame.joints["hip_r"])
            head_distance = frame.joints["chest"].distance_to(frame.joints["head"])
            rest_head_distance = math.hypot(*rest.head_offset_local)

            axial_checks = (
                ("torso_length", torso_length, rest.torso_length),
                ("shoulder_width", shoulder_width, rest.shoulder_width),
                ("hip_width", hip_width, rest.hip_width),
                ("head_offset", head_distance, rest_head_distance),
            )
            for code, actual, expected in axial_checks:
                if abs(actual - expected) > tolerance:
                    issues.append(
                        Issue(
                            frame.frame,
                            code,
                            f"{code}: {actual:.2f}px != {expected:.2f}px",
                        )
                    )


        arm_reach_specs = (
            (
                "shoulder_r",
                "upper_arm_r",
                "forearm_r",
                frame.weapon.grip_main,
                "primary",
            )
            if clip.primary_hand.endswith("_r")
            else (
                "shoulder_l",
                "upper_arm_l",
                "forearm_l",
                frame.weapon.grip_main,
                "primary",
            ),
            (
                "shoulder_l",
                "upper_arm_l",
                "forearm_l",
                frame.weapon.grip_off,
                "secondary",
            )
            if clip.secondary_hand.endswith("_l")
            else (
                "shoulder_r",
                "upper_arm_r",
                "forearm_r",
                frame.weapon.grip_off,
                "secondary",
            ),
        )
        for shoulder_name, upper_name, lower_name, target, role in arm_reach_specs:
            if shoulder_name not in frame.joints:
                continue
            distance = frame.joints[shoulder_name].distance_to(target)
            maximum = (
                rest.bone_lengths[upper_name]
                + rest.bone_lengths[lower_name]
            )
            minimum = abs(
                rest.bone_lengths[upper_name]
                - rest.bone_lengths[lower_name]
            )
            if distance > maximum + tolerance:
                issues.append(
                    Issue(
                        frame.frame,
                        "arm_unreachable",
                        (
                            f"{role} grip is {distance:.2f}px from "
                            f"{shoulder_name}, max reach {maximum:.2f}px"
                        ),
                    )
                )
            elif distance < minimum - tolerance:
                issues.append(
                    Issue(
                        frame.frame,
                        "arm_overcompressed",
                        (
                            f"{role} grip is {distance:.2f}px from "
                            f"{shoulder_name}, min reach {minimum:.2f}px"
                        ),
                    )
                )

        main_hand = frame.joints.get(clip.primary_hand)
        off_hand = frame.joints.get(clip.secondary_hand)
        if main_hand and main_hand.distance_to(frame.weapon.grip_main) > tolerance:
            issues.append(
                Issue(frame.frame, "primary_grip", "Primary hand is detached from sword grip")
            )
        if off_hand and off_hand.distance_to(frame.weapon.grip_off) > tolerance:
            issues.append(
                Issue(frame.frame, "secondary_grip", "Secondary hand is detached from sword grip")
            )

        blade = frame.weapon.grip_main.distance_to(frame.weapon.tip)
        spacing = frame.weapon.grip_main.distance_to(frame.weapon.grip_off)
        if abs(blade - rest.blade_length) > tolerance:
            issues.append(
                Issue(frame.frame, "sword_length", f"Sword length drift: {blade:.2f}px")
            )
        if abs(spacing - rest.grip_spacing) > tolerance:
            issues.append(
                Issue(frame.frame, "grip_spacing", f"Grip spacing drift: {spacing:.2f}px")
            )

        points = list(frame.joints.values()) + [
            frame.weapon.tip,
            frame.weapon.grip_main,
            frame.weapon.grip_off,
        ]
        if any(
            point.x < 0
            or point.y < 0
            or point.x >= clip.width
            or point.y >= clip.height
            for point in points
        ):
            issues.append(Issue(frame.frame, "canvas_clip", "Pose extends outside the canvas"))

    return ValidationReport(issues)
