from __future__ import annotations

from dataclasses import dataclass
import math

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

    preferred = previous_mid or bend_hint
    mid = min(candidates, key=lambda point: point.distance_to(preferred))
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


def normalize_clip(clip: MotionClip) -> MotionClip:
    if not clip.frames:
        return clip

    rest = RestGeometry.from_frame(clip.frames[0])
    normalized: list[FramePose] = []
    previous_mids: dict[str, Vec2] = {}

    for source in clip.frames:
        frame = source

        # 1. Lock planted feet to one explicit ground baseline by translating the whole pose.
        contact_feet = [
            name
            for name in ("foot_l", "foot_r")
            if frame.contacts.get(name) and name in frame.joints
        ]
        if contact_feet:
            avg_y = sum(frame.joints[name].y for name in contact_feet) / len(contact_feet)
            frame = frame.shifted(Vec2(0.0, clip.ground_y - avg_y))

        joints = dict(frame.joints)

        # 2. Rigidize sword from primary grip and requested blade direction.
        direction = (frame.weapon.tip - frame.weapon.grip_main).normalized(Vec2(1.0, 0.0))
        main = frame.weapon.grip_main
        tip = main + direction * rest.blade_length
        off = main + direction * (rest.grip_spacing * rest.off_grip_sign)
        weapon = WeaponPose(main, off, tip)

        # 3. Hard two-handed attachment: hands follow sword, elbows are solved by IK.
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
                    joints[elbow_name],
                    previous_mid=previous_mids.get(elbow_name),
                )
                joints[elbow_name] = elbow
                joints[hand_name] = hand
                previous_mids[elbow_name] = elbow

        # 4. Ground contacts are exact. Knees are solved while preserving bend direction.
        leg_specs = (
            ("foot_l", "knee_l", "hip_l", "thigh_l", "shin_l"),
            ("foot_r", "knee_r", "hip_r", "thigh_r", "shin_r"),
        )
        for foot_name, knee_name, hip_name, thigh_name, shin_name in leg_specs:
            if not {foot_name, knee_name, hip_name}.issubset(joints):
                continue

            foot = joints[foot_name]
            if frame.contacts.get(foot_name):
                foot = Vec2(foot.x, clip.ground_y)
                foot = clamp_target_to_horizontal_reach(
                    joints[hip_name],
                    foot,
                    rest.bone_lengths[thigh_name],
                    rest.bone_lengths[shin_name],
                )

            knee, solved_foot, _ = solve_two_bone(
                joints[hip_name],
                foot,
                rest.bone_lengths[thigh_name],
                rest.bone_lengths[shin_name],
                joints[knee_name],
                previous_mid=previous_mids.get(knee_name),
            )
            joints[knee_name] = knee
            joints[foot_name] = solved_foot
            previous_mids[knee_name] = knee

        normalized.append(
            FramePose(
                frame=frame.frame,
                root=frame.root,
                joints=joints,
                weapon=weapon,
                contacts=dict(frame.contacts),
                label=frame.label,
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
    )


def validate_clip(clip: MotionClip, tolerance: float = 0.75) -> ValidationReport:
    if not clip.frames:
        return ValidationReport([Issue(-1, "empty", "Motion clip has no frames")])

    rest = RestGeometry.from_frame(clip.frames[0])
    issues: list[Issue] = []

    for frame in clip.frames:
        for foot in ("foot_l", "foot_r"):
            if frame.contacts.get(foot) and foot in frame.joints:
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
