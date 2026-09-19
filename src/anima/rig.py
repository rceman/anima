from __future__ import annotations

from dataclasses import dataclass

from .model import FramePose, Vec2


@dataclass(frozen=True)
class Bone:
    parent: str
    child: str
    name: str


HUMANOID_BONES: tuple[Bone, ...] = (
    Bone("shoulder_l", "elbow_l", "upper_arm_l"),
    Bone("elbow_l", "hand_l", "forearm_l"),
    Bone("shoulder_r", "elbow_r", "upper_arm_r"),
    Bone("elbow_r", "hand_r", "forearm_r"),
    Bone("hip_l", "knee_l", "thigh_l"),
    Bone("knee_l", "foot_l", "shin_l"),
    Bone("hip_r", "knee_r", "thigh_r"),
    Bone("knee_r", "foot_r", "shin_r"),
)


def _mid(a: Vec2, b: Vec2) -> Vec2:
    return Vec2((a.x + b.x) * 0.5, (a.y + b.y) * 0.5)


def _local_components(vector: Vec2, axis: Vec2) -> tuple[float, float]:
    perpendicular = Vec2(-axis.y, axis.x)
    return (
        vector.x * axis.x + vector.y * axis.y,
        vector.x * perpendicular.x + vector.y * perpendicular.y,
    )


@dataclass(frozen=True)
class RestGeometry:
    bone_lengths: dict[str, float]
    blade_length: float
    grip_spacing: float
    off_grip_sign: float

    # Axial/body invariants. These prevent the torso itself from morphing
    # while limb IK remains perfectly valid.
    torso_length: float
    shoulder_width: float
    hip_width: float
    head_offset_local: tuple[float, float]
    shoulder_center_offset_local: tuple[float, float]
    hip_center_offset_local: tuple[float, float]
    rest_shoulder_direction: Vec2
    rest_hip_direction: Vec2

    @classmethod
    def from_frame(cls, frame: FramePose) -> "RestGeometry":
        lengths: dict[str, float] = {}
        for bone in HUMANOID_BONES:
            lengths[bone.name] = frame.joints[bone.parent].distance_to(frame.joints[bone.child])

        blade_vec = frame.weapon.tip - frame.weapon.grip_main
        off_vec = frame.weapon.grip_off - frame.weapon.grip_main
        blade_len = blade_vec.length()
        grip_spacing = off_vec.length()
        direction = blade_vec.normalized()
        projection = off_vec.x * direction.x + off_vec.y * direction.y
        sign = 1.0 if projection >= 0 else -1.0

        chest = frame.joints["chest"]
        head = frame.joints["head"]
        shoulder_l = frame.joints["shoulder_l"]
        shoulder_r = frame.joints["shoulder_r"]
        hip_l = frame.joints["hip_l"]
        hip_r = frame.joints["hip_r"]

        torso_vector = chest - frame.root
        torso_length = torso_vector.length()
        torso_axis = torso_vector.normalized(Vec2(0.0, -1.0))

        shoulder_center = _mid(shoulder_l, shoulder_r)
        hip_center = _mid(hip_l, hip_r)

        shoulder_vector = shoulder_r - shoulder_l
        hip_vector = hip_r - hip_l

        return cls(
            bone_lengths=lengths,
            blade_length=blade_len,
            grip_spacing=grip_spacing,
            off_grip_sign=sign,
            torso_length=torso_length,
            shoulder_width=shoulder_vector.length(),
            hip_width=hip_vector.length(),
            head_offset_local=_local_components(head - chest, torso_axis),
            shoulder_center_offset_local=_local_components(
                shoulder_center - chest,
                torso_axis,
            ),
            hip_center_offset_local=_local_components(
                hip_center - frame.root,
                torso_axis,
            ),
            rest_shoulder_direction=shoulder_vector.normalized(Vec2(1.0, 0.0)),
            rest_hip_direction=hip_vector.normalized(Vec2(1.0, 0.0)),
        )
