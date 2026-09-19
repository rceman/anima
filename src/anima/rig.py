from __future__ import annotations

from dataclasses import dataclass
from .model import FramePose


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


@dataclass(frozen=True)
class RestGeometry:
    bone_lengths: dict[str, float]
    blade_length: float
    grip_spacing: float
    off_grip_sign: float

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

        return cls(
            bone_lengths=lengths,
            blade_length=blade_len,
            grip_spacing=grip_spacing,
            off_grip_sign=sign,
        )
