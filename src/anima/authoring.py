from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

from .model import FramePose, MotionClip, Vec2, WeaponPose
from .rig import RestGeometry


def _world_from_local(
    origin: Vec2,
    axis: Vec2,
    local: tuple[float, float],
) -> Vec2:
    perpendicular = Vec2(-axis.y, axis.x)
    return origin + axis * local[0] + perpendicular * local[1]


def _axis_from_vertical(angle_deg: float) -> Vec2:
    """0deg points upward; positive angles lean clockwise/right on screen."""
    radians = math.radians(angle_deg)
    return Vec2(math.sin(radians), -math.cos(radians))


def _screen_direction(angle_deg: float) -> Vec2:
    """0deg points right; positive angles rotate clockwise/down on screen."""
    radians = math.radians(angle_deg)
    return Vec2(math.cos(radians), math.sin(radians))


def _rotate(vector: Vec2, angle_deg: float) -> Vec2:
    radians = math.radians(angle_deg)
    c = math.cos(radians)
    s = math.sin(radians)
    return Vec2(
        vector.x * c - vector.y * s,
        vector.x * s + vector.y * c,
    )


@dataclass(frozen=True)
class PoseRecipe:
    """Compact semantic pose description for model/reference authoring.

    A recipe intentionally does not require elbow/knee coordinates. Those are
    IK results. The author specifies axial pose, weapon pose, feet, contacts,
    and optional pole hints; Anima derives the rest from the canonical rig.
    """

    frame: int
    root: Vec2
    main_grip: Vec2
    sword_angle_deg: float
    foot_l: Vec2
    foot_r: Vec2

    torso_lean_deg: float = 0.0
    shoulder_line_deg: float = 0.0
    hip_line_deg: float = 0.0

    elbow_pole_l: Vec2 | None = None
    elbow_pole_r: Vec2 | None = None
    knee_pole_l: Vec2 | None = None
    knee_pole_r: Vec2 | None = None

    contacts: dict[str, bool | str] = field(default_factory=dict)
    label: str | None = None
    time_s: float | None = None
    kinematic_stop: bool = False
    layer_order: list[str] = field(default_factory=list)
    reference: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PoseRecipe":
        def optional_vec(name: str) -> Vec2 | None:
            value = data.get(name)
            return Vec2.from_any(value) if value is not None else None

        return cls(
            frame=int(data["frame"]),
            root=Vec2.from_any(data["root"]),
            main_grip=Vec2.from_any(data["main_grip"]),
            sword_angle_deg=float(data["sword_angle_deg"]),
            foot_l=Vec2.from_any(data["foot_l"]),
            foot_r=Vec2.from_any(data["foot_r"]),
            torso_lean_deg=float(data.get("torso_lean_deg", 0.0)),
            shoulder_line_deg=float(data.get("shoulder_line_deg", 0.0)),
            hip_line_deg=float(data.get("hip_line_deg", 0.0)),
            elbow_pole_l=optional_vec("elbow_pole_l"),
            elbow_pole_r=optional_vec("elbow_pole_r"),
            knee_pole_l=optional_vec("knee_pole_l"),
            knee_pole_r=optional_vec("knee_pole_r"),
            contacts=dict(data.get("contacts", {})),
            label=data.get("label"),
            time_s=(
                float(data["time_s"])
                if data.get("time_s") is not None
                else None
            ),
            kinematic_stop=bool(data.get("kinematic_stop", False)),
            layer_order=list(data.get("layer_order", [])),
            reference=dict(data.get("reference", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "frame": self.frame,
            "root": self.root.as_list(),
            "main_grip": self.main_grip.as_list(),
            "sword_angle_deg": self.sword_angle_deg,
            "foot_l": self.foot_l.as_list(),
            "foot_r": self.foot_r.as_list(),
            "torso_lean_deg": self.torso_lean_deg,
            "shoulder_line_deg": self.shoulder_line_deg,
            "hip_line_deg": self.hip_line_deg,
            "contacts": dict(self.contacts),
        }
        for name in (
            "elbow_pole_l",
            "elbow_pole_r",
            "knee_pole_l",
            "knee_pole_r",
        ):
            value = getattr(self, name)
            if value is not None:
                out[name] = value.as_list()
        if self.label is not None:
            out["label"] = self.label
        if self.time_s is not None:
            out["time_s"] = self.time_s
        if self.kinematic_stop:
            out["kinematic_stop"] = True
        if self.layer_order:
            out["layer_order"] = list(self.layer_order)
        if self.reference:
            out["reference"] = dict(self.reference)
        return out


class ParametricAuthor:
    """Build full FramePose objects from compact bone/weapon transforms."""

    def __init__(
        self,
        rest_frame: FramePose,
        primary_hand: str = "hand_r",
        secondary_hand: str = "hand_l",
    ) -> None:
        self.rest_frame = rest_frame
        self.rest = RestGeometry.from_frame(rest_frame)
        self.primary_hand = primary_hand
        self.secondary_hand = secondary_hand

    def _rest_relative(
        self,
        joint_name: str,
        anchor_name: str,
        rotation_deg: float,
        anchor: Vec2,
    ) -> Vec2:
        delta = (
            self.rest_frame.joints[joint_name]
            - self.rest_frame.joints[anchor_name]
        )
        return anchor + _rotate(delta, rotation_deg)

    def frame(self, recipe: PoseRecipe) -> FramePose:
        torso_axis = _axis_from_vertical(recipe.torso_lean_deg)
        chest = recipe.root + torso_axis * self.rest.torso_length
        head = _world_from_local(
            chest,
            torso_axis,
            self.rest.head_offset_local,
        )

        shoulder_center = _world_from_local(
            chest,
            torso_axis,
            self.rest.shoulder_center_offset_local,
        )
        shoulder_direction = _screen_direction(recipe.shoulder_line_deg)
        shoulder_half = self.rest.shoulder_width * 0.5
        shoulder_l = shoulder_center - shoulder_direction * shoulder_half
        shoulder_r = shoulder_center + shoulder_direction * shoulder_half

        hip_center = _world_from_local(
            recipe.root,
            torso_axis,
            self.rest.hip_center_offset_local,
        )
        hip_direction = _screen_direction(recipe.hip_line_deg)
        hip_half = self.rest.hip_width * 0.5
        hip_l = hip_center - hip_direction * hip_half
        hip_r = hip_center + hip_direction * hip_half

        sword_direction = _screen_direction(recipe.sword_angle_deg)
        grip_main = recipe.main_grip
        grip_off = grip_main + sword_direction * (
            self.rest.grip_spacing * self.rest.off_grip_sign
        )
        tip = grip_main + sword_direction * self.rest.blade_length
        weapon = WeaponPose(grip_main, grip_off, tip)

        joints: dict[str, Vec2] = {
            "head": head,
            "chest": chest,
            "shoulder_l": shoulder_l,
            "shoulder_r": shoulder_r,
            "hip_l": hip_l,
            "hip_r": hip_r,
            "foot_l": recipe.foot_l,
            "foot_r": recipe.foot_r,
        }

        # Hands are exact weapon targets. The normalizer later solves elbows.
        joints[self.primary_hand] = grip_main
        joints[self.secondary_hand] = grip_off

        # Unspecified poles are transformed from the rest pose. This preserves
        # a stable bend-side default while still allowing a reference author to
        # explicitly place any pole when the motion demands it.
        elbow_l = recipe.elbow_pole_l or self._rest_relative(
            "elbow_l",
            "shoulder_l",
            recipe.torso_lean_deg,
            shoulder_l,
        )
        elbow_r = recipe.elbow_pole_r or self._rest_relative(
            "elbow_r",
            "shoulder_r",
            recipe.torso_lean_deg,
            shoulder_r,
        )
        knee_l = recipe.knee_pole_l or self._rest_relative(
            "knee_l",
            "hip_l",
            recipe.torso_lean_deg,
            hip_l,
        )
        knee_r = recipe.knee_pole_r or self._rest_relative(
            "knee_r",
            "hip_r",
            recipe.torso_lean_deg,
            hip_r,
        )

        joints["elbow_l"] = elbow_l
        joints["elbow_r"] = elbow_r
        joints["knee_l"] = knee_l
        joints["knee_r"] = knee_r

        return FramePose(
            frame=recipe.frame,
            root=recipe.root,
            joints=joints,
            weapon=weapon,
            contacts=dict(recipe.contacts),
            ik_poles={
                "elbow_l": elbow_l,
                "elbow_r": elbow_r,
                "knee_l": knee_l,
                "knee_r": knee_r,
            },
            label=recipe.label,
            time_s=recipe.time_s,
            kinematic_stop=recipe.kinematic_stop,
            layer_order=list(recipe.layer_order),
            reference=dict(recipe.reference),
        )

    def clip(
        self,
        recipes: list[PoseRecipe],
        *,
        width: int,
        height: int,
        ground_y: float,
        fps: float,
        rig: str,
        metadata: dict | None = None,
        dynamics: dict | None = None,
    ) -> MotionClip:
        return MotionClip(
            width=width,
            height=height,
            ground_y=ground_y,
            fps=fps,
            rig=rig,
            frames=[self.frame(recipe) for recipe in recipes],
            primary_hand=self.primary_hand,
            secondary_hand=self.secondary_hand,
            metadata=dict(metadata or {}),
            dynamics=dict(dynamics or {}),
        )



def build_clip_from_recipe_data(
    rest_clip: MotionClip,
    data: dict[str, Any],
) -> MotionClip:
    """Build a full motion clip from compact semantic pose recipes."""
    if not rest_clip.frames:
        raise ValueError("Rest motion has no frames")

    version = int(data.get("version", 1))
    if version != 1:
        raise ValueError(f"Unsupported authoring recipe version: {version}")

    author = ParametricAuthor(
        rest_clip.frames[0],
        primary_hand=str(
            data.get("grip", {}).get(
                "primary_hand",
                rest_clip.primary_hand,
            )
        ),
        secondary_hand=str(
            data.get("grip", {}).get(
                "secondary_hand",
                rest_clip.secondary_hand,
            )
        ),
    )
    recipes = [
        PoseRecipe.from_dict(item)
        for item in data.get("poses", [])
    ]
    if not recipes:
        raise ValueError("Authoring recipe contains no poses")

    canvas = data.get("canvas", {})
    timing = data.get("timing", {})

    return author.clip(
        recipes,
        width=int(canvas.get("width", rest_clip.width)),
        height=int(canvas.get("height", rest_clip.height)),
        ground_y=float(canvas.get("ground_y", rest_clip.ground_y)),
        fps=float(timing.get("fps", rest_clip.fps)),
        rig=str(data.get("rig", rest_clip.rig)),
        metadata={
            **rest_clip.metadata,
            **dict(data.get("metadata", {})),
        },
        dynamics={
            **rest_clip.dynamics,
            **dict(data.get("dynamics", {})),
        },
    )


def build_clip_from_recipe_files(
    rest_motion_path: str | Path,
    recipe_path: str | Path,
) -> MotionClip:
    rest_clip = MotionClip.load(rest_motion_path)
    data = json.loads(Path(recipe_path).read_text(encoding="utf-8"))
    return build_clip_from_recipe_data(rest_clip, data)
