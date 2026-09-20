from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Vec2:
    x: float
    y: float

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vec2":
        return Vec2(self.x * scalar, self.y * scalar)

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def normalized(self, fallback: "Vec2" | None = None) -> "Vec2":
        length = self.length()
        if length <= 1e-9:
            return fallback or Vec2(1.0, 0.0)
        return Vec2(self.x / length, self.y / length)

    def distance_to(self, other: "Vec2") -> float:
        return (self - other).length()

    def rounded(self) -> "Vec2":
        return Vec2(round(self.x), round(self.y))

    def as_list(self) -> list[float]:
        return [self.x, self.y]

    @classmethod
    def from_any(cls, value: Any) -> "Vec2":
        return cls(float(value[0]), float(value[1]))


@dataclass
class WeaponPose:
    grip_main: Vec2
    grip_off: Vec2
    tip: Vec2

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WeaponPose":
        return cls(
            grip_main=Vec2.from_any(data["grip_main"]),
            grip_off=Vec2.from_any(data["grip_off"]),
            tip=Vec2.from_any(data["tip"]),
        )

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "grip_main": self.grip_main.as_list(),
            "grip_off": self.grip_off.as_list(),
            "tip": self.tip.as_list(),
        }


@dataclass
class FramePose:
    frame: int
    root: Vec2
    joints: dict[str, Vec2]
    weapon: WeaponPose
    contacts: dict[str, bool | str] = field(default_factory=dict)
    ik_poles: dict[str, Vec2] = field(default_factory=dict)
    label: str | None = None
    time_s: float | None = None
    kinematic_stop: bool = False
    layer_order: list[str] = field(default_factory=list)
    reference: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FramePose":
        return cls(
            frame=int(data["frame"]),
            root=Vec2.from_any(data["root"]),
            joints={name: Vec2.from_any(point) for name, point in data["joints"].items()},
            weapon=WeaponPose.from_dict(data["weapon"]),
            contacts=dict(data.get("contacts", {})),
            ik_poles={
                name: Vec2.from_any(point)
                for name, point in data.get("ik_poles", {}).items()
            },
            label=data.get("label"),
            time_s=(float(data["time_s"]) if data.get("time_s") is not None else None),
            kinematic_stop=bool(data.get("kinematic_stop", False)),
            layer_order=list(data.get("layer_order", [])),
            reference=dict(data.get("reference", {})),
        )

    def shifted(self, delta: Vec2) -> "FramePose":
        return FramePose(
            frame=self.frame,
            root=self.root + delta,
            joints={name: point + delta for name, point in self.joints.items()},
            weapon=WeaponPose(
                self.weapon.grip_main + delta,
                self.weapon.grip_off + delta,
                self.weapon.tip + delta,
            ),
            contacts=dict(self.contacts),
            ik_poles={name: point + delta for name, point in self.ik_poles.items()},
            label=self.label,
            time_s=self.time_s,
            kinematic_stop=self.kinematic_stop,
            layer_order=list(self.layer_order),
            reference=dict(self.reference),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "frame": self.frame,
            "root": self.root.as_list(),
            "joints": {name: point.as_list() for name, point in self.joints.items()},
            "weapon": self.weapon.to_dict(),
            "contacts": dict(self.contacts),
            "ik_poles": {
                name: point.as_list()
                for name, point in self.ik_poles.items()
            },
        }
        if self.label:
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


@dataclass
class MotionClip:
    width: int
    height: int
    ground_y: float
    fps: float
    rig: str
    frames: list[FramePose]
    primary_hand: str = "hand_r"
    secondary_hand: str = "hand_l"
    metadata: dict[str, Any] = field(default_factory=dict)
    dynamics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MotionClip":
        canvas = data["canvas"]
        timing = data.get("timing", {})
        grip = data.get("grip", {})
        return cls(
            width=int(canvas["width"]),
            height=int(canvas["height"]),
            ground_y=float(canvas["ground_y"]),
            fps=float(timing.get("fps", 12.0)),
            rig=str(data.get("rig", "humanoid_twohand_sword_v1")),
            frames=[FramePose.from_dict(frame) for frame in data["frames"]],
            primary_hand=str(grip.get("primary_hand", "hand_r")),
            secondary_hand=str(grip.get("secondary_hand", "hand_l")),
            metadata=dict(data.get("metadata", {})),
            dynamics=dict(data.get("dynamics", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "MotionClip":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "canvas": {
                "width": self.width,
                "height": self.height,
                "ground_y": self.ground_y,
            },
            "rig": self.rig,
            "timing": {"fps": self.fps},
            "grip": {
                "primary_hand": self.primary_hand,
                "secondary_hand": self.secondary_hand,
            },
            "metadata": self.metadata,
            "dynamics": self.dynamics,
            "frames": [frame.to_dict() for frame in self.frames],
        }

    def times_s(self) -> list[float]:
        """Return monotonically increasing pose timestamps.

        If explicit time_s values are present, every frame must define one.
        Otherwise frame_number/fps is the canonical clock.
        """
        explicit = [getattr(frame, "time_s", None) is not None for frame in self.frames]
        if any(explicit) and not all(explicit):
            raise ValueError("Either all frames define time_s or none of them do")

        if all(explicit) and self.frames:
            times = [
                float(getattr(frame, "time_s"))
                for frame in self.frames
                if getattr(frame, "time_s", None) is not None
            ]
        else:
            times = [frame.frame / self.fps for frame in self.frames]

        for previous, current in zip(times, times[1:]):
            if current <= previous:
                raise ValueError("Frame timestamps must be strictly increasing")
        return times

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
