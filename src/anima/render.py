from __future__ import annotations

from pathlib import Path
import math
from PIL import Image, ImageDraw

from .biomechanics import analyze_body_kinematics
from .contacts import contact_mode
from .dynamics import analyze_weapon_dynamics
from .model import FramePose, MotionClip, Vec2
from .system_dynamics import analyze_system_dynamics

# Colorblind-safe debug palette. Marker shapes are also different so color is
# never the only left/right cue.
DEBUG = {
    "background": "#1E252A",
    "ground": "#AEB8C2",
    "torso": "#F0E6C5",
    "left": "#56B4E9",
    "right": "#E69F00",
    "weapon": "#9B8CFF",
    "joint": "#FFFFFF",
    "velocity": "#00E5FF",
    "acceleration": "#FFD166",
    "support": "#8A98A6",
    "system": "#B3FFFC",
    "force": "#F5F5F5",
    "pole": "#B0BEC5",
    "ghost": "#56616A",
    "trail": "#D8C7FF",
}
CONTROL = {
    "background": "#1E252A",
    "body": "#DAD3B5",
    "ground": "#626E78",
}
PARTS = {
    "background": "#1E252A",
    "ground": "#626E78",
    "head": "#F0E6C5",
    "torso": "#DAD3B5",
    "left_arm": "#56B4E9",
    "right_arm": "#E69F00",
    "left_leg": "#7AA6FF",
    "right_leg": "#F0C05A",
    "weapon": "#9B8CFF",
}


def _xy(p: Vec2) -> tuple[int, int]:
    return round(p.x), round(p.y)


def _line(
    draw: ImageDraw.ImageDraw,
    a: Vec2,
    b: Vec2,
    fill: str,
    width: int,
) -> None:
    draw.line([_xy(a), _xy(b)], fill=fill, width=width)


def _arrow(
    draw: ImageDraw.ImageDraw,
    start: Vec2,
    delta: Vec2,
    fill: str,
    width: int = 1,
) -> None:
    end = start + delta
    draw.line([_xy(start), _xy(end)], fill=fill, width=width)
    length = delta.length()
    if length <= 1e-6:
        return
    direction = delta * (1.0 / length)
    normal = Vec2(-direction.y, direction.x)
    back = end - direction * 4.0
    left = back + normal * 2.0
    right = back - normal * 2.0
    draw.polygon([_xy(end), _xy(left), _xy(right)], fill=fill)


def render_debug_frame(
    clip: MotionClip,
    frame: FramePose,
    scale: int = 4,
    body_diag: dict | None = None,
    weapon_diag: dict | None = None,
    system_diag: dict | None = None,
    previous_frame: FramePose | None = None,
    history: list[FramePose] | None = None,
) -> Image.Image:
    image = Image.new("RGB", (clip.width, clip.height), DEBUG["background"])
    draw = ImageDraw.Draw(image)
    draw.line(
        [(0, round(clip.ground_y)), (clip.width - 1, round(clip.ground_y))],
        fill=DEBUG["ground"],
        width=2,
    )

    caption = f"F{frame.frame:02d}"
    if frame.time_s is not None:
        caption += f" @{frame.time_s:.3f}s"
    if frame.label:
        caption += f" {frame.label}"
    draw.text((3, 3), caption, fill=DEBUG["joint"])


    # Onion-skin the immediately previous solved pose. This is intentionally
    # monochrome/muted so current left/right colors remain unambiguous.
    if previous_frame is not None:
        ghost_chains = (
            ("shoulder_l", "elbow_l", "hand_l"),
            ("shoulder_r", "elbow_r", "hand_r"),
            ("hip_l", "knee_l", "foot_l"),
            ("hip_r", "knee_r", "foot_r"),
        )
        for names in ghost_chains:
            if all(name in previous_frame.joints for name in names):
                _line(
                    draw,
                    previous_frame.joints[names[0]],
                    previous_frame.joints[names[1]],
                    DEBUG["ghost"],
                    1,
                )
                _line(
                    draw,
                    previous_frame.joints[names[1]],
                    previous_frame.joints[names[2]],
                    DEBUG["ghost"],
                    1,
                )
        if "chest" in previous_frame.joints:
            _line(
                draw,
                previous_frame.root,
                previous_frame.joints["chest"],
                DEBUG["ghost"],
                1,
            )
        _line(
            draw,
            previous_frame.weapon.grip_off,
            previous_frame.weapon.tip,
            DEBUG["ghost"],
            1,
        )

    # Motion trails show trajectory rather than only instantaneous pose.
    if history:
        sword_points = [_xy(item.weapon.tip) for item in history]
        root_points = [_xy(item.root) for item in history]
        if len(sword_points) > 1:
            draw.line(sword_points, fill=DEBUG["trail"], width=1)
        if len(root_points) > 1:
            draw.line(root_points, fill=DEBUG["ghost"], width=1)
        for x, y in sword_points:
            draw.point((x, y), fill=DEBUG["trail"])

    if "chest" in frame.joints:
        _line(draw, frame.root, frame.joints["chest"], DEBUG["torso"], 3)
    if {"chest", "head"}.issubset(frame.joints):
        _line(draw, frame.joints["chest"], frame.joints["head"], DEBUG["torso"], 2)
    if {"shoulder_l", "shoulder_r"}.issubset(frame.joints):
        _line(
            draw,
            frame.joints["shoulder_l"],
            frame.joints["shoulder_r"],
            DEBUG["torso"],
            3,
        )
    if {"hip_l", "hip_r"}.issubset(frame.joints):
        _line(
            draw,
            frame.joints["hip_l"],
            frame.joints["hip_r"],
            DEBUG["torso"],
            3,
        )

    chains = (
        (("shoulder_l", "elbow_l", "hand_l"), DEBUG["left"]),
        (("hip_l", "knee_l", "foot_l"), DEBUG["left"]),
        (("shoulder_r", "elbow_r", "hand_r"), DEBUG["right"]),
        (("hip_r", "knee_r", "foot_r"), DEBUG["right"]),
    )
    for names, color in chains:
        if all(name in frame.joints for name in names):
            _line(draw, frame.joints[names[0]], frame.joints[names[1]], color, 3)
            _line(draw, frame.joints[names[1]], frame.joints[names[2]], color, 3)

    _line(draw, frame.weapon.grip_off, frame.weapon.tip, DEBUG["weapon"], 3)

    for name, point in frame.joints.items():
        x, y = _xy(point)
        radius = 3 if name.startswith(("hand", "foot")) else 2
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=DEBUG["joint"],
        )



    joint_labels = {
        "elbow_l": "EL",
        "elbow_r": "ER",
        "hand_l": "HL",
        "hand_r": "HR",
        "knee_l": "KL",
        "knee_r": "KR",
        "foot_l": "FL",
        "foot_r": "FR",
    }
    for name, label in joint_labels.items():
        if name in frame.joints:
            lx, ly = _xy(frame.joints[name])
            draw.text((lx + 3, ly - 4), label, fill=DEBUG["joint"])

    # IK pole targets: small X markers connected to the solved elbow/knee.
    # These make bend-side mistakes visible even when the final joint still
    # reaches its target.
    for name, pole in frame.ik_poles.items():
        if name not in frame.joints:
            continue
        px, py = _xy(pole)
        draw.line([(px - 2, py - 2), (px + 2, py + 2)], fill=DEBUG["pole"], width=1)
        draw.line([(px - 2, py + 2), (px + 2, py - 2)], fill=DEBUG["pole"], width=1)
        draw.line([_xy(frame.joints[name]), (px, py)], fill=DEBUG["pole"], width=1)

    # Explicit contact-mode labels. P = planted world-space anchor,
    # G = grounded/sliding, F = free.
    for foot_name in ("foot_l", "foot_r"):
        if foot_name not in frame.joints:
            continue
        mode = contact_mode(frame.contacts.get(foot_name))
        marker = {"planted": "P", "grounded": "G", "free": "F"}[mode]
        fx, fy = _xy(frame.joints[foot_name])
        draw.text((fx - 2, fy + 3), marker, fill=DEBUG["ground"])

    # Physics overlay. Color is redundant with labels/shapes so the diagnostics
    # remain readable with red/green color-vision deficiencies.
    if body_diag:
        com_data = body_diag.get("com_px")
        if com_data:
            com = Vec2(float(com_data[0]), float(com_data[1]))
            cx, cy = _xy(com)
            draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], outline=DEBUG["joint"], width=1)
            draw.line([(cx - 5, cy), (cx + 5, cy)], fill=DEBUG["joint"], width=1)
            draw.line([(cx, cy - 5), (cx, cy + 5)], fill=DEBUG["joint"], width=1)
            draw.text((cx + 5, cy - 8), "COM", fill=DEBUG["joint"])

            velocity = body_diag.get("com_velocity_m_s", [0.0, 0.0])
            acceleration = body_diag.get("com_acceleration_m_s2", [0.0, 0.0])
            _arrow(
                draw,
                com,
                Vec2(float(velocity[0]) * 5.0, float(velocity[1]) * 5.0),
                DEBUG["velocity"],
                2,
            )
            _arrow(
                draw,
                com,
                Vec2(float(acceleration[0]) * 0.25, float(acceleration[1]) * 0.25),
                DEBUG["acceleration"],
                1,
            )

        support = body_diag.get("support")
        if support:
            y = round(clip.ground_y) + 2
            draw.line(
                [(round(support["min_x"]), y), (round(support["max_x"]), y)],
                fill=DEBUG["support"],
                width=2,
            )

        speed = float(body_diag.get("com_speed_m_s", 0.0))
        accel = body_diag.get("com_acceleration_m_s2", [0.0, 0.0])
        accel_mag = math.hypot(float(accel[0]), float(accel[1]))
        draw.text((3, 13), f"V {speed:.2f}m/s  A {accel_mag:.1f}m/s2", fill=DEBUG["velocity"])

    if weapon_diag:
        omega = float(weapon_diag.get("angular_velocity_deg_s", 0.0))
        torque = float(weapon_diag.get("estimated_torque_nm", 0.0))
        draw.text((3, 23), f"SWORD w {omega:.0f}deg/s  T {torque:.1f}Nm", fill=DEBUG["weapon"])


    if system_diag:
        system_com_data = system_diag.get("system_com_px")
        if system_com_data:
            sys_com = Vec2(float(system_com_data[0]), float(system_com_data[1]))
            sx, sy = _xy(sys_com)
            draw.polygon(
                [(sx, sy - 4), (sx + 4, sy), (sx, sy + 4), (sx - 4, sy)],
                outline=DEBUG["system"],
            )
            draw.text((sx + 5, sy + 2), "SYS", fill=DEBUG["system"])

        grf = system_diag.get("ground_reaction_force")
        support = system_diag.get("support")
        if grf and support:
            ground_start = Vec2(
                (float(support["min_x"]) + float(support["max_x"])) * 0.5,
                clip.ground_y,
            )
            # Visual diagnostic scale only; labels carry the actual values.
            force_vec = Vec2(
                float(grf["horizontal_n"]) / 150.0,
                -float(grf["vertical_up_n"]) / 150.0,
            )
            _arrow(draw, ground_start, force_vec, DEBUG["force"], 1)
            draw.text(
                (3, 33),
                f"GRF {float(grf['magnitude_n']):.0f}N mu {float(grf['required_friction_ratio']):.2f}",
                fill=DEBUG["force"],
            )

    # Root/pelvis anchor: diamond. This makes root drift immediately visible.
    rx, ry = _xy(frame.root)
    draw.polygon(
        [(rx, ry - 4), (rx + 4, ry), (rx, ry + 4), (rx - 4, ry)],
        outline=DEBUG["torso"],
    )

    # Head outline makes torso/head motion easier to inspect.
    if "head" in frame.joints:
        hx, hy = _xy(frame.joints["head"])
        draw.ellipse([hx - 5, hy - 6, hx + 5, hy + 5], outline=DEBUG["torso"], width=2)

    # Redundant marker shapes: left=box, right=circle.
    for name in ("hand_l", "foot_l"):
        if name in frame.joints:
            x, y = _xy(frame.joints[name])
            draw.rectangle([x - 2, y - 2, x + 2, y + 2], outline=DEBUG["left"])
    for name in ("hand_r", "foot_r"):
        if name in frame.joints:
            x, y = _xy(frame.joints[name])
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], outline=DEBUG["right"])

    if scale != 1:
        image = image.resize(
            (clip.width * scale, clip.height * scale),
            Image.Resampling.NEAREST,
        )
    return image


def render_control_frame(
    clip: MotionClip,
    frame: FramePose,
    scale: int = 4,
) -> Image.Image:
    """Render a simple mannequin control image for downstream image generation."""
    image = Image.new("RGB", (clip.width, clip.height), CONTROL["background"])
    draw = ImageDraw.Draw(image)
    draw.line(
        [(0, round(clip.ground_y)), (clip.width - 1, round(clip.ground_y))],
        fill=CONTROL["ground"],
        width=1,
    )
    body = CONTROL["body"]

    chains = (
        ("shoulder_l", "elbow_l", "hand_l"),
        ("shoulder_r", "elbow_r", "hand_r"),
        ("hip_l", "knee_l", "foot_l"),
        ("hip_r", "knee_r", "foot_r"),
    )
    for names in chains:
        if all(name in frame.joints for name in names):
            for a, b in zip(names, names[1:]):
                _line(draw, frame.joints[a], frame.joints[b], body, 5)
            for name in names[1:]:
                x, y = _xy(frame.joints[name])
                draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=body)

    if all(
        name in frame.joints
        for name in ("shoulder_l", "shoulder_r", "hip_r", "hip_l")
    ):
        draw.polygon(
            [
                _xy(frame.joints["shoulder_l"]),
                _xy(frame.joints["shoulder_r"]),
                _xy(frame.joints["hip_r"]),
                _xy(frame.joints["hip_l"]),
            ],
            fill=body,
        )

    head = frame.joints.get("head")
    if head:
        x, y = _xy(head)
        if {"shoulder_l", "shoulder_r"}.issubset(frame.joints):
            shoulder_mid = Vec2(
                (frame.joints["shoulder_l"].x + frame.joints["shoulder_r"].x) * 0.5,
                (frame.joints["shoulder_l"].y + frame.joints["shoulder_r"].y) * 0.5,
            )
            neck_target = Vec2(head.x, head.y + 5.0)
            _line(draw, shoulder_mid, neck_target, body, 5)
        draw.ellipse([x - 5, y - 6, x + 5, y + 5], fill=body)

    _line(draw, frame.weapon.grip_off, frame.weapon.tip, body, 3)
    gx, gy = _xy(frame.weapon.grip_main)
    draw.line([(gx - 4, gy - 1), (gx + 4, gy + 1)], fill=body, width=2)

    if scale != 1:
        image = image.resize(
            (clip.width * scale, clip.height * scale),
            Image.Resampling.NEAREST,
        )
    return image



def render_parts_frame(
    clip: MotionClip,
    frame: FramePose,
    scale: int = 4,
) -> Image.Image:
    """Render a layered/segmentation guide for downstream generative rendering.

    Colors identify semantic body parts; they are not final art colors.
    """
    image = Image.new("RGB", (clip.width, clip.height), PARTS["background"])
    draw = ImageDraw.Draw(image)
    draw.line(
        [(0, round(clip.ground_y)), (clip.width - 1, round(clip.ground_y))],
        fill=PARTS["ground"],
        width=1,
    )

    if all(
        name in frame.joints
        for name in ("shoulder_l", "shoulder_r", "hip_r", "hip_l")
    ):
        draw.polygon(
            [
                _xy(frame.joints["shoulder_l"]),
                _xy(frame.joints["shoulder_r"]),
                _xy(frame.joints["hip_r"]),
                _xy(frame.joints["hip_l"]),
            ],
            fill=PARTS["torso"],
        )

    head = frame.joints.get("head")
    if head:
        x, y = _xy(head)
        if {"shoulder_l", "shoulder_r"}.issubset(frame.joints):
            shoulder_mid = Vec2(
                (frame.joints["shoulder_l"].x + frame.joints["shoulder_r"].x) * 0.5,
                (frame.joints["shoulder_l"].y + frame.joints["shoulder_r"].y) * 0.5,
            )
            neck_target = Vec2(head.x, head.y + 5.0)
            _line(draw, shoulder_mid, neck_target, PARTS["torso"], 5)
        draw.ellipse([x - 5, y - 6, x + 5, y + 5], fill=PARTS["head"])

    segment_groups = (
        (("shoulder_l", "elbow_l", "hand_l"), PARTS["left_arm"]),
        (("shoulder_r", "elbow_r", "hand_r"), PARTS["right_arm"]),
        (("hip_l", "knee_l", "foot_l"), PARTS["left_leg"]),
        (("hip_r", "knee_r", "foot_r"), PARTS["right_leg"]),
    )
    for names, color in segment_groups:
        if not all(name in frame.joints for name in names):
            continue
        _line(draw, frame.joints[names[0]], frame.joints[names[1]], color, 5)
        _line(draw, frame.joints[names[1]], frame.joints[names[2]], color, 5)
        for name in names[1:]:
            x, y = _xy(frame.joints[name])
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)

    _line(draw, frame.weapon.grip_off, frame.weapon.tip, PARTS["weapon"], 3)
    gx, gy = _xy(frame.weapon.grip_main)
    draw.ellipse([gx - 2, gy - 2, gx + 2, gy + 2], fill=PARTS["weapon"])

    if scale != 1:
        image = image.resize(
            (clip.width * scale, clip.height * scale),
            Image.Resampling.NEAREST,
        )
    return image



def render_inspection_frame(
    clip: MotionClip,
    frame: FramePose,
    body_diag: dict | None = None,
    weapon_diag: dict | None = None,
    system_diag: dict | None = None,
    previous_frame: FramePose | None = None,
    history: list[FramePose] | None = None,
    scale: int = 4,
) -> Image.Image:
    """Large diagnostic panel: uncluttered skeleton left, metrics sidebar right."""
    panel_width = clip.width * 2
    image = Image.new("RGB", (panel_width, clip.height), DEBUG["background"])
    draw = ImageDraw.Draw(image)

    # Ground and motion history only occupy the left motion viewport.
    gy = round(clip.ground_y)
    draw.line([(0, gy), (clip.width - 1, gy)], fill=DEBUG["ground"], width=2)

    if history:
        tip_points = [_xy(item.weapon.tip) for item in history]
        root_points = [_xy(item.root) for item in history]
        if len(tip_points) > 1:
            draw.line(tip_points, fill=DEBUG["trail"], width=1)
        if len(root_points) > 1:
            draw.line(root_points, fill=DEBUG["ghost"], width=1)

    if previous_frame is not None:
        for names in (
            ("shoulder_l", "elbow_l", "hand_l"),
            ("shoulder_r", "elbow_r", "hand_r"),
            ("hip_l", "knee_l", "foot_l"),
            ("hip_r", "knee_r", "foot_r"),
        ):
            if all(name in previous_frame.joints for name in names):
                _line(draw, previous_frame.joints[names[0]], previous_frame.joints[names[1]], DEBUG["ghost"], 1)
                _line(draw, previous_frame.joints[names[1]], previous_frame.joints[names[2]], DEBUG["ghost"], 1)
        _line(draw, previous_frame.weapon.grip_off, previous_frame.weapon.tip, DEBUG["ghost"], 1)

    # Current skeleton.
    if "chest" in frame.joints:
        _line(draw, frame.root, frame.joints["chest"], DEBUG["torso"], 3)
    if {"chest", "head"}.issubset(frame.joints):
        _line(draw, frame.joints["chest"], frame.joints["head"], DEBUG["torso"], 2)
    if {"shoulder_l", "shoulder_r"}.issubset(frame.joints):
        _line(draw, frame.joints["shoulder_l"], frame.joints["shoulder_r"], DEBUG["torso"], 3)
    if {"hip_l", "hip_r"}.issubset(frame.joints):
        _line(draw, frame.joints["hip_l"], frame.joints["hip_r"], DEBUG["torso"], 3)

    for names, color in (
        (("shoulder_l", "elbow_l", "hand_l"), DEBUG["left"]),
        (("hip_l", "knee_l", "foot_l"), DEBUG["left"]),
        (("shoulder_r", "elbow_r", "hand_r"), DEBUG["right"]),
        (("hip_r", "knee_r", "foot_r"), DEBUG["right"]),
    ):
        if all(name in frame.joints for name in names):
            _line(draw, frame.joints[names[0]], frame.joints[names[1]], color, 3)
            _line(draw, frame.joints[names[1]], frame.joints[names[2]], color, 3)

    _line(draw, frame.weapon.grip_off, frame.weapon.tip, DEBUG["weapon"], 3)

    # Compact joint markers without overlapping labels.
    for name, point in frame.joints.items():
        if name not in {
            "head", "elbow_l", "elbow_r", "hand_l", "hand_r",
            "knee_l", "knee_r", "foot_l", "foot_r",
        }:
            continue
        x, y = _xy(point)
        radius = 2
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=DEBUG["joint"])

    # IK poles are rendered as crosses.
    for name, pole in frame.ik_poles.items():
        if name not in frame.joints:
            continue
        px, py = _xy(pole)
        draw.line([(px - 2, py - 2), (px + 2, py + 2)], fill=DEBUG["pole"], width=1)
        draw.line([(px - 2, py + 2), (px + 2, py - 2)], fill=DEBUG["pole"], width=1)

    # COM/system COM.
    if body_diag and body_diag.get("com_px"):
        com = Vec2.from_any(body_diag["com_px"])
        cx, cy = _xy(com)
        draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], outline=DEBUG["joint"], width=1)
        velocity = Vec2.from_any(body_diag.get("com_velocity_m_s", [0.0, 0.0]))
        acceleration = Vec2.from_any(body_diag.get("com_acceleration_m_s2", [0.0, 0.0]))
        _arrow(draw, com, velocity * 5.0, DEBUG["velocity"], 2)
        _arrow(draw, com, acceleration * 0.25, DEBUG["acceleration"], 1)

    if system_diag and system_diag.get("system_com_px"):
        sys_com = Vec2.from_any(system_diag["system_com_px"])
        sx, sy = _xy(sys_com)
        draw.polygon(
            [(sx, sy - 4), (sx + 4, sy), (sx, sy + 4), (sx - 4, sy)],
            outline=DEBUG["system"],
        )

    # Sidebar separator.
    sidebar_x = clip.width
    draw.line([(sidebar_x, 0), (sidebar_x, clip.height - 1)], fill=DEBUG["support"], width=1)

    timestamp = frame.time_s if frame.time_s is not None else frame.frame / clip.fps
    lines = [
        f"F{frame.frame:02d} {frame.label or ''}",
        f"t {timestamp:.3f}s",
        ("STOP velocity=0" if frame.kinematic_stop else ""),
        "",
        "LEFT  square/blue",
        "RIGHT circle/orange",
        "P=planted G=grounded",
    ]

    if body_diag:
        velocity = Vec2.from_any(body_diag.get("com_velocity_m_s", [0.0, 0.0]))
        acceleration = Vec2.from_any(body_diag.get("com_acceleration_m_s2", [0.0, 0.0]))
        lines.extend([
            "",
            f"COM v {velocity.length():.2f} m/s",
            f"COM a {acceleration.length():.1f} m/s2",
        ])
        support = body_diag.get("support")
        if support:
            lines.append(f"stability {float(support.get('stability_margin_px', 0.0)):.1f}px")

    if weapon_diag:
        lines.extend([
            "",
            f"sword w {float(weapon_diag.get('angular_velocity_deg_s', 0.0)):.0f} deg/s",
            f"sword a {float(weapon_diag.get('angular_acceleration_deg_s2', 0.0)):.0f}",
            f"torque {float(weapon_diag.get('estimated_torque_nm', 0.0)):.1f} Nm",
            f"handle {float(weapon_diag.get('handle_force_magnitude_n', 0.0)):.0f} N",
            f"KE {float(weapon_diag.get('total_kinetic_energy_j', 0.0)):.1f} J",
        ])

    if system_diag and system_diag.get("ground_reaction_force"):
        grf = system_diag["ground_reaction_force"]
        lines.extend([
            "",
            f"GRF {float(grf.get('magnitude_n', 0.0)):.0f} N",
            f"mu req {float(grf.get('required_friction_ratio', 0.0)):.2f}",
        ])

    y = 3
    for line in lines:
        draw.text((sidebar_x + 4, y), line, fill=DEBUG["joint"])
        y += 8

    # Contact labels directly below feet.
    for foot_name in ("foot_l", "foot_r"):
        if foot_name not in frame.joints:
            continue
        mode = contact_mode(frame.contacts.get(foot_name))
        marker = {"planted": "P", "grounded": "G", "free": "F"}[mode]
        fx, fy = _xy(frame.joints[foot_name])
        draw.text((fx - 2, min(clip.height - 8, fy + 3)), marker, fill=DEBUG["ground"])

    if scale != 1:
        image = image.resize(
            (panel_width * scale, clip.height * scale),
            Image.Resampling.NEAREST,
        )
    return image


def make_sheet(
    frames: list[Image.Image],
    columns: int = 4,
    gap: int = 0,
) -> Image.Image:
    if not frames:
        raise ValueError("No frames")

    width, height = frames[0].size
    rows = (len(frames) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (
            columns * width + (columns - 1) * gap,
            rows * height + (rows - 1) * gap,
        ),
        frames[0].getpixel((0, 0)),
    )
    for index, frame in enumerate(frames):
        x = (index % columns) * (width + gap)
        y = (index // columns) * (height + gap)
        sheet.paste(frame, (x, y))
    return sheet


def _gif_durations_ms(clip: MotionClip) -> list[int]:
    """Return per-frame GIF hold durations using the clip's real timestamps."""
    if not clip.frames:
        return []
    times = clip.times_s()
    if len(times) == 1:
        return [max(1, round(1000.0 / clip.fps))]

    durations = [
        max(1, round((right - left) * 1000.0))
        for left, right in zip(times, times[1:])
    ]
    durations.append(durations[-1])
    return durations


def _scaled(image: Image.Image, scale: int) -> Image.Image:
    if scale == 1:
        return image
    return image.resize(
        (image.width * scale, image.height * scale),
        Image.Resampling.NEAREST,
    )


def export_render_set(
    clip: MotionClip,
    output: str | Path,
    scale: int = 4,
    columns: int = 4,
) -> None:
    """Export canonical 1x assets plus nearest-neighbor previews.

    Canonical frames always retain the clip canvas size (for example 128x128).
    The scale argument affects preview assets only.
    """
    output = Path(output)
    debug_dir = output / "debug_frames"
    control_dir = output / "control_frames"
    parts_dir = output / "parts_frames"
    inspect_dir = output / "inspection_frames"
    debug_dir.mkdir(parents=True, exist_ok=True)
    control_dir.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)
    inspect_dir.mkdir(parents=True, exist_ok=True)

    body_report = analyze_body_kinematics(clip)
    weapon_report = analyze_weapon_dynamics(clip)
    system_report = analyze_system_dynamics(clip, body_report, weapon_report)
    body_by_frame = {item["frame"]: item for item in body_report["frames"]}
    weapon_by_frame = {item["frame"]: item for item in weapon_report["frames"]}
    system_by_frame = {item["frame"]: item for item in system_report["frames"]}

    debug: list[Image.Image] = []
    control: list[Image.Image] = []
    parts: list[Image.Image] = []
    inspection: list[Image.Image] = []
    for index, frame in enumerate(clip.frames):
        debug_frame = render_debug_frame(
            clip,
            frame,
            scale=1,
            body_diag=body_by_frame.get(frame.frame),
            weapon_diag=weapon_by_frame.get(frame.frame),
            system_diag=system_by_frame.get(frame.frame),
            previous_frame=(clip.frames[index - 1] if index > 0 else None),
            history=clip.frames[: index + 1],
        )
        control_frame = render_control_frame(clip, frame, scale=1)
        parts_frame = render_parts_frame(clip, frame, scale=1)
        inspection_frame = render_inspection_frame(
            clip,
            frame,
            body_diag=body_by_frame.get(frame.frame),
            weapon_diag=weapon_by_frame.get(frame.frame),
            system_diag=system_by_frame.get(frame.frame),
            previous_frame=(clip.frames[index - 1] if index > 0 else None),
            history=clip.frames[: index + 1],
            scale=1,
        )
        debug_frame.save(debug_dir / f"frame_{index:02d}.png")
        control_frame.save(control_dir / f"frame_{index:02d}.png")
        parts_frame.save(parts_dir / f"frame_{index:02d}.png")
        inspection_frame.save(inspect_dir / f"frame_{index:02d}.png")
        debug.append(debug_frame)
        control.append(control_frame)
        parts.append(parts_frame)
        inspection.append(inspection_frame)

    debug_sheet = make_sheet(debug, columns=columns)
    control_sheet = make_sheet(control, columns=columns)
    parts_sheet = make_sheet(parts, columns=columns)
    debug_sheet.save(output / "debug_sheet.png")
    control_sheet.save(output / "control_sheet.png")
    parts_sheet.save(output / "parts_sheet.png")

    durations = _gif_durations_ms(clip)

    # Debug is intentionally larger than the control preview: it is a
    # diagnostic instrument, not an art asset. 128px source frames become
    # at least 1024px so elbow flips, foot drift, and root motion are obvious.
    debug_scale = max(8, scale * 2)
    control_scale = max(4, scale)
    debug_preview = [_scaled(frame, debug_scale) for frame in debug]
    control_preview = [_scaled(frame, control_scale) for frame in control]
    parts_preview = [_scaled(frame, control_scale) for frame in parts]
    inspection_preview = [_scaled(frame, 4) for frame in inspection]

    debug_preview[0].save(
        output / "debug_preview.gif",
        save_all=True,
        append_images=debug_preview[1:],
        duration=durations,
        loop=0,
        disposal=2,
    )
    control_preview[0].save(
        output / "control_preview.gif",
        save_all=True,
        append_images=control_preview[1:],
        duration=durations,
        loop=0,
        disposal=2,
    )
    parts_preview[0].save(
        output / "parts_preview.gif",
        save_all=True,
        append_images=parts_preview[1:],
        duration=durations,
        loop=0,
        disposal=2,
    )
    inspection_preview[0].save(
        output / "inspection_preview.gif",
        save_all=True,
        append_images=inspection_preview[1:],
        duration=durations,
        loop=0,
        disposal=2,
    )

    _scaled(debug_sheet, debug_scale).save(output / "debug_sheet.preview.png")
    _scaled(control_sheet, control_scale).save(output / "control_sheet.preview.png")
    _scaled(parts_sheet, control_scale).save(output / "parts_sheet.preview.png")

    # Side-by-side review GIF at the larger debug scale. Useful for checking
    # that the mannequin still matches the solved skeleton.
    control_review = [_scaled(frame, debug_scale) for frame in control]
    review_frames: list[Image.Image] = []
    for debug_frame, control_frame in zip(debug_preview, control_review):
        review = Image.new(
            "RGB",
            (debug_frame.width + control_frame.width, debug_frame.height),
            DEBUG["background"],
        )
        review.paste(debug_frame, (0, 0))
        review.paste(control_frame, (debug_frame.width, 0))
        review_frames.append(review)

    review_frames[0].save(
        output / "review_preview.gif",
        save_all=True,
        append_images=review_frames[1:],
        duration=durations,
        loop=0,
        disposal=2,
    )
