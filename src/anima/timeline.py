from __future__ import annotations

from dataclasses import replace
import math

from .model import FramePose, MotionClip, Vec2, WeaponPose


def _mix(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _mix_vec(a: Vec2, b: Vec2, t: float) -> Vec2:
    return Vec2(_mix(a.x, b.x, t), _mix(a.y, b.y, t))


def _ease(t: float, mode: str) -> float:
    if mode == "linear":
        return t
    if mode == "smoothstep":
        return t * t * (3.0 - 2.0 * t)
    raise ValueError(f"Unsupported easing: {mode}")


def _weapon_angle(frame: FramePose) -> float:
    delta = frame.weapon.tip - frame.weapon.grip_main
    return math.atan2(delta.y, delta.x)


def _unwrap_near(value: float, reference: float) -> float:
    while value - reference > math.pi:
        value -= 2.0 * math.pi
    while value - reference < -math.pi:
        value += 2.0 * math.pi
    return value


def _weapon_from_main_angle(
    source: FramePose,
    main: Vec2,
    angle: float,
) -> WeaponPose:
    blade = source.weapon.tip - source.weapon.grip_main
    off = source.weapon.grip_off - source.weapon.grip_main
    blade_len = blade.length()
    spacing = off.length()
    direction = Vec2(math.cos(angle), math.sin(angle))
    projection = off.x * blade.normalized().x + off.y * blade.normalized().y
    sign = 1.0 if projection >= 0 else -1.0
    return WeaponPose(
        grip_main=main,
        grip_off=main + direction * (spacing * sign),
        tip=main + direction * blade_len,
    )


def _interpolate_pose(
    a: FramePose,
    b: FramePose,
    frame_no: int,
    t: float,
) -> FramePose:
    """Two-key interpolation used for linear/smoothstep modes.

    Sword orientation follows a true angular arc instead of linearly moving the
    tip through space; the weapon therefore remains rotationally coherent even
    before the later rigid-weapon normalization pass.
    """
    names = a.joints.keys() & b.joints.keys()
    joints = {
        name: _mix_vec(a.joints[name], b.joints[name], t)
        for name in names
    }

    contact_source = a if t < 0.5 else b

    pole_names = a.ik_poles.keys() & b.ik_poles.keys()
    ik_poles = {
        name: _mix_vec(a.ik_poles[name], b.ik_poles[name], t)
        for name in pole_names
    }

    main = _mix_vec(a.weapon.grip_main, b.weapon.grip_main, t)
    angle_a = _weapon_angle(a)
    angle_b = _unwrap_near(_weapon_angle(b), angle_a)
    angle = _mix(angle_a, angle_b, t)

    raw_t = (frame_no - a.frame) / max(b.frame - a.frame, 1)
    time_s = (
        _mix(float(a.time_s), float(b.time_s), raw_t)
        if a.time_s is not None and b.time_s is not None
        else None
    )

    return FramePose(
        frame=frame_no,
        root=_mix_vec(a.root, b.root, t),
        joints=joints,
        weapon=_weapon_from_main_angle(a, main, angle),
        contacts=dict(contact_source.contacts),
        ik_poles=ik_poles,
        label=None,
        time_s=time_s,
    )


def _hermite_scalar(
    p1: float,
    p2: float,
    m1: float,
    m2: float,
    segment_dt: float,
    u: float,
) -> float:
    u2 = u * u
    u3 = u2 * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    return (
        h00 * p1
        + h10 * segment_dt * m1
        + h01 * p2
        + h11 * segment_dt * m2
    )


def _hermite_vec(
    p1: Vec2,
    p2: Vec2,
    m1: Vec2,
    m2: Vec2,
    segment_dt: float,
    u: float,
) -> Vec2:
    return Vec2(
        _hermite_scalar(p1.x, p2.x, m1.x, m2.x, segment_dt, u),
        _hermite_scalar(p1.y, p2.y, m1.y, m2.y, segment_dt, u),
    )


def _vec_tangent(
    before: Vec2 | None,
    current: Vec2,
    after: Vec2,
    t_before: float | None,
    t_current: float,
    t_after: float,
) -> Vec2:
    if before is None or t_before is None:
        return (after - current) * (1.0 / max(t_after - t_current, 1e-9))
    return (after - before) * (1.0 / max(t_after - t_before, 1e-9))


def _scalar_tangent(
    before: float | None,
    current: float,
    after: float,
    t_before: float | None,
    t_current: float,
    t_after: float,
) -> float:
    if before is None or t_before is None:
        return (after - current) / max(t_after - t_current, 1e-9)
    return (after - before) / max(t_after - t_before, 1e-9)


def _interpolate_pose_hermite(
    previous: FramePose | None,
    left: FramePose,
    right: FramePose,
    following: FramePose | None,
    frame_no: int,
) -> FramePose:
    """C1-continuous keyframe interpolation.

    Internal keyframes carry velocity through the pose rather than forcing an
    artificial stop at every authored key. That matters particularly for a
    heavy weapon: anticipation, acceleration, impact and follow-through should
    form one continuous trajectory.
    """
    explicit_time = left.time_s is not None and right.time_s is not None
    t1 = float(left.time_s) if explicit_time else float(left.frame)
    t2 = float(right.time_s) if explicit_time else float(right.frame)
    segment_dt = t2 - t1
    raw_frame_u = (frame_no - left.frame) / max(right.frame - left.frame, 1)
    current_t = _mix(t1, t2, raw_frame_u)
    u = (current_t - t1) / segment_dt

    if previous:
        t0 = (
            float(previous.time_s)
            if explicit_time and previous.time_s is not None
            else float(previous.frame)
        )
    else:
        t0 = None
    if following:
        t3 = (
            float(following.time_s)
            if explicit_time and following.time_s is not None
            else float(following.frame)
        )
    else:
        t3 = None

    names = left.joints.keys() & right.joints.keys()
    joints: dict[str, Vec2] = {}
    for name in names:
        p0 = previous.joints.get(name) if previous and name in previous.joints else None
        p1 = left.joints[name]
        p2 = right.joints[name]
        p3 = following.joints.get(name) if following and name in following.joints else None

        m1 = _vec_tangent(p0, p1, p2, t0, t1, t2)
        if p3 is None or t3 is None:
            m2 = (p2 - p1) * (1.0 / max(t2 - t1, 1e-9))
        else:
            m2 = (p3 - p1) * (1.0 / max(t3 - t1, 1e-9))
        joints[name] = _hermite_vec(p1, p2, m1, m2, segment_dt, u)

    root_m1 = _vec_tangent(
        previous.root if previous else None,
        left.root,
        right.root,
        t0,
        t1,
        t2,
    )
    if following is None or t3 is None:
        root_m2 = (right.root - left.root) * (1.0 / max(segment_dt, 1e-9))
    else:
        root_m2 = (following.root - left.root) * (1.0 / max(t3 - t1, 1e-9))
    root = _hermite_vec(left.root, right.root, root_m1, root_m2, segment_dt, u)

    pole_names = left.ik_poles.keys() & right.ik_poles.keys()
    ik_poles: dict[str, Vec2] = {}
    for name in pole_names:
        p0 = previous.ik_poles.get(name) if previous and name in previous.ik_poles else None
        p1 = left.ik_poles[name]
        p2 = right.ik_poles[name]
        p3 = following.ik_poles.get(name) if following and name in following.ik_poles else None
        m1 = _vec_tangent(p0, p1, p2, t0, t1, t2)
        if p3 is None or t3 is None:
            m2 = (p2 - p1) * (1.0 / max(segment_dt, 1e-9))
        else:
            m2 = (p3 - p1) * (1.0 / max(t3 - t1, 1e-9))
        ik_poles[name] = _hermite_vec(p1, p2, m1, m2, segment_dt, u)

    main0 = previous.weapon.grip_main if previous else None
    main1 = left.weapon.grip_main
    main2 = right.weapon.grip_main
    main3 = following.weapon.grip_main if following else None
    main_m1 = _vec_tangent(main0, main1, main2, t0, t1, t2)
    if main3 is None or t3 is None:
        main_m2 = (main2 - main1) * (1.0 / max(segment_dt, 1e-9))
    else:
        main_m2 = (main3 - main1) * (1.0 / max(t3 - t1, 1e-9))
    main = _hermite_vec(main1, main2, main_m1, main_m2, segment_dt, u)

    angle1 = _weapon_angle(left)
    angle2 = _unwrap_near(_weapon_angle(right), angle1)
    angle0 = None
    if previous:
        angle0 = _unwrap_near(_weapon_angle(previous), angle1)
    angle3 = None
    if following:
        angle3 = _unwrap_near(_weapon_angle(following), angle2)

    angle_m1 = _scalar_tangent(angle0, angle1, angle2, t0, t1, t2)
    if angle3 is None or t3 is None:
        angle_m2 = (angle2 - angle1) / max(segment_dt, 1e-9)
    else:
        angle_m2 = (angle3 - angle1) / max(t3 - t1, 1e-9)
    angle = _hermite_scalar(angle1, angle2, angle_m1, angle_m2, segment_dt, u)

    contact_source = left if u < 0.5 else right

    time_s = current_t if explicit_time else None

    return FramePose(
        frame=frame_no,
        root=root,
        joints=joints,
        weapon=_weapon_from_main_angle(left, main, angle),
        contacts=dict(contact_source.contacts),
        ik_poles=ik_poles,
        label=None,
        time_s=time_s,
    )


def densify_clip(
    clip: MotionClip,
    easing: str = "hermite",
) -> MotionClip:
    """Fill missing integer frames between sparse keyframes.

    hermite: C1-continuous trajectories through internal keyframes.
    smoothstep/linear: local two-key interpolation, useful for deliberate stops.
    """
    if len(clip.frames) < 2:
        return clip

    # Validate explicit timestamp mode before interpolating.
    clip.times_s()
    keyframes = sorted(clip.frames, key=lambda frame: frame.frame)
    if len({frame.frame for frame in keyframes}) != len(keyframes):
        raise ValueError("Duplicate keyframe numbers are not allowed")

    output: list[FramePose] = []
    for index, left in enumerate(keyframes[:-1]):
        right = keyframes[index + 1]
        if right.frame <= left.frame:
            raise ValueError("Keyframes must be strictly increasing")

        if not output:
            output.append(left)

        previous = keyframes[index - 1] if index > 0 else None
        following = keyframes[index + 2] if index + 2 < len(keyframes) else None

        gap = right.frame - left.frame
        for frame_no in range(left.frame + 1, right.frame):
            if easing == "hermite":
                interpolated = _interpolate_pose_hermite(
                    previous,
                    left,
                    right,
                    following,
                    frame_no,
                )
            else:
                raw_t = (frame_no - left.frame) / gap
                interpolated = _interpolate_pose(
                    left,
                    right,
                    frame_no,
                    _ease(raw_t, easing),
                )
            output.append(interpolated)
        output.append(right)

    return replace(clip, frames=output)
