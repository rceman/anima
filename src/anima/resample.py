from __future__ import annotations

from dataclasses import replace
import bisect
import math

from .model import FramePose, MotionClip, Vec2
from .timeline import (
    _hermite_scalar,
    _hermite_vec,
    _scalar_tangent,
    _unwrap_near,
    _vec_tangent,
    _weapon_angle,
    _weapon_from_main_angle,
)


def _source_times(clip: MotionClip) -> list[float]:
    times = clip.times_s()
    if not times:
        raise ValueError("Cannot resample an empty motion clip")
    return times


def _sample_times(
    start_s: float,
    end_s: float,
    fps: float,
    required_times: list[float] | None = None,
) -> list[float]:
    if fps <= 0.0:
        raise ValueError("fps must be > 0")
    if end_s < start_s:
        raise ValueError("Motion time range is reversed")
    if math.isclose(start_s, end_s, abs_tol=1e-12):
        return [start_s]

    dt = 1.0 / fps
    duration = end_s - start_s
    whole_steps = int(math.floor(duration / dt + 1e-9))
    times = [start_s + step * dt for step in range(whole_steps + 1)]

    # The exact authored endpoint is authoritative. If it does not land on the
    # requested sample grid, append it as a shorter final interval rather than
    # silently trimming or stretching the motion.
    if not math.isclose(times[-1], end_s, abs_tol=1e-9):
        times.append(end_s)
    else:
        times[-1] = end_s

    # Authored semantic keys are never discarded merely because they fall
    # between the requested regular sample ticks. This keeps impact, contact
    # transitions, explicit rest states, labels and z-order changes exact.
    for authored_time in required_times or []:
        if authored_time < start_s - 1e-9 or authored_time > end_s + 1e-9:
            continue
        if not any(
            math.isclose(authored_time, item, abs_tol=1e-9)
            for item in times
        ):
            times.append(authored_time)

    times.sort()
    return times


def _segment_for_time(
    source_times: list[float],
    time_s: float,
) -> int:
    if len(source_times) < 2:
        return 0
    index = bisect.bisect_right(source_times, time_s) - 1
    return max(0, min(index, len(source_times) - 2))


def _sample_pose(
    clip: MotionClip,
    source_times: list[float],
    time_s: float,
    output_frame: int,
) -> FramePose:
    frames = clip.frames
    if len(frames) == 1:
        return replace(
            frames[0],
            frame=output_frame,
            time_s=time_s,
        )

    # Exact key poses retain their semantic label, stop state, contacts, pole
    # data and explicit z-order.
    for index, authored_time in enumerate(source_times):
        if math.isclose(time_s, authored_time, abs_tol=1e-9):
            return replace(
                frames[index],
                frame=output_frame,
                time_s=time_s,
            )

    index = _segment_for_time(source_times, time_s)
    previous = frames[index - 1] if index > 0 else None
    left = frames[index]
    right = frames[index + 1]
    following = frames[index + 2] if index + 2 < len(frames) else None

    t0 = source_times[index - 1] if index > 0 else None
    t1 = source_times[index]
    t2 = source_times[index + 1]
    t3 = source_times[index + 2] if index + 2 < len(frames) else None
    segment_dt = max(t2 - t1, 1e-9)
    u = min(1.0, max(0.0, (time_s - t1) / segment_dt))

    names = left.joints.keys() & right.joints.keys()
    joints: dict[str, Vec2] = {}
    for name in names:
        p0 = (
            previous.joints.get(name)
            if previous is not None and name in previous.joints
            else None
        )
        p1 = left.joints[name]
        p2 = right.joints[name]
        p3 = (
            following.joints.get(name)
            if following is not None and name in following.joints
            else None
        )

        m1 = _vec_tangent(p0, p1, p2, t0, t1, t2)
        if p3 is None or t3 is None:
            m2 = (p2 - p1) * (1.0 / segment_dt)
        else:
            m2 = (p3 - p1) * (1.0 / max(t3 - t1, 1e-9))
        joints[name] = _hermite_vec(
            p1,
            p2,
            m1,
            m2,
            segment_dt,
            u,
        )

    root_m1 = _vec_tangent(
        previous.root if previous is not None else None,
        left.root,
        right.root,
        t0,
        t1,
        t2,
    )
    if following is None or t3 is None:
        root_m2 = (right.root - left.root) * (1.0 / segment_dt)
    else:
        root_m2 = (
            (following.root - left.root)
            * (1.0 / max(t3 - t1, 1e-9))
        )
    root = _hermite_vec(
        left.root,
        right.root,
        root_m1,
        root_m2,
        segment_dt,
        u,
    )

    pole_names = left.ik_poles.keys() & right.ik_poles.keys()
    poles: dict[str, Vec2] = {}
    for name in pole_names:
        p0 = (
            previous.ik_poles.get(name)
            if previous is not None and name in previous.ik_poles
            else None
        )
        p1 = left.ik_poles[name]
        p2 = right.ik_poles[name]
        p3 = (
            following.ik_poles.get(name)
            if following is not None and name in following.ik_poles
            else None
        )
        m1 = _vec_tangent(p0, p1, p2, t0, t1, t2)
        if p3 is None or t3 is None:
            m2 = (p2 - p1) * (1.0 / segment_dt)
        else:
            m2 = (p3 - p1) * (1.0 / max(t3 - t1, 1e-9))
        poles[name] = _hermite_vec(
            p1,
            p2,
            m1,
            m2,
            segment_dt,
            u,
        )

    main0 = previous.weapon.grip_main if previous is not None else None
    main1 = left.weapon.grip_main
    main2 = right.weapon.grip_main
    main3 = following.weapon.grip_main if following is not None else None
    main_m1 = _vec_tangent(main0, main1, main2, t0, t1, t2)
    if main3 is None or t3 is None:
        main_m2 = (main2 - main1) * (1.0 / segment_dt)
    else:
        main_m2 = (main3 - main1) * (1.0 / max(t3 - t1, 1e-9))
    main = _hermite_vec(
        main1,
        main2,
        main_m1,
        main_m2,
        segment_dt,
        u,
    )

    angle1 = _weapon_angle(left)
    angle2 = _unwrap_near(_weapon_angle(right), angle1)
    angle0 = (
        _unwrap_near(_weapon_angle(previous), angle1)
        if previous is not None
        else None
    )
    angle3 = (
        _unwrap_near(_weapon_angle(following), angle2)
        if following is not None
        else None
    )

    angle_m1 = _scalar_tangent(
        angle0,
        angle1,
        angle2,
        t0,
        t1,
        t2,
    )
    if angle3 is None or t3 is None:
        angle_m2 = (angle2 - angle1) / segment_dt
    else:
        angle_m2 = (angle3 - angle1) / max(t3 - t1, 1e-9)
    angle = _hermite_scalar(
        angle1,
        angle2,
        angle_m1,
        angle_m2,
        segment_dt,
        u,
    )

    semantic_source = left if u < 0.5 else right
    return FramePose(
        frame=output_frame,
        root=root,
        joints=joints,
        weapon=_weapon_from_main_angle(left, main, angle),
        contacts=dict(semantic_source.contacts),
        ik_poles=poles,
        label=None,
        time_s=time_s,
        kinematic_stop=False,
        layer_order=list(semantic_source.layer_order),
    )


def resample_clip(
    clip: MotionClip,
    fps: float,
) -> MotionClip:
    """Sample continuous authored motion on a time-based output grid.

    Unlike integer-frame densification, this respects the clip's real timestamps
    directly. It is the preferred path when a reference video establishes the
    timing and the desired spritesheet needs more poses than the authored keys.
    """
    if not clip.frames:
        return replace(clip, fps=fps)

    source_times = _source_times(clip)
    target_times = _sample_times(
        source_times[0],
        source_times[-1],
        fps,
        required_times=source_times,
    )
    output = [
        _sample_pose(
            clip,
            source_times,
            time_s,
            output_frame=index,
        )
        for index, time_s in enumerate(target_times)
    ]

    dynamics = {
        **clip.dynamics,
        "weapon": dict(clip.dynamics.get("weapon", {})),
    }
    impact_frame = dynamics["weapon"].get("impact_frame")
    if impact_frame is not None:
        source_index = next(
            (
                index
                for index, frame in enumerate(clip.frames)
                if frame.frame == int(impact_frame)
            ),
            None,
        )
        if source_index is not None:
            impact_time = source_times[source_index]
            output_index = next(
                index
                for index, item in enumerate(target_times)
                if math.isclose(item, impact_time, abs_tol=1e-9)
            )
            dynamics["weapon"]["impact_frame"] = output_index

    metadata = {
        **clip.metadata,
        "resampling": {
            "source_pose_count": len(clip.frames),
            "output_pose_count": len(output),
            "target_fps": fps,
            "preserved_authored_key_times": True,
        },
    }

    return replace(
        clip,
        fps=fps,
        frames=output,
        metadata=metadata,
        dynamics=dynamics,
    )
