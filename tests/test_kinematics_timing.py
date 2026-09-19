from anima.kinematics import vec_derivative_times
from anima.model import Vec2


def test_same_displacement_over_more_time_has_lower_velocity():
    positions = [Vec2(0, 0), Vec2(1, 0), Vec2(2, 0)]

    fast = vec_derivative_times(positions, [0.0, 0.1, 0.2])
    slow = vec_derivative_times(positions, [0.0, 0.2, 0.4])

    assert fast[1].x == 10.0
    assert slow[1].x == 5.0


def test_nonuniform_timestamps_affect_local_velocity():
    positions = [Vec2(0, 0), Vec2(1, 0), Vec2(3, 0)]

    velocity = vec_derivative_times(positions, [0.0, 0.1, 0.5])

    assert abs(velocity[1].x - 6.0) < 1e-9


def test_explicit_stop_overrides_endpoint_velocity():
    positions = [Vec2(0, 0), Vec2(1, 0), Vec2(2, 0)]

    velocity = vec_derivative_times(
        positions,
        [0.0, 0.1, 0.2],
        zero_indices={0, 2},
    )

    assert velocity[0] == Vec2(0.0, 0.0)
    assert velocity[1].x == 10.0
    assert velocity[2] == Vec2(0.0, 0.0)
