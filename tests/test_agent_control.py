"""Who may command the arm, and when: shared access plus motion serialization."""

from agent.control import ControlGate, ControlState


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _gate(**kwargs):
    stops, snapshots, clock = [], [], Clock()
    gate = ControlGate(on_stop=lambda: stops.append(1), lease_grace_s=15.0, lease_idle_timeout_s=300.0,
                       on_change=snapshots.append, clock=clock, **kwargs)
    return gate, stops, snapshots, clock


def test_multiple_operators_share_a_session_and_only_idle_accepts_commands():
    gate, _stops, _snaps, _clock = _gate()
    token = gate.acquire_lease()
    assert token
    assert gate.acquire_lease() == token
    assert gate.acquire_lease("stale-token") == token
    assert not gate.try_begin("wrong", "chat")
    assert gate.try_begin(token, "chat") and gate.state is ControlState.BUSY
    assert not gate.try_begin(token, "chat")
    gate.finish(robot_fault=False)
    assert gate.state is ControlState.IDLE


def test_stop_locks_until_home_is_verified():
    gate, stops, _snaps, _clock = _gate()
    token = gate.acquire_lease()
    assert not gate.request_stop()  # nothing is moving
    gate.try_begin(token, "chat")
    assert gate.request_stop() and stops == [1] and gate.state is ControlState.STOPPING
    assert not gate.try_begin_home(token)  # the turn has not unwound yet
    gate.finish(robot_fault=True)
    assert gate.state is ControlState.STOPPED
    assert not gate.try_begin(token, "chat")
    assert not gate.try_begin_home("wrong")
    assert gate.try_begin_home(token) and gate.state is ControlState.HOMING
    gate.finish_home(arm_at_home=False, message="not home")
    assert gate.state is ControlState.STOPPED and gate.message == "not home"
    assert gate.try_begin_home(token)
    gate.finish_home(arm_at_home=True)
    assert gate.state is ControlState.IDLE and gate.try_begin(token, "chat")


def test_robot_fault_without_stop_also_locks():
    gate, _stops, _snaps, _clock = _gate()
    token = gate.acquire_lease()
    gate.try_begin(token, "chat")
    gate.finish(robot_fault=True)
    assert gate.state is ControlState.STOPPED


def test_stop_during_homing_returns_to_stopped():
    gate, _stops, _snaps, _clock = _gate()
    token = gate.acquire_lease()
    gate.try_begin(token, "chat")
    gate.finish(robot_fault=True)
    gate.try_begin_home(token)
    assert gate.request_stop()
    gate.finish_home(arm_at_home=True)
    assert gate.state is ControlState.STOPPED


def test_operator_leaving_mid_motion_stops_and_frees_the_lease():
    gate, stops, _snaps, clock = _gate()
    token = gate.acquire_lease()
    gate.operator_connected(token)
    gate.try_begin(token, "chat")
    gate.operator_disconnected(token)
    clock.now = 10.0
    gate.tick()
    assert stops == [] and gate.check(token)
    clock.now = 20.0
    gate.tick()
    assert stops == [1] and gate.state is ControlState.STOPPING and not gate.check(token)
    newcomer = gate.acquire_lease()
    gate.finish(robot_fault=True)
    assert gate.state is ControlState.STOPPED and gate.try_begin_home(newcomer)


def test_idle_operator_loses_the_lease_after_the_timeout():
    gate, _stops, _snaps, clock = _gate()
    token = gate.acquire_lease()
    gate.operator_connected(token)
    clock.now = 301.0
    gate.tick()
    assert not gate.check(token) and gate.acquire_lease() is not None
