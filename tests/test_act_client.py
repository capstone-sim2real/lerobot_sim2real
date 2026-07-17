import pytest

from pick_stack.config import PolicyConfig
from pick_stack.control import MockRobotIO
from pick_stack.control.robot_io import JOINT_NAMES
from pick_stack.policy import ActPolicyClient, PolicyTransport, RetreatDetector

RETREAT = {
    "shoulder_pan": 10.0, "shoulder_lift": -20.0, "elbow_flex": 15.0,
    "wrist_flex": 0.0, "wrist_roll": 0.0, "gripper": 30.0,
}


def as_vector(pose):
    return [pose[j] for j in JOINT_NAMES]


class FakeTransport(PolicyTransport):
    """Returns scripted action chunks; records what was sent up."""

    def __init__(self, chunks=None):
        self.chunks = list(chunks or [])
        self.sent_observations = []
        self.connected = False

    def connect(self):
        self.connected = True

    def ping(self):
        return self.connected

    def send_observation(self, raw_observation, timestep, must_go):
        self.sent_observations.append({"timestep": timestep, "must_go": must_go})
        return True

    def poll_actions(self):
        # like the real server: no actions before an observation arrives
        if self.sent_observations and self.chunks:
            return self.chunks.pop(0)
        return []

    def close(self):
        self.connected = False


def fast_cfg(**kwargs):
    return PolicyConfig(**{"fps": 0.0, "pick_timeout_s": 1.0, "retreat_hold_ticks": 2, **kwargs})


def make_client(transport, cfg=None):
    robot = MockRobotIO()
    robot.connect()
    return ActPolicyClient(robot, transport, cfg or fast_cfg()), robot


# -- RetreatDetector ---------------------------------------------------------


def test_retreat_detector_streak_and_reset():
    detector = RetreatDetector(RETREAT, tol=1.0, hold_ticks=3, joints=["shoulder_pan"])
    at_pose = {"shoulder_pan": 10.4}
    away = {"shoulder_pan": 50.0}
    assert not detector.update(at_pose)
    assert not detector.update(at_pose)
    assert detector.update(at_pose)  # 3rd consecutive tick
    detector.update(away)  # streak broken
    assert not detector.update(at_pose)


def test_retreat_detector_requires_checked_joints():
    with pytest.raises(ValueError, match="missing"):
        RetreatDetector({"shoulder_pan": 0.0}, tol=1.0, hold_ticks=1, joints=["elbow_flex"])


# -- queue core --------------------------------------------------------------


def test_merge_skips_stale_and_aggregates_overlap():
    client, _ = make_client(FakeTransport(), fast_cfg(aggregate_fn_name="weighted_average", aggregate_weight=0.5))
    client._latest_timestep = 5
    old = [0.0] * 6
    new = [2.0] * 6
    client._merge_actions([(4, old), (6, old), (7, old)])  # 4 is stale
    client._merge_actions([(6, new)])  # overlap -> aggregated
    assert set(client._queue) == {6, 7}
    assert client._queue[6] == [1.0] * 6  # 0.5*0 + 0.5*2

    stepped = client._pop_next_action()
    assert stepped[0] == 6
    assert client._latest_timestep == 6


def test_latest_aggregation_replaces():
    client, _ = make_client(FakeTransport(), fast_cfg(aggregate_fn_name="latest"))
    client._merge_actions([(1, [0.0] * 6)])
    client._merge_actions([(1, [9.0] * 6)])
    assert client._queue[1] == [9.0] * 6


def test_unknown_aggregate_rejected():
    with pytest.raises(ValueError, match="aggregate_fn_name"):
        make_client(FakeTransport(), fast_cfg(aggregate_fn_name="vibes"))


def test_chunk_threshold_gates_observations():
    cfg = fast_cfg(actions_per_chunk=50, chunk_size_threshold=0.5)
    client, _ = make_client(FakeTransport(), cfg)
    client._merge_actions([(i, [0.0] * 6) for i in range(30)])
    assert not client._ready_to_send_observation()  # 30/50 > 0.5
    for _ in range(5):
        client._pop_next_action()
    assert client._ready_to_send_observation()  # 25/50 <= 0.5


def test_action_vector_length_checked():
    client, _ = make_client(FakeTransport())
    with pytest.raises(ValueError, match="length"):
        client._action_to_joints([0.0, 1.0])


# -- run_pick ----------------------------------------------------------------


def test_run_pick_reaches_retreat():
    # one chunk whose actions all command the retreat pose
    chunk = [(i, as_vector(RETREAT)) for i in range(10)]
    transport = FakeTransport(chunks=[chunk])
    client, robot = make_client(transport)

    result = client.run_pick(RETREAT)

    assert result.reached_retreat
    assert result.outcome == "retreat_reached"
    assert result.actions_executed >= 1
    assert robot.read_joints()["shoulder_pan"] == pytest.approx(10.0)
    # first observation of the episode is must_go (empty queue kick-start)
    assert transport.sent_observations[0]["must_go"] is True


def test_run_pick_times_out_without_actions():
    client, _ = make_client(FakeTransport(chunks=[]), fast_cfg(pick_timeout_s=0.05))
    result = client.run_pick(RETREAT)
    assert result.outcome == "timeout"
    assert not result.reached_retreat
    assert result.actions_executed == 0
    assert result.ticks > 0


def test_run_pick_gripper_mismatch_still_retreat():
    # policy leaves the gripper at a different width than the recorded pose
    held = {**RETREAT, "gripper": 77.0}
    chunk = [(i, as_vector(held)) for i in range(10)]
    client, _ = make_client(FakeTransport(chunks=[chunk]))
    assert client.run_pick(RETREAT).reached_retreat
