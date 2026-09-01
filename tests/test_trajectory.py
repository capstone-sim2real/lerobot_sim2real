import pytest

from pick_stack.config import MotionConfig
from pick_stack.control import MockRobotIO, TrajectoryPlayer, interpolate
from pick_stack.control.robot_io import JOINT_NAMES


def full_pose(value=0.0):
    return {j: value for j in JOINT_NAMES}


def fast_cfg(**kwargs):
    # fps=0 -> no sleeps in tests
    return MotionConfig(**{"fps": 0.0, "gripper_action_wait_s": 0.0, "place_settle_s": 0.0, **kwargs})


def test_interpolate_caps_step_size():
    start = {"shoulder_pan": 0.0, "elbow_flex": 0.0}
    goal = {"shoulder_pan": 10.0, "elbow_flex": -5.0}
    steps = interpolate(start, goal, max_step=2.0)
    assert len(steps) == 5
    assert steps[-1] == goal
    previous = start
    for step in steps:
        for joint in goal:
            assert abs(step[joint] - previous[joint]) <= 2.0 + 1e-9
        previous = step


def test_interpolate_noop_and_validation():
    assert interpolate(full_pose(1.0), {j: 1.0 for j in JOINT_NAMES}, 2.0) == []
    with pytest.raises(ValueError, match="max_step"):
        interpolate(full_pose(), full_pose(1.0), 0.0)


def test_interpolate_only_commands_goal_joints():
    steps = interpolate({"shoulder_pan": 0.0, "gripper": 50.0}, {"shoulder_pan": 4.0}, 2.0)
    assert all(set(step) == {"shoulder_pan"} for step in steps)


def test_move_to_reaches_goal_through_bounded_steps():
    robot = MockRobotIO()
    robot.connect()
    player = TrajectoryPlayer(robot, fast_cfg(max_step_per_tick=2.0))
    goal = {**full_pose(0.0), "shoulder_pan": 9.0}
    final = player.move_to(goal)
    assert final["shoulder_pan"] == pytest.approx(9.0)
    # 9.0 / 2.0 -> 5 interpolation ticks, each within the cap
    assert len(robot.sent_actions) == 5
    assert abs(robot.sent_actions[0]["shoulder_pan"]) <= 2.0 + 1e-9


def test_move_to_timeout_when_arm_stuck():
    class StuckRobot(MockRobotIO):
        def send_joints(self, positions):
            self.sent_actions.append(dict(positions))
            return dict(positions)  # joints never move

    robot = StuckRobot()
    robot.connect()
    player = TrajectoryPlayer(robot, fast_cfg(move_timeout_s=0.05, arrival_tol=0.5))
    with pytest.raises(TimeoutError, match="did not reach"):
        player.move_to({**full_pose(0.0), "shoulder_pan": 20.0})


def test_follow_visits_waypoints_in_order():
    robot = MockRobotIO()
    robot.connect()
    player = TrajectoryPlayer(robot, fast_cfg())
    w1 = {**full_pose(0.0), "shoulder_pan": 4.0}
    w2 = {**full_pose(0.0), "shoulder_pan": 4.0, "elbow_flex": -6.0}
    final = player.follow([w1, w2])
    assert final["elbow_flex"] == pytest.approx(-6.0)
    pans = [a.get("shoulder_pan") for a in robot.sent_actions if "shoulder_pan" in a]
    assert pans == sorted(pans)  # monotonic approach, no jumps


class _StallingRobot(MockRobotIO):
    """Gripper that refuses to close past ``stall_at`` — a held block."""

    def __init__(self, stall_at: float):
        super().__init__()
        self._stall_at = stall_at

    def send_joints(self, positions):
        clamped = dict(positions)
        if "gripper" in clamped:
            clamped["gripper"] = max(clamped["gripper"], self._stall_at)
        return super().send_joints(clamped)


def test_set_gripper_returns_when_jaws_stall_on_a_block():
    # closing onto a block cannot reach the goal; that is the grasp signal,
    # not an error (AGENTS.md §10)
    robot = _StallingRobot(stall_at=44.0)
    robot.joints["gripper"] = 95.0
    player = TrajectoryPlayer(robot, MotionConfig(fps=0.0, gripper_action_wait_s=0.0))
    final = player.set_gripper(2.0)
    assert final == pytest.approx(44.0, abs=0.5)


def test_set_gripper_crosses_a_span_wider_than_one_step():
    robot = MockRobotIO()
    robot.joints["gripper"] = 2.0
    player = TrajectoryPlayer(robot, MotionConfig(fps=0.0, gripper_action_wait_s=0.0))
    final = player.set_gripper(95.0)
    assert final == pytest.approx(95.0, abs=0.5)


class StallingRobot(MockRobotIO):
    """Teleports like MockRobotIO until a joint hits ``stop_at``, then holds.

    Stands in for the gripper landing on the block instead of beside it: the
    descent cannot finish, and nothing raises to say so.
    """

    def __init__(self, joint: str, stop_at: float):
        super().__init__()
        self._joint = joint
        self._stop_at = stop_at

    def send_joints(self, positions):
        clamped = dict(positions)
        if self._joint in clamped:
            clamped[self._joint] = max(clamped[self._joint], self._stop_at)
        return super().send_joints(clamped)


def test_descend_reaches_goal_and_reports_clear():
    robot = MockRobotIO()
    robot.connect()
    player = TrajectoryPlayer(robot, fast_cfg(descent_step_per_tick=0.6))
    goal = {**full_pose(0.0), "shoulder_lift": -6.0}
    final, blocked = player.descend(goal)
    assert not blocked
    assert final["shoulder_lift"] == pytest.approx(-6.0)


def test_descend_reports_blocked_when_it_stops_short():
    # the arm gets 1 unit down out of the 6 it was asked for
    robot = StallingRobot("shoulder_lift", stop_at=-1.0)
    robot.connect()
    player = TrajectoryPlayer(robot, fast_cfg(descent_step_per_tick=0.6, descent_settle_s=0.05))
    goal = {**full_pose(0.0), "shoulder_lift": -6.0}
    _, blocked = player.descend(goal)
    assert blocked


def test_descend_never_raises_on_a_blocked_goal():
    # move_to would raise TimeoutError here; a grasp descent must not.
    robot = StallingRobot("shoulder_lift", stop_at=-1.0)
    robot.connect()
    player = TrajectoryPlayer(robot, fast_cfg(descent_step_per_tick=0.6, descent_settle_s=0.05))
    goal = {**full_pose(0.0), "shoulder_lift": -6.0}
    player.descend(goal)  # must not raise
    with pytest.raises(TimeoutError):
        player.move_to(goal, max_step=0.6)


def test_descend_does_not_read_sensors_inside_the_command_loop():
    """A per-tick bus read at fps=30 stalls the descent partway down.

    Regression guard: the interpolation must issue the same command stream
    as move_to, with reads only in the settle phase.
    """
    robot = MockRobotIO()
    robot.connect()
    reads: list[str] = []
    robot.read_loads = lambda: reads.append("load") or {j: 0 for j in JOINT_NAMES}
    player = TrajectoryPlayer(robot, fast_cfg(descent_step_per_tick=0.6))
    player.descend({**full_pose(0.0), "shoulder_lift": -6.0})
    assert reads == []
