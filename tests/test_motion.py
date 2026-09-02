"""CV+IK grasp planning and motion safety contracts."""

import math

import pytest

from config import AppConfig, MotionConfig, SensingConfig
from control import MockRobotIO, TrajectoryPlayer, check_grasp, interpolate
from control import grasp as grasp_mod
from control.grasp import GraspAttempt, GraspOutcome, GraspPlan, biased_grasp_xy, grasp_candidate_points, highest_reachable_hover, plan_grasp_attempts, run_grasp_attempts
from control.ik import IkResult, gripper_frame_offset


class StubIk:
    def solve(self, x_mm, y_mm, z_mm, yaw_deg=None):
        return IkResult({"shoulder_pan": 0.0}, 0.5, 0.1)


def test_gripper_frame_bias_and_candidate_order():
    assert gripper_frame_offset(0.0, 100.0, 0.0, 10.0) == pytest.approx((-10.0, 100.0))
    cfg = AppConfig()
    right = biased_grasp_xy(cfg.motion, 200.0, -80.0)
    left = biased_grasp_xy(cfg.motion, 200.0, 80.0)
    assert math.hypot(left[0] - 200.0, left[1] - 80.0) > math.hypot(right[0] - 200.0, right[1] + 80.0)
    plan = plan_grasp_attempts(StubIk(), cfg, 200.0, -80.0, 9.0)
    assert [a.label for a in plan.attempts] == ["centre", "front", "back", "left", "right"]
    assert [label for label, _ in grasp_candidate_points(cfg.motion, 200.0, -80.0)] == [
        "centre", "front", "back", "left", "right"
    ]


def test_labels_follow_the_configured_offsets():
    """Labels are derived, so re-shaping the offsets cannot mislabel a point."""
    cfg = AppConfig()
    cfg.motion.grasp_retry_offsets_mm = [[10.0, 10.0], [-10.0, -10.0]]
    assert [label for label, _ in grasp_candidate_points(cfg.motion, 200.0, -80.0)] == [
        "centre", "front-left", "back-right"
    ]


def _attempt(label, offset=(0.0, 0.0)):
    solved = IkResult({"shoulder_pan": 0.0}, 0.5, 0.1)
    return GraspAttempt(label, offset, (200.0, 0.0), solved, solved, True)


def _cardinal_plan():
    return GraspPlan(
        (200, 0), (212, 0), 9.0, 80.0,
        [
            _attempt("centre"),
            _attempt("front", (10.0, 0.0)),
            _attempt("back", (-10.0, 0.0)),
            _attempt("left", (0.0, 10.0)),
            _attempt("right", (0.0, -10.0)),
        ],
    )


def _run_queue(monkeypatch, plan, outcomes):
    tried = []

    def fake_attempt(_player, _robot, _cfg, attempt, **_kwargs):
        tried.append(attempt.label)
        return outcomes.get(attempt.label, GraspOutcome.EMPTY), None

    monkeypatch.setattr(grasp_mod, "attempt_grasp", fake_attempt)
    held = run_grasp_attempts(None, None, AppConfig(), plan, log=lambda _message: None)
    return held, tried


def test_blocked_descent_promotes_the_sideways_points(monkeypatch):
    """Stopping short means the depth was right and only the lateral was off."""
    held, tried = _run_queue(
        monkeypatch, _cardinal_plan(),
        {"centre": GraspOutcome.BLOCKED, "right": GraspOutcome.HELD},
    )
    assert held is not None and held.label == "right"
    # left/right jump ahead of the reach-only points
    assert tried == ["centre", "left", "right"]


def test_plain_failure_keeps_the_configured_order(monkeypatch):
    _, tried = _run_queue(monkeypatch, _cardinal_plan(), {})
    assert tried == ["centre", "front", "back", "left", "right"]


def _fast_motion(**overrides):
    return MotionConfig(**{"fps": 0.0, "gripper_action_wait_s": 0.0, "descent_settle_s": 0.01, **overrides})


def test_trajectory_bounds_steps_and_reports_blocked_descent_without_sensor_reads():
    class StallingRobot(MockRobotIO):
        def send_joints(self, positions):
            positions = dict(positions)
            if "shoulder_lift" in positions:
                positions["shoulder_lift"] = max(positions["shoulder_lift"], -1.0)
            return super().send_joints(positions)

    steps = interpolate({"shoulder_pan": 0.0}, {"shoulder_pan": 10.0}, 2.0)
    previous = {"shoulder_pan": 0.0}
    for step in steps:
        assert abs(step["shoulder_pan"] - previous["shoulder_pan"]) <= 2.0
        previous = step
    robot = StallingRobot()
    robot.connect()
    reads = []
    robot.read_loads = lambda: reads.append(True) or {}
    _, blocked = TrajectoryPlayer(robot, _fast_motion()).descend({"shoulder_lift": -6.0})
    assert blocked and not reads


def test_jammed_descent_stops_instead_of_leaning_on_the_block():
    """A gripper that lands on a block must not keep being pushed into it.

    Regression guard: the descent used to send every remaining (deeper) step
    and then re-send an unreachable goal for the whole settle budget, which
    shoved the block out of position and bound the arm against it.
    """
    class StallingRobot(MockRobotIO):
        def send_joints(self, positions):
            positions = dict(positions)
            if "shoulder_lift" in positions:
                positions["shoulder_lift"] = max(positions["shoulder_lift"], -1.0)
            return super().send_joints(positions)

    robot = StallingRobot()
    robot.connect()
    player = TrajectoryPlayer(robot, _fast_motion(descent_step_per_tick=0.6, descent_max_lag=2.0))
    _, blocked = player.descend({"shoulder_lift": -30.0})

    assert blocked
    # -30 at 0.6/tick is 50 commands if it ran to completion; it must bail
    # once the measured pose trails the command by more than descent_max_lag.
    assert len(robot.sent_actions) < 10, robot.sent_actions


def test_normal_descent_is_not_misread_as_jammed():
    robot = MockRobotIO()
    robot.connect()
    player = TrajectoryPlayer(robot, _fast_motion(descent_step_per_tick=0.6))
    final, blocked = player.descend({"shoulder_lift": -6.0})
    assert not blocked
    assert final["shoulder_lift"] == pytest.approx(-6.0)


def test_go_home_can_leave_the_jaws_alone():
    """Homing between CV+IK picks must not close jaws the next pick reopens."""
    from control.poses import PoseRegistry
    from control.motion import MotionController

    robot = MockRobotIO()
    robot.connect()
    poses = PoseRegistry(path=None, poses={"home": {**{j: 0.0 for j in robot.joints}, "gripper": 0.482}})
    motion = MotionController(robot, poses, _fast_motion(), SensingConfig())

    motion.go_home(include_gripper=False)
    assert all("gripper" not in action for action in robot.sent_actions)

    robot.sent_actions.clear()
    motion.go_home()
    assert any("gripper" in action for action in robot.sent_actions)


def test_grasp_sensor_distinguishes_held_from_empty():
    cfg = SensingConfig(grasp_settle_s=0.0, sample_interval_s=0.0, grasp_samples=1)
    held = MockRobotIO(initial_joints={"gripper": 25.0})
    held.loads["gripper"] = 300
    empty = MockRobotIO(initial_joints={"gripper": 3.0})
    empty.loads["gripper"] = 15
    assert check_grasp(held, cfg).grasped
    assert not check_grasp(empty, cfg).grasped


def test_retracting_arm_increases_reachable_hover():
    class EnvelopeIk:
        def solve(self, x_mm, y_mm, z_mm, yaw_deg=None):
            ceiling = 90.0 - max(0.0, math.hypot(x_mm, y_mm) - 195.0) * 0.45
            return IkResult({}, 0.5 if z_mm <= ceiling else 80.0, 0.1)

    cfg = AppConfig()
    far = highest_reachable_hover(EnvelopeIk(), 285.0, 0.0, 10.0, cfg)
    near = highest_reachable_hover(EnvelopeIk(), 195.0, 0.0, 10.0, cfg)
    assert near > far + 20.0
