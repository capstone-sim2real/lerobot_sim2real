"""CV+IK grasp planning and motion safety contracts."""

import math

import pytest

from config import AppConfig, MotionConfig, SensingConfig
from control import MockRobotIO, TrajectoryPlayer, check_grasp, interpolate
from control import grasp as grasp_mod
from control.grasp import GraspAttempt, GraspOutcome, GraspPlan, biased_grasp_xy, highest_reachable_hover, plan_grasp_attempts, run_grasp_attempts
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
    plan = plan_grasp_attempts(StubIk(), cfg, 200.0, -80.0, 9.0, 80.0)
    assert [a.label for a in plan.attempts] == ["centre", "front-left", "front-right", "back-left", "back-right"]


def _attempt(label):
    solved = IkResult({"shoulder_pan": 0.0}, 0.5, 0.1)
    return GraspAttempt(label, (0.0, 0.0), (200.0, 0.0), solved, solved, True)


def test_blocked_descent_promotes_lateral_retry(monkeypatch):
    plan = GraspPlan((200, 0), (212, 0), 9.0, 80.0, [_attempt("centre"), _attempt("diagonal")], [_attempt("left"), _attempt("right")])
    tried = []

    def fake_attempt(_player, _robot, _cfg, attempt):
        tried.append(attempt.label)
        return (GraspOutcome.HELD if attempt.label == "right" else GraspOutcome.BLOCKED), None

    monkeypatch.setattr(grasp_mod, "attempt_grasp", fake_attempt)
    held = run_grasp_attempts(None, None, AppConfig(), plan, log=lambda _message: None)
    assert held is not None and held.label == "right"
    assert tried == ["centre", "left", "right"]


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
