"""CV+IK grasp planning and motion safety contracts."""

import math

import pytest

from config import AppConfig, MotionConfig, SensingConfig
from control import MockRobotIO, TrajectoryPlayer, check_grasp, interpolate
from control.sensing import GraspCheck
from control import grasp as grasp_mod
from control.grasp import GraspAttempt, GraspOutcome, GraspPlan, biased_grasp_xy, grasp_candidate_points, highest_reachable_hover, plan_grasp_attempts, run_grasp_attempts
from control.ik import IkResult, gripper_frame_offset, tangent_square_grasp_yaw_deg


class StubIk:
    def solve(self, x_mm, y_mm, z_mm, yaw_deg=None):
        return IkResult({"shoulder_pan": 0.0}, 0.5, 0.1)

    def grasp_yaw_deg(self, x_mm, y_mm, z_mm, block_angle_deg):
        return self.grasp_yaw_and_rotation_deg(x_mm, y_mm, z_mm, block_angle_deg)[0]

    def grasp_yaw_and_rotation_deg(self, x_mm, y_mm, z_mm, block_angle_deg):
        # stub neutral yaw is 0, so the block angle *is* the jaw rotation
        return block_angle_deg, block_angle_deg


def test_unreachable_displayed_yaw_does_not_fall_back_to_a_perpendicular_grasp(monkeypatch):
    """The robot must either use the displayed yaw or reject the plan."""
    cfg = AppConfig()
    cfg.motion.grasp_offsets_follow_jaw_yaw = True

    class OnlyNeutralIk(StubIk):
        def solve(self, x_mm, y_mm, z_mm, yaw_deg=None):
            miss = yaw_deg not in (None, 0.0)
            return IkResult({"shoulder_pan": 0.0}, 99.0 if miss else 0.5, 0.1)

    plan = plan_grasp_attempts(OnlyNeutralIk(), cfg, 200.0, 0.0, 9.0, block_angle_deg=40.0)
    assert plan.yaw_deg == pytest.approx(40.0)
    assert not plan.attempts[0].reachable


def test_attempt_grasp_tightens_the_hover_on_a_bounded_clock(monkeypatch):
    """The tighten step must not borrow move_to's full timeout: five attempts
    per block would each pay it before descending."""
    cfg = AppConfig()
    calls = {"order": [], "move_tol": [], "settle": []}

    class Player:
        def set_gripper(self, position):
            calls["order"].append("gripper")

        def move_to(self, goal, *, max_step=None, tol=None):
            calls["order"].append("move")
            calls["move_tol"].append(tol)
            return goal

        def settle(self, goal, *, tol, timeout_s):
            calls["order"].append("settle")
            calls["settle"].append((tol, timeout_s))
            return 5.0, False  # servos never reach the tight tolerance

        def descend(self, goal, **_kwargs):
            calls["order"].append("descend")
            return goal, False

    monkeypatch.setattr(
        grasp_mod, "check_grasp", lambda *_a, **_k: GraspCheck(True, 25.0, 300.0, True, True, "x")
    )
    outcome, _check = grasp_mod.attempt_grasp(Player(), MockRobotIO(), cfg, _attempt("centre"))

    assert outcome is GraspOutcome.HELD
    # arrival stays on the fast transit tolerance; only the extra hold is tight
    assert calls["move_tol"][0] == cfg.motion.transit_arrival_tol
    assert calls["settle"] == [
        (cfg.motion.grasp_hover_arrival_tol, cfg.motion.grasp_hover_settle_s)
    ]
    assert cfg.motion.grasp_hover_settle_s < cfg.motion.move_timeout_s
    assert calls["order"].index("settle") < calls["order"].index("descend")


def test_blocked_pick_descent_never_closes_the_gripper(monkeypatch):
    cfg = AppConfig()
    calls = []

    class BlockedPlayer:
        def set_gripper(self, position):
            calls.append(("gripper", position))

        def move_to(self, goal, **_kwargs):
            calls.append(("move", goal))
            return goal

        def settle(self, goal, **_kwargs):
            return 0.0, True

        def descend(self, goal, **_kwargs):
            calls.append(("descend", goal))
            return goal, True

    check_called = False

    def forbidden_check(*_args, **_kwargs):
        nonlocal check_called
        check_called = True
        raise AssertionError("blocked descent must not run grasp verification")

    monkeypatch.setattr(grasp_mod, "check_grasp", forbidden_check)
    outcome, check = grasp_mod.attempt_grasp(
        BlockedPlayer(), MockRobotIO(), cfg, _attempt("centre")
    )

    gripper_commands = [value for kind, value in calls if kind == "gripper"]
    assert outcome is GraspOutcome.BLOCKED and check is None
    assert not check_called
    assert gripper_commands == [cfg.sensing.gripper_open_pos]
    assert calls[-1][0] == "move"  # lift clear before the rotated retry


def _attempt(label, offset=(0.0, 0.0)):
    solved = IkResult({"shoulder_pan": 0.0}, 0.5, 0.1)
    return GraspAttempt(label, offset, (200.0, 0.0), solved, solved, True)


def _cardinal_plan():
    return GraspPlan(
        (200, 0), (212, 0), 9.0, 80.0,
        [
            _attempt("centre"),
            _attempt("left", (0.0, 10.0)),
            _attempt("back", (-10.0, 0.0)),
            _attempt("right", (0.0, -10.0)),
            _attempt("front", (10.0, 0.0)),
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


def test_grasp_sensor_distinguishes_held_from_empty():
    cfg = SensingConfig(grasp_settle_s=0.0, sample_interval_s=0.0, grasp_samples=1)
    held = MockRobotIO(initial_joints={"gripper": 25.0})
    held.loads["gripper"] = 300
    empty = MockRobotIO(initial_joints={"gripper": 3.0})
    empty.loads["gripper"] = 15
    assert check_grasp(held, cfg).grasped
    assert not check_grasp(empty, cfg).grasped
