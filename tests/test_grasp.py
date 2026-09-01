"""Grasp-point biasing and the multi-attempt queue.

No placo and no hardware: ``gripper_frame_offset`` is pure trigonometry and
the attempt planner only needs something with a ``solve()``, so this module
runs in the lightweight test env (AGENTS.md §13) unlike tests/test_ik.py.
"""

from __future__ import annotations

import math

import pytest

from pick_stack.config import AppConfig
from pick_stack.control import grasp as grasp_mod
from pick_stack.control.grasp import (
    GraspAttempt,
    GraspOutcome,
    GraspPlan,
    attempt_grasp,
    biased_grasp_xy,
    plan_grasp_attempts,
    run_grasp_attempts,
)
from pick_stack.control.ik import ARM_JOINTS, IkResult, gripper_frame_offset
from pick_stack.control.robot_io import MockRobotIO
from pick_stack.control.trajectory import TrajectoryPlayer


class StubIk:
    """Returns a perfect solve, or a hopeless one past ``fail_beyond_mm``."""

    def __init__(self, fail_beyond_mm: float = float("inf")):
        self.fail_beyond_mm = fail_beyond_mm
        self.calls: list[tuple[float, float, float]] = []

    def solve(self, x_mm: float, y_mm: float, z_mm: float) -> IkResult:
        self.calls.append((x_mm, y_mm, z_mm))
        bad = math.hypot(x_mm, y_mm) > self.fail_beyond_mm
        return IkResult(
            joints={"shoulder_pan": 0.0},
            position_error_mm=99.0 if bad else 0.5,
            tilt_error_deg=0.4,
        )


# --- gripper_frame_offset ------------------------------------------------


def test_offset_straight_ahead_matches_board_axes():
    # target on +x: radial is +x, and the gripper's left is +y.
    assert gripper_frame_offset(100.0, 0.0, 10.0, 0.0) == pytest.approx((110.0, 0.0))
    assert gripper_frame_offset(100.0, 0.0, 0.0, 10.0) == pytest.approx((100.0, 10.0))


def test_offset_rotates_with_the_arm():
    # target on +y (arm swung 90 deg left): radial is now +y, left is -x.
    assert gripper_frame_offset(0.0, 100.0, 10.0, 0.0) == pytest.approx((0.0, 110.0))
    assert gripper_frame_offset(0.0, 100.0, 0.0, 10.0) == pytest.approx((-10.0, 100.0))


def test_offset_preserves_magnitude_at_an_arbitrary_azimuth():
    x, y = 180.0, -95.0
    nx, ny = gripper_frame_offset(x, y, 10.0, 10.0)
    assert math.hypot(nx - x, ny - y) == pytest.approx(math.hypot(10.0, 10.0))
    # a purely radial push must increase the reach by exactly that much
    rx, ry = gripper_frame_offset(x, y, 12.0, 0.0)
    assert math.hypot(rx, ry) == pytest.approx(math.hypot(x, y) + 12.0)


# --- biased_grasp_xy -----------------------------------------------------


def test_right_half_gets_only_the_global_bias():
    cfg = AppConfig().motion
    x, y = 200.0, -80.0  # right half
    bx, by = biased_grasp_xy(cfg, x, y)
    assert math.hypot(bx - x, by - y) == pytest.approx(cfg.grasp_radial_offset_mm)
    # pushed outward, not inward
    assert math.hypot(bx, by) > math.hypot(x, y)


def test_left_half_adds_the_extra_bias():
    cfg = AppConfig().motion
    x, y = 200.0, 80.0  # left half
    bx, by = biased_grasp_xy(cfg, x, y)
    expected = math.hypot(
        cfg.grasp_radial_offset_mm + cfg.left_half_radial_offset_mm,
        cfg.grasp_tangential_offset_mm + cfg.left_half_tangential_offset_mm,
    )
    assert math.hypot(bx - x, by - y) == pytest.approx(expected)


def test_half_is_decided_before_the_bias_is_applied():
    # A block just right of the divide must not be promoted to the left-half
    # correction by its own tangential nudge.
    cfg = AppConfig().motion
    right = biased_grasp_xy(cfg, 200.0, -1.0)
    assert math.hypot(right[0] - 200.0, right[1] + 1.0) == pytest.approx(cfg.grasp_radial_offset_mm)


# --- plan_grasp_attempts -------------------------------------------------


def test_attempts_are_centre_then_the_four_diagonals():
    cfg = AppConfig()
    plan = plan_grasp_attempts(StubIk(), cfg, 200.0, -80.0, 9.3, 120.0)
    assert [a.label for a in plan.attempts] == [
        "centre",
        "front-left",
        "front-right",
        "back-left",
        "back-right",
    ]
    assert plan.attempts[0].offset_mm == (0.0, 0.0)
    assert plan.attempts[1].offset_mm == (10.0, 10.0)  # front-left
    assert plan.attempts[3].offset_mm == (-10.0, 10.0)  # back-left
    assert [a.label for a in plan.lateral] == ["lateral-left", "lateral-right"]
    assert [a.offset_mm for a in plan.lateral] == [(0.0, 10.0), (0.0, -10.0)]


def test_attempts_share_one_hover_height():
    plan = plan_grasp_attempts(StubIk(), AppConfig(), 200.0, -80.0, 9.3, 118.0)
    assert plan.hover_z_mm == 118.0
    assert all(a.reachable for a in plan.attempts)


def test_out_of_reach_candidates_are_marked_not_dropped():
    # fail anything past the centre's own radius, so the outward nudges fail
    cfg = AppConfig()
    centre_r = math.hypot(*biased_grasp_xy(cfg.motion, 300.0, 0.0))
    plan = plan_grasp_attempts(StubIk(fail_beyond_mm=centre_r + 1.0), cfg, 300.0, 0.0, 9.3, 120.0)
    by_label = {a.label: a for a in plan.attempts}
    assert by_label["centre"].reachable
    assert not by_label["front-left"].reachable
    assert by_label["back-left"].reachable
    assert len(plan.attempts) == 5  # still reported, for the printout


# --- run_grasp_attempts --------------------------------------------------


def _attempt(label: str, reachable: bool = True) -> GraspAttempt:
    r = IkResult(joints={"shoulder_pan": 0.0}, position_error_mm=0.5, tilt_error_deg=0.1)
    return GraspAttempt(
        label=label, offset_mm=(0.0, 0.0), xy_mm=(200.0, 0.0), hover=r, grasp=r, reachable=reachable
    )


def _plan(attempt_labels, lateral_labels=("lateral-left", "lateral-right"), unreachable=()):
    return GraspPlan(
        detected_xy_mm=(200.0, 0.0),
        biased_xy_mm=(212.0, 0.0),
        grasp_z_mm=9.3,
        hover_z_mm=120.0,
        attempts=[_attempt(n, n not in unreachable) for n in attempt_labels],
        lateral=[_attempt(n) for n in lateral_labels],
    )


def _run(monkeypatch, plan, outcomes):
    """Drive the queue with a scripted outcome per label."""
    tried: list[str] = []

    def fake_attempt(player, robot, cfg, attempt):
        tried.append(attempt.label)
        return outcomes.get(attempt.label, GraspOutcome.EMPTY), None

    monkeypatch.setattr(grasp_mod, "attempt_grasp", fake_attempt)
    held = run_grasp_attempts(None, None, AppConfig(), plan, log=lambda _msg: None)
    return held, tried


def test_stops_at_the_first_hold(monkeypatch):
    plan = _plan(["centre", "front-left", "front-right"])
    held, tried = _run(monkeypatch, plan, {"front-left": GraspOutcome.HELD})
    assert held is not None and held.label == "front-left"
    assert tried == ["centre", "front-left"]


def test_blocked_descent_promotes_the_sideways_nudges(monkeypatch):
    plan = _plan(["centre", "front-left", "front-right"])
    held, tried = _run(
        monkeypatch,
        plan,
        {"centre": GraspOutcome.BLOCKED, "lateral-right": GraspOutcome.HELD},
    )
    # the sideways nudges jump ahead of the remaining diagonals
    assert tried == ["centre", "lateral-left", "lateral-right"]
    assert held is not None and held.label == "lateral-right"


def test_sideways_nudges_are_only_injected_once(monkeypatch):
    plan = _plan(["centre", "front-left"])
    _, tried = _run(
        monkeypatch,
        plan,
        {"centre": GraspOutcome.BLOCKED, "lateral-left": GraspOutcome.BLOCKED},
    )
    assert tried.count("lateral-left") == 1
    assert tried == ["centre", "lateral-left", "lateral-right", "front-left"]


def test_unreachable_candidates_are_skipped_not_fatal(monkeypatch):
    plan = _plan(["centre", "front-left", "front-right"], unreachable={"front-left"})
    held, tried = _run(monkeypatch, plan, {"front-right": GraspOutcome.HELD})
    assert tried == ["centre", "front-right"]
    assert held is not None and held.label == "front-right"


def test_returns_none_when_every_point_fails(monkeypatch):
    plan = _plan(["centre", "front-left"], lateral_labels=())
    held, tried = _run(monkeypatch, plan, {})
    assert held is None
    assert tried == ["centre", "front-left"]


# --- attempt_grasp always closes the jaws --------------------------------


class BlockedDescentRobot(MockRobotIO):
    """The jaws land on the block: shoulder_lift cannot get past -1.0."""

    def send_joints(self, positions):
        clamped = dict(positions)
        if "shoulder_lift" in clamped:
            clamped["shoulder_lift"] = max(clamped["shoulder_lift"], -1.0)
        return super().send_joints(clamped)


def _hw_cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.motion.fps = 0.0
    cfg.motion.gripper_action_wait_s = 0.0
    cfg.motion.descent_settle_s = 0.05
    cfg.motion.move_timeout_s = 1.0
    cfg.sensing.grasp_settle_s = 0.0
    cfg.sensing.sample_interval_s = 0.0
    return cfg


def _hw_attempt() -> GraspAttempt:
    arm = {j: 0.0 for j in ARM_JOINTS}
    hover = IkResult(joints=dict(arm), position_error_mm=0.5, tilt_error_deg=0.1)
    grasp = IkResult(joints={**arm, "shoulder_lift": -6.0}, position_error_mm=0.5, tilt_error_deg=0.1)
    return GraspAttempt(
        label="centre", offset_mm=(0.0, 0.0), xy_mm=(200.0, 0.0),
        hover=hover, grasp=grasp, reachable=True,
    )


def _gripper_commands(robot) -> list[float]:
    return [a["gripper"] for a in robot.sent_actions if "gripper" in a]


def test_jaws_close_and_are_checked_even_when_the_descent_stops_short():
    """A short descent must not skip the close.

    Whether a position can hold the block is only knowable by closing the
    jaws on it — reporting failure without closing would retry positions
    that were never actually tried.
    """
    cfg = _hw_cfg()
    robot = BlockedDescentRobot()
    robot.connect()
    player = TrajectoryPlayer(robot, cfg.motion)

    outcome, check = attempt_grasp(player, robot, cfg, _hw_attempt())

    assert outcome is GraspOutcome.BLOCKED  # the descent did stop short...
    assert check is not None  # ...but the jaws were still closed and checked
    commands = _gripper_commands(robot)
    assert cfg.sensing.gripper_close_pos in commands
    # and the cycle ends reopened, ready for the next position
    assert commands[-1] == pytest.approx(cfg.sensing.gripper_open_pos)


def test_clear_descent_that_grabs_nothing_reports_empty():
    cfg = _hw_cfg()
    robot = MockRobotIO()
    robot.connect()
    player = TrajectoryPlayer(robot, cfg.motion)

    outcome, check = attempt_grasp(player, robot, cfg, _hw_attempt())

    assert outcome is GraspOutcome.EMPTY
    assert not check.grasped
    assert cfg.sensing.gripper_close_pos in _gripper_commands(robot)


def test_every_retry_runs_a_full_open_close_open_cycle():
    """close -> fail -> open -> next position -> close -> fail -> open ..."""
    cfg = _hw_cfg()
    robot = MockRobotIO()
    robot.connect()
    player = TrajectoryPlayer(robot, cfg.motion)
    plan = GraspPlan(
        detected_xy_mm=(200.0, 0.0), biased_xy_mm=(212.0, 0.0),
        grasp_z_mm=9.3, hover_z_mm=120.0,
        attempts=[_hw_attempt(), _hw_attempt(), _hw_attempt()], lateral=[],
    )

    held = run_grasp_attempts(player, robot, cfg, plan, log=lambda _m: None)

    assert held is None
    closes = _gripper_commands(robot).count(cfg.sensing.gripper_close_pos)
    assert closes == 3  # one close per position, not one for the whole run
