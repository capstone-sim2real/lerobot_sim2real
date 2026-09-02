"""CV+IK grasp planning and motion safety contracts."""

import math

import pytest

from config import AppConfig, MotionConfig, SensingConfig
from control import MockRobotIO, TrajectoryPlayer, check_grasp, interpolate
from control.sensing import GraspCheck
from control import grasp as grasp_mod
from control.grasp import GraspAttempt, GraspOutcome, GraspPlan, biased_grasp_xy, grasp_candidate_points, highest_reachable_hover, plan_grasp_attempts, run_grasp_attempts
from control.ik import IkResult, gripper_frame_offset


class StubIk:
    def solve(self, x_mm, y_mm, z_mm, yaw_deg=None):
        return IkResult({"shoulder_pan": 0.0}, 0.5, 0.1)

    def grasp_yaw_deg(self, x_mm, y_mm, z_mm, block_angle_deg):
        return self.grasp_yaw_and_rotation_deg(x_mm, y_mm, z_mm, block_angle_deg)[0]

    def grasp_yaw_and_rotation_deg(self, x_mm, y_mm, z_mm, block_angle_deg):
        # stub neutral yaw is 0, so the block angle *is* the jaw rotation
        return block_angle_deg, block_angle_deg


def test_gripper_frame_bias_and_candidate_order():
    assert gripper_frame_offset(0.0, 100.0, 0.0, 10.0) == pytest.approx((-10.0, 100.0))
    cfg = AppConfig()
    right = biased_grasp_xy(cfg.motion, 200.0, -80.0)
    left = biased_grasp_xy(cfg.motion, 200.0, 80.0)
    assert math.hypot(left[0] - 200.0, left[1] - 80.0) > math.hypot(right[0] - 200.0, right[1] + 80.0)
    plan = plan_grasp_attempts(StubIk(), cfg, 200.0, -80.0, 9.0)
    # counter-clockwise from the left in the gripper frame
    assert [a.label for a in plan.attempts] == ["centre", "left", "back", "right", "front"]
    assert [label for label, _ in grasp_candidate_points(cfg.motion, 200.0, -80.0)] == [
        "centre", "left", "back", "right", "front"
    ]


def test_jaw_frame_offsets_rotate_with_the_jaws_only_when_enabled():
    """The offsets' axes are the NEUTRAL-yaw ones unless the flag says the
    jaws carry them. A 90 deg jaw turn swaps radial and tangential, which is
    the sharpest possible statement of the difference."""
    cfg = AppConfig()
    cfg.motion.grasp_radial_offset_mm = 10.0
    cfg.motion.grasp_tangential_offset_mm = 0.0
    cfg.motion.left_half_radial_offset_mm = 0.0
    cfg.motion.left_half_tangential_offset_mm = 0.0

    straight_ahead = (200.0, 0.0)  # radial is +x, tangential is +y
    neutral = biased_grasp_xy(cfg.motion, *straight_ahead, jaw_rot_deg=90.0)
    assert neutral == pytest.approx((210.0, 0.0))  # flag off: rotation ignored

    cfg.motion.grasp_offsets_follow_jaw_yaw = True
    turned = biased_grasp_xy(cfg.motion, *straight_ahead, jaw_rot_deg=90.0)
    assert turned == pytest.approx((200.0, 10.0))  # radial became tangential
    # a zero rotation must stay bit-identical to the neutral behaviour
    assert biased_grasp_xy(cfg.motion, *straight_ahead, jaw_rot_deg=0.0) == pytest.approx((210.0, 0.0))


def test_jaw_frame_retry_points_keep_their_requested_labels():
    """'left' must name the direction asked for, not the rotated vector."""
    cfg = AppConfig()
    cfg.motion.grasp_offsets_follow_jaw_yaw = True
    labels = [label for label, _ in grasp_candidate_points(cfg.motion, 200.0, 0.0, jaw_rot_deg=90.0)]
    assert labels == ["centre", "left", "back", "right", "front"]

    # and the point itself did rotate: 'left' at a 90 deg jaw turn moves the
    # aim back along the reach instead of sideways. The retry is measured
    # from the *biased* centre, whose azimuth the bias itself shifted by a
    # few degrees, so compare against that frame rather than the board axes.
    points = dict(grasp_candidate_points(cfg.motion, 200.0, 0.0, jaw_rot_deg=90.0))
    base = biased_grasp_xy(cfg.motion, 200.0, 0.0, jaw_rot_deg=90.0)
    step = (points["left"][0] - base[0], points["left"][1] - base[1])
    phi = math.atan2(base[1], base[0])
    radial = step[0] * math.cos(phi) + step[1] * math.sin(phi)
    tangential = -step[0] * math.sin(phi) + step[1] * math.cos(phi)
    # magnitude comes from the configured 'left' offset, not a hardcoded mm
    (_radial, expected) = cfg.motion.grasp_retry_offsets_mm[0]
    assert radial == pytest.approx(-expected, abs=1e-6)
    assert tangential == pytest.approx(0.0, abs=1e-6)


def test_neutral_yaw_fallback_does_not_rotate_the_offsets(monkeypatch):
    """When the turned jaws are unreachable the plan reverts to the neutral
    yaw — the offsets have to revert with them or the aim point is skewed by
    a rotation the wrist never made."""
    cfg = AppConfig()
    cfg.motion.grasp_offsets_follow_jaw_yaw = True

    class OnlyNeutralIk(StubIk):
        def solve(self, x_mm, y_mm, z_mm, yaw_deg=None):
            miss = yaw_deg not in (None, 0.0)
            return IkResult({"shoulder_pan": 0.0}, 99.0 if miss else 0.5, 0.1)

    plan = plan_grasp_attempts(OnlyNeutralIk(), cfg, 200.0, 0.0, 9.0, block_angle_deg=40.0)
    assert plan.yaw_deg is None
    assert plan.attempts[0].xy_mm == pytest.approx(biased_grasp_xy(cfg.motion, 200.0, 0.0))


def test_grasp_hover_settles_tighter_than_a_plain_transit():
    """The hover before a descent is aimed, not just transited: descend()
    starts from the measured pose, so slack left here is swept through the
    block on the way down."""
    cfg = AppConfig().motion
    assert cfg.grasp_hover_arrival_tol < cfg.arrival_tol < cfg.transit_arrival_tol


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


def test_settle_gives_up_on_time_instead_of_raising():
    cfg = _fast_motion(move_timeout_s=10.0)
    robot = MockRobotIO(initial_joints={"shoulder_pan": 40.0})
    robot.send_joints = lambda _pose: None  # a servo that refuses to move
    player = TrajectoryPlayer(robot, cfg)

    err, reached = player.settle({"shoulder_pan": 0.0}, tol=1.0, timeout_s=0.05)
    assert not reached and err == pytest.approx(40.0)

    err, reached = player.settle({"shoulder_pan": 40.0}, tol=1.0, timeout_s=0.05)
    assert reached and err == pytest.approx(0.0)


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


def test_blocked_descent_promotes_the_sideways_points(monkeypatch):
    """Stopping short means the depth was right and only the lateral was off."""
    held, tried = _run_queue(
        monkeypatch, _cardinal_plan(),
        {"centre": GraspOutcome.BLOCKED, "right": GraspOutcome.HELD},
    )
    assert held is not None and held.label == "right"
    # left/right jump ahead of the reach-only points
    assert tried == ["centre", "left", "right"]


def test_blocked_descent_promotion_reaches_past_a_reach_only_point(monkeypatch):
    """With left already first, the promotion must still pull 'right' forward."""
    _, tried = _run_queue(
        monkeypatch, _cardinal_plan(),
        {"left": GraspOutcome.BLOCKED, "front": GraspOutcome.HELD},
    )
    # 'right' is promoted ahead of 'back', which only changes reach
    assert tried == ["centre", "left", "right", "back", "front"]


def test_plain_failure_keeps_the_configured_order(monkeypatch):
    _, tried = _run_queue(monkeypatch, _cardinal_plan(), {})
    assert tried == ["centre", "left", "back", "right", "front"]


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


def test_grasp_sensor_accepts_a_block_held_standing_on_edge():
    """A block caught on edge closes the jaws about half as far as one lying
    flat (measured flat: pos 44.1..44.3; empty: 3.4). The position gate exists
    to reject empty, so it must not also reject the narrower hold."""
    cfg = SensingConfig(grasp_settle_s=0.0, sample_interval_s=0.0, grasp_samples=1)
    for pos in (24.0, 18.0, 14.0):
        edgewise = MockRobotIO(initial_joints={"gripper": pos})
        edgewise.loads["gripper"] = 500
        check = check_grasp(edgewise, cfg)
        assert check.grasped, f"edge-on hold at pos={pos} misread as EMPTY"
        assert check.pos_says_held and check.load_says_held

    # an empty gripper still has room to spare below the threshold
    empty = MockRobotIO(initial_joints={"gripper": 3.4})
    empty.loads["gripper"] = 40
    result = check_grasp(empty, cfg)
    assert not result.grasped and not result.pos_says_held and not result.load_says_held


def test_retracting_arm_increases_reachable_hover():
    class EnvelopeIk:
        def solve(self, x_mm, y_mm, z_mm, yaw_deg=None):
            ceiling = 90.0 - max(0.0, math.hypot(x_mm, y_mm) - 195.0) * 0.45
            return IkResult({}, 0.5 if z_mm <= ceiling else 80.0, 0.1)

    cfg = AppConfig()
    far = highest_reachable_hover(EnvelopeIk(), 285.0, 0.0, 10.0, cfg)
    near = highest_reachable_hover(EnvelopeIk(), 195.0, 0.0, 10.0, cfg)
    assert near > far + 20.0


def test_reduced_bias_keeps_the_sideways_correction():
    """The reachability backoff must not throw away the left/right fix.

    Radial is what pushes a block past the reach envelope; a 10mm tangential
    nudge moves reach by 0.17mm. Scaling it down buys nothing and costs the
    whole left-half correction exactly where it is needed, since the left
    half hits the envelope sooner *because* of its extra radial offset.
    """
    cfg = AppConfig().motion
    det = (200.0, 80.0)  # left half
    full = biased_grasp_xy(cfg, *det, scale=1.0)
    none = biased_grasp_xy(cfg, *det, scale=0.0)
    # radial shrinks to nothing...
    assert math.hypot(*none) == pytest.approx(math.hypot(*det), abs=0.5)
    assert math.hypot(*full) > math.hypot(*det) + 20.0
    # ...while the sideways component survives untouched
    def tangential_of(p):
        phi = math.atan2(det[1], det[0])
        return -(p[0] - det[0]) * math.sin(phi) + (p[1] - det[1]) * math.cos(phi)
    assert tangential_of(none) == pytest.approx(tangential_of(full), abs=1e-9)
    assert tangential_of(none) == pytest.approx(
        cfg.grasp_tangential_offset_mm + cfg.left_half_tangential_offset_mm,
        abs=1e-9,
    )


@pytest.mark.parametrize("det", [(240.0, 0.0), (180.0, 140.0), (180.0, -140.0)])
def test_default_grasp_bias_is_ten_mm_to_relative_left_everywhere(det):
    cfg = AppConfig().motion
    aimed = biased_grasp_xy(cfg, *det)
    phi = math.atan2(det[1], det[0])
    tangential = -(aimed[0] - det[0]) * math.sin(phi) + (aimed[1] - det[1]) * math.cos(phi)

    assert tangential == pytest.approx(10.0, abs=1e-9)


def test_left_ramp_is_off_by_default_and_grows_with_y():
    cfg = AppConfig().motion
    near, far = (200.0, 40.0), (200.0, 160.0)
    assert cfg.left_ramp_tangential_mm_per_100mm == 0.0  # unsupported by the calibration data
    plain_near = biased_grasp_xy(cfg, *near)
    cfg.left_ramp_tangential_mm_per_100mm = 10.0
    ramped_near = biased_grasp_xy(cfg, *near)
    ramped_far = biased_grasp_xy(cfg, *far)
    shift_near = math.hypot(ramped_near[0] - plain_near[0], ramped_near[1] - plain_near[1])
    assert shift_near == pytest.approx(4.0, abs=1e-6)  # 0.4 of 100mm
    plain_far = biased_grasp_xy(AppConfig().motion, *far)
    shift_far = math.hypot(ramped_far[0] - plain_far[0], ramped_far[1] - plain_far[1])
    assert shift_far > shift_near
    # the right half is untouched either way
    assert biased_grasp_xy(cfg, 200.0, -80.0) == biased_grasp_xy(AppConfig().motion, 200.0, -80.0)


def test_block_angle_turns_the_jaws_and_is_recorded():
    cfg = AppConfig()
    plan = plan_grasp_attempts(StubIk(), cfg, 200.0, -80.0, 9.0, block_angle_deg=35.0)
    assert plan.yaw_deg == pytest.approx(35.0)
    assert all(a.yaw_deg == pytest.approx(35.0) for a in plan.attempts)
    # no angle given -> neutral yaw, as before
    assert plan_grasp_attempts(StubIk(), cfg, 200.0, -80.0, 9.0).yaw_deg is None


def test_plan_records_the_bias_scale_it_settled_on():
    plan = plan_grasp_attempts(StubIk(), AppConfig(), 200.0, -80.0, 9.0)
    assert plan.bias_scale == 1.0  # StubIk reaches everything
