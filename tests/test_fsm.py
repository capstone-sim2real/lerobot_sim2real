"""FSM budget/verification rules and CV+IK PICK adapter contracts."""

import numpy as np

from config import AppConfig, FsmConfig, SensingConfig
from control import MockRobotIO
from control.grasp import GraspAttempt, GraspPlan
from control.ik import IkResult
from fsm import ik_handler
from fsm.flows import build_pick_lift_lower_states, build_task1_states, build_task2_states
from fsm.handlers import VerifyState
from fsm.ik_handler import CvIkPickState
from fsm.machine import StateMachine
from fsm.states import RunContext, State, StateName
from perception.detector import BlockDetection
from perception.homography import PlaneCalibration
from perception.select import SelectionResult
from runners.run_task import make_pick_state


class _Select(State):
    name = StateName.SELECT

    def step(self, ctx):
        if ctx.extras.get("placed"):
            return StateName.DONE
        ctx.target_id = "block"
        return StateName.PICK


class _Pick(State):
    name = StateName.PICK

    def step(self, ctx):
        ctx.record_attempt(ctx.target_id)
        return StateName.VERIFY


class _Verify(State):
    name = StateName.VERIFY

    def step(self, ctx):
        if ctx.should_skip(ctx.target_id):
            ctx.skip(ctx.target_id)
            return StateName.DONE
        return StateName.TRANSPORT


class _Transport(State):
    name = StateName.TRANSPORT

    def step(self, ctx):
        return StateName.PLACE


class _Place(State):
    name = StateName.PLACE

    def step(self, ctx):
        ctx.placed_count += 1
        ctx.extras["placed"] = True
        return StateName.SELECT


def _states():
    return {StateName.SELECT: _Select(), StateName.PICK: _Pick(), StateName.VERIFY: _Verify(), StateName.TRANSPORT: _Transport(), StateName.PLACE: _Place()}


def test_fsm_places_a_block_and_stops_at_time_budget():
    ctx = RunContext(fsm=FsmConfig(num_blocks=1, reserve_time_s=0.0))
    assert StateMachine(_states(), ctx).run().placed_count == 1
    expired = RunContext(fsm=FsmConfig(time_budget_s=-1.0))
    assert StateMachine(_states(), expired).run().placed_count == 0


def test_verify_never_transports_an_empty_gripper():
    class Motion:
        opened = 0

        def open_gripper(self):
            self.opened += 1

    robot, motion = MockRobotIO(initial_joints={"gripper": 3.0}), Motion()
    robot.loads["gripper"] = 10
    ctx = RunContext(fsm=FsmConfig(max_retries_per_block=1))
    ctx.target_id = "block"
    ctx.record_attempt("block")
    assert VerifyState(robot, SensingConfig(grasp_settle_s=0.0, sample_interval_s=0.0, grasp_samples=1), motion).step(ctx) is StateName.SELECT
    assert ctx.target_id in ctx.skipped and motion.opened == 1


class _FakeIk:
    def solve(self, x_mm, y_mm, z_mm, yaw_deg=None):
        return IkResult({"shoulder_pan": 0.0, "shoulder_lift": -20.0, "elbow_flex": 30.0, "wrist_flex": 10.0, "wrist_roll": 0.0}, 0.1, 0.1)


class _FakePlayer:
    def __init__(self):
        self.goals = []

    def move_to(self, goal, **kwargs):
        self.goals.append(dict(goal))


class _FakeMotion:
    def __init__(self):
        self.opened = 0

    def open_gripper(self):
        self.opened += 1


def _cv_ik_context():
    target = BlockDetection("green", (220.0, -20.0), 1600.0, 1.0, 1.0, 1.0, [])
    ctx = RunContext(fsm=FsmConfig(max_retries_per_block=1))
    ctx.target_id = "green:6,0"
    ctx.extras["selection"] = SelectionResult(target, ctx.target_id, 1, [target])
    return ctx


def _cv_ik_state(motion, player):
    return CvIkPickState(robot=object(), motion=motion, cfg=AppConfig(), grasp_z_mm=8.0,
                          retreat_pose={"shoulder_pan": 5.0, "gripper": 2.0}, ik=_FakeIk(), player=player)


def test_cv_ik_pick_holds_then_retreats_without_opening_gripper(monkeypatch):
    motion, player, ctx = _FakeMotion(), _FakePlayer(), _cv_ik_context()
    monkeypatch.setattr(ik_handler, "run_grasp_attempts", lambda _p, _r, _c, plan, **_kw: plan.attempts[0])
    assert _cv_ik_state(motion, player).step(ctx) is StateName.VERIFY
    assert motion.opened == 0 and player.goals[1] == {"shoulder_pan": 5.0}


def test_cv_ik_pick_skips_after_empty_attempt(monkeypatch):
    motion, ctx = _FakeMotion(), _cv_ik_context()
    monkeypatch.setattr(ik_handler, "run_grasp_attempts", lambda *_args, **_kwargs: None)
    assert _cv_ik_state(motion, _FakePlayer()).step(ctx) is StateName.SELECT
    assert ctx.target_id in ctx.skipped and motion.opened == 1


def test_cv_ik_pick_uses_a_reachable_offset_when_centre_is_unreachable(monkeypatch):
    motion, player, ctx = _FakeMotion(), _FakePlayer(), _cv_ik_context()
    solved = IkResult({"shoulder_pan": 0.0}, 0.1, 0.1)
    centre = GraspAttempt("centre", (0.0, 0.0), (220.0, -20.0), solved, solved, False)
    offset = GraspAttempt("back-left", (-10.0, 10.0), (210.0, -10.0), solved, solved, True)
    plan = GraspPlan((220.0, -20.0), (220.0, -20.0), 8.0, 48.0, [centre, offset], [])
    monkeypatch.setattr(ik_handler, "highest_reachable_hover", lambda *_args: 48.0)
    monkeypatch.setattr(ik_handler, "plan_grasp_attempts", lambda *_args: plan)
    attempted = []

    def run(_player, _robot, _cfg, received_plan, **_kwargs):
        attempted.extend(received_plan.attempts)
        return offset

    monkeypatch.setattr(ik_handler, "run_grasp_attempts", run)
    assert _cv_ik_state(motion, player).step(ctx) is StateName.VERIFY
    assert attempted == [centre, offset] and motion.opened == 0


def test_runner_builds_cv_ik_pick_without_policy_server():
    state = make_pick_state(
        "cv_ik",
        robot=object(),
        motion=_FakeMotion(),
        cfg=AppConfig(),
        calib=PlaneCalibration(H=np.eye(3), image_size=(1, 1), square_mm=25.0, meta={"grasp_z_mm_mean": 8.0}),
        retreat_pose={"shoulder_pan": 0.0},
    )
    assert isinstance(state, CvIkPickState)


def test_flows_compose_only_the_states_each_workflow_needs():
    pick = _Pick()
    robot, motion = object(), _FakeMotion()
    perceive = lambda _skipped: SelectionResult(None, None, 0, [])

    task1 = build_task1_states(
        robot=robot, motion=motion, perceive=perceive, pick_state=pick, sensing_cfg=SensingConfig()
    )
    task2 = build_task2_states(
        robot=robot, motion=motion, perceive=perceive, pick_state=pick, sensing_cfg=SensingConfig()
    )
    smoke = build_pick_lift_lower_states(
        robot=robot, motion=motion, perceive=perceive, pick_state=pick, cfg=AppConfig()
    )

    assert set(task1) == {StateName.SELECT, StateName.PICK, StateName.VERIFY, StateName.TRANSPORT, StateName.PLACE}
    assert set(task2) == set(task1)
    assert set(smoke) == {
        StateName.SELECT, StateName.PICK, StateName.VERIFY, StateName.LIFT,
        StateName.LOWER, StateName.RELEASE, StateName.RETURN,
    }
