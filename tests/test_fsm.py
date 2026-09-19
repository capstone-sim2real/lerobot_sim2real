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
from fsm.task1 import Task1Perception
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
    expired_without_task1_cutoff = RunContext(fsm=FsmConfig(num_blocks=1, time_budget_s=-1.0))
    assert StateMachine(_states(), expired_without_task1_cutoff, enforce_time_budget=False).run().placed_count == 1


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

    def grasp_yaw_deg(self, x_mm, y_mm, z_mm, block_angle_deg):
        return self.grasp_yaw_and_rotation_deg(x_mm, y_mm, z_mm, block_angle_deg)[0]

    def grasp_yaw_and_rotation_deg(self, x_mm, y_mm, z_mm, block_angle_deg):
        # stub neutral yaw is 0, so the block angle *is* the jaw rotation
        return block_angle_deg, block_angle_deg


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
