"""Task 2 tower ladder, level bookkeeping, and contact-descent contracts."""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from config import AppConfig, load_config, validate_task2
from control.grasp import GraspAttempt
from control.ik import IkResult
from control.motion import MotionController
from control.robot_io import MockRobotIO
from control.task1_transport import push_out_from_base
from control.task2_stack import (
    Task2StackPlan,
    Task2StackPlanner,
)
from control.trajectory import TrajectoryPlayer
from fsm.flows import build_task2_stack_states
from fsm.states import RunContext, StateName
from fsm.task1 import Task1Perception
from fsm.task2 import Task2PlaceState, Task2SelectState, Task2TransportState
from perception import BlockDetection, PlaneCalibration
from perception.zone import point_in_zone, zone_slot_centres

DEFAULT_YAML = "src/configs/default.yaml"


def _block(color: str, x: float, y: float) -> BlockDetection:
    return BlockDetection(color, (x, y), 1600.0, 1.0, 1.0, 1.0, [])


def _calibration() -> PlaneCalibration:
    return PlaneCalibration(
        H=np.eye(3),
        image_size=(500, 400),
        square_mm=1.0,
        base_xy_mm=(0.0, 0.0),
        zone_polygon_mm=[(300.0, 100.0), (300.0, -100.0), (200.0, -100.0), (200.0, 100.0)],
        meta={"grasp_z_mm_mean": 10.0},
    )


class _Motion:
    def __init__(self):
        self.home_calls = 0

    def go_home(self, *, include_gripper=True):
        assert include_gripper is False
        self.home_calls += 1


class _Samples:
    def __init__(self, values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


class _AlwaysReachableIk:
    def solve(self, x_mm, y_mm, z_mm, yaw_deg=None, radial_tilt_deg=0.0):
        return IkResult(
            {"wrist_flex": float(z_mm)},
            position_error_mm=0.0,
            tilt_error_deg=abs(radial_tilt_deg),
        )


class _CeilingIk:
    """Solves cleanly up to ``ceiling_mm`` and misses badly above it."""

    def __init__(self, ceiling_mm: float):
        self.ceiling_mm = ceiling_mm

    def solve(self, x_mm, y_mm, z_mm, yaw_deg=None, radial_tilt_deg=0.0):
        error = 0.0 if z_mm <= self.ceiling_mm else 50.0
        return IkResult(
            {"wrist_flex": float(z_mm)},
            position_error_mm=error,
            tilt_error_deg=abs(radial_tilt_deg),
        )


def _planner(cfg: AppConfig | None = None, ik=None) -> Task2StackPlanner:
    cfg = cfg or AppConfig()
    return Task2StackPlanner(_calibration(), cfg, ik or _AlwaysReachableIk())


# --------------------------------------------------------------------------
# ladder geometry
# --------------------------------------------------------------------------


def test_an_unreachable_hover_alone_blocks_a_level():
    """Isolates the hover re-gate from the floor gate.

    The ceiling here clears every level's descent floor, so if the planner
    trusted the height highest_reachable_hover handed back, every level would
    look reachable and the tower clearance would be fictional. It also sits
    below level two's *squeeze* floor, so the level is refused on the hover
    alone rather than rescued by a lower approach.
    """
    cfg = AppConfig()
    planner = _planner(cfg, _CeilingIk(ceiling_mm=39.0))
    level_two = planner.levels[1]

    assert planner.levels[0].reachable is True
    assert level_two.floor.position_error_mm == 0.0  # the floor is fine...
    assert level_two.reachable is False  # ...and the level is still refused
    assert "hover" in level_two.reason


# --------------------------------------------------------------------------
# level bookkeeping / TRANSPORT
# --------------------------------------------------------------------------


def _transport_doubles():
    result = IkResult({"wrist_flex": 0.0}, position_error_mm=0.0, tilt_error_deg=0.0)
    held = GraspAttempt("centre", (0.0, 0.0), (100.0, 0.0), result, result, True)

    class Planner:
        def __init__(self, reachable=(True,) * 5):
            self.indices = []
            self.levels = tuple(
                type(
                    "Level",
                    (),
                    {
                        "level": index + 1,
                        "reachable": ok,
                        "reason": "" if ok else "too high",
                        "hover_squeezed": False,
                    },
                )()
                for index, ok in enumerate(reachable)
            )

        def plan(self, _held, level_index):
            self.indices.append(level_index)
            level = type(
                "Level",
                (),
                {"level": level_index + 1, "hover": result, "release": result},
            )()
            return type("Plan", (), {"slot": level, "carry": ()})()

    class Player:
        def __init__(self):
            self.moves = 0
            self.goals = []

        def move_to(self, goal, **_kwargs):
            self.moves += 1
            self.goals.append(goal)

    return held, Planner, Player


# --------------------------------------------------------------------------
# SELECT guard
# --------------------------------------------------------------------------


def _select_state(cfg, detections):
    samples = _Samples([Task1Perception(detections, 1, 1000.0)])
    return Task2SelectState(_Motion(), samples, _calibration(), cfg)


# --------------------------------------------------------------------------
# PLACE
# --------------------------------------------------------------------------

_ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def _place_plan(
    hover_offset: float = 100.0, *, contact_descent: bool = True, level: int = 1
) -> Task2StackPlan:
    cfg = AppConfig()
    hover = IkResult({j: 0.0 for j in _ARM}, 0.0, 0.0)
    floor = IkResult({j: -hover_offset for j in _ARM}, 0.0, 0.0)
    release = IkResult(
        {j: -hover_offset + cfg.task2.place_overshoot_mm for j in _ARM}, 0.0, 0.0
    )
    plan = type(
        "Level",
        (),
        {
            "level": level,
            "hover": hover,
            "release": release,
            "floor": floor,
            "contact_descent": contact_descent,
            "hover_z_mm": 45.0,
            "place_z_mm": 12.0,
            "floor_z_mm": 6.0,
            "radial_tilt_deg": 0.0,
        },
    )()
    return Task2StackPlan(slot=plan, carry=())


class _RecordingPlayer(TrajectoryPlayer):
    def __init__(self, robot, cfg):
        super().__init__(robot, cfg)
        self.descend_kwargs = []

    def descend(self, goal, **kwargs):
        self.descend_kwargs.append(kwargs)
        return super().descend(goal, **kwargs)


def _place_harness(cfg, robot):
    # Exercise the unreachable legacy helper explicitly. Validated runtime
    # config is fixed at zero and bypasses all of this contact logic.
    cfg.task2.contact_descent_levels = 1
    cfg.motion.fps = 0
    cfg.motion.place_settle_s = 0.0
    cfg.motion.descent_settle_s = 0.0
    cfg.sensing.gripper_action_wait_s = 0.0
    player = _RecordingPlayer(robot, cfg.motion)
    motion = MotionController(robot, _EmptyPoses(), cfg.motion, cfg.sensing)
    state = Task2PlaceState(robot, motion, player, cfg)
    ctx = RunContext(cfg.fsm)
    ctx.extras["task2_stack_plan"] = _place_plan()
    state.enter(ctx)
    return state, ctx, player


class _EmptyPoses:
    def get(self, name):
        raise KeyError(name)


def _run_place(state, ctx, limit=10):
    for _ in range(limit):
        result = state.step(ctx)
        if result is not None:
            return result
    raise AssertionError("PLACE never returned a next state")


class _TrailingRobot(MockRobotIO):
    """Lags a fixed distance behind a descending command, like a loaded arm.

    Only on the way down: gravity leaves the steady-state offset in the
    direction of travel, and a rising command has the block's weight helping
    rather than resisting.
    """

    def __init__(self, trail: float):
        super().__init__()
        self.trail = trail

    def send_joints(self, positions):
        trailed = {}
        for joint, value in positions.items():
            current = self.joints[joint]
            # min(): the lag builds up before the arm starts moving at all,
            # instead of the servo jumping backwards on the first tick.
            trailed[joint] = min(current, value + self.trail) if value < current else value
        return super().send_joints(trailed)


class _TowerRobot(MockRobotIO):
    """Simulates the tower itself: the arm cannot descend past its top.

    The stub IK maps a joint value straight to z in mm, so clamping one joint
    is a faithful stand-in for a block meeting the stack. The top rises by a
    block height every time the jaws open.
    """

    def __init__(self, first_top: float, block_height: float, open_pos: float):
        super().__init__()
        self.tower_top = first_top
        self.block_height = block_height
        self.open_pos = open_pos

    def send_joints(self, positions):
        positions = dict(positions)
        if "wrist_flex" in positions:
            positions["wrist_flex"] = max(positions["wrist_flex"], self.tower_top)
        opening = positions.get("gripper") == self.open_pos
        result = super().send_joints(positions)
        if opening:
            self.tower_top += self.block_height
        return result


class _JammingRobot(MockRobotIO):
    """Stops following once a joint passes ``stop_at`` -- the lag signal."""

    def __init__(self, stop_at: float):
        super().__init__()
        self.stop_at = stop_at

    def send_joints(self, positions):
        positions = dict(positions)
        for joint in positions:
            positions[joint] = max(positions[joint], self.stop_at)
        return super().send_joints(positions)


def test_place_stops_on_joint_lag_and_releases_above_the_floor():
    cfg = AppConfig()
    robot = _JammingRobot(stop_at=-75.0)  # three quarters down a 100-unit descent
    state, ctx, _player = _place_harness(cfg, robot)

    assert _run_place(state, ctx) is StateName.SELECT
    record = ctx.extras["stack_contacts"][-1]
    assert record["source"] == "lag"
    assert record["contact"] is True
    assert record["early"] is False
    assert record["descent_fraction"] == pytest.approx(0.75)
    assert record["shortfall"] == pytest.approx(25.0)
    assert ctx.placed_count == 1
    assert ctx.extras["task2_tower_height"] == 1

    # The backoff must lift the arm before the jaws part, or they drag the
    # tower sideways as they open. Anchor on the deepest commanded tick: the
    # descent itself is full of poses "above" the contact depth on the way
    # down, so only what happens after the bottom counts.
    depths = [
        (action["shoulder_lift"], i)
        for i, action in enumerate(robot.sent_actions)
        if "shoulder_lift" in action
    ]
    bottom = min(depths)[1]
    after = robot.sent_actions[bottom + 1 :]
    lift = next(
        i
        for i, action in enumerate(after)
        if "shoulder_lift" in action and action["shoulder_lift"] > depths[bottom][0]
    )
    gripper_open = next(
        i
        for i, action in enumerate(after)
        if action.get("gripper") == cfg.sensing.gripper_open_pos
    )
    assert lift < gripper_open


# --------------------------------------------------------------------------
# flow composition + config
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# end to end through the real StateMachine
# --------------------------------------------------------------------------


def test_a_whole_tower_cycles_through_the_real_state_machine(monkeypatch):
    """SELECT -> PICK -> VERIFY -> TRANSPORT -> PLACE -> SELECT, for real.

    The individual states are covered above; this is the wiring: that the
    level climbs once per released block, that a plan from one cycle never
    leaks into the next, and that the run ends on the empty-frame proof
    rather than on a block count.
    """
    from fsm.machine import StateMachine
    from fsm.states import State

    cfg = AppConfig()
    cfg.task1.scan_interval_s = 0.0
    cfg.motion.fps = 0
    cfg.motion.place_settle_s = 0.0
    cfg.sensing.gripper_action_wait_s = 0.0

    clock = {"wall": 1000.0, "mono": 10.0}
    monkeypatch.setattr("fsm.task1.time.time", lambda: clock["wall"])
    monkeypatch.setattr("fsm.task1.time.monotonic", lambda: clock["mono"])

    blocks = ["green", "blue", "wood"]
    frames = []
    for i in range(len(blocks)):
        frames.append(
            Task1Perception([_block(c, 120.0 + 10 * n, 0.0) for n, c in enumerate(blocks[i:])], i + 1, 1000.0)
        )
    # Then the outside region is empty long enough to prove the run is done.
    frames.append(Task1Perception([], 90, 1000.0))
    frames.append(Task1Perception([], 91, 1000.0))

    def perceive():
        sample = frames.pop(0)
        if not sample.detections:
            clock["mono"] += cfg.task1.empty_timeout_s
        return sample

    robot = _TowerRobot(
        first_top=10.0 + cfg.task2.release_clearance_mm,  # grasp_z + clearance
        block_height=cfg.task2.block_height_mm,
        open_pos=cfg.sensing.gripper_open_pos,
    )
    planner = _planner(cfg)
    player = _RecordingPlayer(robot, cfg.motion)
    motion = MotionController(robot, _EmptyPoses(), cfg.motion, cfg.sensing)

    class _StubPick(State):
        name = StateName.PICK

        def step(self, ctx):
            # A real pick closes the jaws; the tower simulator counts each
            # open as one block laid down, so the cycle has to be complete.
            robot.send_joints({"gripper": cfg.sensing.gripper_close_pos})
            result = IkResult({j: 0.0 for j in _ARM}, 0.0, 0.0)
            ctx.extras["ik_pick_attempt"] = GraspAttempt(
                "centre", (0.0, 0.0), (120.0, 0.0), result, result, True, grasp_z_mm=10.0
            )
            return StateName.VERIFY

    class _StubVerify(State):
        name = StateName.VERIFY

        def step(self, ctx):
            return StateName.TRANSPORT

    states = {
        StateName.SELECT: Task2SelectState(_Motion(), perceive, _calibration(), cfg),
        StateName.PICK: _StubPick(),
        StateName.VERIFY: _StubVerify(),
        StateName.TRANSPORT: Task2TransportState(planner, player, cfg),
        StateName.PLACE: Task2PlaceState(robot, motion, player, cfg),
    }
    ctx = RunContext(cfg.fsm)
    StateMachine(states, ctx, enforce_time_budget=False).run()

    assert ctx.placed_count == len(blocks)
    assert ctx.extras["task2_tower_height"] == len(blocks)
    assert ctx.extras["task1_complete"] is True  # the empty-frame proof, not a count
    # One placement record per block, and the level climbed every time.
    assert [r["level"] for r in ctx.extras["stack_contacts"]] == [1, 2, 3]
    # Every level is released directly, including the first one.
    records = ctx.extras["stack_contacts"]
    assert [r["mode"] for r in records] == [
        "transport_release", "transport_release", "transport_release"
    ]
    assert all(r["source"] == "transport" for r in records)
    assert ctx.extras["task2_level_index"] == len(blocks) - 1
