"""Task 2 tower ladder and level bookkeeping."""

from __future__ import annotations


import numpy as np
import pytest

from config import AppConfig, load_config
from control.grasp import GraspAttempt
from control.ik import IkResult
from control.motion import MotionController
from control.robot_io import MockRobotIO
from control.task2_stack import Task2StackPlanner
from control.trajectory import TrajectoryPlayer
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


def test_task2_stage_is_outside_target_zone():
    planner = _planner()
    level = planner.levels[0]
    # Fake IK encodes z only; separately check the computed XY with a spy.
    requested = []

    class RecordingIk(_AlwaysReachableIk):
        def solve(self, x_mm, y_mm, z_mm, yaw_deg=None, radial_tilt_deg=0.0):
            requested.append((x_mm, y_mm, z_mm))
            return super().solve(x_mm, y_mm, z_mm, yaw_deg, radial_tilt_deg)

    planner._ik = RecordingIk()
    planner.outside_stage(level)
    assert not point_in_zone(requested[-1][:2], _calibration())
    assert requested[-1][2] == level.hover_z_mm


def test_task2_upper_entry_is_above_nominal_hover_with_folded_arm():
    cfg = AppConfig()
    planner = _planner(cfg)
    level = planner.levels[4]
    requested = []

    class RecordingIk(_AlwaysReachableIk):
        def solve(self, x_mm, y_mm, z_mm, yaw_deg=None, radial_tilt_deg=0.0):
            requested.append((z_mm, radial_tilt_deg))
            return super().solve(x_mm, y_mm, z_mm, yaw_deg, radial_tilt_deg)

    planner._ik = RecordingIk()
    planner.entry_pose(level)
    assert requested[-1] == (
        level.place_z_mm + cfg.task2.upper_entry_clearance_mm,
        cfg.task2.upper_entry_radial_tilt_deg,
    )


# --------------------------------------------------------------------------
# ladder geometry
# --------------------------------------------------------------------------


def test_task2_tower_starts_at_bottom_center_of_task1_row():
    cfg = load_config(DEFAULT_YAML)
    planner = _planner(cfg)
    left, right = zone_slot_centres(_calibration(), cfg.task1.slot_uv)[-2:]
    expected = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)

    assert planner.raw_xy_mm == pytest.approx(expected)
    assert all(level.xy_mm == pytest.approx(planner.stack_xy_mm)
               for level in planner.levels)


def test_an_unreachable_hover_alone_blocks_a_level():
    """The ceiling sits below level two's squeeze floor, so the level is
    refused on the hover alone rather than rescued by a lower approach."""
    cfg = AppConfig()
    planner = _planner(cfg, _CeilingIk(ceiling_mm=39.0))
    level_two = planner.levels[1]

    assert planner.levels[0].reachable is True
    assert level_two.reachable is False
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


class _EmptyPoses:
    def get(self, name):
        raise KeyError(name)


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
    player = TrajectoryPlayer(robot, cfg.motion)
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
        StateName.PLACE: Task2PlaceState(motion),
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
