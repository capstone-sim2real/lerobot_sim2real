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


def test_every_level_shares_one_xy_and_climbs_by_one_block_height():
    cfg = AppConfig()
    levels = _planner(cfg).levels

    assert len(levels) == cfg.task2.max_levels
    assert len({level.xy_mm for level in levels}) == 1  # a tower, not a row
    grasp_z, height = 10.0, cfg.task2.block_height_mm
    assert [level.place_z_mm for level in levels] == pytest.approx(
        [grasp_z + cfg.task2.release_clearance_mm + height * n for n in range(5)]
    )


def test_stack_point_lands_inside_the_zone_but_is_commanded_further_out():
    cfg = AppConfig()
    calib = _calibration()
    planner = _planner(cfg)
    raw = zone_slot_centres(calib, [list(cfg.task2.stack_uv)])[0]

    # The landing point must be inside the zone: that is what makes the
    # detector ignore placed blocks, which is what lets the empty-timeout
    # completion criterion work for Task 2 at all.
    assert planner.raw_xy_mm == pytest.approx(raw)
    assert point_in_zone(planner.raw_xy_mm, calib)
    assert planner.stack_xy_mm == pytest.approx(
        push_out_from_base(raw, calib.base_xy_mm, cfg.task2.stack_radial_offset_mm)
    )
    assert math.hypot(*planner.stack_xy_mm) - math.hypot(*raw) == pytest.approx(
        cfg.task2.stack_radial_offset_mm
    )


def test_stack_point_is_nearer_the_base_than_task1_far_slots():
    """Lift is bought by folding the arm in, so the tower goes near-edge."""
    cfg = AppConfig()
    calib = _calibration()
    task1_radii = [
        math.hypot(*xy) for xy in zone_slot_centres(calib, cfg.task1.slot_uv)
    ]
    assert math.hypot(*_planner(cfg).raw_xy_mm) < max(task1_radii)


def test_descent_floor_is_commanded_below_the_nominal_release_height():
    cfg = AppConfig()
    for level in _planner(cfg).levels:
        assert level.floor_z_mm == pytest.approx(
            level.place_z_mm - cfg.task2.place_overshoot_mm
        )
    # Deliberate interference at level one: contact must fire before the goal.
    assert _planner(cfg).levels[0].floor_z_mm < 10.0


def test_level_tilt_ramp_starts_late_and_is_capped():
    cfg = AppConfig()
    assert [lv.radial_tilt_deg for lv in _planner(cfg).levels] == pytest.approx(
        [0.0, 0.0, -1.5, -3.0, -4.5]
    )

    steep = AppConfig()
    steep.task2.level_tilt_per_level_deg = 4.0
    tilts = [lv.radial_tilt_deg for lv in _planner(steep).levels]
    assert tilts[-1] == pytest.approx(-steep.task2.level_tilt_max_deg)
    assert all(abs(t) <= steep.ik.max_tilt_error_deg for t in tilts)


def test_unreachable_upper_level_is_reported_not_clipped():
    cfg = AppConfig()
    planner = _planner(cfg, _CeilingIk(ceiling_mm=40.0))
    levels = planner.levels

    assert len(levels) == cfg.task2.max_levels  # nothing truncated
    assert levels[0].reachable is True
    assert levels[-1].reachable is False
    assert levels[-1].reason  # the dry-run prints this verbatim

    # Reported, but never refused: the planner still hands back a plan for
    # the level so a grasped block is always carried to the tower.
    held = GraspAttempt("centre", (0.0, 0.0), (100.0, 0.0), None, None, True)
    plan = planner.plan(held, 4)
    assert plan.level is levels[-1]


def test_hover_search_floor_is_regated_not_trusted():
    """highest_reachable_hover returns its floor when nothing solves.

    That floor is an *unreachable* height dressed up as an answer. If the
    planner trusted it, the tower clearance would be fictional.
    """
    cfg = AppConfig()
    planner = _planner(cfg, _CeilingIk(ceiling_mm=40.0))
    blocked = [lv for lv in planner.levels if not lv.reachable]

    assert blocked, "the stub should put at least one level out of reach"
    for level in blocked:
        # The floor still came back looking like a height...
        assert level.hover_z_mm >= level.place_z_mm + cfg.task2.hover_min_clearance_mm
        # ...and the planner refused it anyway.
        assert level.hover.position_error_mm > cfg.task2.hover_gate_mm


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


def test_a_level_is_squeezed_rather_than_refused_when_only_the_band_is_too_high():
    """A level the arm can place at is never given up for approach clearance.

    The ceiling clears level two's squeeze floor but not its preferred
    clearance band. Refusing here would cost a whole block, and the tower may
    be shorter than placed_count claims anyway -- it can collapse -- so the
    approach is attempted rather than predicted away.
    """
    cfg = AppConfig()
    level_two = _planner(cfg, _CeilingIk(ceiling_mm=46.0)).levels[1]
    clearance = level_two.hover_z_mm - level_two.place_z_mm

    assert level_two.reachable is True
    assert level_two.hover_squeezed is True
    assert level_two.hover.position_error_mm <= cfg.task2.hover_gate_mm
    # Below the preferred band, but never below the squeeze floor.
    assert cfg.task2.hover_squeeze_clearance_mm <= clearance
    assert clearance < cfg.task2.hover_min_clearance_mm


def test_a_level_that_needs_no_squeeze_is_not_flagged_as_one():
    for level in _planner().levels:
        assert level.hover_squeezed is False


def test_a_hover_that_cannot_hold_its_tilt_blocks_a_level():
    """Position and attitude are separate failures.

    A hover solved at the right point but the wrong attitude is not the pose
    the descent under it was planned for, and the position gate says nothing
    about that.
    """

    class _FlatteningIk:
        """Reaches every point, but never manages more than 1 degree of tilt."""

        def solve(self, x_mm, y_mm, z_mm, yaw_deg=None, radial_tilt_deg=0.0):
            achieved = 99.0 if radial_tilt_deg else 0.0
            return IkResult({"wrist_flex": float(z_mm)}, 0.0, achieved)

    cfg = AppConfig()
    planner = _planner(cfg, _FlatteningIk())
    tilted = planner.levels[2]  # first level the ramp tilts

    assert planner.levels[0].reachable is True  # untilted levels are fine
    assert tilted.hover.position_error_mm == 0.0  # the point is reachable...
    assert tilted.reachable is False  # ...but the attitude is not
    assert "tilt" in tilted.reason


def test_an_unreachable_release_pose_alone_blocks_a_level():
    """The release pose is the only one a directly released level depends on.

    Its hover is just a waypoint on the way in; if the height the jaws
    actually open at is out of reach, the level is not usable no matter how
    good the approach looks.
    """

    class _BandGapIk:
        """Reaches everything except a narrow band around level 2's release."""

        def solve(self, x_mm, y_mm, z_mm, yaw_deg=None, radial_tilt_deg=0.0):
            miss = 30.0 < z_mm < 35.0
            return IkResult(
                {"wrist_flex": float(z_mm)},
                position_error_mm=50.0 if miss else 0.0,
                tilt_error_deg=abs(radial_tilt_deg),
            )

    cfg = AppConfig()
    planner = _planner(cfg, _BandGapIk())
    level_two = planner.levels[1]

    assert level_two.place_z_mm == pytest.approx(32.0)  # inside the gap
    assert level_two.contact_descent is False  # released from above
    assert level_two.hover.position_error_mm == 0.0  # the approach is fine...
    assert level_two.reachable is False  # ...and the level is still refused
    assert "release" in level_two.reason


def test_reachable_levels_stops_counting_at_the_first_gap():
    planner = _planner(AppConfig(), _CeilingIk(ceiling_mm=40.0))
    assert planner.reachable_levels < AppConfig().task2.max_levels
    assert all(lv.reachable for lv in planner.levels[: planner.reachable_levels])


def test_level_one_out_of_reach_aborts_construction():
    with pytest.raises(ValueError, match="level 1"):
        _planner(AppConfig(), _CeilingIk(ceiling_mm=-1000.0))


def test_describe_names_every_level_and_the_expected_tower():
    text = _planner(AppConfig(), _CeilingIk(ceiling_mm=40.0)).describe()
    assert "reachable levels:" in text
    assert "NO" in text  # the unreachable rows carry their reason


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


def test_level_comes_from_the_tower_height_and_is_never_memoised_per_colour():
    """A colour re-picked after a failure must not get its old level back."""
    cfg = AppConfig()
    held, Planner, Player = _transport_doubles()
    planner = Planner()
    state = Task2TransportState(planner, Player(), cfg)
    ctx = RunContext(cfg.fsm)
    ctx.extras["ik_pick_attempt"] = held

    for color in ("wood", "blue", "blue"):  # the repeated colour is the point
        ctx.target_id = color
        assert state.step(ctx) is StateName.PLACE
        ctx.placed_count += 1  # PLACE does this on the real path

    assert planner.indices == [0, 1, 2]
    assert "task1_slot_by_color" not in ctx.extras


def test_transport_writes_only_its_own_plan_keys():
    cfg = AppConfig()
    held, Planner, Player = _transport_doubles()
    state = Task2TransportState(Planner(), Player(), cfg)
    ctx = RunContext(cfg.fsm)
    ctx.extras["ik_pick_attempt"] = held
    ctx.target_id = "green"

    state.step(ctx)
    assert isinstance(ctx.extras["task2_level_index"], int)
    assert "task2_stack_plan" in ctx.extras
    assert "task1_transport_plan" not in ctx.extras
    assert "task1_slot_index" not in ctx.extras


def test_transport_finishes_at_hover_without_descending_to_release():
    cfg = AppConfig()
    held, _Planner, Player = _transport_doubles()
    hover = IkResult({"wrist_flex": 40.0}, 0.0, 0.0)
    release = IkResult({"wrist_flex": 10.0}, 0.0, 0.0)

    class Planner:
        levels = (
            type(
                "Level",
                (),
                {
                    "level": 1,
                    "reachable": True,
                    "reason": "",
                    "hover_squeezed": False,
                    "hover_z_mm": 40.0,
                    "place_z_mm": 10.0,
                },
            )(),
        )

        def plan(self, _held, _level_index):
            level = type(
                "Level", (), {"level": 1, "hover": hover, "release": release}
            )()
            return type("Plan", (), {"slot": level, "carry": ()})()

    player = Player()
    state = Task2TransportState(Planner(), player, cfg)
    ctx = RunContext(cfg.fsm)
    ctx.extras["ik_pick_attempt"] = held
    ctx.target_id = "green"

    assert state.step(ctx) is StateName.PLACE
    assert player.goals == [hover.joints]


def test_transport_flies_an_unreachable_level_instead_of_stopping():
    """A grasped block is always carried, whatever the ladder predicted.

    The ladder's level comes from placed_count, which is dead reckoning: the
    tower it assumes may have collapsed. Refusing loses the block for certain,
    so the arm goes and is allowed to press into its envelope.
    """
    cfg = AppConfig()
    held, Planner, Player = _transport_doubles()
    planner = Planner(reachable=(True, True, False, False, False))
    player = Player()
    state = Task2TransportState(planner, player, cfg)
    ctx = RunContext(cfg.fsm, placed_count=2)
    ctx.extras["ik_pick_attempt"] = held
    ctx.target_id = "red"

    assert state.step(ctx) is not StateName.DONE
    assert "task2_max_height_reached" not in ctx.extras
    assert planner.indices == [2]  # the unreachable level was planned anyway
    assert player.moves > 0  # ...and flown
    assert ctx.extras["task2_forced_levels"] == [3]


def test_transport_reuses_top_level_when_the_ladder_runs_out():
    cfg = AppConfig()
    held, Planner, Player = _transport_doubles()
    planner = Planner()
    player = Player()
    state = Task2TransportState(planner, player, cfg)
    ctx = RunContext(cfg.fsm, placed_count=5)
    ctx.extras["ik_pick_attempt"] = held
    ctx.target_id = "red"

    assert state.step(ctx) is StateName.PLACE
    assert planner.indices == [4]
    assert player.moves > 0
    assert ctx.extras["task2_reused_top_level"] == 1
    assert "task2_stop_reason" not in ctx.extras


def test_failed_pick_does_not_consume_a_level(monkeypatch):
    cfg = AppConfig()
    cfg.task1.scan_interval_s = 0.0
    clock = {"wall": 1000.0, "mono": 10.0}
    monkeypatch.setattr("fsm.task1.time.time", lambda: clock["wall"])
    monkeypatch.setattr("fsm.task1.time.monotonic", lambda: clock["mono"])
    samples = _Samples([Task1Perception([_block("blue", 120.0, 0.0)], 1, 1000.0)])
    state = Task2SelectState(_Motion(), samples, _calibration(), cfg)
    ctx = RunContext(cfg.fsm)
    state.enter(ctx)

    assert state.step(ctx) is StateName.PICK  # PICK may now fail back to SELECT
    assert ctx.placed_count == 0


# --------------------------------------------------------------------------
# SELECT guard
# --------------------------------------------------------------------------


def _select_state(cfg, detections):
    samples = _Samples([Task1Perception(detections, 1, 1000.0)])
    return Task2SelectState(_Motion(), samples, _calibration(), cfg)


def test_outside_zone_block_near_stack_point_is_never_suppressed(monkeypatch):
    cfg = AppConfig()
    cfg.task1.scan_interval_s = 0.0
    monkeypatch.setattr("fsm.task1.time.time", lambda: 1000.0)
    monkeypatch.setattr("fsm.task1.time.monotonic", lambda: 10.0)
    dropped = _block("red", 290.0, 0.0)  # 40mm outward of the tower
    state = _select_state(cfg, [dropped])
    ctx = RunContext(cfg.fsm)
    state.enter(ctx)

    assert state.step(ctx) is StateName.PICK
    assert ctx.target_id == "red"


def test_a_real_block_between_the_base_and_the_tower_is_never_swallowed(monkeypatch):
    """Parallax only ever displaces a ghost outward, so inward stays eligible."""
    cfg = AppConfig()
    cfg.task1.scan_interval_s = 0.0
    monkeypatch.setattr("fsm.task1.time.time", lambda: 1000.0)
    monkeypatch.setattr("fsm.task1.time.monotonic", lambda: 10.0)
    state = _select_state(cfg, [_block("red", 210.0, 0.0)])  # 40mm inward
    ctx = RunContext(cfg.fsm)
    state.enter(ctx)

    assert state.step(ctx) is StateName.PICK
    assert ctx.target_id == "red"


def test_select_pops_only_its_own_plan_keys(monkeypatch):
    cfg = AppConfig()
    cfg.task1.scan_interval_s = 0.0
    monkeypatch.setattr("fsm.task1.time.time", lambda: 1000.0)
    monkeypatch.setattr("fsm.task1.time.monotonic", lambda: 10.0)
    state = _select_state(cfg, [_block("blue", 120.0, 0.0)])
    ctx = RunContext(cfg.fsm)
    ctx.extras["task1_transport_plan"] = "task1 plan must survive"
    ctx.extras["task2_stack_plan"] = "stale"
    state.enter(ctx)

    assert state.step(ctx) is StateName.PICK
    assert "task2_stack_plan" not in ctx.extras
    assert ctx.extras["task1_transport_plan"] == "task1 plan must survive"


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


def test_a_loaded_arms_steady_state_trail_is_not_reported_as_contact():
    """The trap descend()'s own flag would fall into.

    `blocked` is `jammed or shortfall > descent_blocked_tol`, and that 4.0mm
    tolerance was tuned for an empty gripper. A carried block trails further
    than that on every descent, so reading `blocked` as contact would report
    a landing every single time and make stack_contacts useless.
    """
    cfg = AppConfig()
    # Over motion.descent_blocked_tol (4.0), so descend() raises its flag,
    # but under task2.contact_shortfall, so we do not call it a landing.
    trail = 5.0
    robot = _TrailingRobot(trail)
    state, ctx, player = _place_harness(cfg, robot)

    assert _run_place(state, ctx) is StateName.SELECT
    record = ctx.extras["stack_contacts"][-1]
    assert record["stream_aborted"] is True  # descend() did flag it...
    assert record["source"] == "none"  # ...and we did not believe it
    assert record["contact"] is False
    assert record["shortfall"] == pytest.approx(trail)


def test_a_nominal_landing_on_the_tower_registers_as_contact():
    """The regression test for the ordering the config comment describes.

    A block meeting the tower stops the arm exactly ``place_overshoot_mm``
    above the commanded goal -- no more. If the contact threshold sat above
    that gap, every perfect stack would be logged as "no contact" and the
    backoff would never run.
    """
    cfg = AppConfig()
    plan = _place_plan()
    tower_top = plan.slot.floor.joints["wrist_flex"] + cfg.task2.place_overshoot_mm
    robot = _TowerRobot(
        first_top=tower_top,
        block_height=cfg.task2.block_height_mm,
        open_pos=cfg.sensing.gripper_open_pos,
    )
    state, ctx, _player = _place_harness(cfg, robot)

    assert _run_place(state, ctx) is StateName.SELECT
    record = ctx.extras["stack_contacts"][-1]
    assert record["contact"] is True
    assert record["source"] == "lag"
    assert record["early"] is False
    assert record["shortfall"] == pytest.approx(cfg.task2.place_overshoot_mm)
    assert ctx.placed_count == 1


def test_a_soft_landing_that_never_aborts_the_stream_still_counts():
    """The case `last_descent_jammed` alone would miss.

    The arm keeps following all the way to the bottom but settles short. No
    jam fires, yet the block is resting on the tower.
    """
    cfg = AppConfig()
    # Between contact_shortfall (6.0) and descent_max_lag (10.0): short enough
    # never to abort the descent, far enough to be a landing.
    trail = 8.0
    robot = _TrailingRobot(trail)
    state, ctx, player = _place_harness(cfg, robot)

    assert _run_place(state, ctx) is StateName.SELECT
    record = ctx.extras["stack_contacts"][-1]
    assert record["jammed"] is False  # the stream ran to completion
    assert record["source"] == "lag"
    assert record["contact"] is True
    assert record["shortfall"] == pytest.approx(trail)


def test_place_reports_a_load_spike_when_the_arm_still_followed():
    """The secondary signal: nothing jammed, but the load says we landed."""
    cfg = AppConfig()
    robot = MockRobotIO()
    state, ctx, _player = _place_harness(cfg, robot)

    baseline_reads = {"n": 0}
    real_read_loads = robot.read_loads

    def read_loads():
        baseline_reads["n"] += 1
        if baseline_reads["n"] > cfg.sensing.contact_baseline_samples:
            return dict(robot.loads, shoulder_lift=500.0)
        return real_read_loads()

    robot.read_loads = read_loads

    assert _run_place(state, ctx) is StateName.SELECT
    record = ctx.extras["stack_contacts"][-1]
    assert record["source"] == "load"
    assert record["contact"] is True
    assert ctx.placed_count == 1


def test_place_without_contact_warns_and_still_releases(caplog):
    cfg = AppConfig()
    robot = MockRobotIO()  # teleports all the way down, zero load
    state, ctx, _player = _place_harness(cfg, robot)

    with caplog.at_level(logging.WARNING, logger="fsm.task2"):
        assert _run_place(state, ctx) is StateName.SELECT
    assert "without contact" in caplog.text
    record = ctx.extras["stack_contacts"][-1]
    assert record["contact"] is False
    assert record["source"] == "none"
    # AGENTS.md §5: warn and release. Holding the block hostage stalls the run.
    assert ctx.placed_count == 1
    assert any(
        action.get("gripper") == cfg.sensing.gripper_open_pos
        for action in robot.sent_actions
    )


def test_early_contact_retries_the_descent_once_then_releases(caplog):
    cfg = AppConfig()
    robot = _JammingRobot(stop_at=-2.0)  # jams in the first few percent
    state, ctx, _player = _place_harness(cfg, robot)

    with caplog.at_level(logging.WARNING, logger="fsm.task2"):
        assert _run_place(state, ctx) is StateName.SELECT
    records = ctx.extras["stack_contacts"]
    assert len(records) == cfg.task2.max_descent_retries + 1
    assert [r["attempt"] for r in records] == [0, 1]
    assert all(r["early"] is True for r in records)
    assert "retrying" in caplog.text
    assert ctx.placed_count == 1  # never holds the block hostage


def test_loads_are_read_off_the_descent_hot_path():
    """A per-tick read_loads once stranded the arm partway down.

    The executable form of that warning: the number of bus round trips must
    not grow with the length of the descent.
    """
    cfg = AppConfig()
    counts = {}
    for offset in (40.0, 400.0):  # a ten-times-longer descent
        robot = MockRobotIO()
        reads = {"n": 0}
        real = robot.read_loads

        def read_loads(_real=real, _reads=reads):
            _reads["n"] += 1
            return _real()

        robot.read_loads = read_loads
        state, ctx, _player = _place_harness(cfg, robot)
        ctx.extras["task2_stack_plan"] = _place_plan(hover_offset=offset)
        state.enter(ctx)
        _run_place(state, ctx)
        counts[offset] = reads["n"]

    assert counts[40.0] == counts[400.0]
    assert counts[40.0] == cfg.sensing.contact_baseline_samples + 1


def test_place_never_inherits_the_empty_gripper_descent_settings():
    cfg = AppConfig()
    state, ctx, player = _place_harness(cfg, MockRobotIO())
    _run_place(state, ctx)

    assert player.descend_kwargs
    for kwargs in player.descend_kwargs:
        assert kwargs["settle_s"] == cfg.task2.descent_settle_s
        assert kwargs["max_lag"] == cfg.task2.descent_max_lag
        assert kwargs["max_lag"] != cfg.motion.descent_max_lag


def test_descend_honours_an_injected_lag_threshold():
    """PLACE raises the jam bar because a carried block trails further.

    Asserting the kwarg is passed is not enough -- this checks the loop
    actually uses it, which is what keeps a loaded descent from reading as a
    jam on its first tick.
    """
    cfg = AppConfig()
    cfg.motion.fps = 0
    cfg.motion.descent_settle_s = 0.0
    goal = {j: -40.0 for j in _ARM}

    trail = 10.0  # between motion.descent_max_lag (8.0) and task2's (12.0)
    strict, _ = TrajectoryPlayer(_TrailingRobot(trail), cfg.motion).descend(goal)
    loose, _ = TrajectoryPlayer(_TrailingRobot(trail), cfg.motion).descend(
        goal, max_lag=cfg.task2.descent_max_lag
    )
    # The empty-gripper bar reads the trail as a jam and aborts immediately;
    # the loaded bar lets the descent run to the floor. Assert on where the
    # arm ended up, because descend()'s flag conflates a jam with a shortfall.
    assert strict["shoulder_lift"] > -20.0  # gave up near the top
    assert loose["shoulder_lift"] == pytest.approx(-40.0 + trail)


def test_a_stacked_level_is_released_from_above_without_feeling_for_the_tower():
    """Levels above contact_descent_levels never run a descent at all.

    Pressing down to find the top of a tower topples the thing being
    measured, so a stacked block goes to the solved release height and the
    jaws open there.
    """
    cfg = AppConfig()
    robot = MockRobotIO()
    state, ctx, player = _place_harness(cfg, robot)
    plan = _place_plan(contact_descent=False, level=2)
    ctx.extras["task2_stack_plan"] = plan
    state.enter(ctx)

    assert _run_place(state, ctx) is StateName.SELECT
    assert player.descend_kwargs == []  # no descent, contact-seeking or otherwise
    record = ctx.extras["stack_contacts"][-1]
    assert record["mode"] == "direct"
    assert record["source"] == "direct"
    assert record["level"] == 2
    assert ctx.placed_count == 1

    # It stopped at the release height, not at the descent floor, and the
    # jaws opened only after arriving there.
    release_z = plan.slot.release.joints["shoulder_lift"]
    arrivals = [
        i
        for i, action in enumerate(robot.sent_actions)
        if action.get("shoulder_lift") == pytest.approx(release_z)
    ]
    gripper_open = next(
        i
        for i, action in enumerate(robot.sent_actions)
        if action.get("gripper") == cfg.sensing.gripper_open_pos
    )
    assert arrivals and min(arrivals) < gripper_open
    assert all(
        action.get("shoulder_lift", 0.0) >= release_z - 1e-6
        for action in robot.sent_actions
        if "shoulder_lift" in action
    )


def test_a_directly_released_level_skips_the_backoff():
    """There is nothing pressing on the tower, so there is nothing to relieve."""
    cfg = AppConfig()
    robot = MockRobotIO()
    state, ctx, _player = _place_harness(cfg, robot)
    ctx.extras["task2_stack_plan"] = _place_plan(contact_descent=False, level=3)
    state.enter(ctx)
    _run_place(state, ctx)

    release_z = -100.0 + cfg.task2.place_overshoot_mm
    arrived = next(
        i
        for i, a in enumerate(robot.sent_actions)
        if a.get("shoulder_lift") == pytest.approx(release_z)
    )
    gripper_open = next(
        i
        for i, a in enumerate(robot.sent_actions)
        if a.get("gripper") == cfg.sensing.gripper_open_pos
    )
    # Between arriving and opening, the arm holds the release height: no
    # backoff tick, because nothing is pressing on anything.
    between = robot.sent_actions[arrived:gripper_open]
    assert all(
        a["shoulder_lift"] == pytest.approx(release_z)
        for a in between
        if "shoulder_lift" in a
    )


def test_production_place_only_opens_gripper_without_arm_motion():
    cfg = AppConfig()
    cfg.motion.fps = 0
    cfg.sensing.gripper_action_wait_s = 0.0
    robot = MockRobotIO()
    motion = MotionController(robot, _EmptyPoses(), cfg.motion, cfg.sensing)

    class NoMotionPlayer:
        def move_to(self, *_args, **_kwargs):
            raise AssertionError("Task-2 PLACE must not move the arm")

        def descend(self, *_args, **_kwargs):
            raise AssertionError("Task-2 PLACE must not descend")

    state = Task2PlaceState(robot, motion, NoMotionPlayer(), cfg)
    ctx = RunContext(cfg.fsm)
    ctx.extras["task2_stack_plan"] = _place_plan(contact_descent=True, level=1)
    state.enter(ctx)

    assert state.step(ctx) is StateName.SELECT
    assert ctx.placed_count == 1
    assert all(set(action) == {"gripper"} for action in robot.sent_actions)
    assert ctx.extras["stack_contacts"][-1]["mode"] == "transport_release"


def test_place_without_a_plan_fails_loudly():
    cfg = AppConfig()
    state = Task2PlaceState(
        MockRobotIO(),
        None,
        TrajectoryPlayer(MockRobotIO(), cfg.motion),
        cfg,
    )
    with pytest.raises(RuntimeError, match="no stack plan"):
        state.enter(RunContext(cfg.fsm))


# --------------------------------------------------------------------------
# flow composition + config
# --------------------------------------------------------------------------


def test_task2_stack_flow_composes_the_same_five_states():
    cfg = AppConfig()
    robot = MockRobotIO()
    pick = type("Pick", (), {"name": StateName.PICK})()
    states = build_task2_stack_states(
        robot=robot,
        motion=_Motion(),
        perceive=_Samples([]),
        pick_state=pick,
        cfg=cfg,
        calib=_calibration(),
        planner=_planner(cfg),
    )
    assert set(states) == {
        StateName.SELECT,
        StateName.PICK,
        StateName.VERIFY,
        StateName.TRANSPORT,
        StateName.PLACE,
    }


def test_task2_stack_flow_rejects_a_non_pick_state():
    cfg = AppConfig()
    bad = type("Bad", (), {"name": StateName.VERIFY})()
    with pytest.raises(ValueError, match="PICK"):
        build_task2_stack_states(
            robot=MockRobotIO(),
            motion=_Motion(),
            perceive=_Samples([]),
            pick_state=bad,
            cfg=cfg,
            calib=_calibration(),
            planner=_planner(cfg),
        )


def test_task2_defaults_validate():
    validate_task2(AppConfig())


def test_every_task2_level_uses_direct_release_without_contact_descent():
    cfg = AppConfig()
    assert cfg.task2.contact_descent_levels == 0
    assert all(not level.contact_descent for level in _planner(cfg).levels)


def test_task2_yaml_block_matches_the_dataclass():
    assert load_config(DEFAULT_YAML).task2 == AppConfig().task2


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("level_tilt_max_deg", 7.0, "ik.max_tilt_error_deg"),
        ("place_overshoot_mm", 1.0, "place_overshoot_mm"),
        ("hover_min_clearance_mm", 50.0, "hover_min_clearance_mm"),
        ("block_height_mm", 0.0, "block_height_mm"),
        ("max_levels", 0, "max_levels"),
        ("stack_uv", [0.5, 1.0], "strictly inside"),
        ("descent_probe_segments", 0, "descent_probe_segments"),
        ("min_descent_fraction", 1.0, "min_descent_fraction"),
        ("descent_max_lag", 0.0, "descent_max_lag"),
        ("contact_descent_levels", 1, "contact descent is disabled"),
    ],
)
def test_task2_validation_rejects_contradictory_geometry(field, value, match):
    cfg = AppConfig()
    setattr(cfg.task2, field, value)
    with pytest.raises(ValueError, match=match):
        validate_task2(cfg)


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
