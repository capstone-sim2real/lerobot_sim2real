"""Concrete FSM state handlers.

Wiring rules:
  - handlers receive only interfaces (perceive callable, policy client,
    MotionController, BaseRobotIO) + config — no construction inside;
  - PLACE is a strategy seam: Task 1 injects SlotPlaceStrategy, Task 2
    injects StackPlaceStrategy, and a future policy-driven stack-align can
    slot in behind the same interface (AGENTS.md §4);
  - HARD RULE: VERIFY never advances to TRANSPORT without a confirmed grasp.
"""

from __future__ import annotations

import abc
import logging
from typing import Callable

from config import SensingConfig
from control.motion import MotionController
from control.robot_io import BaseRobotIO
from control.sensing import check_grasp
from control.trajectory import TrajectoryPlayer
from control.poses import Pose
from fsm.states import RunContext, State, StateName
from perception.select import SelectionResult

logger = logging.getLogger(__name__)

# runner wires the real pipeline: capture top frame -> detect -> select
PerceiveFn = Callable[[set[str]], SelectionResult]


class SelectState(State):
    name = StateName.SELECT

    def __init__(self, motion: MotionController, perceive: PerceiveFn, *, next_state: StateName = StateName.PICK):
        self._motion = motion
        self._perceive = perceive
        self._next_state = next_state

    def enter(self, ctx: RunContext) -> None:
        # home first: consistent policy start pose AND the arm clears the
        # top camera's view of the board
        self._motion.go_home()

    def step(self, ctx: RunContext) -> StateName | None:
        if ctx.all_blocks_done():
            ctx.last_note = "all_blocks_placed"
            return StateName.DONE
        result = self._perceive(ctx.skipped)
        ctx.extras["selection"] = result
        if result.target is None:
            ctx.last_note = "no_targets_left"
            return StateName.DONE
        ctx.target_id = result.target_id
        ctx.last_note = f"target={result.target_id} remaining={result.remaining}"
        return self._next_state


class VerifyState(State):
    name = StateName.VERIFY

    def __init__(
        self,
        robot: BaseRobotIO,
        sensing_cfg: SensingConfig,
        motion: MotionController,
        *,
        on_grasped: StateName = StateName.TRANSPORT,
        on_failed: StateName = StateName.SELECT,
    ):
        self._robot = robot
        self._cfg = sensing_cfg
        self._motion = motion
        self._on_grasped = on_grasped
        self._on_failed = on_failed

    def step(self, ctx: RunContext) -> StateName | None:
        check = check_grasp(self._robot, self._cfg)
        ctx.extras["grasp_check"] = check
        if check.grasped:
            ctx.last_note = f"grasp_ok pos={check.gripper_pos:.1f} load={check.gripper_load_abs:.0f}"
            return self._on_grasped

        self._motion.open_gripper()
        if ctx.should_skip(ctx.target_id):
            ctx.skip(ctx.target_id)
            ctx.last_note = "grasp_failed_skip"
        else:
            ctx.last_note = "grasp_failed_retry"
        return self._on_failed


class ContextMotionState(State):
    """Move to a joint goal derived from the current FSM context."""

    def __init__(
        self,
        name: StateName,
        player: TrajectoryPlayer,
        goal_for: Callable[[RunContext], Pose],
        *,
        next_state: StateName,
        max_step: float | None = None,
        tol: float | None = None,
    ):
        self.name = name
        self._player = player
        self._goal_for = goal_for
        self._next_state = next_state
        self._max_step = max_step
        self._tol = tol

    def step(self, ctx: RunContext) -> StateName | None:
        self._player.move_to(self._goal_for(ctx), max_step=self._max_step, tol=self._tol)
        return self._next_state


class ReleaseState(State):
    """Open the gripper as a reusable terminal or placement action."""

    name = StateName.RELEASE

    def __init__(self, motion: MotionController, *, next_state: StateName):
        self._motion = motion
        self._next_state = next_state

    def step(self, ctx: RunContext) -> StateName | None:
        self._motion.open_gripper()
        return self._next_state


class TransportState(State):
    name = StateName.TRANSPORT

    def __init__(self, motion: MotionController):
        self._motion = motion

    def step(self, ctx: RunContext) -> StateName | None:
        self._motion.transport_to_zone()
        return StateName.PLACE


class PlaceStrategy(abc.ABC):
    """Seam for the PLACE stage (rule-based now, policy-driven later)."""

    @abc.abstractmethod
    def place(self, ctx: RunContext) -> None: ...


class SlotPlaceStrategy(PlaceStrategy):
    """Task 1: drop into pre-recorded slot #placed_count."""

    def __init__(self, motion: MotionController):
        self._motion = motion

    def place(self, ctx: RunContext) -> None:
        self._motion.place_in_slot(ctx.placed_count)


class StackPlaceStrategy(PlaceStrategy):
    """Task 2: contact-based descent onto the tower."""

    def __init__(self, motion: MotionController):
        self._motion = motion

    def place(self, ctx: RunContext) -> None:
        contact = self._motion.stack_place()
        ctx.extras.setdefault("stack_contacts", []).append(contact)


class PlaceState(State):
    name = StateName.PLACE

    def __init__(self, strategy: PlaceStrategy):
        self._strategy = strategy

    def step(self, ctx: RunContext) -> StateName | None:
        self._strategy.place(ctx)
        ctx.placed_count += 1
        ctx.last_note = f"placed_count={ctx.placed_count}"
        return StateName.SELECT
