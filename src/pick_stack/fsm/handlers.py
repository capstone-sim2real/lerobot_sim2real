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
from typing import Callable, Protocol

from pick_stack.config import SensingConfig
from pick_stack.control.motion import MotionController
from pick_stack.control.poses import Pose
from pick_stack.control.robot_io import BaseRobotIO
from pick_stack.control.sensing import check_grasp
from pick_stack.fsm.states import RunContext, State, StateName
from pick_stack.perception.select import SelectionResult
from pick_stack.policy.act_client import PickResult

logger = logging.getLogger(__name__)

# runner wires the real pipeline: capture top frame -> detect -> select
PerceiveFn = Callable[[set[str]], SelectionResult]


class PickClient(Protocol):
    def ping(self) -> bool: ...

    def run_pick(self, retreat_pose: Pose) -> PickResult: ...


class SelectState(State):
    name = StateName.SELECT

    def __init__(self, motion: MotionController, perceive: PerceiveFn):
        self._motion = motion
        self._perceive = perceive

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
        return StateName.PICK


class PickState(State):
    name = StateName.PICK

    def __init__(self, client: PickClient, motion: MotionController, retreat_pose: Pose):
        self._client = client
        self._motion = motion
        self._retreat_pose = retreat_pose

    def step(self, ctx: RunContext) -> StateName | None:
        if not self._client.ping():
            # without the policy server there is nothing useful left to do
            ctx.last_note = "policy_server_unreachable"
            return StateName.DONE

        ctx.record_attempt(ctx.target_id)
        result = self._client.run_pick(self._retreat_pose)
        ctx.extras["pick_result"] = result
        if result.reached_retreat:
            return StateName.VERIFY

        # policy never settled at retreat: reset and let SELECT re-detect
        # (the block may have been nudged)
        self._motion.open_gripper()
        if ctx.should_skip(ctx.target_id):
            ctx.skip(ctx.target_id)
            ctx.last_note = f"pick_{result.outcome}_skip"
        else:
            ctx.last_note = f"pick_{result.outcome}_retry"
        return StateName.SELECT


class VerifyState(State):
    name = StateName.VERIFY

    def __init__(self, robot: BaseRobotIO, sensing_cfg: SensingConfig, motion: MotionController):
        self._robot = robot
        self._cfg = sensing_cfg
        self._motion = motion

    def step(self, ctx: RunContext) -> StateName | None:
        check = check_grasp(self._robot, self._cfg)
        ctx.extras["grasp_check"] = check
        if check.grasped:
            ctx.last_note = f"grasp_ok pos={check.gripper_pos:.1f} load={check.gripper_load_abs:.0f}"
            return StateName.TRANSPORT

        self._motion.open_gripper()
        if ctx.should_skip(ctx.target_id):
            ctx.skip(ctx.target_id)
            ctx.last_note = "grasp_failed_skip"
        else:
            ctx.last_note = "grasp_failed_retry"
        return StateName.SELECT


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


def build_states(
    *,
    robot: BaseRobotIO,
    motion: MotionController,
    perceive: PerceiveFn,
    client: PickClient,
    retreat_pose: Pose,
    sensing_cfg: SensingConfig,
    place_strategy: PlaceStrategy,
) -> dict[StateName, State]:
    return {
        StateName.SELECT: SelectState(motion, perceive),
        StateName.PICK: PickState(client, motion, retreat_pose),
        StateName.VERIFY: VerifyState(robot, sensing_cfg, motion),
        StateName.TRANSPORT: TransportState(motion),
        StateName.PLACE: PlaceState(place_strategy),
    }
