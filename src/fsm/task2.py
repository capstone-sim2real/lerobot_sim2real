"""Task 2 states: stack every block at one coordinate.

SELECT, PICK, VERIFY and TRANSPORT are Task 1's -- the two states here
subclass them and change exactly what the mission changes: the destination
is one point instead of a colour's slot, and the level replaces the slot
index. PLACE opens the gripper at the level's release hover.
"""

from __future__ import annotations

import logging

from control.motion import MotionController
from control.task2_stack import Task2StackPlan
from fsm.states import RunContext, State, StateName
from fsm.task1 import Task1SelectState, Task1TransportState

logger = logging.getLogger(__name__)


class Task2SelectState(Task1SelectState):
    """Task 1's SELECT with a bounded final-block failure.

    In-zone detections are already removed by the detector. Every detection
    left outside the zone is treated as a real block and retried, including
    one lying close to the stack point after a failed placement. Unlike Task
    1, once every visible block has exhausted its per-block retry allowance,
    Task 2 returns an explicit incomplete result instead of clearing the skip
    set and spinning until the global time budget.
    """

    index_extra_key = "task2_level_index"
    plan_extra_key = "task2_stack_plan"

    def _retry_sweep_exhausted(
        self, ctx: RunContext, colors: set[str]
    ) -> StateName | None:
        failed = sorted(colors)
        self._archive_round_attempts(ctx, colors)
        ctx.extras["task2_failed_blocks"] = failed
        ctx.extras["task2_stop_reason"] = "pick_retries_exhausted"
        ctx.last_note = f"pick_retries_exhausted={','.join(failed)}"
        return StateName.DONE


class Task2TransportState(Task1TransportState):
    """Task 1's transport with the tower level replacing the colour slot."""

    index_extra_key = "task2_level_index"
    plan_extra_key = "task2_stack_plan"

    def _reserve_slot(self, ctx: RunContext) -> int:
        # The level comes from the tower height, never from a per-colour memo
        # (a re-picked block must go on top, not back to its old level).
        # Past the last planned level, keep reusing the highest one.
        return min(int(ctx.placed_count), len(self._planner.levels) - 1)

    def step(self, ctx: RunContext) -> StateName | None:
        levels = self._planner.levels
        level_index = self._reserve_slot(ctx)
        if ctx.placed_count >= len(levels):
            logger.warning(
                "Task 2 has %d prior release(s), beyond max_levels=%d; reusing "
                "level %d and stacking the outside-zone block anyway",
                ctx.placed_count,
                len(levels),
                levels[level_index].level,
            )
            ctx.extras["task2_reused_top_level"] = (
                int(ctx.extras.get("task2_reused_top_level", 0)) + 1
            )

        # A grasped block is always carried to the tower. The ladder's verdict
        # is dead reckoning off placed_count, and the tower it assumes may have
        # collapsed -- so an "unreachable" level is flown anyway and the arm is
        # allowed to press into its envelope. Deliberate: a refused level is a
        # block lost for certain, a rammed one only maybe.
        level = levels[level_index]
        if not level.reachable:
            logger.warning(
                "Task 2 level %d is outside the IK gate (%s); transporting anyway "
                "-- the arm may press against the tower",
                level.level,
                level.reason,
            )
            ctx.extras["task2_forced_levels"] = [
                *ctx.extras.get("task2_forced_levels", []),
                level.level,
            ]
        elif level.hover_squeezed:
            # Worth a line in the log: this is the level where the approach
            # clearance was traded away to place the block at all.
            logger.warning(
                "Task 2 level %d approaches on a squeezed hover: z=%.1f is only "
                "%.1fmm over the tower top",
                level.level,
                level.hover_z_mm,
                level.hover_z_mm - level.place_z_mm,
            )
        # Keep Task 1's transport implementation: it finishes at the planned
        # P5/stack hover. Task-2 PLACE opens right there, with no subsequent
        # move to the lower release pose.
        return super().step(ctx)


class Task2PlaceState(State):
    """Open the gripper where TRANSPORT stopped (the level's release hover); never move an arm joint."""

    name = StateName.PLACE

    def __init__(self, motion: MotionController):
        self._motion = motion
        self._plan: Task2StackPlan | None = None

    def enter(self, ctx: RunContext) -> None:
        plan = ctx.extras.get("task2_stack_plan")
        if not isinstance(plan, Task2StackPlan):
            raise RuntimeError("Task-2 PLACE has no stack plan")
        self._plan = plan

    def step(self, ctx: RunContext) -> StateName | None:
        assert self._plan is not None
        level = self._plan.slot
        self._motion.open_gripper()
        ctx.placed_count += 1
        ctx.extras["task2_place_actions"] = int(ctx.extras.get("task2_place_actions", 0)) + 1
        ctx.extras["task2_tower_height"] = ctx.placed_count
        ctx.extras.setdefault("stack_contacts", []).append(
            {
                "level": level.level,
                "attempt": 0,
                "mode": "transport_release",
                "contact": False,
                "source": "transport",
                "early": False,
                "hover_z_mm": level.hover_z_mm,
                "place_z_mm": level.place_z_mm,
                "radial_tilt_deg": level.radial_tilt_deg,
            }
        )
        ctx.last_note = f"released_after_transport_level={level.level}"
        return StateName.SELECT
