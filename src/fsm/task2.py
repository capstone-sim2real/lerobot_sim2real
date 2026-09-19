"""Task 2 states: stack every block at one coordinate.

SELECT, PICK, VERIFY and TRANSPORT are Task 1's -- the two states here
subclass them and change exactly what the mission changes: the destination
is one point instead of a colour's slot, and the level replaces the slot
index. PLACE releases directly at the solved level height; Task 2 contact
descent is disabled.
"""

from __future__ import annotations

import enum
import logging
import time

from config import AppConfig
from control.motion import MotionController
from control.robot_io import BaseRobotIO
from control.sensing import ContactMonitor
from control.task2_stack import Task2LevelPlan, Task2StackPlan
from control.trajectory import TrajectoryPlayer, interpolate
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
        # Never memoise per colour the way Task 1 does. If a block is dropped
        # outside the zone after being assigned level 2, another block is
        # stacked, and the first is picked up again, a memo would send it back
        # to level 2 against a taller tower. The tower height is the only
        # thing that decides the level.
        # A failed release can leave the same physical block outside the
        # zone after every pre-solved level has already been attempted. Keep
        # stacking instead of refusing it: reuse the highest defined pose.
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


class _Phase(enum.Enum):
    LAND = "land"
    RELEASE = "release"
    RETREAT = "retreat"


class Task2PlaceState(State):
    """Open the gripper after TRANSPORT; never move an arm joint.

    Validated Task-2 config fixes ``contact_descent_levels`` at zero, so the
    production path calls ``_release_after_transport`` immediately. The old
    contact implementation remains unreachable compatibility code only.

    ``step`` runs one phase per call. Task 2 enforces the 300s budget and the
    machine can only check it between steps, so a place is three preemption
    points rather than one long blocking call (AGENTS.md §3).
    """

    name = StateName.PLACE

    def __init__(
        self,
        robot: BaseRobotIO,
        motion: MotionController,
        player: TrajectoryPlayer,
        cfg: AppConfig,
    ):
        self._robot = robot
        self._motion = motion
        self._player = player
        self._cfg = cfg
        self._plan: Task2StackPlan | None = None
        self._phase = _Phase.LAND
        self._attempt = 0
        self._measured: dict[str, float] = {}
        self._contact = False
        self._source = "none"
        self._fraction = 0.0

    def enter(self, ctx: RunContext) -> None:
        plan = ctx.extras.get("task2_stack_plan")
        if not isinstance(plan, Task2StackPlan):
            raise RuntimeError("Task-2 PLACE has no stack plan")
        self._plan = plan
        self._phase = _Phase.LAND
        self._attempt = 0

    def step(self, ctx: RunContext) -> StateName | None:
        assert self._plan is not None
        level = self._plan.slot
        if self._cfg.task2.contact_descent_levels == 0:
            return self._release_after_transport(ctx, level)
        if self._phase is _Phase.LAND:
            if not level.contact_descent:
                return self._place_from_above(ctx, level)
            return self._descend(ctx, level)
        if self._phase is _Phase.RELEASE:
            return self._release(ctx, level)
        return self._retreat(ctx, level)

    def _release_after_transport(
        self, ctx: RunContext, level: Task2LevelPlan
    ) -> StateName:
        """Open at the pose TRANSPORT already reached, then start the next cycle."""
        self._motion.open_gripper()
        ctx.placed_count += 1
        ctx.extras["task2_place_actions"] = (
            int(ctx.extras.get("task2_place_actions", 0)) + 1
        )
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

    def _place_from_above(self, ctx: RunContext, level: Task2LevelPlan) -> StateName | None:
        """Go to the solved release height and open. No feeling for the tower."""
        self._player.move_to(
            level.release.joints,
            max_step=self._cfg.motion.descent_step_per_tick,
            tol=self._cfg.motion.arrival_tol,
        )
        ctx.extras.setdefault("stack_contacts", []).append(
            {
                "level": level.level,
                "attempt": 0,
                "mode": "direct",
                "contact": False,
                "source": "direct",
                "early": False,
                "hover_z_mm": level.hover_z_mm,
                "place_z_mm": level.place_z_mm,
                "radial_tilt_deg": level.radial_tilt_deg,
            }
        )
        self._measured = self._robot.read_joints()
        # Nothing is pressing on anything, so there is nothing to back off
        # from: RELEASE skips the backoff when contact is False.
        self._contact = False
        self._source = "direct"
        self._fraction = 1.0
        self._phase = _Phase.RELEASE
        return None

    def _descend(self, ctx: RunContext, level: Task2LevelPlan) -> StateName | None:
        t2 = self._cfg.task2
        goal = level.floor.joints
        start = self._robot.read_joints()
        monitor = ContactMonitor(self._robot, self._cfg.sensing)
        # Baseline captured at the hover with the block already held, so it
        # carries the block's weight. TRANSPORT leaves the arm exactly here,
        # and a retry re-baselines because the arm has moved since.
        monitor.start()

        blocked = False
        reading = None
        # descent_probe_segments splits the descent so loads can be read part
        # way down. Keep it at 1: a per-tick read_loads once stranded the arm
        # (control/trajectory.py), and chunking weakens the lag watch too,
        # since descend() re-reads its start pose and lag cannot accumulate
        # across a call boundary.
        segments = max(1, t2.descent_probe_segments)
        waypoints = self._segment_goals(start, goal, segments)
        measured = start
        for waypoint in waypoints:
            measured, blocked = self._player.descend(
                waypoint,
                max_step=self._cfg.motion.descent_step_per_tick,
                tol=self._cfg.motion.arrival_tol,
                settle_s=t2.descent_settle_s,
                max_lag=t2.descent_max_lag,
            )
            reading = monitor.check()
            if blocked or reading.contact:
                break

        # How far short of the goal the arm ended up. The goal is commanded
        # below the nominal surface on purpose, so falling short *is* the
        # landing -- that is the whole point of place_overshoot_mm.
        #
        # Not descend()'s `blocked`: that is `jammed or shortfall >
        # descent_blocked_tol`, and a loaded arm's steady-state trail already
        # exceeds the 4.0 that tolerance was tuned to empty-handed, so it
        # would report a landing on every descent.
        #
        # Not `last_descent_jammed` either, though it is the same event seen
        # from inside the loop: a jam breaks the stream while the command is
        # already descent_max_lag past the arm, so the shortfall it leaves is
        # at least that -- and the validator keeps contact_shortfall below
        # descent_max_lag. A jam therefore always shows up here, while the
        # converse does not: a soft landing can finish the stream and still
        # end short. Testing the shortfall alone catches both. The flag is
        # still recorded, because "did the stream abort" is worth seeing on
        # the bench.
        jammed = self._player.last_descent_jammed
        remaining = max(abs(measured[j] - goal[j]) for j in goal)
        stopped_short = remaining > t2.contact_shortfall
        source = (
            "lag"
            if stopped_short
            else ("load" if reading and reading.contact else "none")
        )
        contact = source != "none"
        fraction = _descent_fraction(start, measured, goal)
        early = contact and fraction < t2.min_descent_fraction

        ctx.extras.setdefault("stack_contacts", []).append(
            {
                "level": level.level,
                "attempt": self._attempt,
                "mode": "contact",
                "contact": contact,
                "source": source,
                "early": early,
                "descent_fraction": round(fraction, 3),
                "shortfall": round(remaining, 2),
                # Which half of the lag signal fired, so a bench run can tell a
                # real stop from a descent that merely finished short.
                "jammed": jammed,
                "stopped_short": stopped_short,
                # descend()'s own conflated flag; diagnostics only.
                "stream_aborted": blocked,
                "hover_z_mm": level.hover_z_mm,
                "place_z_mm": level.place_z_mm,
                "floor_z_mm": level.floor_z_mm,
                "radial_tilt_deg": level.radial_tilt_deg,
                "load_deltas": {
                    joint: round(delta, 1)
                    for joint, delta in (reading.deltas if reading else {}).items()
                },
            }
        )

        if early and self._attempt < t2.max_descent_retries:
            logger.warning(
                "Task-2 level %d contacted at %.0f%% of the descent (%s); "
                "lifting and retrying",
                level.level,
                fraction * 100.0,
                source,
            )
            self._player.move_to(
                level.hover.joints, tol=self._cfg.motion.transit_arrival_tol
            )
            self._attempt += 1
            return None

        if not contact:
            # AGENTS.md §5: warn and release. Holding the block hostage stalls
            # the run, and releasing at the floor is at worst a block-height drop.
            logger.warning(
                "Task-2 level %d descent reached the floor without contact "
                "(fraction %.2f, deltas %s); releasing anyway",
                level.level,
                fraction,
                reading.deltas if reading else {},
            )
        elif early:
            logger.warning(
                "Task-2 level %d contacted early at %.0f%% after %d retr(ies); "
                "releasing anyway",
                level.level,
                fraction * 100.0,
                self._attempt,
            )

        self._measured = measured
        self._contact = contact
        self._source = source
        self._fraction = fraction
        self._phase = _Phase.RELEASE
        return None

    def _release(self, ctx: RunContext, level: Task2LevelPlan) -> StateName | None:
        ticks = self._cfg.motion.contact_backoff_ticks
        if self._contact and ticks > 0:
            # Back off from the pose contact actually stopped at, not from a
            # planned height -- that is what keeps the level count out of the
            # release decision. It also relieves servo pressure so the jaws do
            # not drag the tower sideways as they part.
            step = ticks * self._cfg.motion.descent_step_per_tick
            back = interpolate(self._measured, level.hover.joints, step)
            if back:
                self._player.move_to(
                    back[0], max_step=step, tol=self._cfg.motion.transit_arrival_tol
                )
        if self._cfg.motion.place_settle_s > 0:
            time.sleep(self._cfg.motion.place_settle_s)
        self._motion.open_gripper()
        # Incremented here, not in RETREAT: if the budget cuts between phases
        # the count must reflect that the jaws opened.
        ctx.placed_count += 1
        ctx.extras["task2_place_actions"] = (
            int(ctx.extras.get("task2_place_actions", 0)) + 1
        )
        ctx.extras["task2_tower_height"] = ctx.placed_count
        self._phase = _Phase.RETREAT
        return None

    def _retreat(self, ctx: RunContext, level: Task2LevelPlan) -> StateName | None:
        self._player.move_to(
            level.hover.joints, tol=self._cfg.motion.transit_arrival_tol
        )
        ctx.last_note = (
            f"stacked_level={level.level} contact={self._source} "
            f"fraction={self._fraction:.2f}"
        )
        return StateName.SELECT

    @staticmethod
    def _segment_goals(
        start: dict[str, float], goal: dict[str, float], segments: int
    ) -> list[dict[str, float]]:
        if segments <= 1:
            return [goal]
        return [
            {j: start[j] + (goal[j] - start[j]) * (i / segments) for j in goal}
            for i in range(1, segments + 1)
        ]


def _descent_fraction(
    start: dict[str, float], measured: dict[str, float], goal: dict[str, float]
) -> float:
    """How far down the commanded descent actually got, in joint space.

    Joint space rather than z, so the FSM needs no forward kinematics.
    """
    span = max(abs(start[j] - goal[j]) for j in goal)
    if span == 0.0:
        return 1.0
    remaining = max(abs(measured[j] - goal[j]) for j in goal)
    return max(0.0, min(1.0, 1.0 - remaining / span))
