"""Task-1-only states: fresh-frame completion and dynamic slot placement."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, replace
from typing import Callable

from config import AppConfig
from control.grasp import GraspAttempt
from control.motion import MotionController
from control.task1_transport import (
    Task1TransportPlan,
    Task1TransportPlanner,
    fly_carry,
    release_at,
)
from control.trajectory import TrajectoryPlayer
from fsm.states import RunContext, State, StateName
from perception.detector import BlockDetection
from perception.homography import PlaneCalibration
from perception.select import SelectionResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Task1Perception:
    detections: list[BlockDetection]
    frame_seq: int
    captured_at: float


Task1PerceiveFn = Callable[[], Task1Perception]


def corrected_pick_xy(
    center_mm: tuple[float, float],
    base_xy_mm: tuple[float, float],
    cfg: AppConfig,
) -> tuple[float, float]:
    """Push ultra-near picks outward and leave every other reach raw.

    The oblique-camera far-reach ramp and the P1-9 front-row offsets that
    used to stack here both over-corrected on hardware, so neither survives.
    What is left is one flat boost inside pick_near_boost_max_radius_mm, cut
    hard at that radius: of the measured points only P1 (157 mm) falls in it,
    and that is the band where the arm still visibly under-reaches.
    """
    dx = center_mm[0] - base_xy_mm[0]
    dy = center_mm[1] - base_xy_mm[1]
    radius = math.hypot(dx, dy)
    boost = cfg.task1.pick_near_boost_mm
    if radius == 0.0 or not boost:
        return center_mm
    if radius > cfg.task1.pick_near_boost_max_radius_mm:
        return center_mm
    scale = (radius + boost) / radius
    return base_xy_mm[0] + dx * scale, base_xy_mm[1] + dy * scale


def far_reach_tilt_deg(
    center_mm: tuple[float, float],
    base_xy_mm: tuple[float, float],
    cfg: AppConfig,
) -> float:
    """Distance-ramped outward wrist tilt for Task 1 picks only."""
    radius = math.dist(center_mm, base_xy_mm)
    start = cfg.task1.pick_tilt_start_radius_mm
    end = cfg.task1.pick_tilt_max_radius_mm
    base_tilt = cfg.task1.pick_tilt_base_deg
    maximum = cfg.task1.pick_tilt_max_deg
    if maximum == 0.0:
        return 0.0
    if radius <= start:
        return -base_tilt
    fraction = min(1.0, (radius - start) / (end - start))
    return -(base_tilt + (maximum - base_tilt) * fraction)


class Task1SelectState(State):
    """HOME, then select an outside-zone block or prove 5 s of absence.

    Task 2 stacks with this same state (AGENTS.md §3): the two class
    attributes below name the plan keys it publishes, and
    ``_active_detections`` is where a subclass drops detections it must not
    target. The ``task1_*`` bookkeeping keys are shared with Task 2 on
    purpose -- they belong to the gather pipeline, not to the mission
    number, and ``write_summary`` already emits them.
    """

    name = StateName.SELECT
    index_extra_key = "task1_slot_index"
    plan_extra_key = "task1_transport_plan"

    def __init__(
        self,
        motion: MotionController,
        perceive: Task1PerceiveFn,
        calib: PlaneCalibration,
        cfg: AppConfig,
    ):
        self._motion = motion
        self._perceive = perceive
        self._calib = calib
        self._cfg = cfg
        self._last_frame_seq = -1
        self._empty_since: float | None = None

    def enter(self, ctx: RunContext) -> None:
        # Every successful place and every failed pick comes through here.
        # The camera is consulted only after the arm has cleared its view.
        self._motion.go_home(include_gripper=False)
        self._empty_since = None

    def _pause(self) -> None:
        if self._cfg.task1.scan_interval_s > 0:
            time.sleep(self._cfg.task1.scan_interval_s)

    def _active_detections(self, detections: list[BlockDetection]) -> list[BlockDetection]:
        """Detections this task may act on. Identity for Task 1."""
        return detections

    @staticmethod
    def _archive_attempts(ctx: RunContext, colors: set[str]) -> None:
        totals = ctx.extras.setdefault("task1_attempts_total", {})
        for color in colors:
            totals[color] = int(totals.get(color, 0)) + int(ctx.attempts.pop(color, 0))
        ctx.skipped.difference_update(colors)

    @classmethod
    def _archive_round_attempts(cls, ctx: RunContext, colors: set[str]) -> None:
        cls._archive_attempts(ctx, colors)
        ctx.extras["task1_retry_rounds"] = int(ctx.extras.get("task1_retry_rounds", 0)) + 1

    def _retry_sweep_exhausted(
        self, ctx: RunContext, colors: set[str]
    ) -> StateName | None:
        """Handle a frame in which every visible block is deferred.

        Task 1 deliberately starts another sweep so a physical block is
        never abandoned. Task 2 overrides this hook: repeatedly planning an
        unreachable last block cannot improve the scene and used to spin at
        roughly 1Hz until the global time budget or emergency stop.
        """
        self._archive_round_attempts(ctx, colors)
        return None

    def step(self, ctx: RunContext) -> StateName | None:
        try:
            sample = self._perceive()
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("Task-1 camera sample rejected: %s", exc)
            self._empty_since = None
            self._pause()
            return None

        now_wall = time.time()
        if sample.frame_seq <= self._last_frame_seq:
            if now_wall - sample.captured_at > self._cfg.task1.max_frame_age_s:
                self._empty_since = None
            self._pause()
            return None
        self._last_frame_seq = sample.frame_seq
        if sample.captured_at <= 0 or now_wall - sample.captured_at > self._cfg.task1.max_frame_age_s:
            self._empty_since = None
            self._pause()
            return None

        detections = self._active_detections(sample.detections)
        if not detections:
            now = time.monotonic()
            if self._empty_since is None:
                self._empty_since = now
            empty_for = now - self._empty_since
            ctx.extras["task1_empty_for_s"] = empty_for
            if empty_for >= self._cfg.task1.empty_timeout_s:
                self._archive_attempts(ctx, set(ctx.attempts) | set(ctx.skipped))
                ctx.last_note = f"outside_empty_for={empty_for:.1f}s"
                ctx.extras["task1_complete"] = True
                return StateName.DONE
            self._pause()
            return None

        self._empty_since = None
        colors = {d.color for d in detections}
        eligible = [d for d in detections if d.color not in ctx.skipped]
        if not eligible:
            # max_retries_per_block means "defer for this sweep", never
            # abandon a physical block. Start another sweep indefinitely.
            next_state = self._retry_sweep_exhausted(ctx, colors)
            if next_state is not None:
                return next_state
            eligible = detections

        bx, by = self._calib.base_xy_mm or (0.0, 0.0)
        target = min(
            eligible,
            key=lambda d: (math.hypot(d.center_mm[0] - bx, d.center_mm[1] - by), d.center_mm[0], d.center_mm[1]),
        )
        target_id = target.color  # exactly one physical block per colour
        # Slot assignment belongs after VERIFY. A selected block may fail and
        # be deferred; reserving here would leave slot 0 empty until that block
        # eventually succeeds, producing a visible 5/1/2, 3/4 fill order.
        ctx.extras.pop(self.index_extra_key, None)
        ctx.extras.pop(self.plan_extra_key, None)
        raw_xy = target.center_mm
        pick_xy = corrected_pick_xy(raw_xy, (bx, by), self._cfg)
        pick_target = replace(target, center_mm=pick_xy)
        selection = SelectionResult(pick_target, target_id, len(eligible), detections)
        ctx.extras["selection"] = selection
        ctx.extras["task1_raw_target_xy_mm"] = raw_xy
        ctx.extras["task1_corrected_target_xy_mm"] = pick_xy
        ctx.extras["task1_pick_radial_tilt_deg"] = far_reach_tilt_deg(
            pick_xy, (bx, by), self._cfg
        )
        ctx.target_id = target_id
        correction = math.dist(raw_xy, pick_xy)
        ctx.last_note = (
            f"target={target.color} outside={len(detections)} "
            f"pick_correction={correction:.1f}mm"
        )
        return StateName.PICK


class Task1TransportState(State):
    name = StateName.TRANSPORT
    index_extra_key = "task1_slot_index"
    plan_extra_key = "task1_transport_plan"

    def __init__(self, planner: Task1TransportPlanner, player: TrajectoryPlayer, cfg: AppConfig):
        self._planner = planner
        self._player = player
        self._cfg = cfg

    def _reserve_slot(self, ctx: RunContext) -> int:
        """Assign the next slot only after VERIFY confirmed a held block."""
        color = ctx.target_id
        if color is None:
            raise RuntimeError("Task-1 transport has no verified target")
        assignments = ctx.extras.setdefault("task1_slot_by_color", {})
        if color in assignments:
            return int(assignments[color])
        used = {int(value) for value in assignments.values()}
        slot_index = next(
            (
                index
                for index in range(len(self._cfg.task1.slot_uv))
                if index not in used
            ),
            None,
        )
        if slot_index is None:
            raise RuntimeError(
                "All Task-1 slots are assigned but another colour was grasped; "
                "the arena contract allows one block of each of five colours"
            )
        assignments[color] = slot_index
        return slot_index

    def step(self, ctx: RunContext) -> StateName | None:
        held = ctx.extras.get("ik_pick_attempt")
        if not isinstance(held, GraspAttempt):
            raise RuntimeError("Task-1 transport has no held grasp")
        slot_index = self._reserve_slot(ctx)
        ctx.extras[self.index_extra_key] = slot_index
        plan = self._planner.plan(held, slot_index)
        ctx.extras[self.plan_extra_key] = plan
        fly_carry(self._player, self._cfg, plan)
        ctx.last_note = f"assigned_slot={slot_index}"
        return StateName.PLACE


class Task1PlaceState(State):
    """Release into a reserved slot; completion is never inferred here."""

    name = StateName.PLACE

    def __init__(self, motion: MotionController, player: TrajectoryPlayer, cfg: AppConfig):
        self._motion = motion
        self._player = player
        self._cfg = cfg

    def step(self, ctx: RunContext) -> StateName | None:
        plan = ctx.extras.get("task1_transport_plan")
        if not isinstance(plan, Task1TransportPlan):
            raise RuntimeError("Task-1 PLACE has no transport plan")
        release_at(self._player, self._motion, self._cfg, plan.slot)
        ctx.extras["task1_place_actions"] = int(ctx.extras.get("task1_place_actions", 0)) + 1
        ctx.last_note = f"released_slot={plan.slot.index}"
        return StateName.SELECT
