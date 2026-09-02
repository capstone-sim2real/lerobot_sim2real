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
from control.task1_transport import Task1TransportPlan, Task1TransportPlanner
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
    """Compensate the oblique-camera reach bias without moving ROI geometry."""
    dx = center_mm[0] - base_xy_mm[0]
    dy = center_mm[1] - base_xy_mm[1]
    radius = math.hypot(dx, dy)
    start = cfg.task1.pick_correction_start_radius_mm
    end = cfg.task1.pick_correction_max_radius_mm
    maximum = cfg.task1.pick_correction_max_offset_mm
    if radius <= start or radius == 0.0 or maximum == 0.0:
        return center_mm
    fraction = min(1.0, (radius - start) / (end - start))
    scale = (radius + maximum * fraction) / radius
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
    maximum = cfg.task1.pick_tilt_max_deg
    if radius <= start or maximum == 0.0:
        return 0.0
    fraction = min(1.0, (radius - start) / (end - start))
    return -maximum * fraction


class Task1SelectState(State):
    """HOME, then select an outside-zone block or prove 5 s of absence."""

    name = StateName.SELECT

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

    def _reserve_slot(self, ctx: RunContext, color: str) -> int:
        assignments = ctx.extras.setdefault("task1_slot_by_color", {})
        if color in assignments:
            return int(assignments[color])
        used = {int(value) for value in assignments.values()}
        slot_index = next((index for index in range(len(self._cfg.task1.slot_uv)) if index not in used), None)
        if slot_index is None:
            raise RuntimeError(
                "All Task-1 slots are assigned but another colour was detected; "
                "the arena contract allows one block of each of five colours"
            )
        assignments[color] = slot_index
        return slot_index

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

        detections = sample.detections
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
            self._archive_round_attempts(ctx, colors)
            eligible = detections

        bx, by = self._calib.base_xy_mm or (0.0, 0.0)
        target = min(
            eligible,
            key=lambda d: (math.hypot(d.center_mm[0] - bx, d.center_mm[1] - by), d.center_mm[0], d.center_mm[1]),
        )
        target_id = target.color  # exactly one physical block per colour
        slot_index = self._reserve_slot(ctx, target.color)
        raw_xy = target.center_mm
        pick_xy = corrected_pick_xy(raw_xy, (bx, by), self._cfg)
        pick_target = replace(target, center_mm=pick_xy)
        selection = SelectionResult(pick_target, target_id, len(eligible), detections)
        ctx.extras["selection"] = selection
        ctx.extras["task1_slot_index"] = slot_index
        ctx.extras["task1_raw_target_xy_mm"] = raw_xy
        ctx.extras["task1_corrected_target_xy_mm"] = pick_xy
        ctx.extras["task1_pick_radial_tilt_deg"] = far_reach_tilt_deg(
            pick_xy, (bx, by), self._cfg
        )
        ctx.target_id = target_id
        correction = math.dist(raw_xy, pick_xy)
        ctx.last_note = (
            f"target={target.color} slot={slot_index} outside={len(detections)} "
            f"pick_correction={correction:.1f}mm"
        )
        return StateName.PICK


class Task1TransportState(State):
    name = StateName.TRANSPORT

    def __init__(self, planner: Task1TransportPlanner, player: TrajectoryPlayer, cfg: AppConfig):
        self._planner = planner
        self._player = player
        self._cfg = cfg

    def step(self, ctx: RunContext) -> StateName | None:
        held = ctx.extras.get("ik_pick_attempt")
        slot_index = ctx.extras.get("task1_slot_index")
        if not isinstance(held, GraspAttempt) or not isinstance(slot_index, int):
            raise RuntimeError("Task-1 transport has no held grasp or reserved slot")
        plan = self._planner.plan(held, slot_index)
        ctx.extras["task1_transport_plan"] = plan
        for _name, waypoint in plan.carry:
            self._player.move_to(waypoint.joints, tol=self._cfg.motion.transit_arrival_tol)
        self._player.move_to(plan.slot.hover.joints, tol=self._cfg.motion.transit_arrival_tol)
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
        self._player.move_to(
            plan.slot.drop.joints,
            max_step=self._cfg.motion.descent_step_per_tick,
            tol=self._cfg.motion.arrival_tol,
        )
        if self._cfg.motion.place_settle_s > 0:
            time.sleep(self._cfg.motion.place_settle_s)
        self._motion.open_gripper()
        self._player.move_to(plan.slot.hover.joints, tol=self._cfg.motion.transit_arrival_tol)
        ctx.extras["task1_place_actions"] = int(ctx.extras.get("task1_place_actions", 0)) + 1
        ctx.last_note = f"released_slot={plan.slot.index}"
        return StateName.SELECT
