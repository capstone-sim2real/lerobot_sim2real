"""Single-point tower planning for Task 2.

The mirror of ``task1_transport``: where Task 1 solves N slots at one
height, Task 2 solves N levels at one xy. Every block goes to the same
coordinate and the ladder climbs by one block height per level.

The ladder is a *plan*, not an authority. Task 2 currently uses its solved
release pose directly at every level; contact-seeking descent is disabled.
The ladder also reports how many levels the arm can reach at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from config import AppConfig
from control.grasp import GraspAttempt, highest_reachable_hover
from control.ik import IkResult, TopDownIK
from control.task1_transport import over_ik_gate, push_out_from_base, transit_apex
from perception.homography import PlaneCalibration
from perception.zone import zone_slot_centres


@dataclass(frozen=True)
class Task2LevelPlan:
    level: int  # 1-based physical level
    xy_mm: tuple[float, float]  # identical for every level, by definition
    place_z_mm: float  # nominal gripper-frame release height
    floor_z_mm: float  # commanded descent goal, below place_z on purpose
    hover_z_mm: float
    radial_tilt_deg: float
    hover: IkResult
    release: IkResult  # the nominal release pose, at place_z_mm
    floor: IkResult  # commanded goal for a contact descent only
    contact_descent: bool  # land by feel, or go straight to `release`
    # Advisory only. The dry-run reads these to predict the tower height, but
    # nothing refuses a level at runtime any more: a level called unreachable
    # is still flown, because the tower can collapse and the ladder's height
    # is dead reckoning, not a measurement. See fsm/task2.Task2TransportState.
    reachable: bool
    reason: str = ""  # why not, printed verbatim by the dry-run
    hover_squeezed: bool = False  # approached below the preferred clearance


@dataclass(frozen=True)
class Task2StackPlan:
    # Named 'slot' so Task1TransportState.step drives this unchanged; the
    # shared transport does not care whether a destination is a slot or a
    # level, and duplicating it to rename one attribute would be worse.
    slot: Task2LevelPlan
    carry: tuple[tuple[str, IkResult], ...]

    @property
    def level(self) -> Task2LevelPlan:
        return self.slot


class Task2StackPlanner:
    """Solve the whole tower once, up front, and report what is reachable."""

    def __init__(self, calib: PlaneCalibration, cfg: AppConfig, ik: TopDownIK):
        self._calib = calib
        self._cfg = cfg
        self._ik = ik
        try:
            grasp_z = float(calib.meta["grasp_z_mm_mean"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Calibration metadata is missing grasp_z_mm_mean") from exc
        self._grasp_z = grasp_z
        base = calib.base_xy_mm or (0.0, 0.0)
        self._raw_xy = zone_slot_centres(calib, [list(cfg.task2.stack_uv)])[0]
        # Command the point further out than we want the block, because the
        # arm under-reaches by about that much. The landing point is what has
        # to be inside zone_polygon_mm -- that is what hides placed blocks
        # from the detector, which is what lets Task 1's empty-timeout
        # completion criterion work here unchanged.
        self._xy = push_out_from_base(
            self._raw_xy, base, cfg.task2.stack_radial_offset_mm
        )
        self._levels = self._solve_levels()

    @property
    def levels(self) -> tuple[Task2LevelPlan, ...]:
        return self._levels

    @property
    def stack_xy_mm(self) -> tuple[float, float]:
        return self._xy

    @property
    def raw_xy_mm(self) -> tuple[float, float]:
        return self._raw_xy

    @property
    def grasp_z_mm(self) -> float:
        return self._grasp_z

    @property
    def reachable_levels(self) -> int:
        """Levels reachable counting from the bottom, stopping at the first gap."""
        count = 0
        for level in self._levels:
            if not level.reachable:
                break
            count += 1
        return count

    def place_z_mm(self, level: int) -> float:
        """Gripper-frame height that puts a held block's top face at ``level``.

        The calibration plane is a block's *top face* (AGENTS.md §6), so
        ``grasp_z`` is already one block height above the table.
        """
        t2 = self._cfg.task2
        return self._grasp_z + t2.block_height_mm * (level - 1) + t2.release_clearance_mm

    def level_tilt_deg(self, level: int) -> float:
        """Outward approach tilt for a level; negative tips away from the base."""
        t2 = self._cfg.task2
        steps = max(0, level - t2.level_tilt_start_level)
        return -min(t2.level_tilt_max_deg, t2.level_tilt_per_level_deg * steps)

    def _solve_levels(self) -> tuple[Task2LevelPlan, ...]:
        t2 = self._cfg.task2
        plans: list[Task2LevelPlan] = []
        for level in range(1, t2.max_levels + 1):
            place_z = self.place_z_mm(level)
            tilt = self.level_tilt_deg(level)
            hover_z = highest_reachable_hover(
                self._ik,
                *self._xy,
                place_z,
                self._cfg,
                radial_tilt_deg=tilt,
                clearance_mm=t2.hover_clearance_mm,
                min_clearance_mm=t2.hover_min_clearance_mm,
            )
            floor_z = place_z - t2.place_overshoot_mm
            hover = self._ik.solve(*self._xy, hover_z, radial_tilt_deg=tilt)

            # The preferred clearance band can sit entirely above the arm's
            # ceiling at this level while a lower approach still solves --
            # top-down lift collapses fast with height. Squeeze before giving
            # up: a level refused here costs a whole block, and the tower may
            # not even be as tall as placed_count claims (it can collapse),
            # so the approach must be attempted rather than predicted away.
            squeezed = False
            if hover.position_error_mm > t2.hover_gate_mm:
                squeeze_z = highest_reachable_hover(
                    self._ik,
                    *self._xy,
                    place_z,
                    self._cfg,
                    radial_tilt_deg=tilt,
                    clearance_mm=t2.hover_clearance_mm,
                    min_clearance_mm=t2.hover_squeeze_clearance_mm,
                )
                squeeze = self._ik.solve(*self._xy, squeeze_z, radial_tilt_deg=tilt)
                if squeeze.position_error_mm <= t2.hover_gate_mm:
                    hover_z, hover, squeezed = squeeze_z, squeeze, True

            release = self._ik.solve(*self._xy, place_z, radial_tilt_deg=tilt)
            floor = self._ik.solve(*self._xy, floor_z, radial_tilt_deg=tilt)
            # Task 2 never probes downward for contact. Even level one is
            # released directly at its solved height.
            contact_descent = False

            # Re-gate. highest_reachable_hover returns its search floor when
            # nothing solves, so an unreachable hover comes back looking like
            # a height. Gate it at the same strict threshold the search used,
            # not at ik.max_position_error_mm: a hover 12mm short would eat
            # most of the clearance the held block needs over the tower.
            reason = ""
            if hover.position_error_mm > t2.hover_gate_mm:
                reason = (
                    f"hover z={hover_z:.1f} misses by {hover.position_error_mm:.1f}mm "
                    f"(clearance {hover_z - place_z:.1f}mm over the tower top)"
                )
            elif hover.tilt_error_deg > self._cfg.ik.max_tilt_error_deg:
                # The position gate above says nothing about attitude, and a
                # hover that could not hold the commanded tilt is not the
                # pose the descent below it was planned for.
                reason = (
                    f"hover tilt {hover.tilt_error_deg:.1f}deg exceeds the IK "
                    f"gate ({self._cfg.ik.max_tilt_error_deg:.1f}deg)"
                )
            elif over_ik_gate(release, self._cfg):
                # The pose the block is actually let go at. For a directly
                # released level this is the only thing that has to be right.
                reason = (
                    f"release z={place_z:.1f} misses by "
                    f"{release.position_error_mm:.1f}mm"
                )
            elif contact_descent and over_ik_gate(floor, self._cfg):
                reason = (
                    f"descent floor z={floor_z:.1f} misses by "
                    f"{floor.position_error_mm:.1f}mm"
                )
            elif contact_descent and hover_z - floor_z < t2.min_descent_travel_mm:
                reason = f"only {hover_z - floor_z:.1f}mm of descent travel"

            plans.append(
                Task2LevelPlan(
                    level=level,
                    xy_mm=self._xy,
                    place_z_mm=place_z,
                    floor_z_mm=floor_z,
                    hover_z_mm=hover_z,
                    radial_tilt_deg=tilt,
                    hover=hover,
                    release=release,
                    floor=floor,
                    contact_descent=contact_descent,
                    reachable=not reason,
                    reason=reason,
                    hover_squeezed=squeezed and not reason,
                )
            )

        # An unreachable *upper* level is the answer to the question Task 2
        # asks, not an error -- it is reported, never silently clipped. Level
        # one is different: nothing can be stacked here at all.
        if not plans[0].reachable:
            raise ValueError(
                f"Task-2 level 1 is outside the IK gate at x={self._xy[0]:.1f} "
                f"y={self._xy[1]:.1f} ({plans[0].reason}); nothing can be stacked here"
            )
        return tuple(plans)

    def plan(self, held: GraspAttempt, level_index: int) -> Task2StackPlan:
        if not 0 <= level_index < len(self._levels):
            raise IndexError(f"Task-2 level index {level_index} is not defined")
        level = self._levels[level_index]
        t2 = self._cfg.task2
        carry = [
            item
            for item in (
                # The pick apex is a table-height target, exactly as in Task 1.
                transit_apex(
                    self._ik, self._cfg, "apex_pick", held.xy_mm, held.grasp_z_mm
                ),
                # The place apex needs Task-2 bounds: with motion.hover_* it
                # would search 40..120mm above a tower top, fail, and drop the
                # apex on precisely the levels that most need the arm folded in.
                transit_apex(
                    self._ik,
                    self._cfg,
                    "apex_place",
                    level.xy_mm,
                    level.place_z_mm,
                    clearance_mm=t2.hover_clearance_mm,
                    min_clearance_mm=t2.hover_min_clearance_mm,
                ),
            )
            if item is not None
        ]
        return Task2StackPlan(slot=level, carry=tuple(carry))

    def describe(self) -> str:
        """Human-readable ladder for the dry-run preflight."""
        base = self._calib.base_xy_mm or (0.0, 0.0)
        t2 = self._cfg.task2
        lines = [
            f"stack point: u={t2.stack_uv[0]:.2f} v={t2.stack_uv[1]:.2f}",
            f"  raw       x={self._raw_xy[0]:8.1f} y={self._raw_xy[1]:8.1f}"
            f"  reach {math.dist(self._raw_xy, base):.1f}mm   <- where the block lands",
            f"  commanded x={self._xy[0]:8.1f} y={self._xy[1]:8.1f}"
            f"  reach {math.dist(self._xy, base):.1f}mm"
            f"  (+{t2.stack_radial_offset_mm:.1f}mm radial under-reach)",
            f"grasp plane z={self._grasp_z:.1f}mm  block h={t2.block_height_mm:.1f}mm"
            f"  hover clearance {t2.hover_min_clearance_mm:.1f}..{t2.hover_clearance_mm:.1f}mm",
            "",
            "lvl  place_z  hover_z   clear   tilt  hover_err  rel_err  land      reachable",
        ]
        for lv in self._levels:
            lines.append(
                f"{lv.level:3d}  {lv.place_z_mm:7.1f}"
                f"  {lv.hover_z_mm:7.1f}  {lv.hover_z_mm - lv.place_z_mm:6.1f}"
                f"  {lv.radial_tilt_deg:5.1f}  {lv.hover.position_error_mm:9.2f}"
                f"  {lv.release.position_error_mm:7.2f}"
                f"  {'contact' if lv.contact_descent else 'direct ':8s}  "
                + (
                    ("yes (squeezed)" if lv.hover_squeezed else "yes")
                    if lv.reachable
                    else f"NO   {lv.reason}"
                )
            )
        reachable = self.reachable_levels
        lines += [
            "",
            f"reachable levels: {reachable} of {t2.max_levels}"
            f"  ->  expect a {reachable}-block tower",
            f"  (a run still attempts all {t2.max_levels}: an unreachable level is"
            " flown anyway and the arm may press into the tower)",
        ]
        squeezed = [lv.level for lv in self._levels if lv.hover_squeezed]
        if squeezed:
            lines.append(
                f"  levels {squeezed} approach below the preferred "
                f"{t2.hover_min_clearance_mm:.0f}mm clearance (squeezed toward "
                f"{t2.hover_squeeze_clearance_mm:.0f}mm) -- reachable, but the held "
                f"block passes closer over the tower top."
            )
        if reachable < t2.max_levels:
            lines.append(
                "  to raise the ladder honestly: lower task2.hover_squeeze_clearance_mm, move "
                "task2.stack_uv[1] toward the near edge, or start the tilt ramp "
                "earlier. Do NOT lower stack_radial_offset_mm -- landing inside "
                "the zone is what hides placed blocks from the detector."
            )
        return "\n".join(lines)
