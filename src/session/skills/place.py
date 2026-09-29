"""Releasing a held block: slots, table regions, board cells, and placement checks."""

from __future__ import annotations

import logging
import math
import time
from itertools import product
from typing import Any

from control.task1_transport import release_at
from perception.zone import point_in_zone
from session.grid import bounds as grid_bounds
from session.place_correction import PlaceCorrection
from session.relative import (
    clamp_vector,
    decompose_xy,
    find_free_point,
    offset_xy,
    table_region_xy,
    vector_norm,
)
from session.results import SkillResult
from session.skills.base import XY, _xy

logger = logging.getLogger(__name__)


class PlaceMixin:
    # ── placement correction ─────────────────────────────────────────

    @property
    def place_correction(self) -> PlaceCorrection:
        """Learned command offset for releases (``session/place_correction``)."""
        if self._place_correction is None:
            cfg = self.cfg.agent.place_correction
            self._place_correction = PlaceCorrection(
                forward_mm=cfg.forward_mm, left_mm=cfg.left_mm,
                max_mm=cfg.max_mm, max_sample_mm=cfg.max_sample_mm,
            )
        return self._place_correction

    def _place_plan(self, target_xy: XY, *, prebuilt=None) -> tuple[Any, dict[str, Any]]:
        """Plan for releasing on ``target_xy``, with the learned offset applied.

        The caller has already accepted ``target_xy`` (workspace, zone and
        neighbour checks); this only decides where the arm is actually sent
        so the block lands there. If the offset point misses the IK gate the
        plain target is used and the result says so, because a learned
        offset must never turn a legal request into a refusal.
        """
        cfg = self.cfg.agent.place_correction
        correction = self.place_correction
        if not cfg.enabled or correction.magnitude_mm == 0.0:
            return (prebuilt or self.s.solve_place(target_xy)), {}
        commanded = correction.command_xy(
            target_xy, base_xy_mm=self.s.base_xy, frame=self.cfg.agent.relative.frame
        )
        try:
            return self.s.solve_place(commanded), {"place_correction": correction.as_dict()}
        except ValueError:
            plan = prebuilt or self.s.solve_place(target_xy)
            return plan, {"place_correction_skipped": "ik_gate"}

    def _learn_placement(self, target_xy: XY, verification: dict[str, Any]) -> dict[str, Any]:
        """Fold the post-release camera check into the correction.

        This costs nothing extra: every placement already returns home and
        re-observes to verify itself, so the pair (told, landed) is already
        on the table. Reporting ``miss_mm`` matters as much as learning from
        it -- it is the only number that says whether a placement was off.
        """
        measured = verification.get("measured") or {}
        if not verification.get("verified") or "x_mm" not in measured:
            return {}
        measured_xy = (float(measured["x_mm"]), float(measured["y_mm"]))
        data: dict[str, Any] = {"miss_mm": round(math.dist(measured_xy, target_xy), 1)}
        cfg = self.cfg.agent.place_correction
        if cfg.enabled and cfg.learn and self.place_correction.observe(
            target_xy, measured_xy, base_xy_mm=self.s.base_xy, frame=self.cfg.agent.relative.frame
        ):
            data["place_correction"] = self.place_correction.as_dict()
            logger.info(
                "place correction now forward=%.1fmm left=%.1fmm after %d placements "
                "(this one missed by %.1fmm)",
                self.place_correction.forward_mm, self.place_correction.left_mm,
                self.place_correction.samples, data["miss_mm"],
            )
        return data

    def _measured_cell(self, verification: dict[str, Any]) -> tuple[int, int] | None:
        """Which board cell the camera actually found the block in."""
        measured = verification.get("measured") or {}
        if not verification.get("verified") or "x_mm" not in measured:
            return None
        return self.grid.xy_to_cell((float(measured["x_mm"]), float(measured["y_mm"])))

    @staticmethod
    def _miss_note(correction: dict[str, Any]) -> str:
        """One clause about how far off the release landed, when notable."""
        miss = correction.get("miss_mm")
        if miss is None or miss < 15.0:
            return ""
        return f" 목표에서 {miss:.0f}mm 벗어났습니다."

    # ── placement checks ─────────────────────────────────────────────

    def placement_verdict(
        self,
        xy: XY,
        *,
        allow_zone: bool,
        ignore_color: str | None,
        check_clearance: bool = True,
        check_ik: bool = True,
    ):
        """``(reason_or_None, plan_or_None)`` for releasing a block at ``xy``."""
        s = self.s
        if not s.in_workspace(xy):
            return "out_of_workspace", None
        if not allow_zone and s.in_zone(xy, margin_mm=self.cfg.agent.table_zone_margin_mm):
            return "destination_in_zone", None
        if check_clearance and s.last_scene is not None:
            for block in s.last_scene.all():
                if block.color == ignore_color:
                    continue
                if math.dist(block.center_mm, xy) < self.cfg.agent.place_clear_radius_mm:
                    return "destination_blocked", None
        if not check_ik:
            return None, None
        try:
            return None, s.solve_place(xy)
        except ValueError:
            return "destination_unreachable", None

    _VERDICT_DETAIL = {
        "out_of_workspace": "그 위치는 팔의 작업 부채꼴 밖입니다.",
        "destination_in_zone": "그 위치는 적재 구역과 겹칩니다.",
        "destination_blocked": "그 위치에 다른 블록이 너무 가깝습니다.",
        "destination_unreachable": "그 위치는 IK로 닿을 수 없습니다.",
        "no_free_region": "근처에 비어 있고 닿을 수 있는 자리가 없습니다.",
    }

    def _verdict_failure(self, action: str, reason: str, t0: float, **data) -> SkillResult:
        advice = "retry_ok" if reason in ("destination_blocked", "destination_in_zone") else "do_not_retry"
        return self._result(False, action, reason, self._VERDICT_DETAIL.get(reason, reason),
                            retry_advice=advice, t0=t0, **data)

    def _finish_placement(self, color: str) -> dict[str, Any]:
        """Finish release with home return; visual confirmation is optional observation."""
        self.s.go_home()
        return {"verified": False, "placement_verified": False, "release_completed": True}

    # ── pick ─────────────────────────────────────────────────────────

    def _release_over(self, plan) -> str:
        """Release the held block at ``plan`` from directly above it."""
        s = self.s
        assert s.held is not None
        color = s.held.color
        s.player.move_to(plan.hover.joints, tol=self.cfg.motion.transit_arrival_tol)
        release_at(s.player, s.motion, self.cfg, plan)
        s.last_block_color = color
        s.held = None
        return color

    def _put_back(self) -> None:
        """Set the held block down where it was grasped (for a re-grasp)."""
        s = self.s
        assert s.held is not None
        plan = s.solve_place(s.held.attempt.xy_mm, label="put-back point")
        if s.held.over_xy_mm != s.held.attempt.xy_mm:
            s.carry_and_release(plan)
        else:
            self._release_over(plan)

    def place_at_slot(self, slot_index: int, forward_mm: float = 0.0, left_mm: float = 0.0) -> SkillResult:
        action, t0, s, cfg = "place_at_slot", time.monotonic(), self.s, self.cfg
        if not 0 <= slot_index < len(cfg.task1.slot_uv):
            return self._result(False, action, "invalid_arguments", "없는 칸입니다.", t0=t0)
        if s.held is None:
            return self._no_block(action, t0)
        color = s.held.color
        label, korean = self.slot_label(slot_index), self.slot_korean(slot_index)
        scene = s.last_scene
        occupant = scene.slot_occupancy.get(slot_index) if scene else None
        if occupant is not None and occupant != color:
            free = [self.slot_label(i) for i, c in scene.slot_occupancy.items() if c is None]
            return self._result(
                False, action, "slot_occupied", f"{korean} 칸에는 이미 {occupant} 블록이 있습니다.",
                retry_advice="retry_ok", t0=t0, free_slots=free,
            )

        offset = None
        clamped = False
        if forward_mm or left_mm:
            (f, l), clamped = clamp_vector((forward_mm, left_mm), cfg.agent.relative.max_shift_mm)
            xy = offset_xy(s.slot_centres[slot_index], f, l,
                           frame=cfg.agent.relative.frame, base_xy_mm=s.base_xy)
            reason, plan = self.placement_verdict(xy, allow_zone=True, ignore_color=color)
            if reason is not None:
                return self._verdict_failure(action, reason, t0, target=_xy(xy))
            offset = {"forward": f, "left": l}
            target = xy
        else:
            target = s.slot_centres[slot_index]
        plan, correction = self._place_plan(
            target, prebuilt=s.transport.slots[slot_index] if offset is None else None
        )

        s.carry_and_release(plan)
        verification = self._finish_placement(color)
        correction.update(self._learn_placement(target, verification))
        measured = verification.get("measured") or {}
        in_zone = verification.get("in_zone")
        if offset is None:
            detail = f"{self._label(color)} 블록을 {korean} 칸에 놓았습니다."
            if verification.get("verified") and measured.get("slot") != label:
                detail += f" 다만 카메라로 보니 {measured.get('slot_korean') or '칸 밖'}에 있습니다."
        else:
            detail = f"{self._label(color)} 블록을 {korean} 칸 기준으로 옮겨 놓았습니다."
            if in_zone is False:
                detail += " 결과 위치가 적재 구역 밖입니다."
        detail += self._miss_note(correction)
        return self._result(True, action, "released", detail, t0=t0, color=color, slot=label,
                            target=_xy(target), offset_mm=offset,
                            offset_clamped=clamped or None, **correction, **verification)

    def _choose_table_point(self, column: str | None, row: str | None, *, ignore_color: str | None,
                            reference_xy: XY):
        """``(point, column, row, adjusted_by_mm, reason)`` for a table placement."""
        cfg, regions = self.cfg, self.cfg.agent.table_regions
        if column is not None and column not in regions.columns_deg:
            return None, None, None, 0.0, "invalid_arguments"
        if row is not None and row not in regions.rows_fraction:
            return None, None, None, 0.0, "invalid_arguments"

        def valid(point: XY) -> str | None:
            return self.placement_verdict(point, allow_zone=False, ignore_color=ignore_color)[0]

        if column is not None and row is not None:
            nominal = table_region_xy(column, row, cfg.perception, regions, self.s.base_xy)
            point, moved, reason = find_free_point(
                nominal, is_valid=valid, step_mm=regions.search_step_mm, max_mm=regions.search_max_mm
            )
            if reason is not None:
                return None, column, row, 0.0, (
                    reason if reason in ("out_of_workspace", "destination_unreachable") else "no_free_region"
                )
            return point, column, row, moved, None

        columns = [column] if column is not None else list(regions.columns_deg)
        rows = [row] if row is not None else list(regions.rows_fraction)
        candidates = []
        for c, r in product(columns, rows):
            nominal = table_region_xy(c, r, cfg.perception, regions, self.s.base_xy)
            candidates.append((math.dist(nominal, reference_xy), c, r, nominal))
        for _distance, c, r, nominal in sorted(candidates):
            if valid(nominal) is None:
                return nominal, c, r, 0.0, None
        return None, column, row, 0.0, "no_free_region"

    def place_on_table(self, column: str | None = None, row: str | None = None) -> SkillResult:
        action, t0, s = "place_on_table", time.monotonic(), self.s
        if s.held is None:
            return self._no_block(action, t0)
        color = s.held.color
        point, column, row, moved, reason = self._choose_table_point(
            column, row, ignore_color=color, reference_xy=s.held.over_xy_mm
        )
        if reason == "invalid_arguments":
            return self._result(False, action, reason, "없는 영역 이름입니다.", t0=t0)
        if reason is not None:
            return self._verdict_failure(action, reason, t0, column=column, row=row)
        plan, correction = self._place_plan(point)
        s.carry_and_release(plan)
        regions = self.cfg.agent.table_regions
        name = f"{regions.column_korean[column]} {regions.row_korean[row]}"
        verification = self._finish_placement(color)
        correction.update(self._learn_placement(point, verification))
        detail = f"{self._label(color)} 블록을 부채꼴 {name} 자리에 놓았습니다."
        if moved:
            detail += f" 지정 지점이 막혀 {moved:.0f}mm 옆에 놓았습니다."
        detail += self._miss_note(correction)
        return self._result(True, action, "released", detail, t0=t0, color=color, column=column,
                            row=row, target=_xy(point), adjusted_by_mm=moved or None,
                            **correction, **verification)

    def _cell_point(self, action: str, t0: float, x: int, y: int) -> tuple[XY | None, SkillResult | None]:
        """Resolve a board address to a point, or say why it is not one."""
        point = self.cells.get((int(x), int(y)))
        if point is None:
            b = grid_bounds(self.cells)
            return None, self._result(
                False, action, "invalid_arguments",
                f"({x}, {y})는 부채꼴 안의 칸이 아닙니다. "
                f"x는 {b['x'][0]}~{b['x'][1]}, y는 {b['y'][0]}~{b['y'][1]} 범위에서 "
                "부채꼴 안쪽 칸만 쓸 수 있습니다.",
                retry_advice="do_not_retry", t0=t0,
            )
        return point, None

    def place_at_cell(self, x: int, y: int) -> SkillResult:
        """Release the held block on a chessboard cell of the table."""
        action, t0, s = "place_at_cell", time.monotonic(), self.s
        if s.held is None:
            return self._no_block(action, t0)
        point, failure = self._cell_point(action, t0, x, y)
        if failure is not None:
            return failure
        if point_in_zone(point, s.calib, self.cfg.agent.table_zone_margin_mm):
            return self._result(
                False, action, "invalid_arguments",
                f"({x}, {y}) 칸은 적재 구역 안입니다. 적재 구역에는 칸 이름(좌상단 등)을 쓰세요.",
                retry_advice="do_not_retry", t0=t0,
            )
        color = s.held.color
        regions = self.cfg.agent.table_regions
        # same blocked-point fallback the named regions use
        moved_point, moved, reason = find_free_point(
            point,
            is_valid=lambda p: self.placement_verdict(p, allow_zone=False, ignore_color=color)[0],
            step_mm=regions.search_step_mm,
            max_mm=regions.search_max_mm,
        )
        if reason is not None:
            return self._verdict_failure(
                action,
                reason if reason in ("out_of_workspace", "destination_unreachable") else "no_free_region",
                t0, cell={"x": int(x), "y": int(y)},
            )
        plan, correction = self._place_plan(moved_point)
        s.carry_and_release(plan)
        verification = self._finish_placement(color)
        correction.update(self._learn_placement(moved_point, verification))
        landed = self._measured_cell(verification)
        detail = f"{self._label(color)} 블록을 ({x}, {y}) 칸에 놓았습니다."
        if moved:
            detail += f" 지정 칸이 막혀 {moved:.0f}mm 옆에 놓았습니다."
        if landed is not None and landed != (int(x), int(y)):
            detail += f" 카메라로 보니 ({landed[0]}, {landed[1]}) 칸입니다."
        detail += self._miss_note(correction)
        return self._result(True, action, "released", detail, t0=t0, color=color,
                            cell={"x": int(x), "y": int(y)},
                            measured_cell={"x": landed[0], "y": landed[1]} if landed else None,
                            target=_xy(moved_point), adjusted_by_mm=moved or None,
                            **correction, **verification)

    def place_here(self) -> SkillResult:
        action, t0, s = "place_here", time.monotonic(), self.s
        if s.held is None:
            return self._no_block(action, t0)
        x, y, _z = s.arm_position_mm()
        reason, plan = self.placement_verdict((x, y), allow_zone=True, ignore_color=s.held.color)
        if reason is not None:
            return self._verdict_failure(action, reason, t0, target=_xy((x, y)))
        color = self._release_over(plan)
        verification = self._finish_placement(color)
        return self._result(True, action, "released", f"{self._label(color)} 블록을 현재 위치에 내려놓았습니다.",
                            t0=t0, color=color, target=_xy((x, y)), **verification)

    # ── composites ───────────────────────────────────────────────────

    def _chain(self, action: str, t0: float, first: SkillResult, second_fn) -> SkillResult:
        if not first.ok:
            first.action = action
            first.data["failed_step"] = "pick_block"
            first.elapsed_s = time.monotonic() - t0
            return first
        second = second_fn()
        second.action = action
        second.data.setdefault("pick", {k: v for k, v in first.data.items()
                                        if k in ("grasp_label", "rounds_used", "from_zone")})
        if not second.ok:
            second.data["failed_step"] = "place"
            second.detail = f"블록은 집었지만 내려놓지 못했습니다: {second.detail} (지금 들고 있음)"
        second.elapsed_s = time.monotonic() - t0
        return second

    def move_block_to_slot(self, color: str, slot_index: int) -> SkillResult:
        action, t0, s = "move_block_to_slot", time.monotonic(), self.s
        if s.held is not None and s.held.color == color:
            return self._chain(action, t0, SkillResult(True, action, "held"),
                               lambda: self.place_at_slot(slot_index))
        if s.held is not None:
            return self._result(False, action, "already_holding",
                                f"이미 {s.held.color} 블록을 들고 있습니다.",
                                retry_advice="do_not_retry", t0=t0)
        scene = self._observe_or_fail(action, t0)
        if isinstance(scene, SkillResult):
            return scene
        occupant = scene.slot_occupancy.get(slot_index)
        korean = self.slot_korean(slot_index)
        if occupant == color:
            return self._result(True, action, "ok", f"{color} 블록은 이미 {korean} 칸에 있습니다.", t0=t0)
        if occupant is not None:
            free = [self.slot_label(i) for i, c in scene.slot_occupancy.items() if c is None]
            return self._result(False, action, "slot_occupied",
                                f"{korean} 칸에는 이미 {occupant} 블록이 있습니다.",
                                retry_advice="retry_ok", t0=t0, free_slots=free)
        if scene.find(color) is None:
            return self._result(False, action, "not_detected",
                                f"카메라에서 {color} 블록을 찾지 못했습니다.",
                                retry_advice="ask_operator", t0=t0)
        return self._chain(action, t0, self.pick_block(color), lambda: self.place_at_slot(slot_index))

    def move_block_to_table(self, color: str, column: str | None = None, row: str | None = None) -> SkillResult:
        action, t0, s = "move_block_to_table", time.monotonic(), self.s
        if s.held is not None and s.held.color != color:
            return self._result(False, action, "already_holding",
                                f"이미 {s.held.color} 블록을 들고 있습니다.",
                                retry_advice="do_not_retry", t0=t0)
        if s.held is None:
            scene = self._observe_or_fail(action, t0)
            if isinstance(scene, SkillResult):
                return scene
            block = scene.find(color)
            if block is None:
                return self._result(False, action, "not_detected",
                                    f"카메라에서 {color} 블록을 찾지 못했습니다.",
                                    retry_advice="ask_operator", t0=t0)
            # refuse before grasping if there is nowhere to put it
            point, _c, _r, _moved, reason = self._choose_table_point(
                column, row, ignore_color=color, reference_xy=block.center_mm
            )
            if reason == "invalid_arguments":
                return self._result(False, action, reason, "없는 영역 이름입니다.", t0=t0)
            if reason is not None:
                return self._verdict_failure(action, reason, t0)
            first = self.pick_block(color)
        else:
            first = SkillResult(True, action, "held")
        return self._chain(action, t0, first, lambda: self.place_on_table(column, row))

    def move_block_to_cell(self, color: str, x: int, y: int) -> SkillResult:
        """Pick a block by colour and put it on a chessboard cell."""
        action, t0, s = "move_block_to_cell", time.monotonic(), self.s
        if s.held is not None and s.held.color != color:
            return self._result(False, action, "already_holding",
                                f"이미 {s.held.color} 블록을 들고 있습니다.",
                                retry_advice="do_not_retry", t0=t0)
        # refuse before grasping if that is not a cell at all
        _point, failure = self._cell_point(action, t0, x, y)
        if failure is not None:
            return failure
        first = self.pick_block(color) if s.held is None else SkillResult(True, action, "held")
        return self._chain(action, t0, first, lambda: self.place_at_cell(x, y))

    def shift_block(self, color: str, forward_mm: float, left_mm: float) -> SkillResult:
        action, t0, s, cfg = "shift_block", time.monotonic(), self.s, self.cfg
        rel = cfg.agent.relative
        if not forward_mm and not left_mm:
            return self._result(False, action, "invalid_arguments", "이동 거리가 0입니다.", t0=t0)
        if s.held is not None:
            return self._result(False, action, "already_holding",
                                f"{self._label(s.held.color)} 블록을 들고 있어 다른 블록을 옮길 수 없습니다. 먼저 내려놓으세요.",
                                retry_advice="do_not_retry", t0=t0)
        scene = self._observe_or_fail(action, t0)
        if isinstance(scene, SkillResult):
            return scene
        block = scene.find(color)
        if block is None:
            return self._result(False, action, "not_detected", f"카메라에서 {color} 블록을 찾지 못했습니다.",
                                retry_advice="ask_operator", t0=t0)
        (f, l), clamped = clamp_vector((forward_mm, left_mm), rel.max_shift_mm)
        before = block.center_mm
        target = offset_xy(before, f, l, frame=rel.frame, base_xy_mm=s.base_xy)
        reason, plan = self.placement_verdict(target, allow_zone=True, ignore_color=color)
        if reason is not None:
            return self._verdict_failure(action, reason, t0, target=_xy(target))

        picked = self.pick_block(color)
        if not picked.ok:
            return self._chain(action, t0, picked, lambda: None)
        s.carry_and_release(plan)
        verification = self._finish_placement(color)
        data: dict[str, Any] = {
            "color": color,
            "requested_mm": {"forward": f, "left": l},
            "requested_clamped": clamped or None,
            "before": _xy(before),
            "target": _xy(target),
            **verification,
        }
        detail = f"{color} 블록을 앞 {f:+.0f}mm, 왼쪽 {l:+.0f}mm 만큼 옮겼습니다."
        measured = verification.get("measured")
        if measured:
            after = (measured["x_mm"], measured["y_mm"])
            mf, ml = decompose_xy(before, after, frame=rel.frame, base_xy_mm=s.base_xy)
            data["measured_mm"] = {"forward": mf, "left": ml}
            data["error_mm"] = math.dist(after, target)
            detail += f" 카메라 실측 이동량은 앞 {mf:+.0f}mm, 왼쪽 {ml:+.0f}mm 입니다."
        rms = s.calib.meta.get("rms_mm")
        if rms is not None and vector_norm(f, l) < float(rms):
            detail += f" 요청 거리가 캘리브레이션 오차(약 {float(rms):.0f}mm) 수준이라 부정확할 수 있습니다."
        return self._result(True, action, "moved", detail, t0=t0, **data)

    def place_at_pixel(self, u: int, v: int, calibration_id: str) -> SkillResult:
        action,t0,s = "place_at_pixel",time.monotonic(),self.s
        target,failure=self._pixel_point(action,t0,u,v,calibration_id)
        if failure is not None:
            return failure
        if s.held is None:
            return self._no_block(action,t0)
        point=(target['x_mm'],target['y_mm'])
        reason,plan=self.placement_verdict(point,allow_zone=True,ignore_color=s.held.color)
        if reason is not None:
            return self._verdict_failure(action,reason,t0,target=target)
        s.carry_and_release(plan)
        return self._result(True,action,"released","선택한 픽셀에 해제 동작을 완료했습니다.",
                            t0=t0,target=target,placement_verified=False)
