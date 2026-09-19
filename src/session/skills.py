"""High-level arm skills for the LLM agent. Every method returns a SkillResult.

These compose the mission code; they do not reimplement it:

- ``pick_block`` drives the production ``CvIkPickState`` (centre attempt plus
  the one 90-degree gripper-roll retry live inside ``run_grasp_attempts``),
  applies the same ``corrected_pick_xy``/``far_reach_tilt_deg`` Task 1 applies,
  gates on ``check_grasp`` exactly like VERIFY (AGENTS.md §3 HARD RULE), and
  repeats home -> observe -> pick up to ``fsm.max_retries_per_block`` times,
  which is what SELECT -> PICK -> SELECT does in the FSM.
- Every release uses ``control.task1_transport``'s ``solve_place_point``,
  ``carry_waypoints``, ``fly_carry`` and ``release_at`` -- the motions Task 1
  flies.
- ``run_task`` runs the unmodified FSM flows on this session's robot.

A skill never raises for an expected outcome. ``Cancelled`` (STOP) and
``TimeoutError`` (the arm did not track) propagate to the tool layer, which
turns them into fault results.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any

from control.grasp import plan_grasp_attempts, run_grasp_attempts
from control.sensing import check_grasp
from control.task1_transport import over_ik_gate, place_tilt_deg, release_at
from fsm.states import RunContext, StateName
from fsm.task1 import corrected_pick_xy, far_reach_tilt_deg
from perception.scene import Scene, SceneBlock
from perception.zone import point_in_zone
from perception.select import SelectionResult
from session.arm_session import (
    PICK_TILT_KEY,
    ArmSession,
    CameraError,
    HeldBlock,
    LastPick,
)
from session.cancel import Cancelled, guard
from session.grid import (
    BoardGrid,
    build_grid,
    cells_in_view,
    cells_in_workspace,
    default_anchor_mm,
)
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

logger = logging.getLogger(__name__)

XY = tuple[float, float]


def _xy(point: XY) -> dict[str, float]:
    return {"x_mm": float(point[0]), "y_mm": float(point[1])}


class Skills:
    def __init__(self, session: ArmSession):
        self.s = session
        self._grid: BoardGrid | None = None
        self._cells: dict[tuple[int, int], XY] | None = None
        self._place_correction: PlaceCorrection | None = None

    def close(self) -> None:
        self.s.close()

    # ── board grid ───────────────────────────────────────────────────

    @property
    def grid(self) -> BoardGrid:
        """Chessboard-cell addressing for this venue (``session/grid.py``)."""
        if self._grid is None:
            base = self.s.base_xy
            anchor = default_anchor_mm(self.cfg.perception, self.cfg.agent.table_regions, base)
            self._grid = build_grid(self.s.calib.board_grid, self.cfg.agent.board_grid, anchor)
        return self._grid

    @property
    def cells(self) -> dict[tuple[int, int], XY]:
        """Addressable cells -> centre mm. Built once; the board cannot move."""
        if self._cells is None:
            found = cells_in_workspace(
                self.grid, self.cfg.perception, self.cfg.agent.table_regions, self.s.base_xy,
                self.cfg.agent.board_grid,
            )
            found = cells_in_view(found, self._to_pixel, self.s.calib.image_size)
            self._cells = {(c.x, c.y): c.xy_mm for c in found}
        return self._cells

    def _to_pixel(self, points: list[XY]):
        import numpy as np

        return self.s.calib.board_to_pixel(np.asarray(points, dtype=float))

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

    # ── naming ───────────────────────────────────────────────────────

    @property
    def cfg(self):
        return self.s.cfg

    def slot_label(self, index: int | None) -> str | None:
        if index is None:
            return None
        return self.cfg.agent.zone_slots.labels[index]

    def slot_korean(self, index: int | None) -> str | None:
        if index is None:
            return None
        return self.cfg.agent.zone_slots.korean_labels[index]

    @property
    def colors(self) -> list[str]:
        return sorted(self.cfg.perception.color_prototypes)

    # ── state ────────────────────────────────────────────────────────

    def _block_dict(self, block: SceneBlock) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "color": block.color,
            **_xy(block.center_mm),
            "reach_mm": block.reach_mm,
        }
        if block.in_zone:
            entry["slot"] = self.slot_label(block.slot_index)
            entry["slot_korean"] = self.slot_korean(block.slot_index)
        return entry

    @staticmethod
    def _label(color: str | None) -> str:
        """Korean text for a held/scanned item whose colour may be unknown
        (a manual claw-machine grab never looks it up)."""
        return color or "정체 불명의 물체"

    def state_dict(self, *, read_robot: bool = True) -> dict[str, Any]:
        s = self.s
        state: dict[str, Any] = {
            "holding": ((s.held.color or "unidentified") if s.held else None),
            "last_block_color": s.last_block_color,
            "last_pick_offset": (
                {"color": s.last_pick.color, "forward_mm": s.last_pick.forward_mm,
                 "left_mm": s.last_pick.left_mm}
                if s.last_pick else None
            ),
        }
        if self._place_correction is not None and self._place_correction.samples:
            # what the arm has learned it must aim past to land on target;
            # bake a converged value into agent.place_correction to start
            # the next session there
            state["place_correction"] = self._place_correction.as_dict()
        if read_robot:
            try:
                state["arm_at_home"] = s.arm_at_home()
                x, y, z = s.arm_position_mm()
                state["arm_position_mm"] = {"x": x, "y": y, "z": z}
            except Cancelled:
                raise
            except Exception as exc:  # noqa: BLE001 - state is best effort
                state["arm_state_error"] = str(exc)
        scene = s.last_scene
        if scene is not None:
            state["blocks_outside"] = [self._block_dict(b) for b in scene.outside.values()]
            state["blocks_inside"] = [self._block_dict(b) for b in scene.inside.values()]
            state["zone_slots"] = {
                self.slot_label(i): color for i, color in sorted(scene.slot_occupancy.items())
            }
            if scene.captured_at > 0:
                state["scene_age_s"] = max(0.0, time.time() - scene.captured_at)
        return state

    def _result(self, ok: bool, action: str, reason: str, detail: str = "", *,
                retry_advice: str | None = None, t0: float | None = None,
                **data: Any) -> SkillResult:
        return SkillResult(
            ok=ok,
            action=action,
            reason=reason,
            detail=detail,
            retry_advice=retry_advice,
            data={k: v for k, v in data.items() if v is not None},
            elapsed_s=(time.monotonic() - t0) if t0 is not None else 0.0,
        )

    def _observe_or_fail(self, action: str, t0: float) -> Scene | SkillResult:
        try:
            return self.s.home_and_observe()
        except CameraError as exc:
            return self._result(
                False, action, "camera_stale" if exc.stale else "camera_unreachable",
                f"카메라 프레임을 받지 못했습니다: {exc}", retry_advice="ask_operator", t0=t0,
            )

    # ── read-only ────────────────────────────────────────────────────

    def get_state(self) -> SkillResult:
        return self._result(True, "get_state", "ok")

    def observe_scene(self, include_zone: bool = True) -> SkillResult:
        t0 = time.monotonic()
        scene = self._observe_or_fail("observe_scene", t0)
        if isinstance(scene, SkillResult):
            return scene
        outside = [self._block_dict(b) for b in scene.outside.values()]
        inside = [self._block_dict(b) for b in scene.inside.values()] if include_zone else None
        detail = f"적재 구역 밖 {len(outside)}개" + (
            f", 안 {len(inside)}개" if inside is not None else ""
        ) + " 블록을 보았습니다."
        return self._result(True, "observe_scene", "ok", detail, t0=t0,
                            blocks_outside=outside, blocks_inside=inside)

    def describe_places(self) -> SkillResult:
        """Names the LLM may use: zone cells (and occupancy) and table regions."""
        s = self.s
        scene = s.last_scene
        slots = []
        for index, xy in enumerate(s.slot_centres):
            slots.append({
                "slot": self.slot_label(index),
                "korean": self.slot_korean(index),
                "row": "top(far)" if index < 3 else "bottom(near)",
                **_xy(xy),
                "occupied_by": scene.slot_occupancy.get(index) if scene else None,
            })
        regions_cfg = self.cfg.agent.table_regions
        regions = []
        for column, row in product(regions_cfg.columns_deg, regions_cfg.rows_fraction):
            xy = table_region_xy(column, row, self.cfg.perception, regions_cfg, s.base_xy)
            regions.append({
                "column": column,
                "row": row,
                "korean": f"{regions_cfg.column_korean[column]} {regions_cfg.row_korean[row]}",
                **_xy(xy),
            })
        b = grid_bounds(self.cells)
        return self._result(
            True, "describe_places", "ok",
            "적재 구역 5칸(상단 3, 하단 2), 부채꼴 테이블 영역 이름, 그리고 체스판 칸 좌표입니다.",
            zone_slots=slots, table_regions=regions,
            board_cells={
                "count": len(self.cells),
                "x_range": b["x"],
                "y_range": b["y"],
                "cell_mm": round(self.grid.cell_mm, 1),
                "convention": "화면 기준 (x, y): x+ = 화면 오른쪽, y+ = 로봇에서 멀어지는 쪽. "
                              "(0, 0)은 화면 아래-가운데 기준 칸. 범위 안이어도 부채꼴 밖 칸은 없습니다.",
            },
            frames={
                "zone_and_table_names": "카메라 화면 기준: 상단/위=로봇에서 먼 쪽, 좌=화면 왼쪽",
                "relative_moves": f"{self.cfg.agent.relative.frame} 기준: forward=로봇에서 멀어짐, left=왼쪽, up=높이",
            },
        )

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

    def _verify_block(self, color: str) -> dict[str, Any]:
        """Home, look again, and report where ``color`` actually is."""
        try:
            scene = self.s.home_and_observe()
        except CameraError as exc:
            return {"verified": False, "verification_error": str(exc)}
        block = scene.find(color)
        if block is None:
            return {"verified": False, "measured": None}
        return {"verified": True, "measured": self._block_dict(block), "in_zone": block.in_zone}

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

    def pick_block(
        self,
        color: str,
        forward_mm: float = 0.0,
        left_mm: float = 0.0,
        relative_to_last: bool = False,
    ) -> SkillResult:
        action, t0, s, cfg = "pick_block", time.monotonic(), self.s, self.cfg
        if color not in self.colors:
            return self._result(False, action, "invalid_arguments", f"모르는 색입니다: {color}", t0=t0)

        if s.held is not None:
            if s.held.color == color and (relative_to_last or forward_mm or left_mm):
                self._put_back()
            else:
                return self._result(
                    False, action, "already_holding",
                    f"이미 {self._label(s.held.color)} 블록을 들고 있습니다. 먼저 내려놓아야 합니다.",
                    retry_advice="do_not_retry", t0=t0,
                )

        base_forward = base_left = 0.0
        if relative_to_last:
            if s.last_pick is None or s.last_pick.color != color:
                return self._result(
                    False, action, "precondition",
                    f"직전에 {color} 블록을 집은 기록이 없어 '더' 보정을 할 수 없습니다.",
                    retry_advice="do_not_retry", t0=t0,
                )
            base_forward, base_left = s.last_pick.forward_mm, s.last_pick.left_mm
        (offset_forward, offset_left), clamped = clamp_vector(
            (base_forward + forward_mm, base_left + left_mm), cfg.agent.relative.max_pick_offset_mm
        )

        rounds = max(1, cfg.fsm.max_retries_per_block)
        attempts = 0
        last_note = ""
        for round_index in range(rounds):
            scene = self._observe_or_fail(action, t0)
            if isinstance(scene, SkillResult):
                return scene
            target = scene.find(color)
            if target is None:
                if round_index == 0:
                    return self._result(
                        False, action, "not_detected",
                        f"카메라에서 {color} 블록을 찾지 못했습니다.",
                        retry_advice="ask_operator", t0=t0,
                    )
                last_note = "not_detected"
                break

            pick_xy = corrected_pick_xy(target.center_mm, s.base_xy, cfg)
            pick_xy = offset_xy(
                pick_xy, offset_forward, offset_left,
                frame=cfg.agent.relative.frame, base_xy_mm=s.base_xy,
            )
            ctx = RunContext(fsm=cfg.fsm)
            ctx.target_id = color
            ctx.extras["selection"] = SelectionResult(
                replace(target.detection, center_mm=pick_xy), color, 1, [target.detection]
            )
            ctx.extras[PICK_TILT_KEY] = far_reach_tilt_deg(pick_xy, s.base_xy, cfg)

            attempts += 1
            next_state = s.pick_state.step(ctx)
            last_note = ctx.last_note
            if next_state is StateName.VERIFY:
                check = check_grasp(s.robot, cfg.sensing)
                if check.grasped:
                    held = ctx.extras["ik_pick_attempt"]
                    s.held = HeldBlock(color, held, target.center_mm, target.in_zone, held.xy_mm)
                    s.last_pick = LastPick(color, offset_forward, offset_left)
                    s.last_block_color = color
                    return self._result(
                        True, action, "held",
                        f"{color} 블록을 집었습니다" + (
                            f" (파지 보정 앞 {offset_forward:+.0f}mm, 왼쪽 {offset_left:+.0f}mm)."
                            if offset_forward or offset_left else "."
                        ),
                        t0=t0, rounds_used=round_index + 1, grasp_label=held.label,
                        from_zone=target.in_zone,
                        detected=self._block_dict(target),
                        pick_offset_mm={"forward": offset_forward, "left": offset_left},
                        offset_clamped=clamped or None,
                    )
                s.motion.open_gripper()
                last_note = "verify_empty"
            elif "unreachable" in last_note:
                return self._result(
                    False, action, "unreachable",
                    f"{color} 블록이 팔이 닿을 수 있는 범위 밖입니다. 블록을 로봇 쪽으로 옮겨 주세요.",
                    retry_advice="do_not_retry", t0=t0, internal_retries_exhausted=True,
                    detected=self._block_dict(target),
                )
            elif "motion_timeout" in last_note:
                return self._result(
                    False, action, "motion_timeout",
                    "팔이 명령한 자세를 따라가지 못했습니다. 기계적 문제일 수 있어 멈춥니다.",
                    retry_advice="ask_operator", t0=t0,
                )

        s.go_home()
        if last_note == "not_detected":
            return self._result(
                False, action, "not_detected",
                f"재접근 도중 {color} 블록이 보이지 않게 되었습니다.",
                retry_advice="ask_operator", t0=t0, attempts_used=attempts,
            )
        return self._result(
            False, action, "grasp_empty",
            f"{color} 블록을 {attempts}회 접근했지만 집지 못했습니다. 매 접근마다 그리퍼를 90도 "
            f"돌려 재시도했고 home에서 다시 접근했습니다. 블록이 넘어졌거나 다른 블록에 붙어 있을 수 있습니다.",
            retry_advice="ask_operator", t0=t0, attempts_used=attempts,
            internal_retries_exhausted=True,
        )

    # ── place ────────────────────────────────────────────────────────

    def _no_block(self, action: str, t0: float) -> SkillResult:
        return self._result(False, action, "no_block_held", "들고 있는 블록이 없습니다.",
                            retry_advice="do_not_retry", t0=t0)

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
        verification = self._verify_block(color)
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
        verification = self._verify_block(color)
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
        verification = self._verify_block(color)
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
        verification = self._verify_block(color)
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
        verification = self._verify_block(color)
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

    # ── arm ──────────────────────────────────────────────────────────

    def move_arm(self, forward_mm: float = 0.0, left_mm: float = 0.0, up_mm: float = 0.0) -> SkillResult:
        action, t0, s, cfg = "move_arm", time.monotonic(), self.s, self.cfg
        rel = cfg.agent.relative
        norm = vector_norm(forward_mm, left_mm, up_mm)
        if norm == 0.0:
            return self._result(False, action, "invalid_arguments", "이동 거리가 0입니다.", t0=t0)
        if norm > rel.max_jog_mm:
            return self._result(
                False, action, "limit_exceeded",
                f"한 번에 {rel.max_jog_mm:.0f}mm까지만 움직일 수 있습니다 (요청 {norm:.0f}mm). 나눠서 요청하세요.",
                retry_advice="do_not_retry", t0=t0,
            )
        joints = s.robot.read_joints()
        x, y, z = s.ik.forward_position_mm(joints)
        tx, ty = offset_xy((x, y), forward_mm, left_mm, frame=rel.frame, base_xy_mm=s.base_xy)
        return self._fly_to_xy(action, t0, joints, (x, y, z), (tx, ty), z + up_mm)

    def _fly_to_xy(self, action: str, t0: float, joints, from_xyz, target_xy: XY,
                   tz: float) -> SkillResult:
        """Take the gripper to one xy at height ``tz``, gated like a jog.

        Shared by ``move_arm`` (a bounded relative vector) and
        ``move_to_cell`` (a board address). The height window, workspace
        gate and IK-error gate are the same either way -- only how the
        target was named differs, and only ``move_arm`` caps the distance.
        """
        s, cfg = self.s, self.cfg
        rel = cfg.agent.relative
        x, y, z = from_xyz
        tx, ty = target_xy
        entered = False
        # A pose reached by an earlier jog may settle a few mm past the exact
        # window edge (the same IK/arrival tolerance that gates every move
        # here) -- both the "do we need to re-enter" check and the final
        # acceptance gate below must tolerate that, or a lateral move
        # (up_mm=0, so tz == z) that lands just outside the strict window
        # would be judged already-inside by the first check yet rejected by
        # a second, stricter one, permanently refusing every jog from there.
        slack = rel.jog_max_ik_error_mm
        lo, hi = rel.jog_min_z_mm - slack, rel.jog_max_z_mm + slack
        if not lo <= z <= hi:
            # e.g. from home, whose tool frame sits near table height
            tz = min(max(tz, rel.jog_min_z_mm), rel.jog_max_z_mm)
            entered = True
        elif not lo <= tz <= hi:
            return self._result(
                False, action, "height_limit",
                f"높이는 {rel.jog_min_z_mm:.0f}~{rel.jog_max_z_mm:.0f}mm 범위에서만 움직일 수 있습니다.",
                retry_advice="do_not_retry", t0=t0, from_mm={"x": x, "y": y, "z": z},
            )
        if not s.in_workspace((tx, ty)):
            return self._verdict_failure(action, "out_of_workspace", t0, target=_xy((tx, ty)))
        # keep the jaws (and a held block) turned as they are now; tip outward
        # at far reach exactly as placements do
        tilt = place_tilt_deg((tx, ty), s.base_xy, cfg)
        result = s.ik.solve_holding_wrist_roll(tx, ty, tz, joints["wrist_roll"], radial_tilt_deg=tilt)
        if over_ik_gate(result, cfg) or result.position_error_mm > rel.jog_max_ik_error_mm:
            return self._result(
                False, action, "ik_gate",
                f"그 위치로는 팔을 정확히 보낼 수 없습니다 (IK 오차 {result.position_error_mm:.0f}mm). "
                "로봇에 더 가깝거나 더 높은 곳으로 요청하세요.",
                retry_advice="do_not_retry", t0=t0, target={"x": tx, "y": ty, "z": tz},
                ik_error_mm=result.position_error_mm,
            )
        s.player.move_to(result.joints, max_step=1.0, tol=cfg.motion.transit_arrival_tol)
        if s.held is not None:
            s.held.over_xy_mm = (tx, ty)
        rx, ry, rz = s.arm_position_mm()
        detail = "팔을 움직였습니다."
        if entered:
            detail = f"먼저 작업 높이({tz:.0f}mm)로 올린 뒤 움직였습니다."
        return self._result(True, action, "moved", detail, t0=t0,
                            from_mm={"x": x, "y": y, "z": z}, target={"x": tx, "y": ty, "z": tz},
                            reached={"x": rx, "y": ry, "z": rz}, entered_jog_height=entered or None)

    def move_to_cell(self, x: int, y: int) -> SkillResult:
        """Fly the gripper over one chessboard cell, at the current height.

        Unlike ``move_arm`` this is an address, not a nudge, so the jog
        distance cap does not apply -- the far edge of the board is 300mm
        from the near edge and must be one move. Every other gate (height
        window, workspace sector, IK error) is the jog's.
        """
        action, t0, s = "move_to_cell", time.monotonic(), self.s
        point, failure = self._cell_point(action, t0, x, y)
        if failure is not None:
            return failure
        joints = s.robot.read_joints()
        from_xyz = s.ik.forward_position_mm(joints)
        result = self._fly_to_xy(action, t0, joints, from_xyz, point, from_xyz[2])
        if result.ok:
            result.data["cell"] = {"x": int(x), "y": int(y)}
            result.detail = f"({x}, {y}) 칸 위로 이동했습니다."
        return result

    def rotate_gripper(self, delta_deg: float) -> SkillResult:
        """Spin the jaws about their own axis without moving x/y/z.

        wrist_roll is the last joint before the gripper, so commanding it
        alone (``interpolate`` only touches joints present in the goal) turns
        the jaws in place -- no IK solve needed. No absolute range is
        enforced: lerobot's own max_relative_target clamp and the per-tick
        step limit already bound real motion, and a target past the physical
        stop simply times out (reported as motion_timeout), so a per-call
        delta cap is the only refusal needed here.
        """
        action, t0, s, cfg = "rotate_gripper", time.monotonic(), self.s, self.cfg
        if delta_deg == 0.0:
            return self._result(False, action, "invalid_arguments", "회전 각도가 0입니다.", t0=t0)
        limit = cfg.agent.relative.max_gripper_roll_deg
        if abs(delta_deg) > limit:
            return self._result(
                False, action, "limit_exceeded",
                f"한 번에 {limit:.0f}도까지만 돌릴 수 있습니다 (요청 {delta_deg:+.0f}도). 나눠서 요청하세요.",
                retry_advice="do_not_retry", t0=t0,
            )
        current = s.robot.read_joints()["wrist_roll"]
        target = current + delta_deg
        s.player.move_to({"wrist_roll": target}, max_step=1.0, tol=cfg.motion.transit_arrival_tol)
        reached = s.robot.read_joints()["wrist_roll"]
        return self._result(True, action, "moved", f"그리퍼를 {delta_deg:+.0f}도 돌렸습니다.", t0=t0,
                            from_deg=current, target_deg=target, reached_deg=reached)

    def pick_here(self) -> SkillResult:
        """Manual 'claw machine' grab: attempt a grasp at wherever the arm
        is right now, with no colour/target lookup -- a photo taken with the
        arm already positioned there would just show its own gripper. Uses
        the same grasp planning/execution (centre attempt, the 90-degree
        gripper-roll retry, check_grasp verification) pick_block drives.
        """
        action, t0, s, cfg = "pick_here", time.monotonic(), self.s, self.cfg
        if s.held is not None:
            return self._result(
                False, action, "already_holding",
                f"이미 {self._label(s.held.color)}을(를) 들고 있습니다. 먼저 내려놓으세요.",
                retry_advice="do_not_retry", t0=t0,
            )
        x, y, _z = s.arm_position_mm()
        if not s.in_workspace((x, y)):
            return self._verdict_failure(action, "out_of_workspace", t0, target=_xy((x, y)))
        plan = plan_grasp_attempts(s.ik, cfg, x, y, s.grasp_z_mm, log=logger.info)
        if not any(a.reachable for a in plan.attempts):
            return self._result(
                False, action, "unreachable",
                "이 위치에서는 아래로 내려가 집을 수 없습니다. 조금 더 로봇 쪽으로 옮겨서 다시 시도하세요.",
                retry_advice="do_not_retry", t0=t0, internal_retries_exhausted=True,
            )
        held = run_grasp_attempts(s.player, s.robot, cfg, plan, log=logger.info)
        if held is None:
            return self._result(
                False, action, "grasp_empty",
                "그리퍼를 닫아봤지만 아무것도 집지 못했습니다. 그리퍼를 90도 돌려 한 번 더 시도했습니다.",
                retry_advice="retry_ok", t0=t0,
            )
        s.held = HeldBlock(color=None, attempt=held, picked_xy_mm=(x, y),
                           from_zone=s.in_zone((x, y)), over_xy_mm=held.xy_mm)
        s.last_block_color = None
        return self._result(True, action, "held", "무언가를 집었습니다.", t0=t0, grasp_label=held.label)

    def return_to_home(self) -> SkillResult:
        t0 = time.monotonic()
        lifted, at_home = self.s.return_home_safely()
        return self._result(at_home, "return_to_home", "ok" if at_home else "motion_timeout",
                            "home으로 복귀했습니다." if at_home else "home 자세에 도달하지 못했습니다.",
                            t0=t0, lifted_first=lifted or None, arm_at_home=at_home)

    def recover_and_home(self) -> SkillResult:
        """STOP/fault recovery: drop whatever is held, home, then close the jaws.

        Unlike ``return_to_home`` (a normal LLM tool that keeps a held block
        held), this always opens the gripper first -- the arena's blocks and
        arm are small enough, and the zone is not reachable by students, that
        a dropped block is a non-issue and simplicity wins. Clears the STOP
        flag first so the recovery motion itself is not immediately cancelled.
        """
        t0, s = time.monotonic(), self.s
        s.cancel.clear()
        released = s.held.color if s.held else None
        try:
            s.motion.open_gripper()
        except Exception as exc:  # noqa: BLE001 - still try to get home
            logger.warning("recover_and_home: open_gripper failed: %s", exc)
        s.held = None
        if released is not None:
            s.last_block_color = released
        lifted, at_home = s.return_home_safely()
        if at_home:
            try:
                s.motion.close_gripper()
            except Exception as exc:  # noqa: BLE001 - homing already succeeded
                logger.warning("recover_and_home: close_gripper failed: %s", exc)
        detail = "그리퍼를 열어" + (f" {released} 블록을 내려놓고" if released else "") + \
            (" home으로 복귀하고 그리퍼를 닫았습니다." if at_home else " home으로 복귀를 시도했지만 도달하지 못했습니다.")
        return self._result(
            at_home, "recover_and_home", "ok" if at_home else "motion_timeout", detail,
            t0=t0, released=released, lifted_first=lifted or None, arm_at_home=at_home,
        )

    def open_gripper(self) -> SkillResult:
        t0, s = time.monotonic(), self.s
        released = s.held.color if s.held else None
        s.motion.open_gripper()
        if s.held is not None:
            s.last_block_color = s.held.color
            s.held = None
        return self._result(True, "open_gripper", "released" if released else "ok",
                            f"그리퍼를 열었습니다{f' ({released} 블록을 놓음)' if released else ''}.",
                            t0=t0, released=released)

    # ── missions ─────────────────────────────────────────────────────

    def run_task(self, task: int) -> SkillResult:
        action, t0, s, cfg = f"run_task{task}", time.monotonic(), self.s, self.cfg
        if task not in (1, 2, 3):
            return self._result(False, action, "invalid_arguments", "미션은 1, 2, 3만 있습니다.", t0=t0)
        if task == 3 and not cfg.agent.enable_task3_tool:
            return self._result(False, action, "disabled", "미션 3(데이터 수집)은 설정에서 꺼져 있습니다.",
                                retry_advice="do_not_retry", t0=t0)
        if s.held is not None:
            return self._result(False, action, "already_holding",
                                f"{self._label(s.held.color)} 블록을 들고 있어 미션을 시작할 수 없습니다. 먼저 내려놓으세요.",
                                retry_advice="do_not_retry", t0=t0)
        scene = self._observe_or_fail(action, t0)
        if isinstance(scene, SkillResult):
            return scene
        if task == 3:
            return self._run_task3(t0, scene)

        from fsm.flows import build_task1_states, build_task2_stack_states
        from fsm.machine import StateMachine, TransitionLogger

        perceive = guard(s.cancel, s.task1_perceive())
        ctx = RunContext(fsm=cfg.fsm)
        warnings: list[str] = []
        if task == 1:
            # Pre-reserve cells that already hold a block so Task 1 does not
            # drop a new one on top of an earlier agent placement.
            taken = {color: index for index, color in scene.slot_occupancy.items() if color}
            if taken:
                ctx.extras["task1_slot_by_color"] = dict(taken)
            loose = [b.color for b in scene.inside.values() if b.slot_index is None]
            if loose:
                warnings.append(f"적재 구역 안에 칸에 맞지 않게 놓인 블록이 있습니다: {', '.join(loose)}")
            if len(taken) + len(scene.outside) > len(cfg.task1.slot_uv):
                return self._result(False, action, "precondition",
                                    "빈 칸보다 옮길 블록이 많습니다.", retry_advice="ask_operator", t0=t0)
            states = build_task1_states(robot=s.robot, motion=s.motion, perceive=perceive,
                                        pick_state=s.pick_state, cfg=cfg, calib=s.calib,
                                        planner=s.transport)
        else:
            stack_xy = s.stack.stack_xy_mm
            for block in scene.inside.values():
                if math.dist(block.center_mm, stack_xy) < cfg.agent.place_clear_radius_mm:
                    return self._result(
                        False, action, "precondition",
                        f"적재 지점에 이미 {block.color} 블록이 있어 쌓기를 시작할 수 없습니다.",
                        retry_advice="ask_operator", t0=t0,
                    )
            states = build_task2_stack_states(robot=s.robot, motion=s.motion, perceive=perceive,
                                              pick_state=s.pick_state, cfg=cfg, calib=s.calib,
                                              planner=s.stack)
        run_id = time.strftime(f"agent_task{task}_%Y%m%d_%H%M%S")
        log_dir = Path(cfg.logging.log_dir)
        csv_path = log_dir / f"{run_id}_transitions.csv" if cfg.logging.save_transitions else None
        machine = StateMachine(states, ctx, transition_logger=TransitionLogger(csv_path),
                               enforce_time_budget=task != 1)
        logger.info("agent: running Task %d as %s", task, run_id)
        machine.run()
        s.last_scene = None
        verification = self._observe_or_fail(action, t0)
        verification_failed = isinstance(verification, SkillResult)
        remaining = [] if verification_failed else sorted(verification.outside)
        stop_reason = ctx.extras.get("task2_stop_reason")
        if (
            task == 2
            and stop_reason is None
            and ctx.budget_exhausted()
            and (verification_failed or remaining)
        ):
            stop_reason = "time_budget_exhausted"
        data: dict[str, Any] = {
            "run_id": run_id,
            "remaining_outside": remaining,
            "place_actions": ctx.extras.get("task1_place_actions") if task == 1 else ctx.placed_count,
            "attempts": ctx.extras.get("task1_attempts_total") or ctx.attempts,
            "warnings": warnings or None,
            "stop_reason": stop_reason,
            "failed_blocks": ctx.extras.get("task2_failed_blocks"),
        }
        if task == 1:
            complete = bool(ctx.extras.get("task1_complete"))
            detail = "미션 1 완료: 적재 구역 밖에 블록이 없습니다." if complete else "미션 1이 끝났지만 완료 조건을 확인하지 못했습니다."
        else:
            complete = not verification_failed and not remaining
            if complete:
                detail = f"미션 2 완료: {ctx.placed_count}층을 쌓았습니다."
            elif verification_failed:
                detail = (
                    f"미션 2가 {ctx.placed_count}층에서 끝났지만 카메라로 완료 여부를 "
                    "확인하지 못했습니다."
                )
            else:
                detail = (
                    f"미션 2 미완료: {ctx.placed_count}층을 쌓았고 적재 구역 밖에 "
                    f"{', '.join(remaining)} 블록이 남았습니다."
                )
            if stop_reason:
                detail += f" (종료 사유: {stop_reason})"
        if warnings:
            detail += " " + " ".join(warnings)
        return self._result(complete, action, "ok" if complete else "task_incomplete", detail, t0=t0, **data)

    def _run_task3(self, t0: float, scene: Scene) -> SkillResult:
        """One Task 3 collection round: gather every outside block while recording.

        The terminal prompt that starts a new round is replaced by the end of
        the tool call; the operator rearranges and asks again.
        """
        import copy

        from data.episode_recorder import (
            EpisodeRecorder,
            LeRobotEpisodeSink,
            RecordingRobotIO,
            StopRecording,
            create_dataset,
            remove_empty_dataset,
            resolve_dataset_root,
        )
        from fsm.flows import build_task3_states
        from fsm.machine import StateMachine, TransitionLogger
        from lerobot.datasets import VideoEncodingManager
        from runners.run_task3 import resolve_repo_id, start_frame_sources
        from session.factories import make_pick_state, make_task1_perceive

        class RoundDone(Exception):
            pass

        def end_round(_message: str) -> str:
            raise RoundDone()

        action, s = "run_task3", self.s
        if scene.inside:
            return self._result(False, action, "precondition",
                                "데이터 수집은 빈 적재 구역에서 시작해야 합니다.",
                                retry_advice="ask_operator", t0=t0)
        cfg3 = copy.deepcopy(self.cfg)
        cfg3.motion.fps = cfg3.task3.motion_fps_override
        cfg3.task3.prompt_on_round_complete = True
        repo_id = resolve_repo_id(cfg3, resume=False)
        root = resolve_dataset_root(cfg3.task3, repo_id)
        sources = start_frame_sources(cfg3)
        dataset = None
        recorder = None
        try:
            dataset = create_dataset(cfg3.task3, repo_id, root, resume=False)
            recorder = EpisodeRecorder(LeRobotEpisodeSink(dataset), sources, cfg3.task3)
            robot = RecordingRobotIO(s.robot, recorder, record_fps=cfg3.task3.record_fps,
                                     stop_event=s.cancel.event)
            from control.motion import MotionController

            motion = MotionController(robot, s.poses, cfg3.motion, cfg3.sensing)
            pick = make_pick_state("cv_ik", robot=robot, motion=motion, cfg=cfg3, calib=s.calib,
                                   retreat_pose=None, radial_tilt_extra_key=PICK_TILT_KEY,
                                   max_grasp_attempts=cfg3.task3.max_grasp_attempts, ik=s.ik)
            states = build_task3_states(
                robot=robot, motion=motion,
                perceive=guard(s.cancel, make_task1_perceive(s.calib, cfg3)),
                pick_state=pick, cfg=cfg3, calib=s.calib, planner=s.transport,
                recorder=recorder, prompt=end_round, stop_requested=s.cancel.is_set,
            )
            ctx = RunContext(fsm=cfg3.fsm)
            run_id = time.strftime("agent_task3_%Y%m%d_%H%M%S")
            csv_path = (Path(cfg3.logging.log_dir) / f"{run_id}_transitions.csv"
                        if cfg3.logging.save_transitions else None)
            with VideoEncodingManager(dataset):
                try:
                    StateMachine(states, ctx, transition_logger=TransitionLogger(csv_path),
                                 enforce_time_budget=False).run()
                except RoundDone:
                    pass
                except StopRecording as exc:
                    recorder.abort_episode("interrupted")
                    raise Cancelled(str(exc)) from exc
                finally:
                    recorder.abort_episode("shutdown")
        finally:
            for source in sources.values():
                source.stop()
            if dataset is not None and (recorder is None or recorder.saved_total == 0):
                remove_empty_dataset(Path(dataset.root))
        return self._result(
            True, action, "ok",
            f"데이터 수집 라운드 완료: 에피소드 {recorder.saved_total}개 저장.",
            t0=t0, dataset_root=str(dataset.root), episodes_saved=recorder.saved_total,
            episodes_saved_by_color=recorder.saved_by_color, discard_reasons=recorder.discard_reasons,
        )
