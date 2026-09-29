"""Session state, results and observation shared by every skill."""

from __future__ import annotations

import time
from itertools import product
from typing import Any

from perception.scene import Scene, SceneBlock
from session.arm_session import ArmSession, CameraError
from session.cancel import Cancelled
from session.grid import (
    BoardGrid,
    build_grid,
    cells_in_view,
    cells_in_workspace,
    default_anchor_mm,
)
from session.grid import bounds as grid_bounds
from session.place_correction import PlaceCorrection
from session.relative import table_region_xy
from session.results import SkillResult


XY = tuple[float, float]


def _xy(point: XY) -> dict[str, float]:
    return {"x_mm": float(point[0]), "y_mm": float(point[1])}


class SkillsBase:
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
