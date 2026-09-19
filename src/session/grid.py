"""Chessboard-cell addressing over the workspace sector.

The workspace floor *is* a chessboard (``perception/board.py``), so its
squares are the natural way for an operator to say *where*: "그 블록을
(3, 4)로 옮겨줘". This module turns that integer address into a robot-base
millimetre point and back.

Two things this is **not**:

- It is not a coordinate frame. AGENTS.md §6 stands: every command still
  travels as robot-base mm. A cell is an *address* that is resolved here,
  once, before any motion is planned.
- It is not a reachability claim. ``cells_in_workspace`` only applies the
  detector's sector gate and the same radial band the named table regions
  use; IK still decides whether a concrete pose is solvable.

Axes follow what the operator sees on the camera page, not the robot's own
axes: ``+x`` is image-right, ``+y`` is away from the robot (image-up). In the
robot base frame that is ``u = -y`` and ``v = +x`` -- see ``axis_aligned``.

The lattice itself comes from ``PlaneCalibration.board_grid`` when
``tools/calibrate_board_grid.py`` has measured it; otherwise the axis-aligned
fallback below is used, which is right only if the board happens to be
square with the robot.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from config import BoardGridConfig, PerceptionConfig, TableRegionsConfig
from perception.detector import point_in_workspace, workspace_radius_at_angle

XY = tuple[float, float]


@dataclass(frozen=True)
class BoardGrid:
    """A lattice of chessboard squares on the calibrated plane.

    ``u_mm``/``v_mm`` are whole-cell steps (not unit vectors), so a cell
    centre is ``origin + x * u + y * v``. Keeping them as vectors rather
    than a pitch plus an angle means a board mounted at a slight angle needs
    no special case anywhere downstream.
    """

    origin_mm: XY
    u_mm: XY
    v_mm: XY

    @property
    def cell_mm(self) -> float:
        """Nominal square edge: the mean of the two step lengths."""
        return 0.5 * (math.hypot(*self.u_mm) + math.hypot(*self.v_mm))

    def cell_to_xy(self, x: int, y: int) -> XY:
        return (
            self.origin_mm[0] + x * self.u_mm[0] + y * self.v_mm[0],
            self.origin_mm[1] + x * self.u_mm[1] + y * self.v_mm[1],
        )

    def xy_to_cell(self, xy_mm: XY) -> tuple[int, int]:
        """Nearest cell to a point. Inverse of ``cell_to_xy`` up to rounding."""
        a, b = self._coefficients(xy_mm)
        return int(round(a)), int(round(b))

    def cell_corners_mm(self, x: int, y: int) -> list[XY]:
        """The four corners of one cell, for drawing it in the operator page."""
        cx, cy = self.cell_to_xy(x, y)
        ux, uy = self.u_mm
        vx, vy = self.v_mm
        return [
            (cx - (ux + vx) / 2, cy - (uy + vy) / 2),
            (cx + (ux - vx) / 2, cy + (uy - vy) / 2),
            (cx + (ux + vx) / 2, cy + (uy + vy) / 2),
            (cx - (ux - vx) / 2, cy - (uy - vy) / 2),
        ]

    def snap_origin_to(self, anchor_mm: XY) -> "BoardGrid":
        """Same lattice, with (0, 0) moved to the cell nearest ``anchor_mm``.

        The lattice phase is fixed by the board; only which square we *call*
        the origin is a choice, so re-anchoring must never shift the squares.
        """
        a, b = self._coefficients(anchor_mm)
        return BoardGrid(self.cell_to_xy(int(round(a)), int(round(b))), self.u_mm, self.v_mm)

    def _coefficients(self, xy_mm: XY) -> tuple[float, float]:
        dx = xy_mm[0] - self.origin_mm[0]
        dy = xy_mm[1] - self.origin_mm[1]
        det = self.u_mm[0] * self.v_mm[1] - self.u_mm[1] * self.v_mm[0]
        if det == 0.0:
            raise ValueError("board grid axes are parallel")
        return (
            (dx * self.v_mm[1] - dy * self.v_mm[0]) / det,
            (self.u_mm[0] * dy - self.u_mm[1] * dx) / det,
        )


def axis_aligned(origin_mm: XY, cell_mm: float) -> BoardGrid:
    """Fallback lattice squared with the robot base frame.

    +x (image-right) is -y in the base frame, +y (away) is +x.
    """
    return BoardGrid(origin_mm, (0.0, -cell_mm), (cell_mm, 0.0))


def build_grid(board_grid: dict | None, cfg: BoardGridConfig, anchor_mm: XY) -> BoardGrid:
    """The lattice to address cells with, anchored so ``anchor_mm`` is (0, 0).

    ``board_grid`` is ``PlaneCalibration.board_grid`` -- the measured board,
    when there is one. ``cfg.origin_mm`` overrides ``anchor_mm`` for venues
    that want a different reference square.
    """
    if cfg.origin_mm is not None:
        anchor_mm = (float(cfg.origin_mm[0]), float(cfg.origin_mm[1]))
    if board_grid:
        grid = BoardGrid(
            tuple(float(v) for v in board_grid["origin_mm"]),  # type: ignore[arg-type]
            tuple(float(v) for v in board_grid["u_mm"]),  # type: ignore[arg-type]
            tuple(float(v) for v in board_grid["v_mm"]),  # type: ignore[arg-type]
        )
    else:
        grid = axis_aligned(anchor_mm, float(cfg.cell_mm))
    return grid.snap_origin_to(anchor_mm)


def default_anchor_mm(
    perception_cfg: PerceptionConfig, regions: TableRegionsConfig, base_xy_mm: XY
) -> XY:
    """The table region the operator page calls (0, 0) when config names none.

    ``center``/``near`` is the bottom centre of the reachable band, which is
    where an operator's "기준 칸" naturally sits. A venue that renamed its
    columns falls back to the middle column and the nearest row.
    """
    from session.relative import table_region_xy

    columns = list(regions.columns_deg)
    rows = list(regions.rows_fraction)
    column = "center" if "center" in columns else columns[len(columns) // 2]
    row = "near" if "near" in rows else min(rows, key=lambda r: regions.rows_fraction[r])
    return table_region_xy(column, row, perception_cfg, regions, base_xy_mm)


@dataclass(frozen=True)
class Cell:
    x: int
    y: int
    xy_mm: XY


def in_base_keepout(xy_mm: XY, base_xy_mm: XY, cfg: BoardGridConfig) -> bool:
    """Whether a point falls in the unusable pocket around the robot itself.

    Not a radius: the arm cannot take a top-down pose in a narrow corridor
    straight ahead of the base (the gripper sits ~27mm off the pan axis,
    AGENTS.md §7), yet it reaches points the same distance away once they
    are off that axis. A single inner radius big enough to exclude the
    corridor would throw away every near cell to the left and right, which
    are perfectly pickable.
    """
    dx, dy = xy_mm[0] - base_xy_mm[0], xy_mm[1] - base_xy_mm[1]
    if math.hypot(dx, dy) < cfg.min_radius_mm:
        return True
    return abs(dy) <= cfg.base_keepout_half_width_mm and dx <= cfg.base_keepout_depth_mm


def cells_in_workspace(
    grid: BoardGrid,
    perception_cfg: PerceptionConfig,
    regions: TableRegionsConfig,
    base_xy_mm: XY,
    grid_cfg: BoardGridConfig,
) -> list[Cell]:
    """Every cell the arm can work in: the detector's sector, minus the rims.

    The outer rim is the named points' own ``edge_margin_mm``, so the grid
    never offers ground a named point would have refused. The inner rim is
    *not* the named points' ``min_radius_mm``: that one is an assumption
    sized for free placement, and reusing it emptied the near left and right
    of the board where blocks can in fact be picked and placed. See
    ``in_base_keepout``.
    """
    span = _index_span(grid, perception_cfg, base_xy_mm)
    cells: list[Cell] = []
    for x in range(-span, span + 1):
        for y in range(-span, span + 1):
            xy = grid.cell_to_xy(x, y)
            if not point_in_workspace(xy, perception_cfg, base_xy_mm):
                continue
            if in_base_keepout(xy, base_xy_mm, grid_cfg):
                continue
            dx, dy = xy[0] - base_xy_mm[0], xy[1] - base_xy_mm[1]
            r_max = (
                workspace_radius_at_angle(perception_cfg, math.degrees(math.atan2(dy, dx)))
                - regions.edge_margin_mm
            )
            if math.hypot(dx, dy) > r_max:
                continue
            cells.append(Cell(x, y, xy))
    return cells


def cells_in_view(
    cells: list[Cell],
    board_to_pixel: Callable[[list[XY]], Any],
    image_size: tuple[int, int],
) -> list[Cell]:
    """Drop cells whose centre falls outside the calibrated frame.

    The far left of the band leaves the picture on this rig. A cell nobody
    can see is a cell nobody can click, and a block released there could
    never be verified by the camera afterwards -- so it is not offered.
    ``board_to_pixel`` is passed in to keep this module free of the
    calibration (and OpenCV) import.
    """
    if not cells:
        return []
    width, height = image_size
    pixels = board_to_pixel([cell.xy_mm for cell in cells])
    return [
        cell
        for cell, (u, v) in zip(cells, pixels)
        if 0 <= u < width and 0 <= v < height
    ]


def bounds(cells: Iterable[tuple[int, int]]) -> dict[str, list[int]]:
    """``{"x": [min, max], "y": [min, max]}`` -- the tool schema's integer range.

    Takes addresses, so a ``{(x, y): centre}`` mapping can be passed directly.
    The range is a bounding box: it holds cells that are not addressable,
    which is why every caller still checks membership.
    """
    addresses = list(cells)
    if not addresses:
        return {"x": [0, 0], "y": [0, 0]}
    xs = [x for x, _ in addresses]
    ys = [y for _, y in addresses]
    return {"x": [min(xs), max(xs)], "y": [min(ys), max(ys)]}


def _index_span(grid: BoardGrid, perception_cfg: PerceptionConfig, base_xy_mm: XY) -> int:
    """How far to search in cell indices to be sure the sector is covered."""
    reach = float(perception_cfg.workspace_radius_mm)
    profile = perception_cfg.workspace_radius_by_angle_mm
    if profile:
        reach = min(reach, max(float(pair[1]) for pair in profile))
    offset = math.dist(grid.origin_mm, base_xy_mm)
    step = min(math.hypot(*grid.u_mm), math.hypot(*grid.v_mm))
    return int(math.ceil((reach + offset) / step)) + 1
