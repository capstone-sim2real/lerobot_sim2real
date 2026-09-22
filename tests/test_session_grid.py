"""Board-cell addressing: the lattice, its screen-facing axes, and its extent."""

import math

import numpy as np
import pytest

from agent.service import places_payload
from agent_helpers import make_skills
from config import AppConfig
from perception.homography import PlaneCalibration
from perception.detector import point_in_workspace, workspace_radius_at_angle
from session.grid import (
    BoardGrid,
    axis_aligned,
    bounds,
    build_grid,
    cells_in_view,
    cells_in_workspace,
    default_anchor_mm,
    in_base_keepout,
)
from session.relative import table_region_xy


def _grid(cfg: AppConfig) -> BoardGrid:
    anchor = default_anchor_mm(cfg.perception, cfg.agent.table_regions, (0.0, 0.0))
    return build_grid(None, cfg.agent.board_grid, anchor)


def test_cell_addresses_round_trip_and_step_one_square():
    cfg = AppConfig()
    grid = _grid(cfg)
    for x, y in ((0, 0), (3, 4), (-6, 2), (11, -5)):
        assert grid.xy_to_cell(grid.cell_to_xy(x, y)) == (x, y)
    step = math.dist(grid.cell_to_xy(0, 0), grid.cell_to_xy(1, 0))
    assert step == pytest.approx(cfg.agent.board_grid.cell_mm)
    assert grid.cell_mm == pytest.approx(cfg.agent.board_grid.cell_mm)


def _top_down_calibration(mm_per_px: float = 0.8) -> PlaneCalibration:
    """A camera looking down at the table from behind the robot.

    ``agent_helpers.calibration`` is the identity, which cannot tell image
    directions apart; the screen-facing axis convention needs a mapping that
    actually turns millimetres into a picture.
    """
    width, height = 1280, 720
    H = np.array(
        [[0.0, -mm_per_px, mm_per_px * height], [-mm_per_px, 0.0, mm_per_px * width / 2], [0.0, 0.0, 1.0]]
    )
    return PlaneCalibration(H=H, image_size=(width, height), square_mm=25.0, base_xy_mm=(0.0, 0.0))


def test_cells_reach_the_outer_rim_and_stop_at_the_base_keepout():
    cfg = AppConfig()
    grid = _grid(cfg)
    regions = cfg.agent.table_regions
    board = cfg.agent.board_grid
    cells = cells_in_workspace(grid, cfg.perception, regions, (0.0, 0.0), board)
    assert cells
    for cell in cells:
        assert point_in_workspace(cell.xy_mm, cfg.perception, (0.0, 0.0))
        assert not in_base_keepout(cell.xy_mm, (0.0, 0.0), board)
        azimuth = math.degrees(math.atan2(cell.xy_mm[1], cell.xy_mm[0]))
        r_max = workspace_radius_at_angle(cfg.perception, azimuth) - regions.edge_margin_mm
        assert math.hypot(*cell.xy_mm) <= r_max

    # the near sides are usable ground and must not be thrown away with the
    # corridor in front of the base (that is what the named points' 150mm
    # min_radius used to do here)
    addresses = {(c.x, c.y) for c in cells}
    assert {(-3, -6), (3, -6), (-2, -6), (2, -6)} <= addresses
    assert not ({(0, -6), (0, -4), (1, -4), (-1, -4)} & addresses)
    assert min(math.hypot(*c.xy_mm) for c in cells) < regions.min_radius_mm
    # every named table region is covered by some cell
    for column in regions.columns_deg:
        for row in regions.rows_fraction:
            xy = table_region_xy(column, row, cfg.perception, regions, (0.0, 0.0))
            assert min(math.dist(xy, c.xy_mm) for c in cells) <= grid.cell_mm

    span = bounds((c.x, c.y) for c in cells)
    assert span["x"][0] < 0 < span["x"][1]  # the origin sits mid-board
    assert (0, 0) in {(c.x, c.y) for c in cells}
