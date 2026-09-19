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


def test_origin_is_the_reference_region_and_axes_face_the_operator():
    cfg = AppConfig()
    grid = _grid(cfg)
    anchor = table_region_xy("center", "near", cfg.perception, cfg.agent.table_regions, (0.0, 0.0))
    # (0, 0) is the cell on the reference region, not some lattice corner
    assert math.dist(grid.cell_to_xy(0, 0), anchor) < cfg.agent.board_grid.cell_mm

    origin_px, right_px, away_px = _top_down_calibration().board_to_pixel(
        np.asarray([grid.cell_to_xy(0, 0), grid.cell_to_xy(1, 0), grid.cell_to_xy(0, 1)])
    )
    assert right_px[0] > origin_px[0]  # +x is image right
    assert away_px[1] < origin_px[1]  # +y is image up
    assert math.hypot(*grid.cell_to_xy(0, 1)) > math.hypot(*grid.cell_to_xy(0, 0))  # ... = away


def test_snapping_the_origin_never_moves_the_squares():
    grid = axis_aligned((100.0, 0.0), 25.0)
    moved = grid.snap_origin_to((171.8, 13.0))
    assert moved.u_mm == grid.u_mm and moved.v_mm == grid.v_mm
    # the new origin is a cell of the original lattice
    assert grid.xy_to_cell(moved.origin_mm) == (
        round(grid._coefficients(moved.origin_mm)[0]),
        round(grid._coefficients(moved.origin_mm)[1]),
    )
    assert math.dist(grid.cell_to_xy(*grid.xy_to_cell(moved.origin_mm)), moved.origin_mm) < 1e-9


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


def test_a_measured_lattice_is_used_as_measured():
    cfg = AppConfig()
    angle = math.radians(7.0)
    pitch = 24.0
    measured = {
        "origin_mm": [13.0, -6.0],
        "u_mm": [pitch * math.sin(angle), -pitch * math.cos(angle)],
        "v_mm": [pitch * math.cos(angle), pitch * math.sin(angle)],
    }
    grid = build_grid(measured, cfg.agent.board_grid, (171.8, 0.0))
    assert grid.cell_mm == pytest.approx(pitch)
    # the origin moved onto the anchor's cell, but stayed on the measured lattice
    plain = BoardGrid(tuple(measured["origin_mm"]), tuple(measured["u_mm"]), tuple(measured["v_mm"]))
    assert math.dist(plain.cell_to_xy(*plain.xy_to_cell(grid.origin_mm)), grid.origin_mm) < 1e-9


def test_payload_cells_carry_pixels_and_zone_flags():
    skills, _world, _robot = make_skills({})
    payload = places_payload(skills)
    grid = payload["grid"]
    assert grid["measured"] is False and grid["cells"]
    assert len(payload["sector_px"]["arc"]) > 2
    for cell in grid["cells"]:
        assert len(cell["corners_px"]) == 4
    # the zone is inside the band, so some cells must be flagged as its own
    assert any(cell["in_zone"] for cell in grid["cells"])
    assert {(c["x"], c["y"]) for c in grid["cells"]} == set(skills.cells)
    # the page shows one layer at a time and starts on the named points
    assert grid["default_layer"] == "regions"
    assert payload["regions"]


def test_cells_the_camera_cannot_see_are_not_offered():
    cfg = AppConfig()
    grid = _grid(cfg)
    cells = cells_in_workspace(grid, cfg.perception, cfg.agent.table_regions, (0.0, 0.0),
                               cfg.agent.board_grid)
    calib = _top_down_calibration()
    to_pixel = lambda points: calib.board_to_pixel(np.asarray(points))  # noqa: E731

    seen = cells_in_view(cells, to_pixel, calib.image_size)
    assert 0 < len(seen) <= len(cells)
    width, height = calib.image_size
    for cell in seen:
        u, v = to_pixel([cell.xy_mm])[0]
        assert 0 <= u < width and 0 <= v < height

    # a frame cropped to the left half drops every cell on the right
    cropped = cells_in_view(cells, to_pixel, (width // 2, height))
    assert {(c.x, c.y) for c in cropped} < {(c.x, c.y) for c in seen}
    assert all(to_pixel([c.xy_mm])[0][0] < width // 2 for c in cropped)
