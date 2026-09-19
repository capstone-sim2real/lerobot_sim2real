"""Measure the chessboard lattice on the calibrated plane.

The workspace floor is a chessboard, so its squares can address places the
operator points at ("(3, 4)로 옮겨줘"). Nothing in the repo knew where those
squares *are*, though: H maps pixels straight to robot millimetres and the
board's own origin, angle and pitch were never recorded (AGENTS.md §6 keeps
the board frame out of the control path deliberately). This tool measures
them once and stores them in the calibration file as ``board_grid``, for
display and addressing only.

Preview only (default):
    uv run python -m tools.calibrate_board_grid

Persist after inspecting the preview:
    uv run python -m tools.calibrate_board_grid --write

The lattice rides on H, so it dies with H: re-run this after any camera
move, right after re-running the homography calibration.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from camera.client import DEFAULT_SHOULDER_SNAPSHOT_URL, fetch_snapshot
from config import load_config
from perception import PlaneCalibration
from perception.board import detect_corners
from session.grid import BoardGrid, cells_in_view, cells_in_workspace

# A chessboard repeats every two squares, so nothing here recovers a global
# origin -- only the lattice. Which square is called (0, 0) stays a display
# choice, re-anchored at runtime by ``session.grid.build_grid``.
MIN_CORNERS = 30


def fit_lattice(corners_mm: np.ndarray) -> dict:
    """Fit origin/pitch/angle to corner positions already in plane mm.

    Fitting in millimetres rather than pixels means perspective is already
    undone: real squares are squares here, so one pitch and one angle
    describe the whole board.
    """
    if len(corners_mm) < MIN_CORNERS:
        raise ValueError(f"need at least {MIN_CORNERS} corners, got {len(corners_mm)}")
    distances = np.linalg.norm(corners_mm[:, None, :] - corners_mm[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    pitch = float(np.median(distances.min(axis=1)))
    if not math.isfinite(pitch) or pitch <= 0:
        raise ValueError("could not measure a square pitch from the detected corners")

    # Neighbour vectors: only the four around each corner, so diagonals
    # (pitch * sqrt(2)) never enter the angle estimate.
    near = distances <= pitch * 1.4
    rows, cols = np.nonzero(near)
    vectors = corners_mm[cols] - corners_mm[rows]
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    # A square lattice is 90-degree symmetric, so average over 4*angle and
    # divide back: +0 deg and +90 deg then reinforce instead of cancelling.
    angle = float(np.angle(np.mean(np.exp(4j * angles))) / 4.0)
    pitch = float(np.median(np.linalg.norm(vectors, axis=1)))

    e1 = np.array([math.cos(angle), math.sin(angle)])
    e2 = np.array([-math.sin(angle), math.cos(angle)])
    # Lattice phase: where the corner grid sits along each axis, as a
    # circular mean of the coordinates modulo the pitch.
    phases = []
    for axis in (e1, e2):
        projected = corners_mm @ axis
        turns = 2 * math.pi * projected / pitch
        phases.append(float(np.angle(np.mean(np.exp(1j * turns))) * pitch / (2 * math.pi)))

    origin = phases[0] * e1 + phases[1] * e2
    residual = _residual_mm(corners_mm, origin, e1 * pitch, e2 * pitch)
    return {
        "origin_mm": [float(origin[0]), float(origin[1])],
        "e1": e1,
        "e2": e2,
        "pitch_mm": pitch,
        "angle_deg": math.degrees(angle),
        "rms_mm": residual,
        "num_corners": int(len(corners_mm)),
    }


def _residual_mm(points: np.ndarray, origin: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
    det = u[0] * v[1] - u[1] * v[0]
    delta = points - origin
    a = (delta[:, 0] * v[1] - delta[:, 1] * v[0]) / det
    b = (u[0] * delta[:, 1] - u[1] * delta[:, 0]) / det
    nearest = origin + np.round(a)[:, None] * u + np.round(b)[:, None] * v
    return float(np.sqrt(np.mean(np.sum((points - nearest) ** 2, axis=1))))


def screen_axes_mm(calib: PlaneCalibration) -> tuple[np.ndarray, np.ndarray]:
    """Millimetre directions of one pixel right and one pixel *up* on screen."""
    width, height = calib.image_size
    centre = np.array([[width / 2.0, height / 2.0]], dtype=np.float64)
    probe = np.vstack([centre, centre + [1.0, 0.0], centre + [0.0, 1.0]])
    origin, right, down = calib.pixel_to_board(probe)
    return right - origin, origin - down


def orient(fit: dict, calib: PlaneCalibration) -> tuple[list[float], list[float]]:
    """Pick which lattice axis is +x (image-right) and which is +y (away).

    The lattice is 90-degree symmetric, so the fit alone cannot say. The
    operator's screen decides, which is the convention the whole UI uses.
    """
    right_mm, up_mm = screen_axes_mm(calib)
    right_mm = right_mm / np.linalg.norm(right_mm)
    up_mm = up_mm / np.linalg.norm(up_mm)
    candidates = [fit["e1"], -fit["e1"], fit["e2"], -fit["e2"]]
    u = max(candidates, key=lambda axis: float(axis @ right_mm))
    v = max((c for c in candidates if abs(float(c @ u)) < 0.5), key=lambda axis: float(axis @ up_mm))
    pitch = fit["pitch_mm"]
    return [float(u[0] * pitch), float(u[1] * pitch)], [float(v[0] * pitch), float(v[1] * pitch)]


def _annotate(
    frame: np.ndarray, calib: PlaneCalibration, grid: BoardGrid, cfg
) -> tuple[np.ndarray, int]:
    out = frame.copy()
    cells = cells_in_workspace(grid, cfg.perception, cfg.agent.table_regions,
                               calib.base_xy_mm or (0.0, 0.0), cfg.agent.board_grid)
    cells = cells_in_view(cells, lambda pts: calib.board_to_pixel(np.asarray(pts)), calib.image_size)
    for cell in cells:
        corners = calib.board_to_pixel(np.asarray(grid.cell_corners_mm(cell.x, cell.y)))
        cv2.polylines(out, [np.rint(corners).astype(np.int32)], True, (0, 165, 255), 1)
    ox, oy = (int(v) for v in np.rint(calib.board_to_pixel(np.asarray([grid.origin_mm]))[0]))
    cv2.drawMarker(out, (ox, oy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.putText(out, "(0,0)", (ox + 8, oy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return out, len(cells)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", help="saved BGR image instead of the camera snapshot")
    source.add_argument("--snapshot", default=None, help="camera snapshot URL")
    parser.add_argument("--config", default="src/configs/default.yaml")
    parser.add_argument("--calib", default=None, help="calibration JSON to update")
    parser.add_argument("--preview", default="/tmp/so101-board-grid.png")
    parser.add_argument("--write", action="store_true", help="persist board_grid after previewing")
    parser.add_argument("--max-rms-fraction", type=float, default=0.1,
                        help="refuse to write above this fit RMS, as a fraction of one square")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    calib_path = Path(args.calib or cfg.perception.calibration_path)
    calib = PlaneCalibration.load(calib_path)

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {args.image}")
    else:
        frame = fetch_snapshot(args.snapshot or cfg.perception.snapshot_url or DEFAULT_SHOULDER_SNAPSHOT_URL)

    corners_px = detect_corners(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if len(corners_px) < MIN_CORNERS:
        print(f"ERROR: only {len(corners_px)} chessboard corners found (need {MIN_CORNERS}). "
              "Check focus, lighting and framing, or keep the axis-aligned fallback "
              "(agent.board_grid.cell_mm).")
        return 2
    fit = fit_lattice(calib.pixel_to_board(corners_px))
    u_mm, v_mm = orient(fit, calib)
    grid = BoardGrid(tuple(fit["origin_mm"]), tuple(u_mm), tuple(v_mm))

    preview, cell_count = _annotate(frame, calib, grid, cfg)
    Path(args.preview).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(args.preview, preview):
        raise OSError(f"Could not write preview: {args.preview}")

    print(f"corners: {fit['num_corners']}")
    print(f"pitch:   {fit['pitch_mm']:.2f} mm  (measure a square and compare)")
    print(f"angle:   {fit['angle_deg']:+.2f} deg to the robot x axis")
    print(f"fit RMS: {fit['rms_mm']:.2f} mm  ({100 * fit['rms_mm'] / fit['pitch_mm']:.1f}% of a square)")
    print(f"+x (image right) = {u_mm[0]:+.2f}, {u_mm[1]:+.2f} mm per cell")
    print(f"+y (away)        = {v_mm[0]:+.2f}, {v_mm[1]:+.2f} mm per cell")
    print(f"cells inside the workspace band: {cell_count}")
    print(f"preview (verify the lines sit on real squares): {args.preview}")

    if not args.write:
        print("\nPreview only. Re-run with --write to store it in the calibration.")
        return 0

    # A lattice that does not fit is worse than no lattice: cells would be
    # drawn and addressed where no square is. The usual cause is a stale H
    # (the camera moved) -- re-run the homography calibration first.
    if fit["rms_mm"] > args.max_rms_fraction * fit["pitch_mm"]:
        print(f"\nREFUSED: fit RMS is {100 * fit['rms_mm'] / fit['pitch_mm']:.1f}% of a square, "
              f"above the {100 * args.max_rms_fraction:.0f}% gate. The calibration H probably no "
              "longer matches this camera -- check tools.camera_drift_check and re-run "
              "so101-calibrate. Override with --max-rms-fraction if you accept the error.")
        return 2

    calib.board_grid = {
        "origin_mm": fit["origin_mm"],
        "u_mm": u_mm,
        "v_mm": v_mm,
        "pitch_mm": fit["pitch_mm"],
        "angle_deg": fit["angle_deg"],
        "rms_mm": fit["rms_mm"],
        "num_corners": fit["num_corners"],
        "source": args.image or "camera snapshot",
        "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    calib.save(calib_path)
    print(f"\nboard_grid written to {calib_path}")
    print(json.dumps(calib.board_grid, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
