"""Chessboard corner extraction for the fixed top-down camera.

The workspace floor *is* the chessboard, so it is a permanent fiducial: the
board and the robot are both bolted to the table, so their relative geometry
never changes. Only the camera can move. That makes the detected corner set
the reference for two things:

  - drift detection (tools/camera_drift_check.py), and
  - re-fitting the pixel->robot homography when the camera has shifted,
    without redoing the manual FK point procedure (AGENTS.md §6, §8).

``findChessboardCornersSB`` only returns a *complete rectangular* grid, so the
arm parked in the camera's view splits the board and costs every corner below
it — measured at 35% of the board on the team's rig. ``detect_corners`` works
around that by detecting over overlapping tiles and merging, which needs no
globally consistent grid indexing: nothing downstream uses the grid indices,
only the corner pixel positions and their stored robot-frame coordinates.
"""

from __future__ import annotations

import cv2
import numpy as np

# CALIB_CB_LARGER lets a tile return a grid bigger than the requested minimum,
# so one coarse min_pattern works for both the full frame and small tiles.
_SB_FLAGS = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_LARGER

# (x0_frac, y0_frac, x1_frac, y1_frac) tiles, overlapping so a corner falling on
# a tile seam is still fully inside a neighbouring tile.
DEFAULT_TILES: tuple[tuple[float, float, float, float], ...] = (
    (0.0, 0.0, 1.0, 1.0),      # whole frame first: best-conditioned when it works
    (0.0, 0.0, 0.6, 0.7),
    (0.4, 0.0, 1.0, 0.7),
    (0.0, 0.3, 0.6, 1.0),
    (0.4, 0.3, 1.0, 1.0),
    (0.0, 0.5, 0.45, 1.0),     # left of a centre-parked arm
    (0.55, 0.5, 1.0, 1.0),     # right of a centre-parked arm
)


def _detect_tile(gray: np.ndarray, min_pattern: tuple[int, int]) -> np.ndarray | None:
    try:
        found, corners, _meta = cv2.findChessboardCornersSBWithMeta(gray, min_pattern, _SB_FLAGS)
    except cv2.error:
        return None
    if not found or corners is None or len(corners) < 4:
        return None
    return corners.reshape(-1, 2).astype(np.float64)


def detect_corners(
    gray: np.ndarray,
    *,
    min_pattern: tuple[int, int] = (4, 4),
    tiles: tuple[tuple[float, float, float, float], ...] = DEFAULT_TILES,
    merge_tol_px: float = 3.0,
    edge_margin_px: float = 3.0,
) -> np.ndarray:
    """Return (N, 2) chessboard corner pixels, merged across overlapping tiles.

    Corners closer than ``merge_tol_px`` are treated as the same corner (the
    first detection wins). Corners within ``edge_margin_px`` of the frame
    border are dropped: they are poorly localised and move in and out of view
    as the camera drifts, which would corrupt drift matching.
    """
    if gray.ndim != 2:
        raise ValueError(f"detect_corners expects a grayscale image, got shape {gray.shape}")
    h, w = gray.shape[:2]
    found: list[np.ndarray] = []
    for x0f, y0f, x1f, y1f in tiles:
        x0, y0 = int(x0f * w), int(y0f * h)
        x1, y1 = int(x1f * w), int(y1f * h)
        if x1 - x0 < 40 or y1 - y0 < 40:
            continue
        pts = _detect_tile(gray[y0:y1, x0:x1], min_pattern)
        if pts is not None:
            found.append(pts + np.array([x0, y0], dtype=np.float64))
    if not found:
        return np.empty((0, 2), dtype=np.float64)

    merged: list[np.ndarray] = []
    for pts in found:
        for p in pts:
            if merged and np.min(np.linalg.norm(np.asarray(merged) - p, axis=1)) < merge_tol_px:
                continue
            merged.append(p)
    out = np.asarray(merged, dtype=np.float64)
    keep = (
        (out[:, 0] >= edge_margin_px)
        & (out[:, 1] >= edge_margin_px)
        & (out[:, 0] <= w - 1 - edge_margin_px)
        & (out[:, 1] <= h - 1 - edge_margin_px)
    )
    out = out[keep]
    return out[np.lexsort((out[:, 0], out[:, 1]))]


def match_corners(
    reference_px: np.ndarray, current_px: np.ndarray, max_match_px: float
) -> tuple[np.ndarray, np.ndarray]:
    """Pair each reference corner with its nearest current corner.

    Returns ``(ref_matched, cur_matched)``. Pairs further apart than
    ``max_match_px`` are dropped: beyond roughly half a chessboard square the
    nearest neighbour may be the *wrong* corner, because a chessboard repeats
    every two squares and so has no globally unique origin (AGENTS.md §6).
    """
    if len(reference_px) == 0 or len(current_px) == 0:
        empty = np.empty((0, 2), dtype=np.float64)
        return empty, empty
    d = np.linalg.norm(reference_px[:, None, :] - current_px[None, :, :], axis=2)
    nearest = np.argmin(d, axis=1)
    within = d[np.arange(len(reference_px)), nearest] <= max_match_px
    return reference_px[within], current_px[nearest[within]]


def median_square_px(corners_px: np.ndarray) -> float:
    """Median nearest-neighbour spacing — one chessboard square, in pixels.

    Used to scale the drift matching radius to the rig instead of hard-coding
    a pixel budget that only holds at one camera distance.
    """
    if len(corners_px) < 2:
        return float("nan")
    d = np.linalg.norm(corners_px[:, None, :] - corners_px[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(np.median(np.min(d, axis=1)))
