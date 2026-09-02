"""Fixed target-zone geometry in the robot-base millimetre frame."""

from __future__ import annotations

import math

import cv2
import numpy as np

from config import PerceptionConfig
from perception.homography import PlaneCalibration


def point_in_zone(point_mm: tuple[float, float], calib: PlaneCalibration, margin_mm: float = 0.0) -> bool:
    """Whether a detected centre belongs to the fixed target zone."""
    if not calib.zone_polygon_mm:
        return False
    polygon = np.asarray(calib.zone_polygon_mm, dtype=np.float32)
    return cv2.pointPolygonTest(polygon, point_mm, measureDist=True) >= -margin_mm


def ordered_zone_corners(
    polygon_mm: list[tuple[float, float]], base_xy_mm: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return far-left, far-right, near-right, near-left corners."""
    points = np.asarray(polygon_mm, dtype=np.float64).reshape(-1, 2)
    if len(points) != 4:
        raise ValueError(f"Target zone needs exactly four corners, got {len(points)}")
    hull = cv2.convexHull(points.astype(np.float32), clockwise=False).reshape(-1, 2).astype(np.float64)
    if len(hull) != 4:
        raise ValueError("Target-zone corners must form a convex quadrilateral")

    edges = [(i, (i + 1) % 4, float(np.linalg.norm(hull[(i + 1) % 4] - hull[i]))) for i in range(4)]
    longest = max(edges, key=lambda item: item[2])
    opposite = edges[(longest[0] + 2) % 4]
    edge_a = (hull[longest[0]], hull[longest[1]])
    edge_b = (hull[opposite[0]], hull[opposite[1]])
    base = np.asarray(base_xy_mm, dtype=np.float64)

    def mean_radius(edge) -> float:
        return sum(float(np.linalg.norm(p - base)) for p in edge) / 2.0

    far, near = (edge_a, edge_b) if mean_radius(edge_a) >= mean_radius(edge_b) else (edge_b, edge_a)
    # In this robot-base frame +y is camera-left across the zone.
    far_left, far_right = sorted(far, key=lambda p: float(p[1]), reverse=True)
    near_points = list(near)
    near_left = min(near_points, key=lambda p: float(np.linalg.norm(p - far_left)))
    near_right = near_points[1] if np.array_equal(near_left, near_points[0]) else near_points[0]
    return far_left, far_right, near_right, near_left


def zone_slot_centres(calib: PlaneCalibration, slot_uv: list[list[float]]) -> list[tuple[float, float]]:
    if calib.base_xy_mm is None:
        raise ValueError("Calibration has no robot-base origin")
    if not calib.zone_polygon_mm:
        raise ValueError("Calibration has no target-zone polygon")
    far_left, far_right, near_right, near_left = ordered_zone_corners(
        calib.zone_polygon_mm, calib.base_xy_mm
    )
    slots: list[tuple[float, float]] = []
    for pair in slot_uv:
        if len(pair) != 2:
            raise ValueError(f"Each task1.slot_uv item must be [u, v], got {pair!r}")
        u, v = float(pair[0]), float(pair[1])
        if not (0.0 < u < 1.0 and 0.0 < v < 1.0):
            raise ValueError(f"Slot [u, v] must be strictly inside the zone, got {pair!r}")
        far_point = (1.0 - u) * far_left + u * far_right
        near_point = (1.0 - u) * near_left + u * near_right
        point = (1.0 - v) * far_point + v * near_point
        slots.append((float(point[0]), float(point[1])))
    return slots


def detect_zone_inner_polygon(
    frame_bgr: np.ndarray, calib: PlaneCalibration, cfg: PerceptionConfig
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Find the inner hole of the large red tape ring for one-time setup."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    bands = cfg.hsv_ranges.get("red")
    if not bands:
        raise ValueError("perception.hsv_ranges has no red band")
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo_h, lo_s, lo_v, hi_h, hi_s, hi_v in bands:
        mask |= cv2.inRange(hsv, (lo_h, lo_s, lo_v), (hi_h, hi_s, hi_v))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        raise RuntimeError("No red tape ring found")

    holes: list[tuple[float, np.ndarray]] = []
    for index, contour in enumerate(contours):
        parent = int(hierarchy[0, index, 3])
        if parent < 0:
            continue
        area = float(cv2.contourArea(contour))
        parent_area = float(cv2.contourArea(contours[parent]))
        if area > 0 and parent_area > area:
            holes.append((area, contour))
    if not holes:
        raise RuntimeError("Red contour has no enclosed target-zone hole")
    contour = max(holes, key=lambda item: item[0])[1]
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True).reshape(-1, 2)
    if len(approx) != 4:
        approx = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float64)
    pixels = approx.astype(np.float64).reshape(-1, 2)
    millimetres = calib.pixel_to_board(pixels)
    ordered = ordered_zone_corners(
        [tuple(p) for p in millimetres], calib.base_xy_mm or (0.0, 0.0)
    )
    ordered_mm = [tuple(float(v) for v in p) for p in ordered]
    ordered_px_array = calib.board_to_pixel(np.asarray(ordered_mm, dtype=np.float64))

    lengths = [math.dist(ordered_mm[i], ordered_mm[(i + 1) % 4]) for i in range(4)]
    long_mean = (lengths[0] + lengths[2]) / 2.0
    short_mean = (lengths[1] + lengths[3]) / 2.0
    if not (140.0 <= long_mean <= 240.0 and 60.0 <= short_mean <= 140.0):
        raise RuntimeError(
            f"Detected red hole has implausible size {long_mean:.1f} x {short_mean:.1f} mm"
        )
    ordered_px = [tuple(float(v) for v in p) for p in ordered_px_array]
    return ordered_mm, ordered_px
