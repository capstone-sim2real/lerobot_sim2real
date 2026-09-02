"""Block detection: colour + shape combined, never colour alone.

Runs on the homography-rectified metric view so every threshold is in mm —
independent of where the camera sits. A red block and red tape share hue but
not geometry: tape is thin/elongated/hollow at corners, a block is a filled
~40x40 mm square. The aspect/solidity/fill filters encode exactly that
(AGENTS.md §9), so red-on-red scenes resolve by form.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import PerceptionConfig
from perception.homography import PlaneCalibration


@dataclass
class BlockDetection:
    color: str
    center_mm: tuple[float, float]
    area_mm2: float
    aspect: float
    solidity: float
    fill: float
    # minAreaRect corners in board mm, for debug rendering
    box_mm: list[tuple[float, float]]


def detect_blocks(
    frame: np.ndarray,
    calib: PlaneCalibration,
    cfg: PerceptionConfig,
    *,
    is_rgb: bool = True,
) -> list[BlockDetection]:
    """Detect candidate blocks in a top-camera frame.

    ``is_rgb`` is True for frames from lerobot cameras (RGB) and False for
    images loaded with cv2.imread (BGR).
    """
    rectified, (origin_x, origin_y) = calib.rectify(frame, cfg.rectified_mm_per_px)
    hsv = cv2.cvtColor(rectified, cv2.COLOR_RGB2HSV if is_rgb else cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_kernel_px, cfg.morph_kernel_px))
    mm2_per_px2 = cfg.rectified_mm_per_px**2

    detections: list[BlockDetection] = []
    for color, bands in cfg.hsv_ranges.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo_h, lo_s, lo_v, hi_h, hi_s, hi_v in bands:
            mask |= cv2.inRange(hsv, (lo_h, lo_s, lo_v), (hi_h, hi_s, hi_v))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area_mm2 = cv2.contourArea(contour) * mm2_per_px2
            if not (cfg.area_mm2_min <= area_mm2 <= cfg.area_mm2_max):
                continue
            (cx, cy), (rw, rh), _angle = cv2.minAreaRect(contour)
            if min(rw, rh) <= 0:
                continue
            aspect = max(rw, rh) / min(rw, rh)
            if aspect > cfg.aspect_ratio_max:
                continue
            fill = (cv2.contourArea(contour)) / (rw * rh)
            if fill < cfg.fill_min:
                continue
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0:
                continue
            solidity = cv2.contourArea(contour) / hull_area
            if solidity < cfg.solidity_min:
                continue

            def to_mm(px: float, py: float) -> tuple[float, float]:
                return (
                    origin_x + px * cfg.rectified_mm_per_px,
                    origin_y + py * cfg.rectified_mm_per_px,
                )

            box_px = cv2.boxPoints(((cx, cy), (rw, rh), _angle))
            detections.append(
                BlockDetection(
                    color=color,
                    center_mm=to_mm(cx, cy),
                    area_mm2=float(area_mm2),
                    aspect=float(aspect),
                    solidity=float(solidity),
                    fill=float(fill),
                    box_mm=[to_mm(px, py) for px, py in box_px],
                )
            )
    detections.sort(key=lambda d: (d.center_mm[1], d.center_mm[0]))
    return detections
