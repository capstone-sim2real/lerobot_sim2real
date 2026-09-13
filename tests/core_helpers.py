"""Synthetic block and plane fixtures shared across contract tests."""

import cv2
import numpy as np

from config import AppConfig
from perception import (
    BlockDetection,
    PlaneCalibration,
    calibrate_from_pairs,
)


def _block_bgr(color: str, value: int = 170) -> tuple[int, int, int]:
    """A BGR fill matching the configured reference colour for ``color``.

    Fixtures used pure saturated ink before, which no real block is anywhere
    near; the detector now names a blob by how close it sits to the measured
    reference, so a fixture has to look like the block it stands for.
    """
    hue, sat = AppConfig().perception.color_prototypes[color][0]
    patch = np.uint8([[[hue, sat, value]]])
    b, g, r = cv2.cvtColor(patch, cv2.COLOR_HSV2BGR)[0][0]
    return (int(b), int(g), int(r))


def _calibration() -> PlaneCalibration:
    pairs = [
        ((0.0, 0.0), (0.0, 0.0)),
        ((100.0, 0.0), (50.0, 0.0)),
        ((100.0, 100.0), (50.0, 50.0)),
        ((0.0, 100.0), (0.0, 50.0)),
    ]
    return PlaneCalibration(
        H=calibrate_from_pairs(pairs),
        image_size=(600, 400),
        square_mm=25.0,
        base_xy_mm=(250.0, 300.0),
        zone_polygon_mm=[(10.0, 10.0), (110.0, 10.0), (110.0, 60.0), (10.0, 60.0)],
    )


def _block(color: str, x: float, y: float) -> BlockDetection:
    return BlockDetection(color, (x, y), 1600.0, 1.0, 1.0, 1.0, [])
