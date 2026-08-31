import cv2
import numpy as np
import pytest

from pick_stack.config import PerceptionConfig
from pick_stack.perception import PlaneCalibration, detect_blocks

# identity-scale calibration: 1 px == 1 mm, so scene coords are board mm
CALIB = PlaneCalibration(H=np.eye(3), image_size=(600, 400), square_mm=25.0)

BGR = {
    "blue": (255, 0, 0),
    "green": (0, 180, 0),
    "red": (0, 0, 220),
    "yellow": (0, 230, 230),
    "wood": (125, 170, 205),
}


def make_scene():
    """Checkerboard background + 5 blocks (one rotated) + red tape stripe."""
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    for r in range(0, 400, 25):
        for c in range(0, 600, 25):
            shade = 235 if ((r + c) // 25) % 2 == 0 else 25
            img[r : r + 25, c : c + 25] = shade

    blocks = {
        "blue": (100, 80),
        "green": (300, 90),
        "yellow": (480, 100),
        "wood": (150, 300),
    }
    for color, (cx, cy) in blocks.items():
        cv2.rectangle(img, (cx - 20, cy - 20), (cx + 20, cy + 20), BGR[color], -1)

    # red block, rotated 45 degrees
    red_center = (420, 300)
    rect = ((red_center), (40, 40), 45.0)
    cv2.fillPoly(img, [cv2.boxPoints(rect).astype(np.int32)], BGR["red"])
    blocks["red"] = red_center

    # red tape: long thin stripe — same hue as the red block, different shape
    cv2.rectangle(img, (200, 191), (400, 209), BGR["red"], -1)
    return img, blocks


def test_blocks_found_tape_rejected():
    img, blocks = make_scene()
    detections = detect_blocks(img, CALIB, PerceptionConfig(), is_rgb=False)

    by_color = {d.color: d for d in detections}
    assert set(by_color) == set(blocks), f"got {sorted(by_color)}"
    assert len(detections) == 5  # exactly one per colour -> the tape stripe was rejected

    for color, (cx, cy) in blocks.items():
        det = by_color[color]
        assert det.center_mm == pytest.approx((cx, cy), abs=4)
        assert 1200 <= det.area_mm2 <= 2100
        assert det.aspect <= 1.3


def test_rgb_frames_supported():
    img_bgr, blocks = make_scene()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    detections = detect_blocks(img_rgb, CALIB, PerceptionConfig(), is_rgb=True)
    assert {d.color for d in detections} == set(blocks)


def test_area_filter_drops_odd_sizes():
    img = np.full((400, 600, 3), 235, dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (60, 60), BGR["blue"], -1)  # 10x10 mm: too small
    cv2.rectangle(img, (200, 100), (300, 200), BGR["blue"], -1)  # 100x100 mm: too big
    assert detect_blocks(img, CALIB, PerceptionConfig(), is_rgb=False) == []
