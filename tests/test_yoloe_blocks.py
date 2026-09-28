"""Geometry contract tests need no neural model, SDK or hardware."""
import cv2
import json
import numpy as np
import pytest
from config import PerceptionConfig, load_config
from perception.homography import PlaneCalibration
from perception.yoloe_blocks import hybrid_color_geometry, mask_geometry


def setup_geometry():
    return PlaneCalibration(np.eye(3), (100, 100), 1, (0, 0)), PerceptionConfig()


def test_square_and_json_contract():
    calib, cfg = setup_geometry()
    mask = np.zeros((100,100), np.uint8)
    mask[20:61, 20:61] = 1
    row = mask_geometry(mask, calib, cfg)
    assert row["center_mm"] == pytest.approx([40,40], abs=1)
    assert row["area_mm2"] == pytest.approx(1600, abs=100)
    assert row["grasp_ready"] is False
    json.dumps(row, allow_nan=False)


def test_mapping_is_robot_mm():
    calib, cfg = setup_geometry()
    calib.H = np.array([[1,0,100],[0,1,-40],[0,0,1]], dtype=float)
    mask = np.zeros((100,100), np.uint8)
    mask[20:61,20:61] = 1
    row = mask_geometry(mask, calib, cfg)
    assert row["center_mm"] == pytest.approx([140,0], abs=1)


def test_empty_and_mismatched_resolution():
    calib, cfg = setup_geometry()
    assert mask_geometry(np.zeros((100,100)), calib, cfg)["reason"] == "empty_mask"
    with pytest.raises(ValueError, match="resolution"):
        mask_geometry(np.ones((50,50)), calib, cfg)


def test_tape_rejected_and_unknown_config_rejected(tmp_path):
    calib, cfg = setup_geometry()
    mask = np.zeros((100,100), np.uint8)
    mask[20:25,10:90] = 1
    assert not mask_geometry(mask, calib, cfg)["accepted"]
    f = tmp_path / "bad.yaml"
    f.write_text("yoloe:\n  confience: 0.2\n")
    with pytest.raises(ValueError, match="confience"):
        load_config(f)


def test_non_base_calibration_rejected():
    calib, cfg = setup_geometry()
    calib.base_xy_mm = None
    with pytest.raises(ValueError, match="Robot-base"):
        mask_geometry(np.ones((100,100)), calib, cfg)


def test_hybrid_mask_separates_red_block_from_connected_red_tape():
    calib, cfg = setup_geometry()
    cfg.workspace_radius_mm = 0
    hsv = np.zeros((100, 100, 3), np.uint8)
    hsv[:] = (0, 0, 255)
    hsv[38:45, :] = (2, 168, 180)  # connected same-colour tape
    hsv[20:61, 20:61] = (2, 168, 180)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    instance = np.zeros((100, 100), np.uint8)
    instance[20:61, 20:61] = 1

    result = hybrid_color_geometry(frame, instance, calib, cfg)

    assert result["accepted"] is True
    assert result["color"] == "red"
    assert result["center_mm"] == pytest.approx([40, 40], abs=1)
    assert result["area_mm2"] == pytest.approx(1600, abs=150)
    assert result["grasp_ready"] is False
    json.dumps(result, allow_nan=False)


def test_hybrid_rejects_frame_mask_resolution_mismatch():
    calib, cfg = setup_geometry()
    with pytest.raises(ValueError, match="resolutions differ"):
        hybrid_color_geometry(
            np.zeros((99, 100, 3), np.uint8),
            np.ones((100, 100), np.uint8),
            calib,
            cfg,
        )


def test_hybrid_serializes_rejected_color_contour():
    calib, cfg = setup_geometry()
    cfg.workspace_radius_mm = 0
    hsv = np.zeros((100, 100, 3), np.uint8)
    hsv[:] = (0, 0, 255)
    hsv[35:55, 10:90] = (2, 168, 180)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    instance = np.zeros((100, 100), np.uint8)
    instance[35:55, 10:90] = 1

    result = hybrid_color_geometry(frame, instance, calib, cfg)

    assert result["accepted"] is False
    assert result["color"] == "red"
    assert result["reason"] == "aspect"


def test_tensorrt_letterbox_and_box_roundtrip():
    from perception.yoloe_tensorrt import letterbox_bgr, scale_boxes
    frame = np.zeros((720, 1280, 3), np.uint8)
    tensor, info = letterbox_bgr(frame, 640)
    assert tensor.shape == (1, 3, 640, 640)
    assert info.gain == pytest.approx(0.5)
    assert (info.pad_left, info.pad_top) == (0, 140)
    original = scale_boxes(np.array([[50, 150, 100, 200]], np.float32), info)
    assert original[0] == pytest.approx([100, 20, 200, 120])


def test_tensorrt_nms_removes_overlapping_lower_score():
    from perception.yoloe_tensorrt import nms_indices
    boxes = np.array([[0, 0, 20, 20], [1, 1, 21, 21], [50, 50, 60, 60]], np.float32)
    scores = np.array([.9, .8, .7], np.float32)
    assert nms_indices(boxes, scores, .5, 20).tolist() == [0, 2]
