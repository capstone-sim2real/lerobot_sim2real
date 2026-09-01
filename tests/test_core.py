"""Configuration, camera geometry, detection, and target-selection contracts."""

import cv2
import numpy as np
import pytest

from camera.server import FrameRecorder
from config import AppConfig, load_config
from perception import BlockDetection, PlaneCalibration, calibrate_from_pairs, detect_blocks, select_target, target_id_for
from runners import run_task


def test_config_loads_overrides_and_rejects_unknown_key(tmp_path):
    cfg = load_config(overrides=["fsm.time_budget_s=180", "robot.cameras={}"])
    assert cfg.fsm.time_budget_s == 180.0
    assert cfg.robot.cameras == {}
    bad = tmp_path / "bad.yaml"
    bad.write_text("fsm:\n  typo_key: 1\n")
    with pytest.raises(ValueError, match="typo_key"):
        load_config(bad)


def _calibration() -> PlaneCalibration:
    pairs = [
        ((0.0, 0.0), (0.0, 0.0)), ((100.0, 0.0), (50.0, 0.0)),
        ((100.0, 100.0), (50.0, 50.0)), ((0.0, 100.0), (0.0, 50.0)),
    ]
    return PlaneCalibration(
        H=calibrate_from_pairs(pairs), image_size=(600, 400), square_mm=25.0,
        base_xy_mm=(250.0, 300.0), zone_polygon_mm=[(10.0, 10.0), (110.0, 10.0), (110.0, 60.0), (10.0, 60.0)],
    )


def test_calibration_round_trip():
    pixels = np.array([[40.0, 80.0], [500.0, 10.0]])
    calib = _calibration()
    assert calib.board_to_pixel(calib.pixel_to_board(pixels)) == pytest.approx(pixels, abs=1e-6)


def test_detector_accepts_blocks_and_rejects_same_colour_tape_in_rgb_and_bgr():
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(image, (80, 80), (120, 120), (0, 180, 0), -1)
    cv2.rectangle(image, (380, 280), (420, 320), (0, 0, 220), -1)
    cv2.rectangle(image, (180, 191), (400, 209), (0, 0, 220), -1)
    calib = PlaneCalibration(H=np.eye(3), image_size=(600, 400), square_mm=25.0)
    for frame, is_rgb in ((image, False), (cv2.cvtColor(image, cv2.COLOR_BGR2RGB), True)):
        assert {d.color for d in detect_blocks(frame, calib, AppConfig().perception, is_rgb=is_rgb)} == {"green", "red"}


def _block(color: str, x: float, y: float) -> BlockDetection:
    return BlockDetection(color, (x, y), 1600.0, 1.0, 1.0, 1.0, [])


def test_selection_excludes_zone_and_skipped_blocks_with_stable_ids():
    cfg = AppConfig().select
    inside, skipped, candidate = _block("red", 50, 30), _block("green", 220, 280), _block("blue", 240, 290)
    assert target_id_for(_block("blue", 238, 292), cfg.target_cell_mm) == target_id_for(candidate, cfg.target_cell_mm)
    result = select_target([inside, skipped, candidate], _calibration(), cfg, {target_id_for(skipped, cfg.target_cell_mm)})
    assert result.target is candidate and result.remaining == 1


def test_runner_perception_uses_http_snapshot_as_bgr(monkeypatch):
    cfg, calibration = AppConfig(), _calibration()
    candidate = _block("blue", 240, 290)
    calls = {}
    monkeypatch.setattr(run_task, "fetch_snapshot", lambda url: calls.setdefault("url", url) and np.zeros((1, 1, 3), dtype=np.uint8))
    monkeypatch.setattr(run_task, "detect_blocks", lambda frame, calib, perception, *, is_rgb: calls.setdefault("is_rgb", is_rgb) or [candidate])
    result = run_task.make_perceive(calibration, cfg)(set())
    assert result.target is candidate
    assert calls == {"url": cfg.perception.snapshot_url, "is_rgb": False}


def test_camera_frame_recorder_saves_atomically_at_the_requested_interval(tmp_path):
    recorder = FrameRecorder(tmp_path, interval_s=1.0)
    first = recorder.record("shoulder", b"one", now=100.123)
    assert first is not None and first.read_bytes() == b"one"
    assert recorder.record("shoulder", b"two", now=100.5) is None
    second = recorder.record("shoulder", b"three", now=101.124)
    assert second is not None and second.read_bytes() == b"three"
    assert recorder.saved == 2
    assert not list((tmp_path / "shoulder").glob("*.tmp"))
