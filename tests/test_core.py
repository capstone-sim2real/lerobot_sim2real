"""Configuration, camera geometry, detection, and target-selection contracts."""

import cv2
import numpy as np
import pytest

from camera.overlay import detection_metadata, workspace_boundary_metadata
from camera.server import DEFAULT_OVERLAY_CONFIG, FrameRecorder, parse_args
from camera.vision_worker import VisionWorker
from camera.web_ui import render_camera_page
from config import AppConfig, WorkspaceBoundaryConfig, load_config
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


def test_camera_cli_enables_default_overlay_and_honours_explicit_options():
    defaults = parse_args([])
    assert defaults.host == "0.0.0.0"
    assert defaults.port == 8090
    assert defaults.overlay_config == str(DEFAULT_OVERLAY_CONFIG)
    assert defaults.no_overlay is False

    custom = parse_args(
        ["--host", "127.0.0.1", "--port", "9000", "--overlay-config", "custom.yaml"]
    )
    assert (custom.host, custom.port, custom.overlay_config) == (
        "127.0.0.1", 9000, "custom.yaml"
    )
    assert parse_args(["--no-overlay"]).no_overlay is True


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


def test_workspace_arc_is_projected_from_robot_base_and_stays_display_only():
    calibration = PlaneCalibration(
        H=np.eye(3), image_size=(600, 400), square_mm=25.0, base_xy_mm=(10.0, 20.0)
    )
    cfg = WorkspaceBoundaryConfig(
        outer_radius_mm=100.0,
        angle_min_deg=-90.0,
        angle_max_deg=90.0,
        sample_step_deg=90.0,
    )
    boundary = workspace_boundary_metadata(calibration, cfg)
    assert boundary is not None
    assert boundary["kind"] == "nominal_topdown_outer"
    np.testing.assert_allclose(
        boundary["points_px"],
        [[10.0, -80.0], [110.0, 20.0], [10.0, 120.0]],
        atol=1e-6,
    )


def test_overlay_metadata_draws_existing_box_without_changing_detection_geometry():
    calibration = PlaneCalibration(H=np.eye(3), image_size=(600, 400), square_mm=25.0)
    detection = BlockDetection(
        "green", (100.0, 120.0), 1600.0, 1.0, 1.0, 1.0,
        [(80.0, 100.0), (120.0, 100.0), (120.0, 140.0), (80.0, 140.0)],
        angle_deg=0.0,
    )
    metadata = detection_metadata(detection, calibration, AppConfig())
    assert metadata["center_px"] == pytest.approx([100.0, 120.0])
    np.testing.assert_allclose(metadata["box_px"], detection.box_mm, atol=1e-6)
    assert metadata["block_angle_deg"] == 0.0
    assert metadata["display_plan"] == "nominal_full_bias"
    assert "target_label" not in metadata


def test_camera_page_keeps_mjpeg_source_while_canvas_overlay_is_enabled():
    page = render_camera_page(
        [("shoulder", "/dev/video0")], overlay_cameras={"shoulder"}
    ).decode()
    assert 'src="/video/shoulder.mjpg"' in page
    assert 'canvas class="camera-overlay"' in page
    assert 'option value="" selected' in page
    assert "/events/" in page
    assert "/overlay/shoulder.jpg" not in page


def test_vision_worker_returns_metadata_without_blocking_the_caller():
    frame = cv2.imread("docs/calibration/p1_top.jpg")
    assert frame is not None
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok

    worker = VisionWorker(DEFAULT_OVERLAY_CONFIG)
    worker.start()
    worker.subscribe("shoulder")
    try:
        worker.submit("shoulder", encoded.tobytes(), frame_seq=7, captured_at=100.0)
        payload, revision = worker.wait_for_update("shoulder", 0, timeout_s=10.0)
        assert revision == 1
        assert payload is not None
        assert payload["frame_seq"] == 7
        assert payload["display_only"] is True
        assert "detections" in payload
    finally:
        worker.unsubscribe("shoulder")
        worker.stop()


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


def test_camera_frame_recorder_can_save_only_significant_scene_changes(tmp_path):
    def jpeg(value: int) -> bytes:
        ok, encoded = cv2.imencode(".jpg", np.full((90, 160, 3), value, dtype=np.uint8))
        assert ok
        return encoded.tobytes()

    recorder = FrameRecorder(tmp_path, interval_s=1.0, change_threshold=8.0, max_interval_s=5.0)
    assert recorder.record("shoulder", jpeg(0), now=100.0) is not None
    assert recorder.record("shoulder", jpeg(0), now=101.0) is None
    assert recorder.record("shoulder", jpeg(40), now=102.0) is not None
    assert recorder.record("shoulder", jpeg(40), now=103.0) is None
    assert recorder.record("shoulder", jpeg(40), now=107.0) is not None
    assert recorder.saved == 3


def test_detects_block_rotation_for_jaw_alignment():
    """A diamond-oriented block must report its edge angle, not 0.

    The gripper grips two faces only if it can be turned to the block's
    edges; without an angle it always meets two corners and slips.
    """
    pairs = [((0.0, 0.0), (0.0, 0.0)), ((400.0, 0.0), (400.0, 0.0)),
             ((400.0, 300.0), (400.0, 300.0)), ((0.0, 300.0), (0.0, 300.0))]
    calib = PlaneCalibration(H=calibrate_from_pairs(pairs), image_size=(400, 300),
                             square_mm=25.0, base_xy_mm=(200.0, 400.0))
    cfg = AppConfig().perception
    cfg.rectified_mm_per_px = 1.0
    for drawn in (0, 20, 45, 70):
        frame = np.full((300, 400, 3), 200, np.uint8)
        box = cv2.boxPoints(((200.0, 150.0), (40.0, 40.0), float(drawn)))
        cv2.fillPoly(frame, [box.astype(np.int32)], (60, 200, 60))
        found = detect_blocks(frame, calib, cfg, is_rgb=False)
        assert found, drawn
        # folded to [0, 90): a square grasps identically every quarter turn
        off_by = ((found[0].angle_deg - drawn) + 45.0) % 90.0 - 45.0
        assert abs(off_by) < 3.0, (drawn, found[0].angle_deg)
