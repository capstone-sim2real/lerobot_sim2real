"""Camera contracts."""

import cv2
import numpy as np
import pytest

from camera.overlay import (
    detection_metadata,
    target_zone_metadata,
    workspace_boundary_metadata,
)
from camera.client import fetch_snapshot_with_metadata
from camera.server import (
    CrossCalibration,
    DEFAULT_OVERLAY_CONFIG,
    FrameRecorder,
    parse_args,
)
from camera.vision_worker import VisionWorker
from camera.web_ui import render_camera_page
from config import AppConfig, PerceptionConfig, WorkspaceBoundaryConfig
from perception import (
    BlockDetection,
    PlaneCalibration,
)
from perception import detector
from runners import run_task


from pathlib import Path
from core_helpers import _calibration, _block


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
        "127.0.0.1",
        9000,
        "custom.yaml",
    )
    assert parse_args(["--no-overlay"]).no_overlay is True


def test_snapshot_client_requires_and_decodes_freshness_headers(monkeypatch):
    image = np.full((10, 12, 3), 80, np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok

    class Response:
        headers = {"X-Frame-Seq": "42", "X-Captured-At": "1234.5"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return encoded.tobytes()

    monkeypatch.setattr("camera.client.urlopen", lambda *_args, **_kwargs: Response())
    snapshot = fetch_snapshot_with_metadata("http://camera/snapshot.jpg")
    assert snapshot.frame_seq == 42
    assert snapshot.captured_at == 1234.5
    assert snapshot.frame.shape == image.shape


def test_workspace_arc_is_projected_from_robot_base_and_matches_the_detector():
    calibration = PlaneCalibration(
        H=np.eye(3), image_size=(600, 400), square_mm=25.0, base_xy_mm=(10.0, 20.0)
    )
    perception = PerceptionConfig(
        workspace_radius_mm=100.0,
        workspace_angle_min_deg=-90.0,
        workspace_angle_max_deg=90.0,
        workspace_radius_by_angle_mm=[],
    )
    boundary = workspace_boundary_metadata(
        calibration, WorkspaceBoundaryConfig(sample_step_deg=90.0), perception
    )
    assert boundary is not None
    assert boundary["kind"] == "nominal_topdown_outer"
    np.testing.assert_allclose(
        boundary["points_px"],
        [[10.0, -80.0], [110.0, 20.0], [10.0, 120.0]],
        atol=1e-6,
    )
    # radial legs run from the robot base to the two arc ends, closing the sector
    np.testing.assert_allclose(boundary["base_px"], [10.0, 20.0], atol=1e-6)
    # the drawn arc reports the detector's own gate, not a separate number
    assert boundary["radius_mm"] == perception.workspace_radius_mm
    assert boundary["angle_min_deg"] == perception.workspace_angle_min_deg
    assert boundary["angle_max_deg"] == perception.workspace_angle_max_deg


def test_default_workspace_arc_spans_the_full_half_plane():
    cfg = AppConfig().perception
    assert (cfg.workspace_angle_min_deg, cfg.workspace_angle_max_deg) == (-90.0, 90.0)


def test_workspace_overlay_uses_same_angle_dependent_radii_as_detector():
    calibration = PlaneCalibration(
        H=np.eye(3), image_size=(600, 400), square_mm=25.0, base_xy_mm=(0.0, 0.0)
    )
    perception = PerceptionConfig(
        workspace_radius_mm=320.0,
        workspace_angle_min_deg=-90.0,
        workspace_angle_max_deg=90.0,
        workspace_radius_by_angle_mm=[[-90.0, 250.0], [0.0, 320.0], [90.0, 270.0]],
    )
    boundary = workspace_boundary_metadata(
        calibration, WorkspaceBoundaryConfig(sample_step_deg=90.0), perception
    )
    assert boundary["kind"] == "ik_reach_envelope"
    assert boundary["label"] == "IK reach"
    np.testing.assert_allclose(
        boundary["points_px"], [[0.0, -250.0], [320.0, 0.0], [0.0, 270.0]], atol=1e-6
    )
    for point in boundary["points_px"]:
        assert detector._in_workspace(tuple(point), perception, (0.0, 0.0))


def test_target_zone_overlay_uses_the_detector_exclusion_polygon():
    calibration = _calibration()
    zone = target_zone_metadata(calibration)
    assert zone is not None and zone["kind"] == "excluded_target_zone"
    np.testing.assert_allclose(zone["points_mm"], calibration.zone_polygon_mm)
    np.testing.assert_allclose(
        zone["points_px"],
        calibration.board_to_pixel(np.asarray(calibration.zone_polygon_mm)),
    )


def test_overlay_metadata_draws_existing_box_without_changing_detection_geometry():
    calibration = PlaneCalibration(H=np.eye(3), image_size=(600, 400), square_mm=25.0)
    detection = BlockDetection(
        "green",
        (100.0, 120.0),
        1600.0,
        1.0,
        1.0,
        1.0,
        [(80.0, 100.0), (120.0, 100.0), (120.0, 140.0), (80.0, 140.0)],
        angle_deg=0.0,
    )
    metadata = detection_metadata(detection, calibration, AppConfig())
    assert metadata["center_px"] == pytest.approx([100.0, 120.0])
    np.testing.assert_allclose(metadata["box_px"], detection.box_mm, atol=1e-6)
    assert metadata["block_angle_deg"] == 0.0
    assert "grasp_yaw_deg" in metadata
    assert len(metadata["grasp_axis_px"]) == 2
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


def test_cross_calibration_fits_marker_pixels_to_table_points(tmp_path):
    session = CrossCalibration(DEFAULT_OVERLAY_CONFIG, tmp_path / "cross.json")
    for marker, reference in [
        ([10, 10], [100, 100]),
        ([40, 10], [300, 80]),
        ([40, 30], [280, 260]),
        ([10, 30], [90, 220]),
    ]:
        state = session.add(marker, reference)
    assert state["ready"] is True
    assert state["rms_mm"] < 0.001
    assert len(state["marker_to_table_h"]) == 3
    reloaded = CrossCalibration(DEFAULT_OVERLAY_CONFIG, tmp_path / "cross.json")
    assert len(reloaded.status()["samples"]) == 4


def test_vision_worker_returns_metadata_without_blocking_the_caller():
    frame = cv2.imread(str(Path(__file__).parent / "fixtures" / "p1_top.jpg"))
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


def test_runner_perception_uses_http_snapshot_as_bgr(monkeypatch):
    cfg, calibration = AppConfig(), _calibration()
    candidate = _block("blue", 240, 290)
    calls = {}
    monkeypatch.setattr(
        run_task,
        "fetch_snapshot",
        lambda url: (
            calls.setdefault("url", url) and np.zeros((1, 1, 3), dtype=np.uint8)
        ),
    )
    monkeypatch.setattr(
        run_task,
        "detect_blocks",
        lambda frame, calib, perception, *, is_rgb: (
            calls.setdefault("is_rgb", is_rgb) or [candidate]
        ),
    )
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

    recorder = FrameRecorder(
        tmp_path, interval_s=1.0, change_threshold=8.0, max_interval_s=5.0
    )
    assert recorder.record("shoulder", jpeg(0), now=100.0) is not None
    assert recorder.record("shoulder", jpeg(0), now=101.0) is None
    assert recorder.record("shoulder", jpeg(40), now=102.0) is not None
    assert recorder.record("shoulder", jpeg(40), now=103.0) is None
    assert recorder.record("shoulder", jpeg(40), now=107.0) is not None
    assert recorder.saved == 3
