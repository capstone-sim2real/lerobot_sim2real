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
