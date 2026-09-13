"""HTTP contracts after separating service assembly from request handling."""

import json
import threading
from urllib.request import urlopen

from camera.http import CameraServer


def test_fk_endpoint_serves_publisher_file_and_handles_missing_file(tmp_path):
    server = CameraServer(("127.0.0.1", 0), {})
    server.gripper_reference_path = tmp_path / "fk.json"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/gripper-reference.json"
        with urlopen(url, timeout=3) as response:
            assert json.load(response) == {"available": False}
        payload = dict(available=True, xyz_mm=[1, 2, 3], measured_at=10)
        server.gripper_reference_path.write_text(json.dumps(payload))
        with urlopen(url, timeout=3) as response:
            assert json.load(response) == payload
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_camera_server_retains_legacy_class_imports():
    from camera import server, http, stream, recorder, cross_calibration

    assert server.CameraRequestHandler is http.CameraRequestHandler
    assert server.CameraStream is stream.CameraStream
    assert server.FrameRecorder is recorder.FrameRecorder
    assert server.CrossCalibration is cross_calibration.CrossCalibration
