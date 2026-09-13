"""Camera http implementation."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from camera.cross_calibration import CrossCalibration
from camera.stream import CameraStream
from camera.web_ui import render_camera_page

BOUNDARY = "frame"


class CameraServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        cameras: dict[str, CameraStream],
        overlay: Any | None = None,
        *,
        overlay_cameras: set[str] | None = None,
        cross_calibration: CrossCalibration | None = None,
    ) -> None:
        super().__init__(server_address, CameraRequestHandler)
        self.cameras = cameras
        self.overlay = overlay
        self.overlay_cameras = overlay_cameras or set()
        self.cross_calibration = cross_calibration


class CameraRequestHandler(BaseHTTPRequestHandler):
    server: CameraServer
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if self.server.cross_calibration is None:
            self.send_error(HTTPStatus.NOT_FOUND, "cross calibration unavailable")
            return
        if path == "/tools/cross-calibration/reset":
            self._send_json(self.server.cross_calibration.reset())
            return
        if path != "/tools/cross-calibration/points":
            self.send_error(HTTPStatus.NOT_FOUND, "not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            self._send_json(
                self.server.cross_calibration.add(
                    body["marker_px"], body["reference_px"]
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, f"invalid calibration point: {exc}")

    def do_GET(self) -> None:
        request = urlparse(self.path)
        path = request.path
        color = parse_qs(request.query).get("color", [""])[0] or None
        if path in {"/", "/index.html"}:
            self._send_html()
            return
        if path == "/gripper-reference.json":
            try:
                reference_path = getattr(self.server, "gripper_reference_path", None)
                data = (
                    json.loads(Path(reference_path).read_text())
                    if reference_path
                    else {}
                )
                self._send_json(data)
            except (OSError, ValueError, TypeError):
                self._send_json({"available": False})
            return
        if path == "/health":
            data: dict[str, Any] = {
                "ok": True,
                "cameras": [camera.status() for camera in self.server.cameras.values()],
                "transport": "mjpeg-over-http",
            }
            if self.server.overlay is not None:
                data["overlay"] = self.server.overlay.status()
            if self.server.cross_calibration is not None:
                data["cross_calibration"] = self.server.cross_calibration.status()
            self._send_json(data)
            return
        if path.startswith("/snapshot/") and path.endswith(".jpg"):
            self._send_snapshot(path.removeprefix("/snapshot/").removesuffix(".jpg"))
            return
        if path == "/tools/cross-calibration.json":
            if self.server.cross_calibration is None:
                self.send_error(HTTPStatus.NOT_FOUND, "cross calibration unavailable")
            else:
                self._send_json(self.server.cross_calibration.status())
            return
        if path.startswith("/overlay/") and path.endswith(".jpg"):
            self.send_error(
                HTTPStatus.GONE,
                "Server-composited overlays were replaced by the live browser canvas",
            )
            return
        if path.startswith("/overlay-config/") and path.endswith(".json"):
            self._send_overlay_config(
                path.removeprefix("/overlay-config/").removesuffix(".json")
            )
            return
        if path.startswith("/detections/") and path.endswith(".json"):
            self._send_detections(
                path.removeprefix("/detections/").removesuffix(".json"),
                color=color,
            )
            return
        if path.startswith("/events/"):
            self._send_events(path.removeprefix("/events/"))
            return
        if path.startswith("/video/") and path.endswith(".mjpg"):
            self._send_stream(path.removeprefix("/video/").removesuffix(".mjpg"))
            return
        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.client_address[0]} - {fmt % args}")

    def _send_html(self) -> None:
        payload = render_camera_page(
            [(name, camera.device) for name, camera in self.server.cameras.items()],
            overlay_cameras=self.server.overlay_cameras,
        )
        self._send_bytes(payload, "text/html; charset=utf-8")

    def _send_json(self, data: dict[str, Any]) -> None:
        payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self._send_bytes(payload, "application/json; charset=utf-8")

    def _send_snapshot(self, name: str) -> None:
        camera = self.server.cameras.get(name)
        if camera is None:
            self.send_error(HTTPStatus.NOT_FOUND, f"unknown camera: {name}")
            return
        jpeg, frame_seq, captured_at = camera.latest_frame()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Frame-Seq", str(frame_seq))
        self.send_header("X-Captured-At", f"{captured_at:.6f}")
        self.end_headers()
        self.wfile.write(jpeg)

    def _overlay_available(self, name: str) -> bool:
        if self.server.overlay is None or name not in self.server.overlay_cameras:
            self.send_error(
                HTTPStatus.NOT_FOUND, f"overlay unavailable for camera: {name}"
            )
            return False
        return True

    def _send_overlay_config(self, name: str) -> None:
        if not self._overlay_available(name):
            return
        data = self.server.overlay.static_metadata
        data["camera"] = name
        self._send_json(data)

    def _send_detections(self, name: str, *, color: str | None) -> None:
        if not self._overlay_available(name):
            return
        self._send_json(self.server.overlay.latest(name, color=color))

    def _send_events(self, name: str) -> None:
        if not self._overlay_available(name):
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        revision = 0
        self.server.overlay.subscribe(name)
        try:
            while True:
                payload, revision = self.server.overlay.wait_for_update(
                    name, revision, timeout_s=5.0
                )
                if payload is None:
                    message = b": keepalive\n\n"
                else:
                    compact = json.dumps(
                        payload, ensure_ascii=False, separators=(",", ":")
                    ).encode("utf-8")
                    message = b"data: " + compact + b"\n\n"
                self.wfile.write(message)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        finally:
            self.server.overlay.unsubscribe(name)

    def _send_stream(self, name: str) -> None:
        camera = self.server.cameras.get(name)
        if camera is None:
            self.send_error(HTTPStatus.NOT_FOUND, f"unknown camera: {name}")
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header(
            "Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}"
        )
        self.end_headers()

        last_ts = 0.0
        while True:
            jpeg, last_ts = camera.wait_for_frame(last_ts)
            try:
                self.wfile.write(f"--{BOUNDARY}\r\n".encode("ascii"))
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                break

    def _send_bytes(self, payload: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(payload)
