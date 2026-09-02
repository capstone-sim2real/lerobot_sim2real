#!/usr/bin/env python3
"""Serve SO-101 USB cameras as MJPEG streams over HTTP."""

from __future__ import annotations

import argparse
import json
import signal
import socket
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from camera.web_ui import render_camera_page


BOUNDARY = "frame"
DEFAULT_OVERLAY_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
FrameCallback = Callable[[str, bytes, int, float], None]


class FrameRecorder:
    """Persist periodic frames, optionally only when the scene has changed."""

    def __init__(
        self,
        directory: Path | str,
        interval_s: float,
        *,
        change_threshold: float | None = None,
        max_interval_s: float | None = None,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("save interval must be positive")
        if change_threshold is not None and change_threshold < 0:
            raise ValueError("change threshold must be non-negative")
        if max_interval_s is not None and max_interval_s <= 0:
            raise ValueError("maximum save interval must be positive")
        self._directory = Path(directory).resolve()
        self._interval_s = interval_s
        self._change_threshold = change_threshold
        self._max_interval_s = max_interval_s
        self._next_save_at: dict[str, float] = {}
        self._last_saved_at: dict[str, float] = {}
        self._previous_frames: dict[str, np.ndarray] = {}
        self._saved = 0
        self._lock = threading.Lock()

    @property
    def interval_s(self) -> float:
        return self._interval_s

    @property
    def saved(self) -> int:
        with self._lock:
            return self._saved

    @property
    def saves_on_change(self) -> bool:
        return self._change_threshold is not None

    def record(
        self,
        camera_name: str,
        jpeg: bytes,
        *,
        now: float | None = None,
    ) -> Path | None:
        now = time.time() if now is None else now
        with self._lock:
            if now < self._next_save_at.get(camera_name, 0.0):
                return None
            self._next_save_at[camera_name] = now + self._interval_s
            if self._change_threshold is not None:
                previous = self._previous_frames.get(camera_name)
                forced = (
                    self._max_interval_s is not None
                    and now - self._last_saved_at.get(camera_name, now)
                    >= self._max_interval_s
                )
                current = self._comparison_frame(jpeg)
                changed = (
                    previous is None
                    or self._change_score(previous, current) >= self._change_threshold
                )
                self._previous_frames[camera_name] = current
                if not changed and not forced:
                    return None
                self._last_saved_at[camera_name] = now

        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
        millis = int((now % 1) * 1000)
        directory = self._directory / camera_name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{stamp}_{millis:03d}.jpg"
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_bytes(jpeg)
        temporary.replace(path)
        with self._lock:
            self._saved += 1
        return path

    @staticmethod
    def _comparison_frame(jpeg: bytes) -> np.ndarray:
        frame = cv2.imdecode(
            np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
        )
        if frame is None:
            raise ValueError("cannot compare an invalid JPEG")
        frame = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(frame, (5, 5), 0)

    @staticmethod
    def _change_score(previous: np.ndarray, current: np.ndarray) -> float:
        return float(np.mean(cv2.absdiff(previous, current)))


class CameraStream:
    def __init__(
        self,
        name: str,
        device: str,
        width: int,
        height: int,
        fps: int,
        fourcc: str,
        jpeg_quality: int,
        recorder: FrameRecorder | None = None,
        frame_callback: FrameCallback | None = None,
    ) -> None:
        self.name = name
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self.jpeg_quality = jpeg_quality
        self._recorder = recorder
        self._frame_callback = frame_callback
        self._condition = threading.Condition()
        self._latest_jpeg = self._make_status_jpeg(f"{name}: starting")
        self._latest_ts = 0.0
        self._frames = 0
        self._error = ""
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._capture_loop, name=f"camera-{name}", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def status(self) -> dict[str, Any]:
        with self._condition:
            return {
                "name": self.name,
                "device": self.device,
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "frames": self._frames,
                "latest_ts": self._latest_ts,
                "age_s": (
                    round(time.time() - self._latest_ts, 3)
                    if self._latest_ts
                    else None
                ),
                "error": self._error,
                "recording": self._recorder is not None,
                "saved_frames": self._recorder.saved if self._recorder else 0,
            }

    def latest_jpeg(self) -> bytes:
        with self._condition:
            return self._latest_jpeg

    def latest_frame(self) -> tuple[bytes, int, float]:
        """Return one atomic image/sequence/timestamp snapshot."""
        with self._condition:
            return self._latest_jpeg, self._frames, self._latest_ts

    def wait_for_frame(
        self, last_ts: float, timeout: float = 2.0
    ) -> tuple[bytes, float]:
        deadline = time.time() + timeout
        with self._condition:
            while self._latest_ts <= last_ts and not self._stop.is_set():
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
            return self._latest_jpeg, self._latest_ts

    def _capture_loop(self) -> None:
        backoff_s = 1.0
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self._opencv_device())
            if not cap.isOpened():
                self._publish_error(f"{self.name}: cannot open {self.device}")
                cap.release()
                time.sleep(backoff_s)
                backoff_s = min(backoff_s * 1.5, 5.0)
                continue

            backoff_s = 1.0
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            if self.fourcc:
                cap.set(
                    cv2.CAP_PROP_FOURCC,
                    cv2.VideoWriter_fourcc(*self.fourcc[:4]),
                )

            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    self._publish_error(f"{self.name}: frame read failed")
                    break

                ok, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                )
                if not ok:
                    self._publish_error(f"{self.name}: jpeg encode failed")
                    continue

                jpeg = encoded.tobytes()
                captured_at = time.time()
                with self._condition:
                    self._latest_jpeg = jpeg
                    self._latest_ts = captured_at
                    self._frames += 1
                    frame_seq = self._frames
                    self._error = ""
                    self._condition.notify_all()

                if self._frame_callback is not None:
                    try:
                        self._frame_callback(
                            self.name, jpeg, frame_seq, captured_at
                        )
                    except Exception as exc:
                        print(f"{self.name}: overlay submit failed: {exc}")
                if self._recorder is not None:
                    try:
                        self._recorder.record(
                            self.name, jpeg, now=captured_at
                        )
                    except OSError as exc:
                        print(f"{self.name}: frame save failed: {exc}")

            cap.release()

    def _opencv_device(self) -> int | str:
        if self.device.isdigit():
            return int(self.device)
        return self.device

    def _publish_error(self, message: str) -> None:
        with self._condition:
            self._latest_jpeg = self._make_status_jpeg(message)
            self._latest_ts = time.time()
            self._error = message
            self._condition.notify_all()

    @staticmethod
    def _make_status_jpeg(message: str) -> bytes:
        frame = np.zeros((240, 640, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            message[:80],
            (24, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        ok, encoded = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        )
        return encoded.tobytes() if ok else b""


class CameraServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        cameras: dict[str, CameraStream],
        overlay: Any | None = None,
        *,
        overlay_cameras: set[str] | None = None,
    ) -> None:
        super().__init__(server_address, CameraRequestHandler)
        self.cameras = cameras
        self.overlay = overlay
        self.overlay_cameras = overlay_cameras or set()


class CameraRequestHandler(BaseHTTPRequestHandler):
    server: CameraServer
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        request = urlparse(self.path)
        path = request.path
        color = parse_qs(request.query).get("color", [""])[0] or None
        if path in {"/", "/index.html"}:
            self._send_html()
            return
        if path == "/health":
            data: dict[str, Any] = {
                "ok": True,
                "cameras": [camera.status() for camera in self.server.cameras.values()],
                "transport": "mjpeg-over-http",
            }
            if self.server.overlay is not None:
                data["overlay"] = self.server.overlay.status()
            self._send_json(data)
            return
        if path.startswith("/snapshot/") and path.endswith(".jpg"):
            self._send_snapshot(
                path.removeprefix("/snapshot/").removesuffix(".jpg")
            )
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
            self._send_stream(
                path.removeprefix("/video/").removesuffix(".mjpg")
            )
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
            self.send_error(HTTPStatus.NOT_FOUND, f"overlay unavailable for camera: {name}")
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
                self.wfile.write(
                    f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                )
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


def local_addresses() -> list[str]:
    addresses = ["127.0.0.1"]
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            address = sock.getsockname()[0]
            if address not in addresses:
                addresses.append(address)
    except OSError:
        pass
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            address = info[4][0]
            if address not in addresses:
                addresses.append(address)
    except socket.gaierror:
        pass
    return addresses


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve SO-101 cameras as a browser MJPEG page."
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="HTTP bind host. Default: 0.0.0.0"
    )
    parser.add_argument(
        "--port", type=int, default=8090, help="HTTP port. Default: 8090"
    )
    parser.add_argument(
        "--shoulder-device",
        default="/dev/video0",
        help="Shoulder camera device. Default: /dev/video0",
    )
    parser.add_argument(
        "--wrist-device", default="", help="Optional wrist camera device"
    )
    parser.add_argument(
        "--width", type=int, default=1280, help="Capture width. Default: 1280"
    )
    parser.add_argument(
        "--height", type=int, default=720, help="Capture height. Default: 720"
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Capture FPS. Default: 30"
    )
    parser.add_argument(
        "--fourcc", default="MJPG", help="Capture FOURCC. Default: MJPG"
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=80,
        help="Stream JPEG quality 1-100. Default: 80",
    )
    parser.add_argument(
        "--save-dir",
        default="",
        help="Directory for periodic JPEGs; disabled unless --save-interval-s is positive.",
    )
    parser.add_argument(
        "--save-interval-s",
        type=float,
        default=0.0,
        help="Seconds between saved frames per camera. Default: 0 (disabled).",
    )
    parser.add_argument(
        "--save-on-change",
        action="store_true",
        help="Save only when the scene differs enough from the last saved frame.",
    )
    parser.add_argument(
        "--change-threshold",
        type=float,
        default=8.0,
        help="Mean grayscale difference required by --save-on-change. Default: 8.",
    )
    parser.add_argument(
        "--max-save-interval-s",
        type=float,
        default=10.0,
        help="Force a status frame this often with --save-on-change. Default: 10.",
    )
    overlay_group = parser.add_mutually_exclusive_group()
    overlay_group.add_argument(
        "--overlay-config",
        default=str(DEFAULT_OVERLAY_CONFIG),
        help="Detection overlay project YAML. Enabled by default.",
    )
    overlay_group.add_argument(
        "--no-overlay",
        action="store_true",
        help="Disable operator perception metadata and canvas overlay.",
    )
    args = parser.parse_args(argv)
    if bool(args.save_dir) != (args.save_interval_s > 0):
        parser.error(
            "--save-dir and a positive --save-interval-s must be supplied together"
        )
    if args.save_on_change and not args.save_dir:
        parser.error("--save-on-change requires --save-dir and --save-interval-s")
    if args.save_on_change and args.max_save_interval_s <= 0:
        parser.error(
            "--max-save-interval-s must be positive with --save-on-change"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overlay = None
    if not args.no_overlay:
        from camera.vision_worker import VisionWorker

        overlay = VisionWorker(args.overlay_config)
        overlay.start()

    recorder = (
        FrameRecorder(
            args.save_dir,
            args.save_interval_s,
            change_threshold=(
                args.change_threshold if args.save_on_change else None
            ),
            max_interval_s=(
                args.max_save_interval_s if args.save_on_change else None
            ),
        )
        if args.save_dir
        else None
    )
    cameras = {
        "shoulder": CameraStream(
            "shoulder",
            args.shoulder_device,
            args.width,
            args.height,
            args.fps,
            args.fourcc,
            args.jpeg_quality,
            recorder,
            frame_callback=overlay.submit if overlay is not None else None,
        )
    }
    if args.wrist_device:
        cameras["wrist"] = CameraStream(
            "wrist",
            args.wrist_device,
            args.width,
            args.height,
            args.fps,
            args.fourcc,
            args.jpeg_quality,
            recorder,
        )

    server: CameraServer | None = None
    try:
        for camera in cameras.values():
            camera.start()
        overlay_cameras = {"shoulder"} if overlay is not None else set()
        server = CameraServer(
            (args.host, args.port),
            cameras,
            overlay,
            overlay_cameras=overlay_cameras,
        )

        def shutdown(_signum: int, _frame: Any) -> None:
            assert server is not None
            threading.Thread(target=server.shutdown, daemon=True).start()

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        print("SO-101 camera web server")
        for address in local_addresses():
            print(f"  http://{address}:{args.port}")
        print(
            "Routes: /, /health, /snapshot/<camera>.jpg, /video/<camera>.mjpg"
        )
        if overlay is not None:
            print(
                "Overlay: browser canvas + /events/<camera> + "
                "/detections/<camera>.json"
            )
        if recorder is not None:
            mode = (
                "when the scene changes" if recorder.saves_on_change else "periodically"
            )
            print(
                f"Saving JPEGs {mode}, checked every {recorder.interval_s:g}s "
                f"under {args.save_dir}"
            )
        server.serve_forever()
    finally:
        if server is not None:
            server.server_close()
        for camera in cameras.values():
            camera.stop()
        if overlay is not None:
            overlay.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
