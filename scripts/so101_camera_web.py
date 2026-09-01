#!/usr/bin/env python3
"""Serve SO-101 USB cameras as MJPEG streams over HTTP."""

from __future__ import annotations

import argparse
import html
import json
import signal
import socket
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np


BOUNDARY = "frame"


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
    ) -> None:
        self.name = name
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self.jpeg_quality = jpeg_quality
        self._condition = threading.Condition()
        self._latest_jpeg = self._make_status_jpeg(f"{name}: starting")
        self._latest_ts = 0.0
        self._frames = 0
        self._error = ""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._capture_loop, name=f"camera-{name}", daemon=True)

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
                "age_s": round(time.time() - self._latest_ts, 3) if self._latest_ts else None,
                "error": self._error,
            }

    def latest_jpeg(self) -> bytes:
        with self._condition:
            return self._latest_jpeg

    def wait_for_frame(self, last_ts: float, timeout: float = 2.0) -> tuple[bytes, float]:
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
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc[:4]))

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

                with self._condition:
                    self._latest_jpeg = encoded.tobytes()
                    self._latest_ts = time.time()
                    self._frames += 1
                    self._error = ""
                    self._condition.notify_all()

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
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return b""
        return encoded.tobytes()


class CameraServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], cameras: dict[str, CameraStream]) -> None:
        super().__init__(server_address, CameraRequestHandler)
        self.cameras = cameras


class CameraRequestHandler(BaseHTTPRequestHandler):
    server: CameraServer

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._send_html()
            return
        if path == "/health":
            self._send_json({"ok": True, "cameras": [cam.status() for cam in self.server.cameras.values()]})
            return
        if path.startswith("/snapshot/") and path.endswith(".jpg"):
            self._send_snapshot(path.removeprefix("/snapshot/").removesuffix(".jpg"))
            return
        if path.startswith("/video/") and path.endswith(".mjpg"):
            self._send_stream(path.removeprefix("/video/").removesuffix(".mjpg"))
            return
        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.client_address[0]} - {fmt % args}")

    def _send_html(self) -> None:
        host = self.headers.get("Host", "localhost")
        camera_tiles = "\n".join(
            f"""
            <section class="camera">
              <header>
                <h2>{html.escape(name)}</h2>
                <code>{html.escape(camera.device)}</code>
              </header>
              <img src="/video/{html.escape(name)}.mjpg" alt="{html.escape(name)} camera stream">
            </section>
            """
            for name, camera in self.server.cameras.items()
        )
        body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SO-101 Cameras</title>
  <style>
    :root {{
      color-scheme: light dark;
      font-family: Arial, sans-serif;
      background: #111;
      color: #f5f5f5;
    }}
    body {{
      margin: 0;
      padding: 24px;
    }}
    main {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
      max-width: 1400px;
      margin: 0 auto;
    }}
    .topbar {{
      max-width: 1400px;
      margin: 0 auto 16px;
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 16px;
    }}
    h1, h2 {{
      margin: 0;
      font-weight: 600;
    }}
    h1 {{
      font-size: 22px;
    }}
    h2 {{
      font-size: 16px;
    }}
    code {{
      color: #b8d7ff;
      word-break: break-all;
    }}
    .camera {{
      border: 1px solid #333;
      border-radius: 8px;
      overflow: hidden;
      background: #1b1b1b;
    }}
    .camera header {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 12px;
      border-bottom: 1px solid #333;
    }}
    img {{
      display: block;
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: contain;
      background: #000;
    }}
  </style>
</head>
<body>
  <div class="topbar">
    <h1>SO-101 Cameras</h1>
    <code>http://{html.escape(host)}</code>
  </div>
  <main>
    {camera_tiles}
  </main>
</body>
</html>
"""
        self._send_bytes(body.encode("utf-8"), "text/html; charset=utf-8")

    def _send_json(self, data: dict[str, Any]) -> None:
        payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self._send_bytes(payload, "application/json; charset=utf-8")

    def _send_snapshot(self, name: str) -> None:
        camera = self.server.cameras.get(name)
        if camera is None:
            self.send_error(HTTPStatus.NOT_FOUND, f"unknown camera: {name}")
            return
        self._send_bytes(camera.latest_jpeg(), "image/jpeg")

    def _send_stream(self, name: str) -> None:
        camera = self.server.cameras.get(name)
        if camera is None:
            self.send_error(HTTPStatus.NOT_FOUND, f"unknown camera: {name}")
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
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
            except (BrokenPipeError, ConnectionResetError):
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve SO-101 cameras as a browser MJPEG page.")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP bind host. Default: 0.0.0.0")
    parser.add_argument("--port", type=int, default=8090, help="HTTP port. Default: 8090")
    parser.add_argument("--shoulder-device", default="/dev/video0", help="Shoulder camera device. Default: /dev/video0")
    parser.add_argument("--wrist-device", default="/dev/video2", help="Wrist camera device. Default: /dev/video2")
    parser.add_argument("--width", type=int, default=1280, help="Capture width. Default: 1280")
    parser.add_argument("--height", type=int, default=720, help="Capture height. Default: 720")
    parser.add_argument("--fps", type=int, default=30, help="Capture FPS. Default: 30")
    parser.add_argument("--fourcc", default="MJPG", help="Capture FOURCC. Default: MJPG")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="Stream JPEG quality 1-100. Default: 80")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cameras = {
        "shoulder": CameraStream(
            "shoulder",
            args.shoulder_device,
            args.width,
            args.height,
            args.fps,
            args.fourcc,
            args.jpeg_quality,
        ),
        "wrist": CameraStream(
            "wrist",
            args.wrist_device,
            args.width,
            args.height,
            args.fps,
            args.fourcc,
            args.jpeg_quality,
        ),
    }

    for camera in cameras.values():
        camera.start()

    server = CameraServer((args.host, args.port), cameras)

    def shutdown(_signum: int, _frame: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("SO-101 camera web server")
    for address in local_addresses():
        print(f"  http://{address}:{args.port}")
    print("Routes: /, /health, /video/shoulder.mjpg, /video/wrist.mjpg")

    try:
        server.serve_forever()
    finally:
        server.server_close()
        for camera in cameras.values():
            camera.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
