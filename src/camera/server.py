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
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np


BOUNDARY = "frame"


class FrameRecorder:
    """Persist the newest frame from each camera at a bounded interval."""

    def __init__(self, directory: Path | str, interval_s: float) -> None:
        if interval_s <= 0:
            raise ValueError("save interval must be positive")
        # The IK preview temporarily changes the process cwd while loading
        # its URDF. Recording must not follow that transient cwd.
        self._directory = Path(directory).resolve()
        self._interval_s = interval_s
        self._next_save_at: dict[str, float] = {}
        self._saved = 0
        self._lock = threading.Lock()

    @property
    def interval_s(self) -> float:
        return self._interval_s

    @property
    def saved(self) -> int:
        with self._lock:
            return self._saved

    def record(self, camera_name: str, jpeg: bytes, *, now: float | None = None) -> Path | None:
        now = time.time() if now is None else now
        with self._lock:
            if now < self._next_save_at.get(camera_name, 0.0):
                return None
            self._next_save_at[camera_name] = now + self._interval_s

        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
        millis = int((now % 1) * 1000)
        directory = self._directory / camera_name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{stamp}_{millis:03d}.jpg"
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_bytes(jpeg)
        temporary.replace(path)  # readers never see a partially-written JPEG
        with self._lock:
            self._saved += 1
        return path


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
    ) -> None:
        self.name = name
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self.jpeg_quality = jpeg_quality
        self._recorder = recorder
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
                "recording": self._recorder is not None,
                "saved_frames": self._recorder.saved if self._recorder else 0,
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

                if self._recorder is not None:
                    try:
                        self._recorder.record(self.name, encoded.tobytes(), now=self._latest_ts)
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
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return b""
        return encoded.tobytes()


class CameraServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], cameras: dict[str, CameraStream], overlay: Any | None = None) -> None:
        super().__init__(server_address, CameraRequestHandler)
        self.cameras = cameras
        self.overlay = overlay


class CameraRequestHandler(BaseHTTPRequestHandler):
    server: CameraServer

    def do_GET(self) -> None:
        request = urlparse(self.path)
        path = request.path
        color = parse_qs(request.query).get("color", [""])[0] or None
        if path in {"/", "/index.html"}:
            self._send_html()
            return
        if path == "/health":
            self._send_json({"ok": True, "cameras": [cam.status() for cam in self.server.cameras.values()]})
            return
        if path.startswith("/snapshot/") and path.endswith(".jpg"):
            self._send_snapshot(path.removeprefix("/snapshot/").removesuffix(".jpg"))
            return
        if path.startswith("/overlay/") and path.endswith(".jpg"):
            self._send_overlay(path.removeprefix("/overlay/").removesuffix(".jpg"), color=color)
            return
        if path.startswith("/detections/") and path.endswith(".json"):
            self._send_detections(path.removeprefix("/detections/").removesuffix(".json"), color=color)
            return
        if path.startswith("/video/") and path.endswith(".mjpg"):
            self._send_stream(path.removeprefix("/video/").removesuffix(".mjpg"))
            return
        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.client_address[0]} - {fmt % args}")

    def _send_html(self) -> None:
        camera_tiles = "\n".join(
            f"""
            <section class="camera-layout">
              <div class="camera-view">
                <img class="camera-image" data-camera="{html.escape(name)}" data-mode="raw" src="/video/{html.escape(name)}.mjpg" alt="{html.escape(name)} camera stream">
                <div class="camera-info">{html.escape(name)} · {html.escape(camera.device)}</div>
              </div>
              {"""<aside class="legend" aria-label="overlay legend">
                <div class="legend-title">Overlay</div>
                <div class="key"><i class="cyan"></i><b>C</b><span>detected centre</span></div>
                <div class="key"><i class="orange"></i><b>B</b><span>bias-corrected centre</span></div>
                <div class="key"><i class="magenta"></i><b>FL / FR</b><span>front retries</span></div>
                <div class="key"><i class="magenta"></i><b>BL / BR</b><span>back retries</span></div>
                <div class="key"><i class="red"></i><b>T</b><span>first IK-reachable target</span></div>
                <section class="details">Select an overlay to show live values.</section>
              </aside>""" if self.server.overlay is not None else ""}
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
      max-width: 1280px;
      margin: 0 auto;
    }}
    .topbar {{
      max-width: 1280px;
      margin: 0 auto 16px;
      display: flex;
      align-items: center;
      justify-content: flex-end;
    }}
    .camera-layout {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 210px;
      gap: 20px;
      align-items: start;
    }}
    img {{
      display: block;
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: contain;
      background: #000;
    }}
    .camera-info {{
      padding: 8px 0 0;
      color: #a8a8a8;
      font-size: 13px;
    }}
    .legend {{
      display: grid;
      gap: 9px;
      color: #b0b0b0;
      font-size: 13px;
      line-height: 1.3;
    }}
    .legend-title {{ color: #f5f5f5; font-size: 14px; font-weight: 600; margin-bottom: 3px; }}
    .key {{ display: grid; grid-template-columns: 10px 58px 1fr; gap: 8px; align-items: center; }}
    .legend b {{ color: #f5f5f5; font-weight: 600; }}
    .key i {{ width: 9px; height: 9px; border-radius: 50%; display: block; }}
    .key .cyan {{ background: #00ffff; }}
    .key .orange {{ background: #ffa500; }}
    .key .magenta {{ background: #ff00ff; }}
    .key .red {{ background: #ff0000; }}
    .details {{
      display: grid;
      gap: 6px;
      margin-top: 8px;
      padding-top: 12px;
      border-top: 1px solid #444;
      color: #a8a8a8;
    }}
    .detail-block {{ display: grid; gap: 3px; }}
    .detail-block + .detail-block {{ padding-top: 8px; border-top: 1px solid #333; }}
    .detail-name {{ color: #f5f5f5; font-weight: 600; text-transform: capitalize; }}
    .detail-row {{ display: grid; grid-template-columns: 44px 1fr; gap: 8px; }}
    select {{
      color: #f5f5f5;
      background: #111;
      border: 1px solid #666;
      padding: 5px 7px;
      font: inherit;
    }}
    @media (max-width: 760px) {{
      .camera-layout {{ grid-template-columns: 1fr; }}
      .legend {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .legend-title, .details {{ grid-column: 1 / -1; }}
    }}
  </style>
</head>
<body>
  <div class="topbar">
    {"""<select id="overlay-color" aria-label="overlay colour">
      <option value="none" selected>none</option><option value="">all</option>
      <option value="green">green</option>
      <option value="yellow">yellow</option><option value="blue">blue</option>
      <option value="red">red</option><option value="wood">wood</option>
    </select>""" if self.server.overlay is not None else ""}
  </div>
  <main>
    {camera_tiles}
  </main>
  <script>
    const pointText = (point) => `(${{point.map((value) => Number(value).toFixed(1)).join(', ')}}) mm`;
    const addDetailRow = (parent, name, value) => {{
      const row = document.createElement('div');
      row.className = 'detail-row';
      const key = document.createElement('span');
      key.textContent = name;
      const text = document.createElement('span');
      text.textContent = value;
      row.append(key, text);
      parent.append(row);
    }};
    let detailsInFlight = false;
    let detailsColor = '';
    let lastDetailsAt = 0;
    const refreshDetails = async (color) => {{
      const panel = document.querySelector('.details');
      if (!panel) return;
      if (color === 'none' || color === undefined) {{
        panel.textContent = 'Select an overlay to show live values.';
        return;
      }}
      const now = Date.now();
      if (detailsInFlight || (detailsColor === color && now - lastDetailsAt < 2000)) return;
      detailsInFlight = true;
      detailsColor = color;
      lastDetailsAt = now;
      const camera = document.querySelector('.camera-image')?.dataset.camera;
      if (!camera) return;
      try {{
        const response = await fetch(`/detections/${{camera}}.json?color=${{encodeURIComponent(color)}}`);
        if (!response.ok) throw new Error('request failed');
        const data = await response.json();
        panel.replaceChildren();
        if (!data.detections.length) {{
          panel.textContent = 'No matching block detected.';
          return;
        }}
        data.detections.forEach((detection) => {{
          const block = document.createElement('div');
          block.className = 'detail-block';
          const name = document.createElement('div');
          name.className = 'detail-name';
          name.textContent = detection.color;
          block.append(name);
          addDetailRow(block, 'C', pointText(detection.center_mm));
          addDetailRow(block, 'B', pointText(detection.biased_center_mm));
          addDetailRow(block, 'T', detection.target_label ? `T → ${{detection.target_label}}` : 'no reachable target');
          addDetailRow(block, 'retry', `${{detection.candidates_mm.length - 1}} points`);
          panel.append(block);
        }});
      }} catch (_error) {{
        panel.textContent = 'Live values unavailable.';
      }} finally {{
        detailsInFlight = false;
      }}
    }};
    const refreshOverlays = () => {{
      const color = document.querySelector('#overlay-color')?.value;
      document.querySelectorAll('img.camera-image').forEach((image) => {{
        if (color === 'none' || color === undefined) {{
          if (image.dataset.mode !== 'raw') {{
            image.src = `/video/${{image.dataset.camera}}.mjpg`;
            image.dataset.mode = 'raw';
            image.dataset.loading = '0';
          }}
          return;
        }}
        // A cold IK target preview can take several seconds. Do not replace
        // that request on every timer tick or the browser will cancel it
        // before the JPEG ever reaches the screen.
        if (image.dataset.loading === '1') return;
        image.dataset.loading = '1';
        image.onload = () => {{ image.dataset.loading = '0'; }};
        image.onerror = () => {{ image.dataset.loading = '0'; }};
        image.src = `/overlay/${{image.dataset.camera}}.jpg?color=${{encodeURIComponent(color)}}&t=${{Date.now()}}`;
        image.dataset.mode = 'overlay';
      }});
      refreshDetails(color);
    }};
    document.querySelector('#overlay-color')?.addEventListener('change', refreshOverlays);
    setInterval(refreshOverlays, 1000);
    refreshOverlays();
  </script>
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

    def _get_overlay(self, name: str, *, color: str | None) -> tuple[bytes, list[dict[str, object]]] | None:
        if self.server.overlay is None:
            self.send_error(HTTPStatus.NOT_FOUND, "overlay is disabled")
            return None
        camera = self.server.cameras.get(name)
        if camera is None:
            self.send_error(HTTPStatus.NOT_FOUND, f"unknown camera: {name}")
            return None
        try:
            return self.server.overlay.render(camera.latest_jpeg(), color=color)
        except Exception as exc:
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, f"overlay failed: {exc}")
            return None

    def _send_overlay(self, name: str, *, color: str | None) -> None:
        rendered = self._get_overlay(name, color=color)
        if rendered is not None:
            self._send_bytes(rendered[0], "image/jpeg")

    def _send_detections(self, name: str, *, color: str | None) -> None:
        rendered = self._get_overlay(name, color=color)
        if rendered is not None:
            self._send_json({"camera": name, "detections": rendered[1]})

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
    parser.add_argument("--wrist-device", default="", help="Optional wrist camera device (disabled by default)")
    parser.add_argument("--width", type=int, default=1280, help="Capture width. Default: 1280")
    parser.add_argument("--height", type=int, default=720, help="Capture height. Default: 720")
    parser.add_argument("--fps", type=int, default=30, help="Capture FPS. Default: 30")
    parser.add_argument("--fourcc", default="MJPG", help="Capture FOURCC. Default: MJPG")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="Stream JPEG quality 1-100. Default: 80")
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
        "--overlay-config",
        default="",
        help="Enable detection/grasp-plan overlay using this project YAML config.",
    )
    args = parser.parse_args()
    if bool(args.save_dir) != (args.save_interval_s > 0):
        parser.error("--save-dir and a positive --save-interval-s must be supplied together")
    return args


def main() -> int:
    args = parse_args()
    overlay = None
    if args.overlay_config:
        from camera.overlay import OverlayRenderer

        overlay = OverlayRenderer(args.overlay_config)
    recorder = FrameRecorder(args.save_dir, args.save_interval_s) if args.save_dir else None
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
        ),
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

    for camera in cameras.values():
        camera.start()

    server = CameraServer((args.host, args.port), cameras, overlay)

    def shutdown(_signum: int, _frame: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("SO-101 camera web server")
    for address in local_addresses():
        print(f"  http://{address}:{args.port}")
    print("Routes: /, /health, /snapshot/<camera>.jpg, /video/<camera>.mjpg")
    if overlay is not None:
        print("Overlay: /overlay/<camera>.jpg, /detections/<camera>.json")
    if recorder is not None:
        print(f"Saving one JPEG per camera every {recorder.interval_s:g}s under {args.save_dir}")

    try:
        server.serve_forever()
    finally:
        server.server_close()
        for camera in cameras.values():
            camera.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
