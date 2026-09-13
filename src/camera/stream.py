"""Camera stream implementation."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable
from camera.recorder import FrameRecorder
import cv2
import numpy as np

FrameCallback = Callable[[str, bytes, int, float], None]


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
                    round(time.time() - self._latest_ts, 3) if self._latest_ts else None
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
                        self._frame_callback(self.name, jpeg, frame_seq, captured_at)
                    except Exception as exc:
                        print(f"{self.name}: overlay submit failed: {exc}")
                if self._recorder is not None:
                    try:
                        self._recorder.record(self.name, jpeg, now=captured_at)
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
        return encoded.tobytes() if ok else b""
