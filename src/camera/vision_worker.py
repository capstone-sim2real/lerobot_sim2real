"""Low-priority, latest-frame-only process for operator overlay metadata."""

from __future__ import annotations

import copy
import multiprocessing
import os
import queue
import signal
import threading
import time
from pathlib import Path
from typing import Any

from camera.overlay import OverlayAnalyzer


def _replace_queue_item(target: Any, item: Any) -> bool:
    """Best-effort latest-only put for a bounded multiprocessing queue."""
    try:
        target.put_nowait(item)
        return True
    except queue.Full:
        try:
            target.get_nowait()
        except queue.Empty:
            return False
        try:
            target.put_nowait(item)
            return True
        except queue.Full:
            return False


def _worker_main(config_path: str, input_queue: Any, output_queue: Any) -> None:
    # Observational work yields to the normal-priority robot process whenever
    # both need CPU. OpenCV is constrained before any frame processing begins.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    analyzer = OverlayAnalyzer(config_path)
    worker_cfg = analyzer.cfg.camera.overlay
    if worker_cfg.worker_nice > 0:
        try:
            os.nice(worker_cfg.worker_nice)
        except OSError:
            pass

    import cv2

    cv2.setNumThreads(max(1, int(worker_cfg.opencv_threads)))
    while True:
        item = input_queue.get()
        if item is None:
            return
        camera_name, jpeg, frame_seq, captured_at = item
        try:
            payload = analyzer.analyse(
                jpeg,
                camera_name=camera_name,
                frame_seq=frame_seq,
                captured_at=captured_at,
            )
        except Exception as exc:
            payload = {
                "camera": camera_name,
                "frame_seq": int(frame_seq),
                "captured_at": float(captured_at),
                "analysed_at": time.time(),
                "display_only": True,
                "detections": [],
                "error": str(exc),
            }
        _replace_queue_item(output_queue, payload)


class VisionWorker:
    """Own the overlay subprocess and expose its newest immutable payload."""

    def __init__(self, config_path: Path | str) -> None:
        self._config_path = str(Path(config_path).resolve())
        analyzer = OverlayAnalyzer(self._config_path)
        self._static = analyzer.static_metadata()
        self._analysis_fps = float(analyzer.cfg.camera.overlay.analysis_fps)
        if self._analysis_fps <= 0:
            raise ValueError("camera.overlay.analysis_fps must be positive")

        context = multiprocessing.get_context("spawn")
        self._input = context.Queue(maxsize=1)
        self._output = context.Queue(maxsize=1)
        self._process = context.Process(
            target=_worker_main,
            args=(self._config_path, self._input, self._output),
            name="camera-overlay-vision",
            daemon=True,
        )
        self._condition = threading.Condition()
        self._latest: dict[str, dict[str, Any]] = {}
        self._revisions: dict[str, int] = {}
        self._subscribers: dict[str, int] = {}
        self._active_until: dict[str, float] = {}
        self._next_submit_at: dict[str, float] = {}
        self._submitted = 0
        self._dropped = 0
        self._stop = threading.Event()
        self._collector = threading.Thread(
            target=self._collect_results,
            name="camera-overlay-results",
            daemon=True,
        )

    @property
    def static_metadata(self) -> dict[str, Any]:
        return copy.deepcopy(self._static)

    def start(self) -> None:
        self._process.start()
        self._collector.start()

    def stop(self) -> None:
        self._stop.set()
        _replace_queue_item(self._input, None)
        if self._process.is_alive():
            self._process.join(timeout=3.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
        self._collector.join(timeout=1.0)
        self._input.close()
        self._output.close()

    def subscribe(self, camera_name: str) -> None:
        with self._condition:
            self._subscribers[camera_name] = self._subscribers.get(camera_name, 0) + 1

    def unsubscribe(self, camera_name: str) -> None:
        with self._condition:
            self._subscribers[camera_name] = max(
                0, self._subscribers.get(camera_name, 0) - 1
            )

    def request_sample(self, camera_name: str, duration_s: float = 1.0) -> None:
        with self._condition:
            self._active_until[camera_name] = max(
                self._active_until.get(camera_name, 0.0),
                time.monotonic() + duration_s,
            )

    def submit(
        self,
        camera_name: str,
        jpeg: bytes,
        frame_seq: int,
        captured_at: float,
    ) -> None:
        now = time.monotonic()
        with self._condition:
            active = (
                self._subscribers.get(camera_name, 0) > 0
                or now < self._active_until.get(camera_name, 0.0)
            )
            if not active or now < self._next_submit_at.get(camera_name, 0.0):
                return
            self._next_submit_at[camera_name] = now + 1.0 / self._analysis_fps
        if _replace_queue_item(self._input, (camera_name, jpeg, frame_seq, captured_at)):
            self._submitted += 1
        else:
            self._dropped += 1

    def latest(self, camera_name: str, *, color: str | None = None) -> dict[str, Any]:
        self.request_sample(camera_name)
        with self._condition:
            payload = copy.deepcopy(
                self._latest.get(
                    camera_name,
                    {
                        "camera": camera_name,
                        "ready": False,
                        "display_only": True,
                        "detections": [],
                    },
                )
            )
        if color:
            payload["detections"] = [
                item
                for item in payload.get("detections", [])
                if item.get("color") == color
            ]
        return payload

    def wait_for_update(
        self,
        camera_name: str,
        revision: int,
        *,
        timeout_s: float = 5.0,
    ) -> tuple[dict[str, Any] | None, int]:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while (
                self._revisions.get(camera_name, 0) <= revision
                and not self._stop.is_set()
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None, revision
                self._condition.wait(remaining)
            current = self._revisions.get(camera_name, revision)
            payload = self._latest.get(camera_name)
            return (copy.deepcopy(payload) if payload is not None else None), current

    def status(self) -> dict[str, Any]:
        with self._condition:
            latest = {
                name: {
                    "frame_seq": payload.get("frame_seq"),
                    "analysis_ms": payload.get("analysis_ms"),
                    "age_s": round(
                        time.time() - payload.get("captured_at", time.time()), 3
                    ),
                    "error": payload.get("error", ""),
                }
                for name, payload in self._latest.items()
            }
            subscribers = dict(self._subscribers)
        return {
            "enabled": True,
            "analysis_fps": self._analysis_fps,
            "worker_alive": self._process.is_alive(),
            "submitted": self._submitted,
            "dropped": self._dropped,
            "subscribers": subscribers,
            "latest": latest,
        }

    def _collect_results(self) -> None:
        while not self._stop.is_set():
            try:
                payload = self._output.get(timeout=0.5)
            except queue.Empty:
                continue
            camera_name = str(payload["camera"])
            with self._condition:
                self._latest[camera_name] = payload
                self._revisions[camera_name] = self._revisions.get(camera_name, 0) + 1
                self._condition.notify_all()
