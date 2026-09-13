"""Camera recorder implementation."""

from __future__ import annotations

import threading
import time
from pathlib import Path
import cv2
import numpy as np


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
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise ValueError("cannot compare an invalid JPEG")
        frame = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(frame, (5, 5), 0)

    @staticmethod
    def _change_score(previous: np.ndarray, current: np.ndarray) -> float:
        return float(np.mean(cv2.absdiff(previous, current)))
