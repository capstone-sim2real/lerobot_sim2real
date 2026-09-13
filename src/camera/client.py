"""Client-side access to frames owned by :mod:`camera.server`."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.request import urlopen

import cv2
import numpy as np


DEFAULT_SHOULDER_SNAPSHOT_URL = "http://127.0.0.1:8090/snapshot/shoulder.jpg"


@dataclass(frozen=True)
class CameraSnapshot:
    frame: np.ndarray
    frame_seq: int
    captured_at: float


def _decode_jpeg(data: bytes, url: str) -> np.ndarray:
    if not data.startswith(b"\xff\xd8"):
        raise RuntimeError(f"Snapshot from {url} is not a JPEG")
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not decode the JPEG returned by {url}")
    return frame


def fetch_snapshot(url: str = DEFAULT_SHOULDER_SNAPSHOT_URL, *, timeout_s: float = 5.0) -> np.ndarray:
    """Fetch and decode one latest BGR frame from the camera service."""
    with urlopen(url, timeout=timeout_s) as response:  # nosec B310 -- caller controls the camera URL
        data = response.read()
    return _decode_jpeg(data, url)


def fetch_snapshot_with_metadata(
    url: str = DEFAULT_SHOULDER_SNAPSHOT_URL, *, timeout_s: float = 5.0
) -> CameraSnapshot:
    """Fetch a frame whose capture sequence can prove that it is fresh."""
    with urlopen(url, timeout=timeout_s) as response:  # nosec B310 -- caller controls the camera URL
        data = response.read()
        seq_text = response.headers.get("X-Frame-Seq")
        captured_text = response.headers.get("X-Captured-At")
    if seq_text is None or captured_text is None:
        raise RuntimeError(
            "Camera snapshot has no freshness metadata; restart so101-camera with the current code"
        )
    return CameraSnapshot(
        frame=_decode_jpeg(data, url),
        frame_seq=int(seq_text),
        captured_at=float(captured_text),
    )
