"""Client-side access to frames owned by :mod:`camera.server`."""

from __future__ import annotations

from urllib.request import urlopen

import cv2
import numpy as np


DEFAULT_SHOULDER_SNAPSHOT_URL = "http://127.0.0.1:8090/snapshot/shoulder.jpg"


def fetch_snapshot(url: str = DEFAULT_SHOULDER_SNAPSHOT_URL, *, timeout_s: float = 5.0) -> np.ndarray:
    """Fetch and decode one latest BGR frame from the camera service."""
    with urlopen(url, timeout=timeout_s) as response:  # nosec B310 -- caller controls the camera URL
        data = response.read()
    if not data.startswith(b"\xff\xd8"):
        raise RuntimeError(f"Snapshot from {url} is not a JPEG")
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not decode the JPEG returned by {url}")
    return frame
