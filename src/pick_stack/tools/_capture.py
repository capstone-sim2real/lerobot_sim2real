"""Minimal standalone frame grab for the CLI tools (BGR, like cv2.imread).

Deliberately does not use lerobot's camera stack: the tools must run while
no other process holds the camera, and cv2.VideoCapture keeps them
dependency-free. Frames returned here are BGR — pass is_rgb=False downstream.
"""

from __future__ import annotations

import cv2
import numpy as np


def grab_frame(
    index_or_path: int | str,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    fourcc: str = "MJPG",
    warmup_frames: int = 10,
) -> np.ndarray:
    cap = cv2.VideoCapture(index_or_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {index_or_path!r} (busy? wrong /dev/video*?)")
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        frame = None
        for _ in range(max(1, warmup_frames)):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Camera {index_or_path!r} returned no frame")
        return frame
    finally:
        cap.release()


def grab_snapshot(url: str, timeout_s: float = 5.0) -> np.ndarray:
    """Fetch one BGR frame from the camera web server's JPEG snapshot endpoint.

    Needed because scripts/so101_camera_web.py holds /dev/video* open for the
    live view: cv2.VideoCapture would fail on a busy device, so tools that run
    alongside the browser view must go through HTTP instead.
    """
    from urllib.request import urlopen

    with urlopen(url, timeout=timeout_s) as response:  # nosec B310 -- local camera URL
        data = response.read()
    if not data.startswith(b"\xff\xd8"):
        raise RuntimeError(f"Snapshot from {url} is not a JPEG")
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not decode the JPEG returned by {url}")
    return frame


def grab(source: str, **kwargs) -> np.ndarray:
    """Grab a frame from either an http(s) snapshot URL or a camera device."""
    if source.startswith(("http://", "https://")):
        return grab_snapshot(source, timeout_s=kwargs.get("timeout_s", 5.0))
    return grab_frame(int(source) if source.isdigit() else source, **kwargs)
