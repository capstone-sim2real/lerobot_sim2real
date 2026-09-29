"""Minimal standalone frame grab for the CLI tools (BGR, like cv2.imread).

Deliberately does not use lerobot's camera stack: the tools must run while
no other process holds the camera, and cv2.VideoCapture keeps them
dependency-free. Frames returned here are BGR — pass is_rgb=False downstream.
"""

from __future__ import annotations

import cv2
import numpy as np

from camera.client import fetch_snapshot


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


grab_snapshot = fetch_snapshot


def grab(source: str, **kwargs) -> np.ndarray:
    """Grab a frame from either an http(s) snapshot URL or a camera device."""
    if source.startswith(("http://", "https://")):
        return grab_snapshot(source, timeout_s=kwargs.get("timeout_s", 5.0))
    return grab_frame(int(source) if source.isdigit() else source, **kwargs)
