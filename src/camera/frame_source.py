"""Control-loop-rate frames from :mod:`camera.server`, without owning a device.

Task 3 needs a camera image on *every* control tick while the CV detector is
simultaneously polling the same physical camera. Only ``camera.server`` may
open ``/dev/video*`` (AGENTS.md §8), so the frames have to come from it — but
``client.fetch_snapshot`` opens a fresh HTTP connection and decodes a
1280x720 JPEG per call, which is far too slow to sit inside a 30 Hz loop.

So this module mirrors what lerobot's own cameras do: a background thread
holds one persistent MJPEG connection, decodes into the shape the dataset
wants, and publishes the latest frame. ``latest()`` on the control thread is
a dict lookup and an array copy — the decode cost never lands on the loop
that is driving the arm.

Frames are published as **RGB**: lerobot cameras default to
``ColorMode.RGB`` and ACT's ResNet backbone is pretrained on RGB statistics,
while OpenCV hands us BGR. Converting here (rather than at the call site)
is what keeps a recorded dataset and a live rollout on the same pipeline.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from urllib.request import urlopen

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_BOUNDARY_PREFIX = b"--"
# The server sends Content-Length on every part, so parts are read exactly
# rather than scanned for the next boundary.
_CONTENT_LENGTH = b"content-length:"


@dataclass(frozen=True)
class SourceFrame:
    """One decoded frame plus the evidence that it is recent."""

    image: np.ndarray  # (H, W, 3) uint8, RGB
    seq: int
    received_at: float  # time.monotonic() when this frame finished decoding


class MjpegFrameSource:
    """Latest frame from one ``camera.server`` MJPEG stream.

    ``received_at`` is the local decode time, not the server's capture
    timestamp: the stream parts carry no per-frame header, and for the
    staleness question the recorder actually asks ("is this image still
    describing now?") the decode time is the conservative answer anyway,
    since it is always later than capture.
    """

    def __init__(
        self,
        name: str,
        url: str,
        *,
        width: int,
        height: int,
        connect_timeout_s: float = 5.0,
        reconnect_backoff_s: float = 1.0,
    ):
        self.name = name
        self.url = url
        self._width = int(width)
        self._height = int(height)
        self._connect_timeout_s = connect_timeout_s
        self._reconnect_backoff_s = reconnect_backoff_s
        self._lock = threading.Lock()
        self._latest: SourceFrame | None = None
        self._first_frame = threading.Event()
        self._error = ""
        self._decoded = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"frame-source-{name}", daemon=True
        )

    # ── lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)

    def wait_for_first_frame(self, timeout_s: float = 10.0) -> bool:
        """Block until one frame has been decoded. False on timeout."""
        return self._first_frame.wait(timeout=timeout_s)

    # ── reading ──────────────────────────────────────────────────────

    def latest(self) -> SourceFrame | None:
        with self._lock:
            return self._latest

    def age_s(self, now: float | None = None) -> float:
        """Seconds since the latest frame; ``inf`` when there is none."""
        frame = self.latest()
        if frame is None:
            return float("inf")
        return (time.monotonic() if now is None else now) - frame.received_at

    def status(self) -> dict[str, object]:
        with self._lock:
            latest = self._latest
            error = self._error
            decoded = self._decoded
        return {
            "name": self.name,
            "url": self.url,
            "shape": None if latest is None else tuple(latest.image.shape),
            "decoded": decoded,
            "age_s": round(self.age_s(), 3) if latest is not None else None,
            "error": error,
        }

    # ── worker ───────────────────────────────────────────────────────

    def _publish(self, image: np.ndarray) -> None:
        now = time.monotonic()
        with self._lock:
            self._decoded += 1
            self._latest = SourceFrame(image=image, seq=self._decoded, received_at=now)
            self._error = ""
        self._first_frame.set()

    def _publish_error(self, message: str) -> None:
        logger.warning("%s", message)
        with self._lock:
            self._error = message

    def _decode(self, jpeg: bytes) -> np.ndarray | None:
        bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        if (bgr.shape[1], bgr.shape[0]) != (self._width, self._height):
            bgr = cv2.resize(
                bgr, (self._width, self._height), interpolation=cv2.INTER_AREA
            )
        # np.ascontiguousarray so the array handed to the image writer owns a
        # plain buffer; cvtColor already returns one, but the copy is also
        # what makes ``latest()`` safe to hand out without the lock held.
        return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def _run(self) -> None:
        backoff_s = self._reconnect_backoff_s
        while not self._stop.is_set():
            try:
                with urlopen(  # nosec B310 -- the camera URL comes from our own config
                    self.url, timeout=self._connect_timeout_s
                ) as response:
                    backoff_s = self._reconnect_backoff_s
                    self._read_stream(response)
            except Exception as exc:  # noqa: BLE001 - any transport fault reconnects
                if self._stop.is_set():
                    return
                self._publish_error(f"{self.name}: stream {self.url} failed: {exc}")
            if self._stop.wait(timeout=backoff_s):
                return
            backoff_s = min(backoff_s * 1.5, 5.0)

    def _read_stream(self, response) -> None:
        while not self._stop.is_set():
            length = self._read_part_headers(response)
            if length is None:
                raise RuntimeError("multipart stream ended or sent no Content-Length")
            payload = response.read(length)
            if len(payload) != length:
                raise RuntimeError("multipart part truncated")
            image = self._decode(payload)
            if image is None:
                self._publish_error(f"{self.name}: JPEG decode failed")
                continue
            self._publish(image)

    @staticmethod
    def _read_part_headers(response) -> int | None:
        """Consume one boundary + header block, returning its Content-Length."""
        length: int | None = None
        saw_boundary = False
        while True:
            line = response.readline()
            if not line:
                return None
            stripped = line.strip()
            if not stripped:
                # Blank line ends the header block -- but only once a boundary
                # has been seen, since parts are separated by a trailing CRLF.
                if saw_boundary:
                    return length
                continue
            if stripped.startswith(_BOUNDARY_PREFIX):
                saw_boundary = True
                length = None
                continue
            if stripped.lower().startswith(_CONTENT_LENGTH):
                try:
                    length = int(stripped.split(b":", 1)[1])
                except ValueError:
                    return None


def build_frame_sources(
    cameras: dict[str, str], *, width: int, height: int
) -> dict[str, MjpegFrameSource]:
    """Create (but do not start) one source per configured camera."""
    return {
        name: MjpegFrameSource(name, url, width=width, height=height)
        for name, url in cameras.items()
    }
