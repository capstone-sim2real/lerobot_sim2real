"""Lazy bridge from the Python 3.12 agent to the Jetson Python 3.10 TRT worker."""
from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import threading
from urllib.request import Request, urlopen

from config import AppConfig

_HEADER = struct.Struct("!Q")


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def yoloe_web_status(cfg: AppConfig, root: Path) -> dict:
    python = _resolve(root, cfg.yoloe.worker_python)
    engine = _resolve(root, cfg.yoloe.engine)
    missing = []
    if not python.is_file():
        missing.append("worker_python")
    if not engine.is_file():
        missing.append("engine")
    return {"available": not missing, "missing": missing}


class YoloeOverlayClient:
    def __init__(self, cfg: AppConfig, root: Path):
        self.cfg = cfg
        self.root = root
        self._process: subprocess.Popen | None = None
        self._config_path: Path | None = None
        self._lock = threading.RLock()

    def _start(self) -> subprocess.Popen:
        status = yoloe_web_status(self.cfg, self.root)
        if not status["available"]:
            raise RuntimeError("YOLOE runtime unavailable: " + ", ".join(status["missing"]))
        env = os.environ.copy()
        env.update(PYTHONPATH=str(self.root / "src"), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        if self._config_path is None:
            handle = tempfile.NamedTemporaryFile(
                mode="w", prefix="so101-yoloe-", suffix=".json", delete=False, encoding="utf-8"
            )
            with handle:
                json.dump(self.cfg.to_dict(), handle)
            self._config_path = Path(handle.name)
        process = subprocess.Popen(
            [str(_resolve(self.root, self.cfg.yoloe.worker_python)), "-m", "tools.yoloe_worker",
             "--config", str(self._config_path),
             "--calibration", str(_resolve(self.root, self.cfg.perception.calibration_path))],
            cwd=self.root, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._process = process
        return process

    @staticmethod
    def _read_exact(stream, size: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            chunk = stream.read(size - len(chunks))
            if not chunk:
                raise RuntimeError("YOLOE worker closed its output")
            chunks.extend(chunk)
        return bytes(chunks)

    def _fetch_latest(self) -> tuple[bytes, int, float]:
        with urlopen(Request(self.cfg.yoloe.snapshot_url, headers={"Cache-Control": "no-cache"}),
                     timeout=self.cfg.yoloe.http_timeout_s) as response:  # nosec B310
            return (
                response.read(), int(response.headers.get("X-Frame-Seq", -1)),
                float(response.headers.get("X-Captured-At", 0.0)),
            )

    def _analyse_jpeg(self, jpeg: bytes, frame_seq: int, captured_at: float) -> dict:
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                process = self._start()
            assert process.stdin is not None and process.stdout is not None
            try:
                process.stdin.write(_HEADER.pack(len(jpeg)))
                process.stdin.write(jpeg)
                process.stdin.flush()
                size = _HEADER.unpack(self._read_exact(process.stdout, _HEADER.size))[0]
                result = json.loads(self._read_exact(process.stdout, size))
            except (BrokenPipeError, OSError, RuntimeError):
                self.close()
                raise
        result.update(frame_seq=frame_seq, captured_at=captured_at)
        return result

    def analyse_latest(self) -> dict:
        return self._analyse_jpeg(*self._fetch_latest())

    def analyse_latest_with_snapshot(self):
        import cv2
        import numpy as np
        from camera.client import CameraSnapshot

        jpeg, frame_seq, captured_at = self._fetch_latest()
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("Camera returned an invalid JPEG")
        return self._analyse_jpeg(jpeg, frame_seq, captured_at), CameraSnapshot(
            frame=frame, frame_seq=frame_seq, captured_at=captured_at
        )

    def close(self) -> None:
        with self._lock:
            process, self._process = self._process, None
            if process is None:
                if self._config_path is not None:
                    self._config_path.unlink(missing_ok=True)
                    self._config_path = None
                return
            if process.stdin is not None:
                process.stdin.close()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
            if self._config_path is not None:
                self._config_path.unlink(missing_ok=True)
                self._config_path = None


class PerceptionBackendController:
    """Thread-safe detector selection shared by web diagnostics and ArmSession."""

    def __init__(self, cfg: AppConfig, root: Path):
        self.cfg = cfg
        self.root = root
        self.yoloe = YoloeOverlayClient(cfg, root)
        self._backend = "cv"
        self._backend_lock = threading.RLock()

    @property
    def backend(self) -> str:
        with self._backend_lock:
            return self._backend

    def status(self) -> dict:
        yoloe = yoloe_web_status(self.cfg, self.root)
        return {
            "cv": {"available": True},
            "yoloe": {**yoloe, "control_capable": yoloe["available"]},
            "selected": self.backend,
            "control_backend": self.backend,
        }

    def set_backend(self, backend: str) -> str:
        if backend not in {"cv", "yoloe"}:
            raise ValueError("지원하지 않는 검출 backend입니다.")
        if backend == "yoloe" and not yoloe_web_status(self.cfg, self.root)["available"]:
            raise RuntimeError("YOLOE runtime을 사용할 수 없습니다.")
        with self._backend_lock:
            self._backend = backend
        return backend

    def analyse_latest(self) -> dict:
        return self.yoloe.analyse_latest()

    @staticmethod
    def _block(data: dict):
        from perception.detector import BlockDetection

        return BlockDetection(
            color=data["color"], center_mm=tuple(data["center_mm"]),
            area_mm2=float(data["area_mm2"]), aspect=float(data["aspect"]),
            solidity=float(data["solidity"]), fill=float(data["fill"]),
            box_mm=[tuple(point) for point in data["box_mm"]],
            angle_deg=float(data["angle_deg"]), hue_sat=tuple(data.get("hue_sat", (0.0, 0.0))),
        )

    def observe_scene(self, calib, slot_xy, *, after, cancel, clock):
        """Return the same Scene contract as CV using a fresh YOLOE snapshot."""
        import time
        from perception.scene import build_scene
        from perception.zone import point_in_zone

        deadline = time.monotonic() + self.cfg.agent.camera_fresh_timeout_s
        while True:
            cancel.raise_if_set()
            packet, snapshot = self.yoloe.analyse_latest_with_snapshot()
            if not packet.get("ready"):
                raise RuntimeError(packet.get("error", "YOLOE analysis failed"))
            age = clock() - snapshot.captured_at
            if age <= self.cfg.task1.max_frame_age_s and (
                after is None or snapshot.captured_at >= after
            ):
                break
            if time.monotonic() > deadline:
                raise RuntimeError(f"no fresh YOLOE frame (age {age:.1f}s)")
            time.sleep(self.cfg.task1.scan_interval_s)
        detections = [self._block(row["detection"]) for row in packet.get("detections", [])]
        outside = [item for item in detections if not point_in_zone(item.center_mm, calib)]
        inside = [item for item in detections if point_in_zone(item.center_mm, calib)]
        scene = build_scene(
            outside, inside, calib, slot_xy,
            snap_radius_mm=self.cfg.agent.slot_snap_radius_mm,
            frame_seq=snapshot.frame_seq, captured_at=snapshot.captured_at,
        )
        return scene, snapshot

    def close(self) -> None:
        self.yoloe.close()
