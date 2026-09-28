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
    return {"available": not missing, "display_only": True, "missing": missing}


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

    def analyse_latest(self) -> dict:
        with urlopen(Request(self.cfg.yoloe.snapshot_url, headers={"Cache-Control": "no-cache"}),
                     timeout=self.cfg.yoloe.http_timeout_s) as response:  # nosec B310
            jpeg = response.read()
            frame_seq = int(response.headers.get("X-Frame-Seq", -1))
            captured_at = float(response.headers.get("X-Captured-At", 0.0))
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
