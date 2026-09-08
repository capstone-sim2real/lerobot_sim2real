"""Camera cross calibration implementation."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from config import load_config
from perception import PlaneCalibration


class CrossCalibration:
    """Collect manual marker-pixel to table-point pairs from the web UI."""

    def __init__(self, config_path: Path | str, session_path: Path | str) -> None:
        cfg = load_config(config_path)
        self._calibration = PlaneCalibration.load(cfg.perception.calibration_path)
        self._session_path = Path(session_path).resolve()
        self._samples: list[dict[str, list[float]]] = self._load()
        self._lock = threading.Lock()

    def _load(self) -> list[dict[str, list[float]]]:
        if not self._session_path.is_file():
            return []
        try:
            data = json.loads(self._session_path.read_text())
            samples = data.get("samples", [])
            return samples if isinstance(samples, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    def _persist(self) -> None:
        self._session_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._session_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"samples": self._samples}, indent=2))
        temporary.replace(self._session_path)

    def status(self) -> dict[str, Any]:
        with self._lock:
            marker = np.asarray(
                [item["marker_px"] for item in self._samples], dtype=np.float64
            )
            foot = np.asarray(
                [item["foot_mm"] for item in self._samples], dtype=np.float64
            )
        result: dict[str, Any] = {"samples": self._samples, "ready": len(marker) >= 4}
        if len(marker) >= 4:
            transform, _ = cv2.findHomography(marker, foot, method=0)
            assert transform is not None
            projected = cv2.perspectiveTransform(
                marker.reshape(-1, 1, 2), transform
            ).reshape(-1, 2)
            result["marker_to_table_h"] = transform.tolist()
            result["rms_mm"] = float(
                np.sqrt(np.mean(np.sum((projected - foot) ** 2, axis=1)))
            )
        return result

    def add(self, marker_px: list[float], reference_px: list[float]) -> dict[str, Any]:
        marker = np.asarray(marker_px, dtype=np.float64).reshape(2)
        reference = np.asarray(reference_px, dtype=np.float64).reshape(1, 2)
        foot = self._calibration.pixel_to_board(reference)[0]
        with self._lock:
            self._samples.append(
                {
                    "marker_px": marker.tolist(),
                    "reference_px": reference[0].tolist(),
                    "foot_mm": foot.tolist(),
                }
            )
            self._persist()
        return self.status()

    def reset(self) -> dict[str, Any]:
        with self._lock:
            self._samples.clear()
            self._persist()
        return self.status()
