"""Persistent length-prefixed YOLOE TensorRT worker for the agent web UI.

The worker reads JPEG requests from stdin and writes JSON responses to stdout.
It owns no camera and performs no robot operation.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time

import cv2
import numpy as np

from camera.overlay import detection_metadata, reject_metadata
from config import load_config
from perception.detector import BlockDetection, RejectedCandidate
from perception.homography import PlaneCalibration
from perception.yoloe_blocks import hybrid_color_geometry, prepare_hybrid_frame
from perception.yoloe_tensorrt import TensorRTEngine, decode_yoloe_segmentation, letterbox_bgr

_HEADER = struct.Struct("!Q")


def _read_exact(size: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sys.stdin.buffer.read(size - len(chunks))
        if not chunk:
            return None
        chunks.extend(chunk)
    return bytes(chunks)


def _write(payload: dict) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    sys.stdout.buffer.write(_HEADER.pack(len(encoded)))
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


def _block(row: dict) -> BlockDetection:
    return BlockDetection(
        color=row["color"],
        center_mm=tuple(row["center_mm"]),
        area_mm2=float(row["area_mm2"]),
        aspect=float(row["aspect"]),
        solidity=float(row["solidity"]),
        fill=float(row["fill"]),
        box_mm=[tuple(point) for point in row["box_mm"]],
        angle_deg=float(row["angle_deg"]),
        hue_sat=tuple(row.get("hue_sat", (0.0, 0.0))),
    )


def _reject(row: dict) -> RejectedCandidate | None:
    required = {"color", "center_mm", "box_mm", "area_mm2", "aspect", "solidity", "fill", "reason"}
    if not required.issubset(row) or row["reason"] is None:
        return None
    return RejectedCandidate(
        color=row["color"], center_mm=tuple(row["center_mm"]),
        box_mm=[tuple(point) for point in row["box_mm"]],
        area_mm2=float(row["area_mm2"]), aspect=float(row["aspect"]),
        solidity=float(row["solidity"]), fill=float(row["fill"]), reason=row["reason"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/configs/default.yaml")
    parser.add_argument("--calibration", default="src/configs/calib/venue_lab.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    calib = PlaneCalibration.load(args.calibration)
    cv2.setNumThreads(cfg.yoloe.cpu_threads)
    warmed = False
    with TensorRTEngine(cfg.yoloe.engine) as engine:
        while True:
            header = _read_exact(_HEADER.size)
            if header is None:
                return
            data = _read_exact(_HEADER.unpack(header)[0])
            if data is None:
                return
            started = time.perf_counter()
            try:
                frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if frame is None or frame.shape[1::-1] != tuple(calib.image_size):
                    raise ValueError("JPEG resolution differs from calibration")
                tensor, info = letterbox_bgr(frame, cfg.yoloe.imgsz)
                if not warmed:
                    for _ in range(cfg.yoloe.trt_warmup_runs):
                        engine.infer(tensor)
                    warmed = True
                detections, prototypes, gpu_ms = engine.infer(tensor)
                decoded = decode_yoloe_segmentation(
                    detections, prototypes, info, cfg.yoloe.trt_confidence,
                    cfg.yoloe.iou, cfg.yoloe.max_detections,
                )
                context = prepare_hybrid_frame(frame, calib, cfg.perception)
                accepted, rejected = [], []
                for item in decoded:
                    hybrid = hybrid_color_geometry(
                        frame, item["mask"], calib, cfg.perception, context=context
                    )
                    if hybrid.get("accepted"):
                        metadata = detection_metadata(_block(hybrid), calib, cfg)
                        metadata.update(confidence=item["confidence"], detector="yoloe")
                        accepted.append(metadata)
                    else:
                        candidate = _reject(hybrid)
                        if candidate is not None:
                            metadata = reject_metadata(candidate, calib)
                            metadata.update(confidence=item["confidence"], detector="yoloe")
                            rejected.append(metadata)
                _write({
                    "ready": True, "backend": "yoloe", "display_only": True,
                    "image_size": list(calib.image_size), "detections": accepted,
                    "rejects": rejected, "gpu_and_transfer_ms": round(gpu_ms, 2),
                    "analysis_ms": round((time.perf_counter() - started) * 1000, 2),
                })
            except Exception as exc:  # keep the worker available after one bad frame
                _write({"ready": False, "backend": "yoloe", "display_only": True,
                        "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
