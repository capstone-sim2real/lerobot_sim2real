"""Read-only TensorRT YOLOE snapshot experiment."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

import cv2
import numpy as np

from config import load_config
from perception.homography import PlaneCalibration
from perception.yoloe_blocks import hybrid_color_geometry, mask_geometry, prepare_hybrid_frame
from perception.yoloe_tensorrt import TensorRTEngine, decode_yoloe_segmentation, letterbox_bgr


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "system"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/configs/default.yaml")
    parser.add_argument("--calibration", default="src/configs/calib/venue_lab.json")
    parser.add_argument("--image", help="Saved image; otherwise GET configured HTTP snapshot")
    parser.add_argument("--output", required=True, help="New directory, refuses overwrite")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    yc = cfg.yoloe
    if args.repeats < 1 or yc.trt_warmup_runs < 0 or yc.max_detections < 1 or not 0 <= yc.trt_confidence <= 1:
        parser.error("Invalid repeats/trt_warmup_runs/max_detections")
    if yc.imgsz != 640:
        parser.error("The checked-in engine contract requires yoloe.imgsz=640")

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    raw = Path(args.image).read_bytes() if args.image else urlopen(
        Request(yc.snapshot_url, headers={"Cache-Control": "no-cache"}),
        timeout=yc.http_timeout_s,
    ).read()
    acquired_at = time.time()
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    calib = PlaneCalibration.load(args.calibration)
    if frame is None or frame.shape[1::-1] != tuple(calib.image_size):
        raise ValueError("Image missing or resolution differs from calibration")
    (output / "input.jpg").write_bytes(raw)
    (output / "calibration.json").write_bytes(Path(args.calibration).read_bytes())

    engine_path = Path(yc.engine).resolve()
    stages = {key: [] for key in ("preprocess_ms", "gpu_and_transfer_ms", "decode_ms", "hybrid_ms", "total_ms")}
    final = []
    with TensorRTEngine(engine_path) as engine:
        warm_tensor, _ = letterbox_bgr(frame, yc.imgsz)
        for _ in range(yc.trt_warmup_runs):
            engine.infer(warm_tensor)
        for _ in range(args.repeats):
            total_start = time.perf_counter()
            start = time.perf_counter()
            tensor, info = letterbox_bgr(frame, yc.imgsz)
            stages["preprocess_ms"].append((time.perf_counter() - start) * 1000)
            detections, prototypes, gpu_ms = engine.infer(tensor)
            stages["gpu_and_transfer_ms"].append(gpu_ms)
            start = time.perf_counter()
            decoded = decode_yoloe_segmentation(
                detections, prototypes, info, yc.trt_confidence, yc.iou, yc.max_detections
            )
            stages["decode_ms"].append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            context = prepare_hybrid_frame(frame, calib, cfg.perception)
            rows = []
            for index, item in enumerate(decoded):
                mask = item.pop("mask")
                row = mask_geometry(mask, calib, cfg.perception)
                hybrid = hybrid_color_geometry(
                    frame, mask, calib, cfg.perception, context=context
                )
                row.update(
                    id=index, label="block", confidence=item["confidence"],
                    bbox_px=item["bbox_px"], hybrid_color_geometry=hybrid,
                    _mask=mask,
                )
                if hybrid["accepted"]:
                    row["candidate_color"] = hybrid["color"]
                    row["candidate_center_mm"] = hybrid["center_mm"]
                rows.append(row)
            stages["hybrid_ms"].append((time.perf_counter() - start) * 1000)
            stages["total_ms"].append((time.perf_counter() - total_start) * 1000)
            final = rows

    overlay = frame.copy()
    tint = np.zeros_like(frame)
    for row in final:
        index, mask = row["id"], row.pop("_mask")
        cv2.imwrite(str(output / f"mask_{index:02d}.png"), mask * 255)
        tint[mask.astype(bool)] = (255, 80, 0)
        x1, y1, x2, y2 = np.asarray(row["bbox_px"]).round().astype(int)
        hybrid = row["hybrid_color_geometry"]
        color = (0, 255, 0) if hybrid["accepted"] else (0, 0, 255)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"{index} {row['confidence']:.2f}"
        if hybrid.get("center_mm"):
            xy = hybrid["center_mm"]
            label += f" {hybrid['color']} ({xy[0]:.1f},{xy[1]:.1f})mm"
        cv2.putText(overlay, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
    overlay = cv2.addWeighted(overlay, 0.75, tint, 0.25, 0)
    cv2.imwrite(str(output / "overlay.jpg"), overlay)

    report = {
        "engine": str(engine_path),
        "engine_sha256": hashlib.sha256(engine_path.read_bytes()).hexdigest(),
        "config": cfg.to_dict(),
        "calibration": str(Path(args.calibration).resolve()),
        "calibration_sha256": hashlib.sha256(Path(args.calibration).read_bytes()).hexdigest(),
        "calibration_meta": calib.meta,
        "acquired_at_unix": acquired_at,
        "source": args.image or yc.snapshot_url,
        "versions": {
            "tensorrt": _version("tensorrt"), "cuda-python": _version("cuda-python"),
            "numpy": _version("numpy"), "opencv": cv2.__version__,
        },
        "timings": stages,
        "instances": final,
        "limitation": "Fixed visual-prompt FP16 engine. Hybrid colour geometry is not proven to be only the top face. Camera drift and physical accuracy remain unverified. No robot motion.",
    }
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    summary = {
        "output": str(output), "instances": len(final),
        "hybrid_accepted": sum(r["hybrid_color_geometry"]["accepted"] for r in final),
        "median_ms": {key: float(np.median(value)) for key, value in stages.items()},
        "p95_total_ms": float(np.percentile(stages["total_ms"], 95)),
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
