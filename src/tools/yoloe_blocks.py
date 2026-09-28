"""Read-only YOLOE snapshot experiment: python -m tools.yoloe_blocks --help."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen

import cv2
import numpy as np
from config import load_config
from perception.homography import PlaneCalibration
from perception.yoloe_blocks import (
    hybrid_color_geometry,
    mask_geometry,
    prepare_hybrid_frame,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/configs/default.yaml")
    parser.add_argument("--calibration", default="src/configs/calib/venue_lab.json")
    parser.add_argument("--image", help="Saved image; otherwise GET configured HTTP snapshot")
    parser.add_argument("--visual-prompts", help="JSON: image path and bboxes on that reference image")
    parser.add_argument("--output", required=True, help="New directory, refuses overwrite")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--repeats", type=int, default=1, help="Benchmark the same saved frame")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    yc = cfg.yoloe
    if args.repeats < 1 or not 0 <= yc.confidence <= 1 or yc.imgsz < 32 or yc.cpu_threads < 1:
        parser.error("Invalid repeats/confidence/imgsz/cpu_threads")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    raw = Path(args.image).read_bytes() if args.image else urlopen(
        Request(yc.snapshot_url, headers={"Cache-Control": "no-cache"}),
        timeout=yc.http_timeout_s).read()
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    calib = PlaneCalibration.load(args.calibration)
    if frame is None or frame.shape[1::-1] != tuple(calib.image_size):
        raise ValueError("Image missing or resolution differs from calibration")
    (output / "input.jpg").write_bytes(raw)
    acquired_at = time.time()
    (output / "calibration.json").write_bytes(Path(args.calibration).read_bytes())
    for folder in ("runtime", "cache"):
        (output / folder).mkdir()
    # Keep generated settings and text-encoder caches inside the experiment.
    os.environ.setdefault("YOLO_CONFIG_DIR", str(output / "runtime"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output / "cache"))
    from ultralytics import YOLOE  # optional, lazy dependency
    import torch
    torch.set_num_threads(yc.cpu_threads)
    model_path = Path(yc.model).resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    old_cwd = Path.cwd()
    try:
        # Ultralytics downloads its text encoder relative to cwd.
        os.chdir(model_path.parent)
        model = YOLOE(str(model_path))
        if not args.visual_prompts:
            model.set_classes(yc.prompts)
    finally:
        os.chdir(old_cwd)
    visual = None
    prediction_args = {}
    if args.visual_prompts:
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
        visual = json.loads(Path(args.visual_prompts).read_text())
        boxes = np.asarray(visual["bboxes"], dtype=np.float32)
        if boxes.ndim != 2 or boxes.shape[1] != 4 or not len(boxes):
            raise ValueError("Visual boxes must have shape (N,4)")
        reference_path = Path(visual["image"]).resolve()
        visual["image"] = str(reference_path)
        visual["sha256"] = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        (output / "reference_image.jpg").write_bytes(reference_path.read_bytes())
        prediction_args = dict(visual_prompts={"bboxes": boxes, "cls": np.zeros(len(boxes), dtype=int)},
            refer_image=visual["image"], predictor=YOLOEVPSegPredictor)
    durations = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        result = model.predict(frame, device=yc.device, imgsz=yc.imgsz,
            conf=yc.confidence, iou=yc.iou, retina_masks=True, verbose=False, **prediction_args)[0]
        prediction_args = {}  # reference embedding is cached after the first prediction
        durations.append((time.perf_counter() - start) * 1000)
    hybrid_context = prepare_hybrid_frame(frame, calib, cfg.perception)
    rows = []
    overlay = result.plot()
    if result.masks is not None:
        for i, (mask, box) in enumerate(zip(result.masks.data.cpu().numpy(), result.boxes)):
            binary = (mask > 0).astype(np.uint8)
            row = mask_geometry(binary, calib, cfg.perception)
            hybrid = hybrid_color_geometry(
                frame, binary, calib, cfg.perception, context=hybrid_context
            )
            row.update(id=i, label=result.names[int(box.cls.item())],
                       confidence=float(box.conf.item()),
                       bbox_px=box.xyxy[0].cpu().tolist(),
                       hybrid_color_geometry=hybrid)
            if hybrid["accepted"]:
                row["candidate_color"] = hybrid["color"]
                row["candidate_center_mm"] = hybrid["center_mm"]
            rows.append(row)
            cv2.imwrite(str(output / f"mask_{i:02d}.png"), binary * 255)
            if hybrid.get("center_mm"):
                xy = hybrid["center_mm"]
                box_px = calib.board_to_pixel(
                    np.asarray(hybrid["box_mm"])
                ).round().astype(int)
                color = (0, 255, 0) if hybrid["accepted"] else (0, 0, 255)
                cv2.polylines(overlay, [box_px], True, color, 2)
                pixel = calib.board_to_pixel(np.asarray([xy]))[0].round().astype(int)
                cv2.drawMarker(overlay, tuple(pixel), color)
                cv2.putText(
                    overlay,
                    f"{i} {hybrid['color']}: ({xy[0]:.1f},{xy[1]:.1f}) mm",
                    tuple(pixel), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1,
                )
    cv2.imwrite(str(output / "overlay.jpg"), overlay)
    report = {"model": str(model_path), "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "config": cfg.to_dict(), "calibration": str(Path(args.calibration).resolve()),
        "calibration_sha256": hashlib.sha256(Path(args.calibration).read_bytes()).hexdigest(),
        "calibration_meta": calib.meta, "acquired_at_unix": acquired_at, "visual_prompt": visual,
        "source": args.image or yc.snapshot_url, "device": yc.device,
        "versions": {name: importlib.metadata.version(name) for name in ("ultralytics", "torch", "torchvision", "numpy")},
        "prediction_ms": durations, "instances": rows,
        "limitation": "Hybrid colour geometry is clipped to a YOLOE instance but is not proven to be only the top face. Camera drift and physical accuracy remain unverified. No robot motion."}
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"output": str(output), "instances": len(rows),
        "hybrid_accepted": sum(r["hybrid_color_geometry"]["accepted"] for r in rows),
        "prediction_ms": durations}))


if __name__ == "__main__":
    main()
