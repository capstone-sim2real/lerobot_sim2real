"""Register the fixed red-tape target zone without replacing calibration H.

Preview only (default):
    so101-zone-calibrate

Persist ``zone_polygon_mm`` after inspecting the preview:
    so101-zone-calibrate --write
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np

from camera.client import DEFAULT_SHOULDER_SNAPSHOT_URL, fetch_snapshot
from config import load_config
from perception import PlaneCalibration, detect_zone_inner_polygon, zone_slot_centres
from perception.zone import ordered_zone_corners


def _annotate(
    frame: np.ndarray,
    polygon_px: list[tuple[float, float]],
    slots_mm: list[tuple[float, float]],
    calib: PlaneCalibration,
) -> np.ndarray:
    out = frame.copy()
    points = np.rint(np.asarray(polygon_px)).astype(np.int32)
    cv2.polylines(out, [points], True, (255, 0, 255), 3)
    for index, pixel in enumerate(calib.board_to_pixel(np.asarray(slots_mm))):
        p = tuple(np.rint(pixel).astype(int))
        cv2.drawMarker(out, p, (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(out, str(index), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", help="saved BGR image instead of the camera snapshot")
    source.add_argument("--snapshot", default=None, help="camera snapshot URL")
    parser.add_argument("--config", default="src/configs/default.yaml")
    parser.add_argument("--calib", default=None, help="calibration JSON to update")
    parser.add_argument("--preview", default="/tmp/so101-zone-preview.png")
    parser.add_argument("--samples", type=int, default=5, help="fresh snapshots to median (ignored with --image)")
    parser.add_argument("--sample-interval-s", type=float, default=0.1)
    parser.add_argument("--write", action="store_true", help="persist zone_polygon_mm after previewing")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    calib_path = Path(args.calib or cfg.perception.calibration_path)
    calib = PlaneCalibration.load(calib_path)
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {args.image}")
        samples = [frame]
    else:
        if args.samples <= 0:
            raise ValueError("--samples must be positive")
        samples = []
        for index in range(args.samples):
            samples.append(fetch_snapshot(args.snapshot or cfg.perception.snapshot_url or DEFAULT_SHOULDER_SNAPSHOT_URL))
            if index + 1 < args.samples and args.sample_interval_s > 0:
                time.sleep(args.sample_interval_s)
        frame = samples[-1]

    measured_px = []
    for sample in samples:
        _polygon_mm, polygon_px = detect_zone_inner_polygon(sample, calib, cfg.perception)
        measured_px.append(polygon_px)
    median_px = np.median(np.asarray(measured_px, dtype=np.float64), axis=0)
    median_mm = calib.pixel_to_board(median_px)
    ordered_mm_array = ordered_zone_corners(
        [tuple(p) for p in median_mm], calib.base_xy_mm or (0.0, 0.0)
    )
    polygon_mm = [tuple(float(v) for v in point) for point in ordered_mm_array]
    polygon_px = [
        tuple(float(v) for v in point)
        for point in calib.board_to_pixel(np.asarray(polygon_mm, dtype=np.float64))
    ]
    candidate = PlaneCalibration(
        H=calib.H.copy(), image_size=calib.image_size, square_mm=calib.square_mm,
        base_xy_mm=calib.base_xy_mm, zone_polygon_mm=polygon_mm, meta=dict(calib.meta),
    )
    slots = zone_slot_centres(candidate, cfg.task1.slot_uv)
    preview = Path(args.preview)
    preview.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(preview), _annotate(frame, polygon_px, slots, candidate)):
        raise OSError(f"Could not write preview: {preview}")

    lengths = [math.dist(polygon_mm[i], polygon_mm[(i + 1) % 4]) for i in range(4)]
    print("zone corners (far-left, far-right, near-right, near-left):")
    for index, (pixel, mm) in enumerate(zip(polygon_px, polygon_mm, strict=True)):
        print(f"  {index}: px=({pixel[0]:.1f}, {pixel[1]:.1f})  base=({mm[0]:.1f}, {mm[1]:.1f}) mm")
    print("edge lengths:", ", ".join(f"{length:.1f}mm" for length in lengths))
    print("slots:", ", ".join(f"({x:.1f}, {y:.1f})" for x, y in slots))
    print(f"preview: {preview}")

    if not args.write:
        print("preview only; pass --write to save without changing H/base/grasp metadata")
        return 0

    candidate.meta.update(
        {
            "zone_source": "red_tape_inner_contour",
            "zone_polygon_px": [[float(x), float(y)] for x, y in polygon_px],
            "zone_calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    candidate.save(calib_path)
    print(f"saved zone polygon: {calib_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
