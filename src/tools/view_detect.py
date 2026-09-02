"""Detection debug CLI: run the detector on a frame and save an annotated view.

    python -m tools.view_detect \
        --snapshot http://127.0.0.1:8090/snapshot/shoulder.jpg \
        --calib src/configs/calib/lab.json \
        --out /tmp/detect.png

Use --image for saved frames, --set to try thresholds without editing YAML:

    python -m tools.view_detect --image frame.png --calib lab.json \
        --set perception.area_mm2_min=800
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from config import load_config
from camera.client import DEFAULT_SHOULDER_SNAPSHOT_URL, fetch_snapshot
from perception import PlaneCalibration, detect_blocks, select_target, target_id_for
from tools._capture import grab_frame

_COLORS_BGR = {
    "red": (0, 0, 255),
    "yellow": (0, 220, 220),
    "green": (0, 200, 0),
    "blue": (255, 80, 0),
    "wood": (80, 150, 210),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--snapshot", default=DEFAULT_SHOULDER_SNAPSHOT_URL, help="camera.server JPEG URL (default: shoulder)")
    source.add_argument("--camera", help="direct camera index or /dev/video* path; stop camera.server first")
    source.add_argument("--image", help="saved frame (BGR, e.g. from calibrate_homography)")
    parser.add_argument("--calib", required=True)
    parser.add_argument("--config", default="src/configs/default.yaml")
    parser.add_argument("--set", action="append", default=[], dest="overrides", help="key.path=value")
    parser.add_argument("--out", default="/tmp/pick_stack_detect.png")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=args.overrides)
    calib = PlaneCalibration.load(args.calib)

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Cannot read image: {args.image}", file=sys.stderr)
            return 2
    elif args.camera:
        cam = int(args.camera) if args.camera.isdigit() else args.camera
        frame = grab_frame(cam, width=1280, height=720)
    else:
        frame = fetch_snapshot(args.snapshot)

    detections = detect_blocks(frame, calib, cfg.perception, is_rgb=False)
    try:
        selection = select_target(detections, calib, cfg.select)
    except ValueError as e:  # calibration without base point: still show detections
        print(f"NOTE: {e}")
        selection = None

    mm_per_px = cfg.perception.rectified_mm_per_px
    rectified, (ox, oy) = calib.rectify(frame, mm_per_px)
    canvas = rectified.copy()

    def mm_to_px(pt_mm) -> tuple[int, int]:
        return (int((pt_mm[0] - ox) / mm_per_px), int((pt_mm[1] - oy) / mm_per_px))

    if calib.zone_polygon_mm:
        pts = np.array([mm_to_px(p) for p in calib.zone_polygon_mm], dtype=np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=(255, 0, 255), thickness=2)
    if calib.base_xy_mm is not None:
        cv2.drawMarker(canvas, mm_to_px(calib.base_xy_mm), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)

    print(f"{len(detections)} detection(s):")
    for det in detections:
        det_id = target_id_for(det, cfg.select.target_cell_mm)
        is_target = selection is not None and selection.target is det
        print(
            f"  {'>> ' if is_target else '   '}{det_id:<16} center=({det.center_mm[0]:7.1f},{det.center_mm[1]:7.1f})mm "
            f"area={det.area_mm2:6.0f}mm2 aspect={det.aspect:.2f} solidity={det.solidity:.2f} fill={det.fill:.2f}"
        )
        box = np.array([mm_to_px(p) for p in det.box_mm], dtype=np.int32)
        color = _COLORS_BGR.get(det.color, (255, 255, 255))
        cv2.polylines(canvas, [box], isClosed=True, color=color, thickness=3 if is_target else 1)
        cv2.putText(canvas, det_id, mm_to_px(det.center_mm), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    if selection is not None:
        print(f"selected: {selection.target_id}  (eligible remaining: {selection.remaining})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, canvas)
    print(f"Annotated rectified view saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
