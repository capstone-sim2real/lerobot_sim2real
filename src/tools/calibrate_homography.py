"""Per-venue calibration CLI: chessboard homography + base point + zone polygon.

Run this once per venue/session (and after any camera-mount change), with the
arm parked at home so it does not occlude the board:

    # 1) grab a frame and fit the homography from the chessboard
    python -m tools.calibrate_homography \
        --camera /dev/video0 --square-mm 25 --venue lab \
        --out src/configs/calib/lab.json

    # 2) open the saved <out>.frame.png, read off pixel coords, then re-run
    #    adding the robot base pixel and the 4 zone corner pixels:
    python -m tools.calibrate_homography \
        --image src/configs/calib/lab.json.frame.png \
        --square-mm 25 --venue lab \
        --base-px 320,470 --zone-px "200,100 400,100 400,200 200,200" \
        --out src/configs/calib/lab.json

If the chessboard detector fails (glare, blur), pass >= 4 manual pairs:
    --pair 100,50:0,0 --pair 500,60:400,0 --pair 480,400:400,300 --pair 90,380:0,300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from perception.homography import (
    PlaneCalibration,
    calibrate_from_chessboard,
    calibrate_from_pairs,
)
from tools._capture import grab_frame


def _parse_xy(text: str) -> tuple[float, float]:
    x, y = text.split(",")
    return (float(x), float(y))


def _parse_pair(text: str) -> tuple[tuple[float, float], tuple[float, float]]:
    px, mm = text.split(":")
    return (_parse_xy(px), _parse_xy(mm))


def _annotate(frame_bgr: np.ndarray, calib: PlaneCalibration) -> np.ndarray:
    out = frame_bgr.copy()
    # 50 mm grid lines projected back into the image, for eyeball validation
    corners_mm = calib.pixel_to_board(
        np.array([[0, 0], [out.shape[1] - 1, 0], [out.shape[1] - 1, out.shape[0] - 1], [0, out.shape[0] - 1]])
    )
    x_min, y_min = corners_mm.min(axis=0)
    x_max, y_max = corners_mm.max(axis=0)
    for x in np.arange(np.floor(x_min / 50) * 50, x_max, 50):
        pts = calib.board_to_pixel(np.array([[x, y_min], [x, y_max]])).astype(int)
        cv2.line(out, tuple(pts[0]), tuple(pts[1]), (0, 255, 0), 1)
    for y in np.arange(np.floor(y_min / 50) * 50, y_max, 50):
        pts = calib.board_to_pixel(np.array([[x_min, y], [x_max, y]])).astype(int)
        cv2.line(out, tuple(pts[0]), tuple(pts[1]), (0, 255, 0), 1)
    if calib.base_xy_mm is not None:
        px = calib.board_to_pixel(np.array([calib.base_xy_mm])).astype(int)[0]
        cv2.drawMarker(out, tuple(px), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(out, "base", tuple(px + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    if calib.zone_polygon_mm:
        pts = calib.board_to_pixel(np.array(calib.zone_polygon_mm)).astype(np.int32)
        cv2.polylines(out, [pts], isClosed=True, color=(255, 0, 255), thickness=2)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--camera", help="camera index or /dev/video* path")
    source.add_argument("--image", help="use a saved frame instead of grabbing one")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--square-mm", type=float, required=True, help="chessboard square edge (measure it)")
    parser.add_argument("--min-pattern", default="5,5", help="minimal inner-corner grid to search, e.g. 5,5")
    parser.add_argument("--pair", action="append", default=[], help="manual px_x,px_y:mm_x,mm_y pair (>=4 to use)")
    parser.add_argument("--base-px", help="robot base position in the image, e.g. 320,470")
    parser.add_argument("--zone-px", help="4 zone corner pixels: 'x,y x,y x,y x,y'")
    parser.add_argument("--venue", default="default")
    parser.add_argument("--out", default=None, help="default: src/configs/calib/<venue>.json")
    args = parser.parse_args(argv)

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Cannot read image: {args.image}", file=sys.stderr)
            return 2
    else:
        cam = int(args.camera) if args.camera.isdigit() else args.camera
        frame = grab_frame(cam, width=args.width, height=args.height)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if len(args.pair) >= 4:
        H = calibrate_from_pairs([_parse_pair(p) for p in args.pair])
        info = {"mode": "manual_pairs", "num_pairs": len(args.pair)}
    else:
        min_pattern = tuple(int(v) for v in args.min_pattern.split(","))
        try:
            H, info = calibrate_from_chessboard(gray, args.square_mm, min_pattern)
            info["mode"] = "chessboard"
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1

    calib = PlaneCalibration(
        H=H,
        image_size=(frame.shape[1], frame.shape[0]),
        square_mm=args.square_mm,
        base_xy_mm=None,
        zone_polygon_mm=None,
        meta={"venue": args.venue, **info},
    )
    if args.base_px:
        calib.base_xy_mm = tuple(calib.pixel_to_board(np.array([_parse_xy(args.base_px)]))[0])
    if args.zone_px:
        corners_px = np.array([_parse_xy(p) for p in args.zone_px.split()])
        if len(corners_px) != 4:
            print("ERROR: --zone-px needs exactly 4 points", file=sys.stderr)
            return 2
        calib.zone_polygon_mm = [tuple(p) for p in calib.pixel_to_board(corners_px)]

    out = Path(args.out) if args.out else Path(f"src/configs/calib/{args.venue}.json")
    calib.save(out)
    cv2.imwrite(str(out) + ".frame.png", frame)
    cv2.imwrite(str(out) + ".annotated.png", _annotate(frame, calib))

    print(f"Calibration saved: {out}")
    print(f"  mode: {info.get('mode')}  detail: {info}")
    if calib.base_xy_mm is None:
        print("  WARNING: no --base-px given; SELECT cannot rank targets until it is set.")
    if calib.zone_polygon_mm is None:
        print("  WARNING: no --zone-px given; placed blocks will not be excluded from SELECT.")
    print(f"  Inspect {out}.annotated.png: the green grid must land on the board's 50 mm lines.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
