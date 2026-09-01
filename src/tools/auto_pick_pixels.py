"""Auto-fill points.csv pixel columns from a solid-color calibration block.

Alternative to tools/pick_pixels.py's manual click when every calibration
point used the same block colour against the black/white board: a colour
threshold finds the block far more precisely (sub-pixel centroid via image
moments) than a mouse click, and needs no human per point.

    python -m tools.auto_pick_pixels --color green

Writes u_px/v_px into points.csv and saves an annotated overlay per point
next to each source image (<name>_detected.png) so the fit can be sanity
checked before trusting it.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

# Loose HSV bands (OpenCV hue 0-179), wide enough to catch a single saturated
# block colour under normal room lighting. Tightened later per AGENTS.md §9
# tuning is for the real-time detector, not this one-shot calibration helper.
_HSV_BANDS = {
    "red": [(0, 90, 60, 8, 255, 255), (172, 90, 60, 179, 255, 255)],
    "yellow": [(20, 90, 80, 34, 255, 255)],
    "green": [(40, 60, 50, 85, 255, 255)],
    "blue": [(95, 90, 50, 130, 255, 255)],
    "wood": [(8, 60, 60, 22, 200, 255)],
}


def find_block_centroid(frame_bgr: np.ndarray, color: str, min_area_px: float = 200.0):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo_h, lo_s, lo_v, hi_h, hi_s, hi_v in _HSV_BANDS[color]:
        mask |= cv2.inRange(hsv, (lo_h, lo_s, lo_v), (hi_h, hi_s, hi_v))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area_px]
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    m = cv2.moments(largest)
    if m["m00"] == 0:
        return None
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    return cx, cy, largest, cv2.contourArea(largest)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", type=Path, default=Path("docs/calibration/points.csv"))
    ap.add_argument("--color", required=True, choices=sorted(_HSV_BANDS))
    ap.add_argument("--min-area-px", type=float, default=200.0)
    args = ap.parse_args(argv)

    rows = list(csv.DictReader(args.points.open(newline="", encoding="utf-8")))
    image_dir = args.points.parent
    found, missing = 0, []
    for row in rows:
        img_path = image_dir / row["image"]
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  {row['name']}: cannot read {img_path}")
            missing.append(row["name"])
            continue
        result = find_block_centroid(frame, args.color, args.min_area_px)
        if result is None:
            print(f"  {row['name']}: no {args.color} blob found in {img_path.name}")
            missing.append(row["name"])
            continue
        cx, cy, contour, area = result
        row["u_px"], row["v_px"] = f"{cx:.2f}", f"{cy:.2f}"
        found += 1

        overlay = frame.copy()
        cv2.drawContours(overlay, [contour], -1, (0, 255, 255), 2)
        cv2.drawMarker(overlay, (int(round(cx)), int(round(cy))), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
        out_path = image_dir / f"{img_path.stem}_detected.png"
        cv2.imwrite(str(out_path), overlay)
        print(f"  {row['name']}: centroid=({cx:.1f},{cy:.1f})  area_px={area:.0f}  -> {out_path.name}")

    with args.points.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"\n{found}/{len(rows)} points filled" + (f", missing: {missing}" if missing else ""))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
