"""Fit the pixel -> robot-base-frame homography from points.csv.

Each row pairs a pixel (u_px, v_px) with an FK position (x_m, y_m, z_m)
recorded while the gripper held a block at that pixel (AGENTS.md §6: the
calibration plane is the block's own height, not the bare table, so there is
no parallax term to correct for). The robot base is the origin, so
``base_xy_mm = (0, 0)`` and select.py's nearest-first math needs no offset.

    python -m tools.calibrate_base_frame \
        --points docs/calibration/points.csv \
        --out src/configs/calib/venue_lab.json

Reports fit RMS and a leave-one-out (LOO) max error: refit on 8 points and
predict the 9th, repeated for each point. LOO is the honest number — RMS
alone can look good even when the fit is quietly overfitting 9 points.
Gate (AGENTS.md §6): RMS < 5mm, LOO max < 8mm.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from perception.homography import PlaneCalibration, calibrate_from_pairs


def load_points(csv_path: Path) -> list[dict]:
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    incomplete = [r["name"] for r in rows if not r["u_px"] or not r["x_m"]]
    if incomplete:
        raise ValueError(f"points.csv rows missing pixel or FK data: {incomplete}")
    return rows


def to_pairs(rows: list[dict]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    return [
        ((float(r["u_px"]), float(r["v_px"])), (float(r["x_m"]) * 1000.0, float(r["y_m"]) * 1000.0))
        for r in rows
    ]


def residuals_mm(H: np.ndarray, pairs) -> np.ndarray:
    px = np.array([p for p, _ in pairs], dtype=np.float64)
    mm = np.array([m for _, m in pairs], dtype=np.float64)
    proj = cv2.perspectiveTransform(px.reshape(-1, 1, 2), H).reshape(-1, 2)
    return np.linalg.norm(proj - mm, axis=1)


def leave_one_out_errors(pairs) -> np.ndarray:
    errors = np.empty(len(pairs))
    for i in range(len(pairs)):
        train = pairs[:i] + pairs[i + 1 :]
        H = calibrate_from_pairs(train)
        px = np.array([pairs[i][0]], dtype=np.float64)
        mm = np.array(pairs[i][1], dtype=np.float64)
        proj = cv2.perspectiveTransform(px.reshape(-1, 1, 2), H).reshape(-1, 2)[0]
        errors[i] = float(np.linalg.norm(proj - mm))
    return errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", type=Path, default=Path("docs/calibration/points.csv"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--image-size", default="1280x720", help="WxH the calibration frames were captured at")
    ap.add_argument("--rms-max-mm", type=float, default=5.0)
    ap.add_argument("--loo-max-mm", type=float, default=8.0)
    args = ap.parse_args(argv)

    rows = load_points(args.points)
    if len(rows) < 6:
        raise ValueError(f"Need several well-spread points for a trustworthy LOO check, got {len(rows)}")
    pairs = to_pairs(rows)

    H = calibrate_from_pairs(pairs)
    res = residuals_mm(H, pairs)
    loo = leave_one_out_errors(pairs)

    print(f"{len(rows)} points: {[r['name'] for r in rows]}")
    print(f"{'name':6s} {'resid_mm':>10s} {'loo_mm':>10s}")
    for r, e_fit, e_loo in zip(rows, res, loo):
        flag = " <-- LOO exceeds gate" if e_loo > args.loo_max_mm else ""
        print(f"{r['name']:6s} {e_fit:10.2f} {e_loo:10.2f}{flag}")

    rms_mm = float(np.sqrt((res**2).mean()))
    loo_max_mm = float(loo.max())
    z_vals = np.array([float(r["z_m"]) for r in rows]) * 1000.0
    print()
    print(f"fit RMS      : {rms_mm:.2f} mm  (gate < {args.rms_max_mm} mm)")
    print(f"fit max      : {res.max():.2f} mm")
    print(f"LOO max      : {loo_max_mm:.2f} mm  (gate < {args.loo_max_mm} mm)")
    print(f"LOO mean     : {loo.mean():.2f} mm")
    print(f"grasp z (mm) : mean={z_vals.mean():.1f} std={z_vals.std():.1f} range=[{z_vals.min():.1f}, {z_vals.max():.1f}]")

    passed = rms_mm < args.rms_max_mm and loo_max_mm < args.loo_max_mm
    print(f"\n{'PASS' if passed else 'FAIL'}")

    w, h = (int(v) for v in args.image_size.lower().split("x"))
    calib = PlaneCalibration(
        H=H,
        image_size=(w, h),
        square_mm=1.0,  # unused by this FK-direct calibration mode (AGENTS.md §6)
        base_xy_mm=(0.0, 0.0),
        zone_polygon_mm=None,
        meta={
            "mode": "fk_direct_pairs",
            "num_points": len(rows),
            "rms_mm": rms_mm,
            "loo_max_mm": loo_max_mm,
            "grasp_z_mm_mean": float(z_vals.mean()),
            "grasp_z_mm_std": float(z_vals.std()),
            "points_csv": str(args.points),
            "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    calib.save(args.out)
    print(f"\nsaved: {args.out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
