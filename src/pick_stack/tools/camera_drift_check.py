"""Measure how far the fixed top-down camera has moved since calibration.

The pixel->robot homography is baked at calibration time, so if the camera
shifts, every block coordinate is silently wrong and nothing downstream can
tell (AGENTS.md §8). This tool makes that failure visible and gives the
mount a pass/fail number.

    # 1) after bolting the camera down, store the baseline
    python -m pick_stack.tools.camera_drift_check --save-reference

    # 2) any time later — start of a session, after touching a cable
    python -m pick_stack.tools.camera_drift_check

    # 3) prove the mount holds: sample for 10 minutes, log to CSV
    python -m pick_stack.tools.camera_drift_check --watch 600

Exits non-zero when drift exceeds --max-drift-px, so it can gate a run.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from pick_stack.perception.board import detect_corners, match_corners, median_square_px
from pick_stack.tools._capture import grab

DEFAULT_REFERENCE = Path("docs/calibration/camera_reference.json")
DEFAULT_SOURCE = "http://127.0.0.1:8090/snapshot/shoulder.jpg"


def _measure(reference: np.ndarray, current: np.ndarray, max_match_px: float) -> dict:
    ref_m, cur_m = match_corners(reference, current, max_match_px)
    if len(ref_m) == 0:
        return {"matched": 0, "max_px": float("nan"), "rms_px": float("nan"), "median_px": float("nan")}
    d = np.linalg.norm(cur_m - ref_m, axis=1)
    return {
        "matched": int(len(d)),
        "median_px": float(np.median(d)),
        "p95_px": float(np.percentile(d, 95)),
        "rms_px": float(np.sqrt((d**2).mean())),
        "max_px": float(d.max()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="snapshot URL or /dev/video* / index")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--save-reference", action="store_true", help="store the current frame as the baseline")
    parser.add_argument("--watch", type=float, default=0.0, help="monitor for N seconds instead of one check")
    parser.add_argument("--interval", type=float, default=10.0, help="seconds between samples while watching")
    parser.add_argument("--max-drift-px", type=float, default=2.0, help="fail above this drift")
    parser.add_argument("--csv", type=Path, default=None, help="append samples here (default: docs/calibration/drift_<ts>.csv)")
    args = parser.parse_args(argv)

    frame = grab(args.source)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners = detect_corners(gray)
    if len(corners) < 12:
        print(f"ERROR: only {len(corners)} chessboard corners found — check focus, lighting and framing.", file=sys.stderr)
        return 2
    square_px = median_square_px(corners)

    if args.save_reference:
        args.reference.parent.mkdir(parents=True, exist_ok=True)
        args.reference.write_text(json.dumps({
            "corners_px": corners.tolist(),
            "image_size": [int(frame.shape[1]), int(frame.shape[0])],
            "median_square_px": square_px,
            "source": args.source,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, indent=2))
        preview = args.reference.with_suffix(".preview.png")
        marked = frame.copy()
        for x, y in corners.astype(int):
            cv2.drawMarker(marked, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 6, 1)
        cv2.imwrite(str(preview), marked)
        print(f"Reference saved: {len(corners)} corners, square {square_px:.1f} px -> {args.reference}")
        print(f"Preview (verify the crosses sit on real corners): {preview}")
        return 0

    if not args.reference.exists():
        print(f"ERROR: no reference at {args.reference}. Run with --save-reference first.", file=sys.stderr)
        return 2
    payload = json.loads(args.reference.read_text())
    reference = np.array(payload["corners_px"], dtype=np.float64)
    # half a square: past that, the nearest corner may be the wrong one, since
    # a chessboard repeats every two squares (AGENTS.md §6).
    max_match_px = 0.5 * float(payload.get("median_square_px") or square_px)

    def sample() -> dict:
        f = grab(args.source)
        c = detect_corners(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        m = _measure(reference, c, max_match_px)
        m["detected"] = int(len(c))
        return m

    if args.watch <= 0:
        m = sample()
        ok = m["matched"] > 0 and m["p95_px"] <= args.max_drift_px
        print(f"matched {m['matched']}/{len(reference)} corners  "
              f"median {m['median_px']:.2f}  p95 {m['p95_px']:.2f}  rms {m['rms_px']:.2f}  "
              f"max {m['max_px']:.2f} px")
        print(f"{'PASS' if ok else 'FAIL'} (threshold {args.max_drift_px:.1f} px, match radius {max_match_px:.1f} px)")
        return 0 if ok else 1

    csv_path = args.csv or Path("docs/calibration") / f"drift_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.watch
    worst = 0.0
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(("elapsed_s", "detected", "matched", "median_px", "p95_px", "rms_px", "max_px"))
        start = time.monotonic()
        while True:
            m = sample()
            elapsed = time.monotonic() - start
            writer.writerow((f"{elapsed:.1f}", m["detected"], m["matched"],
                             f"{m['median_px']:.3f}", f"{m['p95_px']:.3f}",
                             f"{m['rms_px']:.3f}", f"{m['max_px']:.3f}"))
            fh.flush()
            if m["matched"]:
                worst = max(worst, m["p95_px"])
            print(f"  t={elapsed:6.1f}s  matched {m['matched']:3d}  p95 {m['p95_px']:5.2f} px  "
                  f"(max {m['max_px']:5.2f}, worst p95 so far {worst:5.2f})")
            if time.monotonic() >= deadline:
                break
            time.sleep(max(0.0, min(args.interval, deadline - time.monotonic())))
    ok = worst <= args.max_drift_px
    print(f"\nworst p95 drift over {args.watch:.0f}s: {worst:.2f} px -> {'PASS' if ok else 'FAIL'} "
          f"(threshold {args.max_drift_px:.1f} px)")
    print(f"log: {csv_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
