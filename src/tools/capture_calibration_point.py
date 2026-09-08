"""Interactively record one FK/image calibration point and a clean snapshot."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from tools.session_io import snapshot_bytes

from config import load_config
from tools.calibration_pixels import complete_pixel_pair, make_pixel_preview


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="point label, e.g. P1")
    parser.add_argument(
        "--snapshot-url", default="http://127.0.0.1:8090/snapshot/shoulder.jpg"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments/current/calibration",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "src/configs/default.yaml"
    )
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument(
        "--detect-image",
        type=Path,
        help="Preview pixels from an existing clean image; no robot access or CSV update",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.overrides)
    if args.detect_image is not None:
        centre, preview = make_pixel_preview(args.detect_image, cfg)
        print(f"candidate pixels: {centre}; preview: {preview}; CSV unchanged")
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean_image = args.output_dir / f"{args.name.lower()}_clean.jpg"
    if clean_image.exists() and not args.overwrite:
        raise FileExistsError(f"{clean_image} exists; choose a new point name")
    print("1/2 Close the jaws around the block, hold the arm steady, then press Enter.")
    input()
    command = [
        sys.executable,
        "-m",
        "tools.record_calibration_point",
        args.name,
        "--output-dir",
        str(args.output_dir),
        "--snapshot-url",
        args.snapshot_url,
    ]
    if args.overwrite:
        command.append("--overwrite")
    subprocess.run(command, check=True)
    print("2/2 Open the jaws, move only the arm away, then press Enter.")
    input()
    clean_image.write_bytes(
        snapshot_bytes(args.snapshot_url, cfg.session_tools.snapshot_timeout_s)
    )
    print(f"clean image saved: {clean_image}")
    centre, preview = make_pixel_preview(clean_image, cfg)
    print(f"Open preview: {preview}")
    print(f"Candidate pixel: u={centre[0]:.3f}, v={centre[1]:.3f}")
    print("Confirm the cross is at the TOP FACE centre and the block did not move.")
    if (
        input("Type yes to save the pixel pair (anything else leaves it incomplete): ")
        .strip()
        .lower()
        != "yes"
    ):
        print(
            "Pixel pair not accepted. Keep the evidence and retry with a new point name."
        )
        return 1
    complete_pixel_pair(
        args.output_dir / "points.csv", args.name, centre, clean_image.name
    )
    print(f"Complete FK/pixel pair saved: {args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
