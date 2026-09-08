"""Interactively record one FK/image calibration point and a clean snapshot."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="point label, e.g. P1")
    parser.add_argument("--snapshot-url", default="http://127.0.0.1:8090/snapshot/shoulder.jpg")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "docs/calibration")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean_image = args.output_dir / f"{args.name.lower()}_top.jpg"
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
    with urlopen(args.snapshot_url, timeout=5) as response:
        clean_image.write_bytes(response.read())
    print(f"clean image saved: {clean_image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
