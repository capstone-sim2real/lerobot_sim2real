"""Record all calibration points in one resumable interactive session."""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from pathlib import Path

from config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
from tools.calibration_records import (
    POINT_NAME,
    PAIR_FIELDS,
    read_rows,
    completed_points,
    next_attempt_name,
    export_accepted,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "src/configs/default.yaml"
    )
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument(
        "--snapshot-url", default="http://127.0.0.1:8090/snapshot/shoulder.jpg"
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config, overrides=args.overrides)
    count = cfg.calibration_capture.point_count
    if not isinstance(count, int) or isinstance(count, bool) or count < 6:
        parser.error("calibration_capture.point_count must be an integer >= 6")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        while True:
            complete = export_accepted(args.output_dir, count)
            if len(complete) == count:
                print(
                    f"\n{count}/{count} complete. Fit input: {args.output_dir / 'accepted_points.csv'}"
                )
                print(
                    "Calibration has not been fitted or activated; original attempts are preserved."
                )
                return 0
            index = next(i for i in range(1, count + 1) if i not in complete)
            _, rows = read_rows(args.output_dir)
            name = next_attempt_name(args.output_dir, rows, index)
            print(
                f"\n=== {len(complete)}/{count} saved; next P{index} ({name}) ===",
                flush=True,
            )
            command = [
                sys.executable,
                "-m",
                "tools.capture_calibration_point",
                name,
                "--output-dir",
                str(args.output_dir),
                "--config",
                str(args.config),
                "--snapshot-url",
                args.snapshot_url,
            ]
            for override in args.overrides:
                command.extend(["--set", override])
            result = subprocess.run(command, check=False)
            if result.returncode:
                answer = input(
                    "Point incomplete. Enter to retry this point, q to stop: "
                )
                if answer.strip().lower() == "q":
                    return 1
    except (KeyboardInterrupt, EOFError):
        print("\nStopped. Run the same command to resume; saved points are retained.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
