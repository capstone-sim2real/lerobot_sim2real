"""Record resumable calibration points through the active teleop process."""

import argparse
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

from tools.calibration_records import (
    read_rows,
    completed_points,
    next_attempt_name,
    export_accepted,
)
from tools.calibration_pixels import complete_pixel_pair, make_pixel_preview
from tools.session_io import (
    parse_session_args,
    read_telemetry,
    request_capture,
    snapshot_bytes,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--count", type=int)
    parser.add_argument("--wrist-limit", type=float)
    parser.add_argument("--snapshot-url")
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--telemetry", type=Path)
    args = parse_session_args(
        parser,
        argv,
        {
            "runtime_dir": "runtime_dir",
            "wrist_limit": "wrist_limit_deg",
            "snapshot_url": "snapshot_url",
            "timeout": "capture_timeout_s",
        },
    )
    if args.count is None:
        args.count = args.app_config.calibration_capture.point_count
    if args.count <= 0 or args.timeout <= 0 or args.wrist_limit < 0:
        parser.error("count/timeout must be positive and wrist-limit non-negative")
    args.request = args.request or args.runtime_dir / "capture_request.json"
    args.telemetry = args.telemetry or args.runtime_dir / "observations.jsonl"
    return args


def choose_attempt(folder, rows, index):
    pattern = re.compile(rf"P{index}(?:_retry\d*)?", re.IGNORECASE)
    candidates = [row for row in rows if pattern.fullmatch(row["name"])]
    if candidates:
        name = candidates[-1]["name"]
        choice = (
            input(
                f"{name} 팔 기록이 있습니다. 블록이 그대로면 Enter, 새로 기록하려면 r: "
            )
            .strip()
            .lower()
        )
        if not choice:
            return name, True
        if choice != "r":
            raise ValueError("Enter 또는 r을 입력하세요. 다시 실행하면 이어집니다.")
    return next_attempt_name(folder, rows, index), False


def capture_pose(args, index, name):
    input(f"[{index}/{args.count}] 기준점을 맞추고 손목 회전을 0 근처로 둔 뒤 Enter: ")
    row = read_telemetry(args.telemetry, args.settings.telemetry_tail_bytes)
    if not 0 <= time.time() - row["time"] < args.settings.capture_stale_s:
        raise RuntimeError("텔레옵 기록이 오래됐습니다. 첫 터미널 상태를 확인하세요.")
    roll = row["follower"]["wrist_roll"]
    print(f"wrist_roll={roll:+.2f} deg")
    if abs(roll) > args.wrist_limit:
        raise RuntimeError("손목을 0 근처로 조정한 후 다시 실행하세요.")
    record = request_capture(
        args.request, args.output_dir, name, args.timeout, args.settings.poll_interval_s
    )
    subprocess.run(
        [sys.executable, "-m", "tools.finalize_teleop_capture", str(record)], check=True
    )


def review_pixels(args, name):
    input(f"{name}: 팔 기록 완료. 집게를 벌려 블록을 그대로 두고 팔만 비킨 뒤 Enter: ")
    image = args.output_dir / (name.lower() + "_clean.jpg")
    if not image.exists():
        image.write_bytes(
            snapshot_bytes(args.snapshot_url, args.settings.snapshot_timeout_s)
        )
    centre, preview = make_pixel_preview(image, args.app_config)
    print(f"블록 중심: {centre}\n확인 이미지: {preview.resolve()}")
    print("원격 실행이면 이미지를 로컬로 복사해서 확인하세요.")
    print("xdg-open " + shlex.quote(str(preview.resolve())))
    if (
        input("중심 표시가 맞고 블록이 움직이지 않았으면 yes: ").strip().lower()
        != "yes"
    ):
        raise RuntimeError("점은 미확정으로 보존했습니다.")
    complete_pixel_pair(args.output_dir / "points.csv", name, centre, image.name)


def run_session(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, args.count + 1):
        _, rows = read_rows(args.output_dir)
        if index in completed_points(rows, args.count):
            print(f"P{index}: 이미 완료, 건너뜁니다.")
            continue
        name, reuse = choose_attempt(args.output_dir, rows, index)
        if not reuse:
            capture_pose(args, index, name)
        review_pixels(args, name)
        export_accepted(args.output_dir, args.count)
        print(f"{name} 완료. 다음 위치로 블록을 옮기세요.")
    export_accepted(args.output_dir, args.count)
    print(f"{args.count}점 기록 완료. 보정 적합/적용은 아직 하지 않았습니다.")


def main(argv=None):
    args = parse_args(argv)
    try:
        run_session(args)
    except (KeyboardInterrupt, EOFError):
        print("\n기록만 종료합니다. 텔레옵은 유지됩니다.")
    except Exception as exc:
        print("중단:", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
