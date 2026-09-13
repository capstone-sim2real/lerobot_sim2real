"""Shared file-based session IO. Never opens a robot serial port."""

import argparse
import json
import time
from pathlib import Path
from urllib.request import urlopen

from config import load_config

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/default.yaml"


def parse_session_args(parser, argv, defaults):
    """Apply YAML/--set defaults, with explicit legacy CLI flags taking precedence."""
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    bootstrap.add_argument("--set", action="append", default=[], dest="overrides")
    known, _ = bootstrap.parse_known_args(argv)
    cfg = load_config(known.config, overrides=known.overrides)
    parser.set_defaults(
        **{name: getattr(cfg.session_tools, field) for name, field in defaults.items()}
    )
    args = parser.parse_args(argv)
    args.app_config = cfg
    args.settings = cfg.session_tools
    for name in (
        "snapshot_timeout_s",
        "poll_interval_s",
        "telemetry_stale_s",
        "capture_stale_s",
        "capture_timeout_s",
        "telemetry_interval_s",
        "startup_poll_s",
    ):
        if getattr(args.settings, name) <= 0:
            parser.error(f"session_tools.{name} must be positive")
    if args.settings.telemetry_tail_bytes <= 0:
        parser.error("session_tools.telemetry_tail_bytes must be positive")
    return args


def read_telemetry(path, tail_bytes):
    """Use the latest complete record, tolerating a concurrent partial append."""
    with Path(path).open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - tail_bytes))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            row = json.loads(line)
            if isinstance(row, dict) and "follower" in row and "time" in row:
                return row
        except (ValueError, UnicodeError):
            continue
    raise ValueError("No complete telemetry record")


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data), encoding="utf-8")
    temporary.replace(path)


def snapshot_bytes(url, timeout_s):
    with urlopen(url, timeout=timeout_s) as response:
        data = response.read()
    if not data.startswith(b"\xff\xd8"):
        raise ValueError("Expected JPEG")
    return data


def request_capture(request, folder, name, timeout_s, poll_interval_s):
    request = Path(request)
    if request.exists() or request.with_suffix(".processing").exists():
        raise RuntimeError("A capture request is already pending")
    result = Path(folder) / (name.lower() + "_live_result.json")
    if result.exists():
        result.unlink()  # Failed status only; retry naming protects actual images/records.
    atomic_json(request, {"name": name, "output_dir": str(Path(folder).resolve())})
    deadline = time.monotonic() + timeout_s
    while not result.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(
                "Capture response timed out; do not repeat the pending request"
            )
        time.sleep(poll_interval_s)
    status = json.loads(result.read_text())
    if not status["ok"]:
        raise RuntimeError(status["error"])
    return Path(status["record"])
