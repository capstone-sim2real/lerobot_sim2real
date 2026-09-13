#!/usr/bin/env python3
"""Serve SO-101 USB cameras as MJPEG streams over HTTP."""

from __future__ import annotations

import argparse
import signal
import socket
import threading
from pathlib import Path
from typing import Any


from camera.http import (
    CameraServer,
    CameraRequestHandler as CameraRequestHandler,
    BOUNDARY as BOUNDARY,
)
from camera.cross_calibration import CrossCalibration
from camera.recorder import FrameRecorder
from camera.stream import CameraStream, FrameCallback as FrameCallback

DEFAULT_OVERLAY_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
)


def local_addresses() -> list[str]:
    addresses = ["127.0.0.1"]
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            address = sock.getsockname()[0]
            if address not in addresses:
                addresses.append(address)
    except OSError:
        pass
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            address = info[4][0]
            if address not in addresses:
                addresses.append(address)
    except socket.gaierror:
        pass
    return addresses


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve SO-101 cameras as a browser MJPEG page."
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="HTTP bind host. Default: 0.0.0.0"
    )
    parser.add_argument(
        "--port", type=int, default=8090, help="HTTP port. Default: 8090"
    )
    parser.add_argument(
        "--shoulder-device",
        default="/dev/video0",
        help="Shoulder camera device. Default: /dev/video0",
    )
    parser.add_argument(
        "--wrist-device", default="", help="Optional wrist camera device"
    )
    parser.add_argument(
        "--width", type=int, default=1280, help="Capture width. Default: 1280"
    )
    parser.add_argument(
        "--height", type=int, default=720, help="Capture height. Default: 720"
    )
    parser.add_argument("--fps", type=int, default=30, help="Capture FPS. Default: 30")
    parser.add_argument(
        "--fourcc", default="MJPG", help="Capture FOURCC. Default: MJPG"
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=80,
        help="Stream JPEG quality 1-100. Default: 80",
    )
    parser.add_argument(
        "--save-dir",
        default="",
        help="Directory for periodic JPEGs; disabled unless --save-interval-s is positive.",
    )
    parser.add_argument(
        "--save-interval-s",
        type=float,
        default=0.0,
        help="Seconds between saved frames per camera. Default: 0 (disabled).",
    )
    parser.add_argument(
        "--save-on-change",
        action="store_true",
        help="Save only when the scene differs enough from the last saved frame.",
    )
    parser.add_argument(
        "--change-threshold",
        type=float,
        default=8.0,
        help="Mean grayscale difference required by --save-on-change. Default: 8.",
    )
    parser.add_argument(
        "--max-save-interval-s",
        type=float,
        default=10.0,
        help="Force a status frame this often with --save-on-change. Default: 10.",
    )
    parser.add_argument(
        "--cross-calibration-path",
        default="/tmp/so101-cross-calibration.json",
        help="Persistent cross-calibration session JSON. Default: /tmp/so101-cross-calibration.json",
    )
    parser.add_argument(
        "--gripper-reference-path",
        type=Path,
        default=Path("/tmp/so101-gripper-reference.json"),
    )
    overlay_group = parser.add_mutually_exclusive_group()
    overlay_group.add_argument(
        "--overlay-config",
        default=str(DEFAULT_OVERLAY_CONFIG),
        help="Detection overlay project YAML. Enabled by default.",
    )
    overlay_group.add_argument(
        "--no-overlay",
        action="store_true",
        help="Disable operator perception metadata and canvas overlay.",
    )
    args = parser.parse_args(argv)
    if bool(args.save_dir) != (args.save_interval_s > 0):
        parser.error(
            "--save-dir and a positive --save-interval-s must be supplied together"
        )
    if args.save_on_change and not args.save_dir:
        parser.error("--save-on-change requires --save-dir and --save-interval-s")
    if args.save_on_change and args.max_save_interval_s <= 0:
        parser.error("--max-save-interval-s must be positive with --save-on-change")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overlay = None
    if not args.no_overlay:
        from camera.vision_worker import VisionWorker

        overlay = VisionWorker(args.overlay_config)
        overlay.start()

    recorder = (
        FrameRecorder(
            args.save_dir,
            args.save_interval_s,
            change_threshold=(args.change_threshold if args.save_on_change else None),
            max_interval_s=(args.max_save_interval_s if args.save_on_change else None),
        )
        if args.save_dir
        else None
    )
    cameras = {
        "shoulder": CameraStream(
            "shoulder",
            args.shoulder_device,
            args.width,
            args.height,
            args.fps,
            args.fourcc,
            args.jpeg_quality,
            recorder,
            frame_callback=overlay.submit if overlay is not None else None,
        )
    }
    if args.wrist_device:
        cameras["wrist"] = CameraStream(
            "wrist",
            args.wrist_device,
            args.width,
            args.height,
            args.fps,
            args.fourcc,
            args.jpeg_quality,
            recorder,
        )

    server: CameraServer | None = None
    try:
        for camera in cameras.values():
            camera.start()
        overlay_cameras = {"shoulder"} if overlay is not None else set()
        cross_calibration = CrossCalibration(
            args.overlay_config, args.cross_calibration_path
        )
        server = CameraServer(
            (args.host, args.port),
            cameras,
            overlay,
            overlay_cameras=overlay_cameras,
            cross_calibration=cross_calibration,
        )

        server.gripper_reference_path = args.gripper_reference_path

        def shutdown(_signum: int, _frame: Any) -> None:
            assert server is not None
            threading.Thread(target=server.shutdown, daemon=True).start()

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        print("SO-101 camera web server")
        for address in local_addresses():
            print(f"  http://{address}:{args.port}")
        print("Routes: /, /health, /snapshot/<camera>.jpg, /video/<camera>.mjpg")
        if overlay is not None:
            print(
                "Overlay: browser canvas + /events/<camera> + /detections/<camera>.json"
            )
        if recorder is not None:
            mode = (
                "when the scene changes" if recorder.saves_on_change else "periodically"
            )
            print(
                f"Saving JPEGs {mode}, checked every {recorder.interval_s:g}s "
                f"under {args.save_dir}"
            )
        server.serve_forever()
    finally:
        if server is not None:
            server.server_close()
        for camera in cameras.values():
            camera.stop()
        if overlay is not None:
            overlay.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
