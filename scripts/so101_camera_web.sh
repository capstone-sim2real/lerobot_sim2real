#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
# shellcheck source=so101_env.sh
source "$SCRIPT_DIR/so101_env.sh"

HOST="${SO101_CAMERA_WEB_HOST:-0.0.0.0}"
PORT="${SO101_CAMERA_WEB_PORT:-8090}"
SHOULDER_DEVICE="${SO101_SHOULDER_CAMERA:-/dev/video0}"
WRIST_DEVICE="${SO101_WRIST_CAMERA:-}"
WIDTH="${SO101_CAMERA_WIDTH:-1280}"
HEIGHT="${SO101_CAMERA_HEIGHT:-720}"
FPS="${SO101_CAMERA_FPS:-30}"
FOURCC="${SO101_CAMERA_FOURCC:-MJPG}"
JPEG_QUALITY="${SO101_CAMERA_JPEG_QUALITY:-80}"
SAVE_DIR="${SO101_CAMERA_SAVE_DIR:-}"
SAVE_INTERVAL_S="${SO101_CAMERA_SAVE_INTERVAL_S:-0}"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

camera_args=(
  --host "$HOST"
  --port "$PORT"
  --shoulder-device "$SHOULDER_DEVICE"
  --wrist-device "$WRIST_DEVICE"
  --width "$WIDTH"
  --height "$HEIGHT"
  --fps "$FPS"
  --fourcc "$FOURCC"
  --jpeg-quality "$JPEG_QUALITY"
)
if [[ -n "$SAVE_DIR" ]]; then
  camera_args+=(--save-dir "$SAVE_DIR" --save-interval-s "$SAVE_INTERVAL_S")
fi

exec python -m camera.server "${camera_args[@]}" "$@"
