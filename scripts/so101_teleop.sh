#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=so101_env.sh
source "$SCRIPT_DIR/so101_env.sh"

FPS="${1:-${SO101_FPS:-30}}"
LEADER_PORT="${SO101_LEADER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6085435-if00}"
FOLLOWER_PORT="${SO101_FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6086462-if00}"
LEADER_ID="${SO101_LEADER_ID:-my_leader}"
FOLLOWER_ID="${SO101_FOLLOWER_ID:-my_follower}"

case "$FPS" in
  30|60) ;;
  -h|--help|help)
    cat <<'EOF'
Usage:
  ./scripts/so101_teleop.sh [30|60]

Environment overrides:
  SO101_FPS
  SO101_LEADER_PORT
  SO101_FOLLOWER_PORT
  SO101_LEADER_ID
  SO101_FOLLOWER_ID
EOF
    exit 0
    ;;
  *)
    echo "Unsupported fps: $FPS (expected 30 or 60)" >&2
    exit 2
    ;;
esac

exec lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id="$LEADER_ID" \
  --fps="$FPS"
