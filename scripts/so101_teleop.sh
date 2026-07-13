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
MAX_RELATIVE_TARGET="${SO101_MAX_RELATIVE_TARGET:-10}"
DISABLE_TORQUE_ON_DISCONNECT="${SO101_DISABLE_TORQUE_ON_DISCONNECT:-false}"

case "$FPS" in
  -h|--help|help)
    cat <<'EOF'
Usage:
  ./scripts/so101_teleop.sh [FPS]

FPS는 양의 정수면 됩니다 (기본 30). 예: 15, 30, 60.
낮은 fps는 통신 부담이 적고, 높은 fps는 반응이 빠릅니다.

Environment overrides:
  SO101_FPS
  SO101_LEADER_PORT
  SO101_FOLLOWER_PORT
  SO101_LEADER_ID
  SO101_FOLLOWER_ID
  SO101_MAX_RELATIVE_TARGET          default: 10
  SO101_DISABLE_TORQUE_ON_DISCONNECT default: false
EOF
    exit 0
    ;;
  ''|*[!0-9]*)
    echo "Invalid fps: '$FPS' (양의 정수를 입력하세요. 예: 15, 30, 60)" >&2
    exit 2
    ;;
  *)
    if [ "$FPS" -lt 1 ]; then
      echo "Invalid fps: '$FPS' (1 이상이어야 합니다)" >&2
      exit 2
    fi
    ;;
esac

exec lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  --robot.max_relative_target="$MAX_RELATIVE_TARGET" \
  --robot.disable_torque_on_disconnect="$DISABLE_TORQUE_ON_DISCONNECT" \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id="$LEADER_ID" \
  --fps="$FPS"
