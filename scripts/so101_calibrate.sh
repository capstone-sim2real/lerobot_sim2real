#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LEADER_PORT="${SO101_LEADER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6085435-if00}"
FOLLOWER_PORT="${SO101_FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6086462-if00}"
LEADER_ID="${SO101_LEADER_ID:-my_leader}"
FOLLOWER_ID="${SO101_FOLLOWER_ID:-my_follower}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/so101_calibrate.sh reset
  ./scripts/so101_calibrate.sh leader
  ./scripts/so101_calibrate.sh follower
  ./scripts/so101_calibrate.sh all

Environment overrides:
  SO101_LEADER_PORT
  SO101_FOLLOWER_PORT
  SO101_LEADER_ID
  SO101_FOLLOWER_ID
EOF
}

reset_calibration() {
  rm -f "$HOME/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/$LEADER_ID.json"
  rm -f "$HOME/.cache/huggingface/lerobot/calibration/robots/so_follower/$FOLLOWER_ID.json"
  echo "Removed SO-101 calibration files."
}

calibrate_leader() {
  # shellcheck source=so101_env.sh
  source "$SCRIPT_DIR/so101_env.sh"
  lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port="$LEADER_PORT" \
    --teleop.id="$LEADER_ID"
}

calibrate_follower() {
  # shellcheck source=so101_env.sh
  source "$SCRIPT_DIR/so101_env.sh"
  lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port="$FOLLOWER_PORT" \
    --robot.id="$FOLLOWER_ID"
}

command="${1:-}"
case "$command" in
  reset)
    reset_calibration
    ;;
  leader)
    calibrate_leader
    ;;
  follower)
    calibrate_follower
    ;;
  all)
    reset_calibration
    calibrate_leader
    calibrate_follower
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
