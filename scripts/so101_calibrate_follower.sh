#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=so101_env.sh
source "$SCRIPT_DIR/so101_env.sh"

exec lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6086462-if00 \
  --robot.id=my_follower

