#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=so101_env.sh
source "$SCRIPT_DIR/so101_env.sh"

python - <<'PY'
from lerobot.motors.feetech import FeetechMotorsBus

ports = [
    "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6085435-if00",
    "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6086462-if00",
]

for port in ports:
    print(f"\nPORT {port}")
    print(FeetechMotorsBus.scan_port(port, protocol_version=0))
PY

