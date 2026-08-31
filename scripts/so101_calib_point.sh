#!/usr/bin/env bash
# Record one calibration point: FK pose with the block grasped, then a clean
# image of the block with the arm moved away (AGENTS.md §6).
#
#   ./scripts/so101_calib_point.sh P1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
NAME="${1:?usage: so101_calib_point.sh <POINT_NAME>   e.g. P1}"
SNAPSHOT_URL="${SO101_SNAPSHOT_URL:-http://127.0.0.1:8090/snapshot/shoulder.jpg}"
IMG="$ROOT/docs/calibration/$(echo "$NAME" | tr '[:upper:]' '[:lower:]')_top.jpg"

echo "[$NAME] 1/2  Close the gripper jaws around the block by hand so the block"
echo "            self-centres, hold the arm steady, then press Enter."
read -r _
"$SCRIPT_DIR/so101_record_calibration_point.sh" "$NAME" --overwrite --snapshot-url "$SNAPSHOT_URL"

echo
echo "[$NAME] 2/2  Open the jaws, move the ARM away — leave the BLOCK exactly"
echo "            where it is — then press Enter to capture the clean image."
read -r _
curl -fsS -m 5 -o "$IMG" "$SNAPSHOT_URL"
echo "[$NAME] clean image saved: $IMG"
echo "[$NAME] done. Next point, or run pick_pixels when all points are recorded."
