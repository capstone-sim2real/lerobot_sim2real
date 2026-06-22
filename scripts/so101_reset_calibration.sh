#!/usr/bin/env bash
set -euo pipefail

rm -f "$HOME/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/my_leader.json"
rm -f "$HOME/.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower.json"

echo "Removed SO-101 calibration files."

