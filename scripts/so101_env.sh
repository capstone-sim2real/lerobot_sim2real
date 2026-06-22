#!/usr/bin/env bash
set -euo pipefail

LEROBOT_DIR="${LEROBOT_DIR:-$HOME/lerobot}"
VENV_ACTIVATE="$LEROBOT_DIR/.venv/bin/activate"

if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "LeRobot virtualenv not found: $VENV_ACTIVATE" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$VENV_ACTIVATE"

