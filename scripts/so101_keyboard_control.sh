#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=so101_env.sh
source "$SCRIPT_DIR/so101_env.sh"

exec python "$SCRIPT_DIR/so101_keyboard_control.py" "$@"

