#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_ACTIVATE="$REPO_ROOT/.venv/bin/activate"

if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "Project virtualenv not found: $VENV_ACTIVATE" >&2
  echo "Run ./scripts/setup_lerobot_env.sh from the repository root first." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$VENV_ACTIVATE"
