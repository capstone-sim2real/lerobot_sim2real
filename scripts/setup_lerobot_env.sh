#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

UV_BIN="${UV_BIN:-uv}"
if ! command -v "$UV_BIN" >/dev/null 2>&1; then
  if [[ -x "$HOME/.local/bin/uv" ]]; then
    UV_BIN="$HOME/.local/bin/uv"
  else
    echo "uv not found. Install it first:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    echo '  echo '\''export PATH="$HOME/.local/bin:$PATH"'\'' >> ~/.bashrc' >&2
    echo "  source ~/.bashrc" >&2
    exit 1
  fi
fi

cd "$REPO_ROOT"
git submodule update --init --recursive
"$UV_BIN" sync --python 3.12 --extra hardware --extra dev

. "$REPO_ROOT/.venv/bin/activate"
python - <<'PY'
import lerobot
import serial
import scservo_sdk
import placo

print("lerobot", lerobot.__version__)
print("pyserial", serial.VERSION)
print("scservo_sdk ok")
print("placo ok")
PY
