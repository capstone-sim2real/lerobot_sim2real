#!/usr/bin/env bash
set -euo pipefail

LEROBOT_DIR="${LEROBOT_DIR:-$HOME/lerobot}"
LEROBOT_REPO="${LEROBOT_REPO:-https://github.com/huggingface/lerobot.git}"
LEROBOT_COMMIT="${LEROBOT_COMMIT:-8a74e0ac6d01706d67fddfed682a09d694d9c8c0}"
INSTALL_KINEMATICS="${INSTALL_KINEMATICS:-0}"

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

if [[ ! -d "$LEROBOT_DIR/.git" ]]; then
  git clone "$LEROBOT_REPO" "$LEROBOT_DIR"
fi

cd "$LEROBOT_DIR"
git fetch --tags origin
git checkout "$LEROBOT_COMMIT"

if [[ "$INSTALL_KINEMATICS" == "1" ]]; then
  "$UV_BIN" sync --extra feetech --extra kinematics
else
  "$UV_BIN" sync --extra feetech
fi

. "$LEROBOT_DIR/.venv/bin/activate"
python - <<'PY'
import lerobot
import serial
import scservo_sdk

print("lerobot", lerobot.__version__)
print("pyserial", serial.VERSION)
print("scservo_sdk ok")
PY
