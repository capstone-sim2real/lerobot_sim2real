#!/usr/bin/env bash
# 장소별 CV 캘리브레이션 (체스판 homography + base/zone 지정)
# 사용 예: ./scripts/pickstack_calibrate.sh --camera /dev/video0 --square-mm 25 --venue lab
# 자세한 절차는 python -m tools.calibrate_homography --help
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
# shellcheck source=so101_env.sh
source "$SCRIPT_DIR/so101_env.sh"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
exec python -m tools.calibrate_homography "$@"
