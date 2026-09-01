#!/usr/bin/env bash
# Task 2 (적층): 블록을 접촉 감지 기반으로 탑 쌓기
# 사용: ./scripts/pickstack_task2.sh [--set key.path=value ...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
# shellcheck source=so101_env.sh
source "$SCRIPT_DIR/so101_env.sh"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
exec python -m runners.run_task --task 2 "$@"
