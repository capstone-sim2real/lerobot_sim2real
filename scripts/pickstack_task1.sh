#!/usr/bin/env bash
# Task 1 (운반): 블록 5개를 zone 슬롯에 적재
# 사용: ./scripts/pickstack_task1.sh [--set key.path=value ...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
# shellcheck source=so101_env.sh
source "$SCRIPT_DIR/so101_env.sh"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
exec python -m pick_stack.runners.run_task --task 1 "$@"
