#!/usr/bin/env bash
# policy_server 컨테이너 실행 (GPU 필요; nvidia-container-toolkit 설치 전제)
#
# 사용:
#   ./docker/run_policy_server.sh [IMAGE_TAG] [PORT]
set -euo pipefail

TAG="${1:-pickstack-policy-server:latest}"
PORT="${2:-8080}"

exec docker run --rm \
  --gpus all \
  -p "${PORT}:8080" \
  --name pickstack-policy-server \
  "$TAG"
