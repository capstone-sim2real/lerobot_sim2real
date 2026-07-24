#!/usr/bin/env bash
# Eval 서버용 policy_server 이미지 빌드 (x86_64 GPU 머신에서 실행)
#
# 사용:
#   ./docker/build_policy_server.sh <MODEL_CHECKPOINT_DIR> [IMAGE_TAG]
# 예:
#   ./docker/build_policy_server.sh \
#     ~/lerobot/outputs/act_pick_v1/checkpoints/last/pretrained_model \
#     pickstack-policy-server:v1
set -euo pipefail

MODEL_SRC="${1:?모델 checkpoint 디렉터리 경로가 필요합니다 (…/pretrained_model)}"
TAG="${2:-pickstack-policy-server:latest}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

if [[ ! -f "$MODEL_SRC/config.json" ]]; then
  echo "ERROR: $MODEL_SRC 가 pretrained_model 디렉터리로 보이지 않습니다 (config.json 없음)" >&2
  exit 1
fi

# 빌드 컨텍스트 안으로 모델 스테이징 (models/ 는 gitignore 대상)
STAGE="$REPO_ROOT/models/pretrained_model"
rm -rf "$STAGE"
mkdir -p "$STAGE"
cp -r "$MODEL_SRC"/. "$STAGE"/

cd "$REPO_ROOT"
docker build \
  --platform linux/amd64 \
  -f docker/policy_server.Dockerfile \
  -t "$TAG" \
  .

echo
echo "빌드 완료: $TAG"
echo "스모크: ./docker/run_policy_server.sh $TAG"
