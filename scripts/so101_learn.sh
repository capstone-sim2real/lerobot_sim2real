#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=so101_env.sh
source "$SCRIPT_DIR/so101_env.sh"

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$HOME/lerobot_data}"
export DATASET_REPO_ID="${DATASET_REPO_ID:-local/so101_pick_place_v1}"
export DATASET_PREFIX="${DATASET_PREFIX:-so101_pick_place_v1}"
export SO101_TASK="${SO101_TASK:-Pick up one block and place it neatly in the target area.}"
export SO101_FPS="${SO101_FPS:-30}"
export SO101_EPISODE_TIME_S="${SO101_EPISODE_TIME_S:-30}"
export SO101_RESET_TIME_S="${SO101_RESET_TIME_S:-10}"
export SO101_NUM_EPISODES="${SO101_NUM_EPISODES:-1}"
export SO101_ENCODER_THREADS="${SO101_ENCODER_THREADS:-2}"
export SO101_MAX_RELATIVE_TARGET="${SO101_MAX_RELATIVE_TARGET:-10}"
export SO101_LEADER_PORT="${SO101_LEADER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6085435-if00}"
export SO101_FOLLOWER_PORT="${SO101_FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6086462-if00}"
export SO101_CAMERAS="${SO101_CAMERAS:-{shoulder: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30, fourcc: MJPG}}}"

mkdir -p "$HF_LEROBOT_HOME"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/so101_learn.sh latest-dataset
  ./scripts/so101_learn.sh record
  ./scripts/so101_learn.sh train-overfit
  ./scripts/so101_learn.sh rollout

Common environment overrides:
  ACTUAL_REPO_ID
  DATASET_REPO_ID
  DATASET_PREFIX
  SO101_RESUME=1
  SO101_NUM_EPISODES=5
  SO101_TASK="..."
  POLICY_PATH=/path/to/pretrained_model
  SO101_ROLLOUT_DURATION=30
EOF
}

latest_dataset_id() {
  find "$HF_LEROBOT_HOME/local" -maxdepth 1 -type d -name "${DATASET_PREFIX}_*" -printf 'local/%f\n' 2>/dev/null | sort | tail -1
}

require_dataset_id() {
  if [[ -n "${ACTUAL_REPO_ID:-}" ]]; then
    printf '%s\n' "$ACTUAL_REPO_ID"
    return
  fi

  local latest
  latest="$(latest_dataset_id)"
  if [[ -z "$latest" ]]; then
    echo "No dataset found for prefix '$DATASET_PREFIX' under $HF_LEROBOT_HOME/local" >&2
    echo "Set ACTUAL_REPO_ID=local/<dataset_id> or record a dataset first." >&2
    exit 1
  fi
  printf '%s\n' "$latest"
}

latest_dataset() {
  local latest
  latest="$(latest_dataset_id)"
  if [[ -z "$latest" ]]; then
    echo "No dataset found for prefix '$DATASET_PREFIX' under $HF_LEROBOT_HOME/local" >&2
    exit 1
  fi
  echo "$latest"
}

record_episode() {
  local target_repo_id
  local resume_arg=()

  if [[ "${SO101_RESUME:-0}" == "1" ]]; then
    target_repo_id="$(require_dataset_id)"
    resume_arg=(--resume=true)
  else
    target_repo_id="$DATASET_REPO_ID"
  fi

  exec lerobot-record \
    "${resume_arg[@]}" \
    --robot.type=so101_follower \
    --robot.port="$SO101_FOLLOWER_PORT" \
    --robot.id=my_follower \
    --robot.cameras="$SO101_CAMERAS" \
    --robot.max_relative_target="$SO101_MAX_RELATIVE_TARGET" \
    --teleop.type=so101_leader \
    --teleop.port="$SO101_LEADER_PORT" \
    --teleop.id=my_leader \
    --dataset.repo_id="$target_repo_id" \
    --dataset.num_episodes="$SO101_NUM_EPISODES" \
    --dataset.episode_time_s="$SO101_EPISODE_TIME_S" \
    --dataset.reset_time_s="$SO101_RESET_TIME_S" \
    --dataset.single_task="$SO101_TASK" \
    --dataset.push_to_hub=false \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads="$SO101_ENCODER_THREADS" \
    --display_data=true \
    --play_sounds=false
}

train_overfit() {
  local actual_repo_id
  actual_repo_id="$(require_dataset_id)"

  local train_dir="${TRAIN_DIR:-$HOME/lerobot_outputs/act_so101_pick_place_overfit}"
  local dataset_episodes="${DATASET_EPISODES:-[0,1,2,3,4]}"

  if [[ "${SO101_CLEAN_TRAIN_DIR:-1}" == "1" ]]; then
    rm -rf "$train_dir"
  fi

  exec lerobot-train \
    --policy.type=act \
    --policy.device="${POLICY_DEVICE:-cuda}" \
    --policy.use_amp="${POLICY_USE_AMP:-true}" \
    --policy.push_to_hub=false \
    --policy.dim_model="${POLICY_DIM_MODEL:-256}" \
    --policy.n_heads="${POLICY_N_HEADS:-4}" \
    --policy.dim_feedforward="${POLICY_DIM_FEEDFORWARD:-1024}" \
    --policy.n_encoder_layers="${POLICY_N_ENCODER_LAYERS:-2}" \
    --policy.n_vae_encoder_layers="${POLICY_N_VAE_ENCODER_LAYERS:-2}" \
    --policy.chunk_size="${POLICY_CHUNK_SIZE:-50}" \
    --policy.n_action_steps="${POLICY_N_ACTION_STEPS:-25}" \
    --dataset.repo_id="$actual_repo_id" \
    --dataset.episodes="$dataset_episodes" \
    --output_dir="$train_dir" \
    --batch_size="${BATCH_SIZE:-2}" \
    --num_workers="${NUM_WORKERS:-0}" \
    --steps="${TRAIN_STEPS:-1000}" \
    --save_freq="${SAVE_FREQ:-500}" \
    --log_freq="${LOG_FREQ:-50}" \
    --eval_freq=0 \
    --wandb.enable=false
}

rollout() {
  local policy_path="${POLICY_PATH:-$HOME/lerobot_outputs/act_so101_pick_place_v1/checkpoints/last/pretrained_model}"

  if [[ ! -d "$policy_path" ]]; then
    echo "Policy path not found: $policy_path" >&2
    echo "Set POLICY_PATH=/path/to/pretrained_model if needed." >&2
    exit 1
  fi

  exec lerobot-rollout \
    --strategy.type=base \
    --policy.path="$policy_path" \
    --robot.type=so101_follower \
    --robot.port="$SO101_FOLLOWER_PORT" \
    --robot.id=my_follower \
    --robot.cameras="$SO101_CAMERAS" \
    --robot.max_relative_target="$SO101_MAX_RELATIVE_TARGET" \
    --fps="$SO101_FPS" \
    --duration="${SO101_ROLLOUT_DURATION:-10}" \
    --task="$SO101_TASK" \
    --display_data=true \
    --return_to_initial_position=true \
    --play_sounds=false
}

command="${1:-}"
case "$command" in
  latest-dataset)
    latest_dataset
    ;;
  record)
    record_episode
    ;;
  train-overfit)
    train_overfit
    ;;
  rollout)
    rollout
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
