# syntax=docker/dockerfile:1
# Eval-server image: lerobot policy_server + baked-in ACT checkpoint.
#
# The eval server provides nvidia-driver-580-open; CUDA userland comes from
# the torch wheels that lerobot's uv.lock pins, so the base stays plain
# Python. lerobot is pinned to the SAME commit the team uses on Orin/WSL
# (scripts/setup_lerobot_env.sh) — protocol drift between robot_client and
# policy_server is not survivable on demo day.
#
# Build (from the repo root; stage the model first — see build_policy_server.sh):
#   docker build --platform linux/amd64 -f docker/policy_server.Dockerfile \
#     -t pickstack-policy-server:latest .

FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ARG LEROBOT_REPO=https://github.com/huggingface/lerobot.git
# keep in sync with LEROBOT_COMMIT in scripts/setup_lerobot_env.sh
ARG LEROBOT_COMMIT=8a74e0ac6d01706d67fddfed682a09d694d9c8c0
RUN git clone "$LEROBOT_REPO" /opt/lerobot \
    && cd /opt/lerobot \
    && git checkout "$LEROBOT_COMMIT" \
    && rm -rf /opt/lerobot/.git

WORKDIR /opt/lerobot
RUN uv sync --locked --extra async --no-dev

# trained checkpoint, staged into the build context by build_policy_server.sh;
# robot-side config must point at this path:
#   policy.pretrained_name_or_path=/models/pretrained_model
ARG MODEL_DIR=models/pretrained_model
COPY ${MODEL_DIR} /models/pretrained_model

ENV HOST=0.0.0.0 \
    PORT=8080 \
    FPS=30
EXPOSE 8080

CMD ["/bin/sh", "-c", "uv run --no-sync python -m lerobot.async_inference.policy_server --host=${HOST} --port=${PORT} --fps=${FPS}"]
