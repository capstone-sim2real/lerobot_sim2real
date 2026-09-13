#!/usr/bin/env bash
set -e
cd /home/ehdrms/lerobot_sim2real-task1-zone-gather
exec env PYTHONPATH=src .venv/bin/python /tmp/so101-relative-teleop/run.py \
 --offsets-json /tmp/so101-relative-teleop/absolute_offsets.json \
 --fps 60 --step 5 --max-relative-target 10 --seconds 0 \
 --temperature-limit 65 --startup-limit 30 --stable-seconds 0 --stable-delta 360 \
 --capture-request /tmp/so101-relative-teleop/capture_request.json \
 --snapshot-url http://127.0.0.1:8090/snapshot/shoulder.jpg --capture-max-delta 1 "$@"
