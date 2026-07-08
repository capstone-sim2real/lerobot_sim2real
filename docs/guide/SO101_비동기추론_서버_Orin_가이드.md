# SO-101 비동기 추론: 서버-Orin 분리 가이드

GPU 서버(데스크탑)에서 ACT/SmolVLA 추론하고, Orin Nano에서 SO-101을 제어하는 비동기 분리 구조 설명서입니다.

전제:

```text
데스크탑: GPU 서버, WSL(Arch Linux), Tailscale 연결
Orin:     SO-101 제어 컴퓨터, LeRobot 설치 완료
```

---

## 1. 구조 개요

```
Orin (robot_client)
  카메라 프레임 + 관절 상태 → [Tailscale/LAN] → 데스크탑 (policy_server)
                                                        ↓ ACT 추론
  SO-101 실행 ← 액션 청크 (50~100 스텝) ←────────────────
```

ACT는 한 번에 50~100 스텝치 액션을 뽑기 때문에 Orin이 청크를 실행하는 동안 다음 추론이 병렬로 진행됩니다. Tailscale 지연(~30ms)도 감당 가능합니다.

---

## 2. 데스크탑 초기 설정 (WSL, Arch Linux, fish)

```fish
cd ~
git clone https://github.com/huggingface/lerobot.git
cd lerobot
uv sync --extra async
```

uv가 없으면 먼저 설치합니다.

```fish
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.config/fish/config.fish
```

GPU 확인:

```fish
cd ~/lerobot
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

학습까지 할 경우 추가 의존성:

```fish
uv sync --extra async --extra training --extra dataset
```

---

## 3. Orin async 의존성 설치

```fish
cd ~/lerobot
~/lerobot/.venv/bin/python -m pip install "lerobot[async]"
```

pip이 없으면:

```fish
~/lerobot/.venv/bin/python -m ensurepip
~/lerobot/.venv/bin/python -m pip install "lerobot[async]"
```

또는 uv가 설치되어 있으면:

```fish
cd ~/lerobot
uv sync --extra async
```

---

## 4. policy_server 실행 (데스크탑)

서버를 먼저 켭니다.

```fish
cd ~/lerobot
uv run python -m lerobot.async_inference.policy_server \
    --host=0.0.0.0 \
    --port=8080 \
    --fps=30
```

`PolicyServer started on 0.0.0.0:8080` 이 뜨면 준비 완료입니다.

---

## 5. robot_client 실행 (Orin)

서버 확인 후 Orin에서 실행합니다.

```fish
cd ~/lerobot
~/lerobot/.venv/bin/python -m lerobot.async_inference.robot_client \
    --robot.type=so101_follower \
    --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6086462-if00 \
    --robot.id=my_follower \
    --robot.cameras='{
        "top":   {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
        "wrist": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}
    }' \
    --server_address=데스크탑_IP:8080 \
    --policy_type=act \
    --pretrained_name_or_path=모델경로_또는_HuggingFace_이름 \
    --actions_per_chunk=50 \
    --fps=30
```

카메라 이름(`top`, `wrist`)은 학습 시 데이터 수집에서 사용한 이름과 반드시 일치해야 합니다.

데스크탑 IP 확인:

```fish
# 데스크탑 WSL에서
ip addr show eth0 | grep "inet "
# 또는 Tailscale IP 확인
tailscale ip
```

---

## 6. 실행 순서 주의

반드시 **서버 먼저, 클라이언트 나중**입니다.

```text
1. 데스크탑: policy_server 실행
2. 서버 로그에서 "PolicyServer started" 확인
3. Orin: robot_client 실행
4. 서버 로그에서 "Client connected and ready" 확인
5. 서버 로그에서 "Receiving policy instructions" 확인
6. 서버 로그에서 "Running inference" 확인
```

서버를 재시작하면 robot_client도 반드시 재시작합니다. 서버만 재시작하면 클라이언트가 이전 세션 상태로 관측값만 보내고 모델이 로드되지 않습니다.

---

## 7. 데이터 수집 (Orin)

데이터 수집은 로컬에서 Orin 단독으로 합니다. 서버 필요 없습니다.

```fish
cd ~/lerobot
~/lerobot/.venv/bin/python -m lerobot.scripts.lerobot_record \
    --robot.type=so101_follower \
    --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6086462-if00 \
    --robot.id=my_follower \
    --robot.cameras='{
        "top":   {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
        "wrist": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}
    }' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6085435-if00 \
    --teleop.id=my_leader \
    --dataset.repo_id=local/block_stacking_v1 \
    --dataset.num_episodes=50 \
    --dataset.single_task="Pick up blocks and stack them." \
    --fps=30
```

녹화 중 조작:

```text
→  현재 에피소드 저장하고 다음으로
←  현재 에피소드 버리고 재녹화
Esc  전체 중지
```

실패한 에피소드는 반드시 버립니다. 저장하면 그 실패 동작도 학습합니다.

---

## 8. 학습 (데스크탑)

```fish
cd ~/lerobot
uv run python -m lerobot.scripts.lerobot_train \
    --policy.type=act \
    --dataset.repo_id=local/block_stacking_v1 \
    --training.num_steps=50000 \
    --output_dir=outputs/act_block_stacking_v1
```

학습 완료 후 모델 경로: `outputs/act_block_stacking_v1/checkpoints/last/pretrained_model`

이 경로를 robot_client의 `--pretrained_name_or_path`에 지정하면 됩니다. 단, 데스크탑에서 학습한 모델을 Orin이 찾을 수 있어야 하므로 경로가 데스크탑 기준임에 주의합니다(서버가 로드하기 때문).

---

## 8-1. SmolVLA (VLA) 파인튜닝 — 언어 조건부 정책

ACT는 언어를 무시하지만 SmolVLA는 task 문장으로 행동을 고릅니다("green block" vs "yellow block"). 색/작업별로 데이터를 나눠 모으고 하나의 모델로 합칠 수 있습니다.

ACT와 다른 핵심 3가지:

1. `--policy.type=act` 대신 **`--policy.path=lerobot/smolvla_base`** (450M 베이스에서 파인튜닝)
2. rollout 때 **`--task="..."` 필수** (없으면 뭘 할지 모름). 학습 때 쓴 문장과 동일하게.
3. base가 카메라 3개(camera1/2/3)를 기대 → 우리 2개(top/wrist)를 **`--rename_map`** 으로 매핑

학습 명령 (데스크탑):

```fish
cd ~/lerobot
uv run python -m lerobot.scripts.lerobot_train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=local/so101_place_gy_v1 \
    --dataset.root=/home/(whoami)/.cache/huggingface/lerobot/local/so101_place_gy_v1 \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
    --batch_size=16 \
    --steps=8000 \
    --save_freq=2000 \
    --output_dir=outputs/smolvla_place_gy_v1 \
    --job_name=smolvla_place_gy \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --wandb.enable=false
```

주의:
- `--policy.push_to_hub=false` 없으면 `'repo_id' argument missing` 에러.
- 로컬 데이터셋은 `--dataset.root` 명시 필수.
- OOM이면 `--batch_size=8`. **학습 중 게임 등 GPU 점유 금지** (CUDA unknown error로 죽음).
- SmolVLA `chunk_size`는 50 고정. `--actions_per_chunk=100`을 줘도 50까지만 나옴.

체크포인트에서 이어서 학습 (중단됐을 때):

```fish
uv run python -m lerobot.scripts.lerobot_train \
    --config_path=outputs/smolvla_place_gy_v1/checkpoints/last/pretrained_model/train_config.json \
    --resume=true
```

`last`는 마지막 `save_freq` 지점을 가리킵니다(예: 5871에서 죽으면 4000부터 재개).

### SmolVLA rollout (robot_client)

robot_client에는 rename_map 인자가 없으므로, **카메라 이름을 아예 `camera1`/`camera2`로** 줍니다(학습 때 매핑과 동일하게, camera1=top=index2, camera2=wrist=index0).

```fish
    --robot.cameras='{
        "camera1": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
        "camera2": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}
    }' \
    --policy_type=smolvla \
    --task="Pick up the green block and place it inside the red tape area." \
```

---

## 8-2. 데이터셋 관리

**이름 규칙:** `local/so101_<동작>_<변형>_v<버전>` (예: `so101_place_green_v1`).
`lerobot-record`는 실행 세션마다 `_YYYYMMDD_HHMMSS` 타임스탬프를 이름에 붙입니다.

**task 문장을 색/작업별로 정확히:** `"Pick up the green block ..."`. 나중에 VLA 색 구분의 근거가 됩니다. (지정구역 테이프가 빨강이므로 **빨강 블록은 task 문장이 겹쳐 혼동** — 다른 색부터 권장)

**이어붙이기(resume):** `--resume=true` + `--dataset.root=<정확한 폴더 경로>` (root 없으면 에러). repo_id는 타임스탬프 포함 정확한 이름으로.

**나쁜 에피소드 삭제:** 이어붙은 mp4를 재인코딩하므로 비쌈. 애초에 녹화 중 `←`로 버리는 게 나음.

```fish
~/lerobot/.venv/bin/python -m lerobot.scripts.lerobot_edit_dataset \
    --repo_id local/so101_place_green_v1_TIMESTAMP \
    --root /home/ehdrms/.cache/huggingface/lerobot/local/so101_place_green_v1_TIMESTAMP \
    --operation.type delete_episodes \
    --operation.episode_indices "[31]"
```

**여러 데이터셋 병합 (색 합쳐 1개 모델):** 리스트로 동시 학습은 미지원. 학습 전 `aggregate_datasets`로 합칩니다. **fps·해상도·카메라 이름이 같아야** 합칠 수 있음.

```fish
uv run python -c 'from lerobot.datasets.aggregate import aggregate_datasets; from pathlib import Path; base = Path.home() / ".cache/huggingface/lerobot/local"; aggregate_datasets(repo_ids=["local/so101_place_green_v1_T1", "local/so101_place_yellow_v1_T2"], aggr_repo_id="local/so101_place_gy_v1", roots=[base / "so101_place_green_v1_T1", base / "so101_place_yellow_v1_T2"], aggr_root=base / "so101_place_gy_v1"); print("병합 완료")'
```

### Orin ↔ 데스크탑 데이터 전송 (Taildrop)

WSL에 ssh 서버가 없어 rsync가 안 되므로 Tailscale 파일 전송을 씁니다. **WSL 노드 IP로** 보내야 함(Windows 본체 노드 아님).

```fish
# Orin: 압축 후 전송
cd ~/.cache/huggingface/lerobot/local && tar czf /tmp/ds.tar.gz <데이터셋폴더>
sudo tailscale file cp /tmp/ds.tar.gz <WSL_Tailscale_IP>:

# 데스크탑 WSL: 수신 후 압축 해제
cd ~/.cache/huggingface/lerobot/local
sudo tailscale file get .
sudo chown $USER ds.tar.gz && tar xzf ds.tar.gz
```

---

## 8-3. Orin Nano 하드웨어 제약 (중요)

- **GPU 학습/추론 불가:** Orin의 torch(`2.11.0+cu128` 일반 휠)는 `sm_87` 커널이 없어 GPU 실행 시 `no kernel image` 에러. **학습·추론은 데스크탑에서.** Orin GPU를 쓰려면 JetPack용 PyTorch 별도 설치 필요.
- **NVENC 없음:** Orin Nano에 하드웨어 비디오 인코더가 없음. `h264_nvenc`는 `Operation not permitted`로 실패. 영상은 소프트웨어 `h264`로만. `--dataset.rgb_encoder.vcodec` 유효값에 `libx264` 없음 → `h264` 사용.
- **인코딩/카메라 부담:** 듀얼 640×480×30fps 수집 시 record loop가 22Hz로 떨어지는 경고가 남(정상, 드랍 감수). `--dataset.streaming_encoding=true`는 Orin에선 오히려 녹화가 버벅여 비권장.

---

## 9. 네트워크 환경별 권장 설정

| 환경 | 설명 |
|---|---|
| 유선 LAN (같은 공유기) | 가장 안정적. 지연 1ms 이하 |
| Tailscale direct | 지연 20~60ms. 청크 추론에는 충분 |
| Tailscale relay 경유 | 지연 100ms+. 불안정할 수 있음 |
| WiFi | 불안정. 평가 당일은 유선 권장 |

Tailscale 연결 상태 확인:

```fish
tailscale status
```

`direct` 표시가 있으면 P2P 직접 연결입니다. relay 경유면 지연이 늘어납니다.

---

## 10. 자주 나는 문제

**`'NoneType' object has no attribute 'config'`**

서버 재시작 후 robot_client를 재시작하지 않은 경우입니다. 둘 다 재시작합니다.

**`config.json not found on the HuggingFace Hub`**

모델 이름이 잘못됐습니다. HuggingFace에서 정확한 모델 ID를 확인합니다.

**`'observation.images.top'` 키 에러**

robot_client의 카메라 이름과 모델이 학습된 카메라 이름이 다릅니다. 우리가 학습한 모델을 쓸 때는 수집 때 쓴 카메라 이름과 동일하게 지정합니다.

**카메라 `width` 필수 에러**

`--robot.cameras` 지정 시 `width`, `height`, `fps` 세 가지가 모두 있어야 합니다.

**`attempted relative import with no known parent package`**

`src/lerobot/async_inference/robot_client.py`를 직접 실행했을 때 납니다. `-m lerobot.async_inference.robot_client`로 모듈 실행해야 합니다.

**로봇 연결 시 `Torque_Enable ... no status packet` / 카메라 `can't open camera by index`**

이전 프로세스(robot_client, record, 카메라 웹 호스팅)가 안 죽고 포트/카메라를 점유 중입니다. `fuser /dev/ttyACM0 /dev/video0 /dev/video2`로 확인 후 해당 PID를 `kill`. 카메라 웹 호스팅과 robot_client/record는 **동시 실행 불가**(같은 카메라 점유).

**모터 `no status packet`이 kill 후에도 지속**

서보 통신이 꼬인 것. SO-101 팔로워 전원 껐다 켜고 USB 재연결 후 `./scripts/so101_scan_motors.sh`로 ID 1~6 확인.

**rollout 급발진 / 멈칫 (control loop 스파이크)**

주로 네트워크(관측 1.75MB를 Tailscale로 전송) + 모델 미학습. 완화책: `--fps=15`로 낮춤, `--robot.max_relative_target=10`(급발진 안전 클램프, `clamped to be safe` 경고는 정상 동작), `--aggregate_fn_name=conservative`(떨림 억제) 또는 `latest_only`. 근본 해결은 같은 LAN 유선 + 데이터 증량. robot_client 로그는 `~/lerobot/logs/robot_client_*.log`에 남고, `QUEUE SIZE`가 0 근처로 자주 떨어지면 네트워크 굶주림.

**학습 중 `CUDA error: unknown error`**

데스크탑에서 게임 등 다른 프로그램이 GPU를 점유한 경우. GPU 독점 후 재시도.

## aggregate_fn_name 옵션 (rollout 부드러움 조절)

| 옵션 | old/new 가중 | 특성 |
|---|---|---|
| `latest_only` | new만 | 반응 빠름, 경계에서 튐 |
| `weighted_average` (기본) | 0.3/0.7 | 균형 |
| `average` | 0.5/0.5 | 중간 |
| `conservative` | 0.7/0.3 | 제일 부드러움, 반응 느림 |
