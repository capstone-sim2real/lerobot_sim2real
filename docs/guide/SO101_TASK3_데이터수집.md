# Task 3 — ACT 데이터셋 자동 수집 가이드

Task 1의 수집 루프를 그대로 돌리면서, 성공한 한 사이클을 LeRobot 에피소드로
녹화합니다. 사람이 리더암으로 30번 반복하는 대신 **블록 5개를 배치하는 일만**
합니다.

한 에피소드 = `home → 파지 → 운반 → 릴리스 → home 복귀`. 시작과 끝이 모두
home이라 학습할 사이클이 닫혀 있습니다.

관련 문서: [CV+IK 파지·운반](./SO101_CV_IK_파지운반.md) ·
[과거 텔레옵 수집 기록](./SO101_데이터수집_관리.md) ·
[학습·추론](./SO101_학습_추론.md)

---

## 1. Task 1과 다른 점

| | Task 1 | Task 3 |
|---|---|---|
| 파지 시도 | 중앙 + 재시도 링 4점 = 5회 | **중앙 1회** (`task3.max_grasp_attempts`) |
| 파지 실패 | 다음 그래스프 포인트로 재시도 | 에피소드 파기 후 SELECT로 복귀 |
| 구역이 비면 | `DONE`, 런 종료 | **재배치 프롬프트** 후 계속 수집 |
| 시간 예산 | 없음(외부 supervisor) | 없음 — Ctrl-C까지 무한 |
| 목적지 | 지정구역 슬롯 5개 | 동일 (`Task1TransportPlanner` 재사용) |

SELECT/PICK/VERIFY/TRANSPORT 핸들러는 Task 1의 것을 **그대로** 주입합니다.
녹화는 `BaseRobotIO`를 감싸는 `RecordingRobotIO`에서 일어나므로 FSM·모션·
그래스프 코드는 한 줄도 바뀌지 않습니다.

**저장 규칙:** 파지에 성공하고 **운반까지 마친** 에피소드만 저장합니다. 실패한
시연을 저장하면 정책이 그 실패를 학습합니다.

---

## 2. 준비

```bash
cd ~/lerobot_sim2real

# 데이터셋 의존성 (한 번만). uv sync가 아니라 uv pip install을 씁니다 —
# 전체 sync는 Orin용 JetPack torch 휠을 덮어쓸 수 있습니다.
uv pip install --python .venv/bin/python "datasets>=4.7.0,<5.0.0" "av>=15.0.0,<16.0.0"

# 카메라 서버 (손목캠까지 쓰려면 --wrist-device 추가)
so101-camera --wrist-device /dev/video2
```

체크리스트:

```text
1. so101-camera 가 설정된 스트림을 서빙 중인가  (curl -s localhost:8090/health)
2. venue 캘리브레이션에 zone_polygon_mm 이 있는가  (so101-zone-calibrate --write)
3. home 자세가 기록돼 있는가  (src/configs/poses.yaml)
4. 팔 주변 충돌 위험 제거
```

---

## 3. 실행

```bash
# 프리플라이트 — 모터 버스 미접속, 데이터셋 미생성 (AGENTS.md §14.4)
so101-collect --dry-run

# 수집
so101-collect

# 데이터셋 이름 지정
so101-collect --set task3.repo_id=local/so101_task3_green
```

`--dry-run`이 출력하는 것: 카메라별 실측 fps와 해상도, 구성될 features dict,
데이터셋 경로, 슬롯 5개의 IK 해, 현재 탐지 결과와 색별 task 문장.

운영 흐름:

```text
1. 블록 5개를 랜덤 배치
2. 팔이 하나씩 파지 → 슬롯으로 운반 → home 복귀   (각 사이클 = 에피소드 1개)
3. 지정구역 밖이 5초간 비면 배너 + 벨 → 재배치 후 Enter
4. 반복. Ctrl-C 로 종료
```

**Ctrl-C:** 진행 중인 미완 에피소드는 버리고, 팔은 home으로 복귀하며, 이미
저장된 에피소드는 모두 보존한 채 `finalize()`까지 마칩니다. 종료 중 두 번째
Ctrl-C는 무시됩니다 — parquet/비디오 finalize가 깨지면 데이터셋 전체를 잃습니다.

이어붙이기:

```bash
so101-collect --resume \
  --set task3.repo_id=local/so101_task3_20260913_101500 \
  --set task3.root=/home/mseoky/.cache/huggingface/lerobot/local/so101_task3_20260913_101500
```

`--resume`은 **정확한 타임스탬프 포함 이름 + root 경로**가 모두 필요합니다.

---

## 4. 손목캠

`task3.cameras`에 한 줄 추가하면 끝입니다. 코드 변경은 없습니다.

```yaml
task3:
  cameras:
    top:   http://127.0.0.1:8090/video/shoulder.mjpg
    wrist: http://127.0.0.1:8090/video/wrist.mjpg
```

주의:

- **하나의 데이터셋 안에서 카메라 구성을 바꾸지 않습니다.** 도중에 손목캠을
  추가하면 앞뒤 에피소드의 feature가 달라져 학습에 못 씁니다. 새 데이터셋을
  시작하세요.
- ACT는 `observation.images.*` 여러 개를 카메라 뷰로 보고 **shape이 같기를
  요구**합니다. 두 캠 모두 `task3.image_width/height`로 리사이즈되므로 자동으로
  맞습니다.
- 손목캠은 탑다운 시야를 가릴 수 있습니다(AGENTS.md §8). `so101-detect`로
  블록 검출이 여전히 되는지 먼저 확인하세요.

---

## 5. 왜 fps를 강제하는가

`TrajectoryPlayer._tick_sleep()`은 방금 한 작업과 무관하게 `1/motion.fps`를
무조건 잡니다. 그래서 실제 주기는 `1/fps + 버스 시간`이 되고, 데이터셋이 주장하는
fps와 어긋납니다. LeRobotDataset은 `timestamp`를 `frame_index / fps`로 **합성**
하므로, 어긋난 채 학습하면 **추론 시 정책이 기록과 다른 속도로 재생**됩니다.

그래서 Task 3은 `motion.fps`를 `task3.motion_fps_override`(300)로 올려 그 슬립을
무시할 수준으로 만들고, `RecordingRobotIO`가 절대 데드라인으로 매 틱을
페이싱합니다. `motion.fps`는 `_tick_sleep` 말고 쓰이는 데가 없습니다 — 틱당 관절
상한은 `motion.max_step_per_tick`이라 안전 특성은 그대로입니다.

**실측 (Orin, 카메라 2대, 버스 왕복 6ms 모사):**

```text
300 ticks in 9.98s -> 30.06 Hz achieved (target 30)
tick mean 33.3ms (target 33.3ms, -0.1%) p95 36.1ms max 51.8ms
```

수집이 끝나면 러너가 같은 통계를 요약에 출력합니다. 평균이 목표에서 10% 넘게
벗어나면 `record_fps`를 실측값으로 낮추고 다시 모으세요.

---

## 6. 검수

```bash
PYTHONPATH=src uv run --no-sync python - <<'PY'
from lerobot.datasets import LeRobotDataset
import cv2, numpy as np
ROOT = "<summary의 dataset_root>"
ds = LeRobotDataset("<repo_id>", root=ROOT)
print(ds.num_episodes, "episodes /", ds.num_frames, "frames @", ds.fps, "fps")
print(list(ds.meta.tasks.index))                     # 색별 문장이 다 있는지
s = ds[0]
print(s["observation.images.top"].shape, s["action"].shape)
img = (s["observation.images.top"].permute(1,2,0).numpy()*255).astype(np.uint8)
cv2.imwrite("/tmp/check.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))   # 색 확인
PY
```

봐야 할 것:

- **이미지 색상**: 팔이 보라색, 테이프가 빨강으로 보이면 RGB 정상. 팔이
  청록색이면 채널이 뒤집힌 것입니다.
- **에피소드 길이 분포**: `task3.min_episode_frames`(60)와
  `max_episode_frames`(3000)는 **가정값**입니다. 첫 라운드 후 실측으로 교체하세요.
  ACT 기본 `chunk_size`가 100프레임이라 그보다 짧은 에피소드는 쓸모가 적습니다.
- **요약 JSON** `logs/pick_stack/task3_<TS>_summary.json`: 색별 저장/파기 수,
  파기 사유, 라운드 수, 틱 주기 통계.

파기 사유별 대응:

| reason | 뜻 | 대응 |
|---|---|---|
| `pick_failed` | 한 번의 파지 시도 실패 | 정상. 잦으면 그래스프 보정 점검 |
| `too_short` | `min_episode_frames` 미만 | 하한이 너무 높은지 확인 |
| `too_long` | `max_episode_frames` 초과 | 팔이 루프에 갇혔는지 확인 |
| `stale_camera` | 신선한 프레임이 연속으로 안 옴 | 카메라 서버/USB 대역폭 점검 |
| `interrupted` | Ctrl-C | 정상 |

---

## 7. 주의

- **빨강 블록과 빨강 테이프**: 지정구역 테이프가 빨강이라 `red` task 문장이
  구역 자체와 의미가 겹칩니다. 첫 데이터셋은 red를 빼고 4색으로 시작하는 편이
  낫습니다.
- **파지 불가 블록**: 구역이 빌 때까지 재시도하므로, IK로 안 닿는 위치의 블록이
  있으면 계속 돕니다. `task3.max_rounds_without_progress`(3) 연속으로 저장이 0이면
  프롬프트에 경고가 붙습니다 — 그때 블록을 작업 반경 안으로 옮기세요.
- **학습은 Orin에서 불가**: sm_87 CUDA 커널이 없습니다. 수집만 Orin,
  학습은 데스크탑 ([SO101_학습_추론.md](./SO101_학습_추론.md)).
- **push_to_hub는 호출하지 않습니다.** 데이터셋은 로컬에만 남습니다.
