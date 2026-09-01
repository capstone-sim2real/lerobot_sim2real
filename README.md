# Capstone Sim2Real

SO-101 로봇팔, Jetson/JetBot 기반 vision-action 실험, 온디바이스 LLM 추론 벤치마크를 정리하는 졸업과제 저장소입니다.

LeRobot은 `third_party/lerobot` submodule로 고정하며, 프로젝트 루트의 단일 `uv` 환경에서 함께 실행합니다.

## Quick Start

처음 받기:

```bash
git clone --recurse-submodules <REPOSITORY_URL> ~/lerobot_sim2real
cd ~/lerobot_sim2real
```

이미 clone한 뒤 서브모듈만 맞출 때:

```bash
git submodule update --init --recursive
```

팀원 초기세팅:

```bash
sudo usermod -aG dialout "$USER"
# 로그아웃 후 다시 로그인

cd ~/lerobot_sim2real
uv sync --python 3.12 --extra hardware --extra dev
so101-scan-motors
```

자세한 설치 절차와 `uv`, `fish`, 권한 문제는 [SO-101 세팅 가이드](docs/guide/SO101_세팅가이드.md)를 봅니다.

## Common Commands

모터 응답 확인:

```bash
so101-scan-motors
```

캘리브레이션 초기화 및 재실행:

```bash
so101-robot-calibrate all --reset
```

리더 없이 원격 키보드 조작:

```bash
so101-keyboard --step 1
```

브라우저로 카메라 실시간 확인:

```bash
so101-camera --host 0.0.0.0 --port 8090
```

에이전트 검토용으로 프레임을 주기 저장하려면 다음처럼 실행합니다.

```bash
so101-camera --host 0.0.0.0 --port 8090 \
  --save-dir /tmp/so101-camera --save-interval-s 2
```

색으로 블록을 찾아 집어서 지정 좌표로 옮기기 (카메라 서버를 먼저 띄워둔 채로):

```bash
uv run python -m tools.demo_pick_and_place \
  --color green --to P13
```

계획만 확인하려면 `--dry-run`. 자세한 내용은 [CV+IK 파지·운반 가이드](docs/guide/SO101_CV_IK_파지운반.md).

## Repository Layout

```text
.
├── docs/
│   ├── guide/                 SO-101 사용/실험 가이드
│   └── report/                착수보고서 등 제출 문서
├── src/tools/                 SO-101 Python CLI 구현
├── third_party/               SO-101 자산·LeRobot submodule
```

`third_party/lerobot`은 Git submodule이고, `.venv/`는 각 팀원이 `uv sync`로 생성하는 로컬 환경입니다. `uv.lock`과 submodule 커밋을 함께 추적해 같은 드라이버·기구학 버전을 재현합니다.

## Documents

- [SO-101 세팅 가이드](docs/guide/SO101_세팅가이드.md)
- [SO-101 문제해결](docs/guide/SO101_문제해결.md)
- [SO-101 원격 조작 가이드](docs/guide/SO101_원격조작.md)
- [SO-101 원격 카메라 연결 가이드](docs/guide/SO101_원격카메라.md)
- [SO-101 데이터 수집 & 관리 가이드 (Orin)](docs/guide/SO101_데이터수집_관리.md)
- [SO-101 학습 & 추론 가이드 (데스크탑)](docs/guide/SO101_학습_추론.md)
- [SO-101 CV+IK 파지·운반 가이드](docs/guide/SO101_CV_IK_파지운반.md)
- [JetBot Vision-Action 착수보고서](docs/report/착수보고서/JetBot_Vision_Action_착수보고서.pdf)

## Datasets

모방학습 데이터셋은 git이 아니라 HuggingFace Hub(public)에 둡니다. 로그인 없이 바로 받을 수 있습니다. 받는 법과 수집·관리는 [데이터 수집 & 관리 가이드](docs/guide/SO101_데이터수집_관리.md) 참고.

- [`142spp/so101_place_v1`](https://huggingface.co/datasets/142spp/so101_place_v1) — 병합 마스터 (초록+노랑, 51 에피소드, 학습용)
- [`142spp/so101_place_green_v1`](https://huggingface.co/datasets/142spp/so101_place_green_v1), [`142spp/so101_place_yellow_v1`](https://huggingface.co/datasets/142spp/so101_place_yellow_v1) — 색별 원본

## Notes

- SO-101 실행 전 로봇팔 전원, 서보 데이지체인 케이블, USB serial 연결을 먼저 확인합니다.
- `/dev/ttyACM0`, `/dev/ttyACM1`은 재부팅이나 재연결 후 바뀔 수 있으므로 가능하면 `/dev/serial/by-id/...` 경로를 사용합니다.
- 원격 키보드 조작 전에는 카메라로 팔 주변에 충돌 위험이 없는지 확인합니다.
- 웹 카메라 서버는 LAN 또는 Tailscale 안에서만 열고, 공인 인터넷에는 직접 노출하지 않습니다.
- 대용량 모델 파일, 빌드 디렉토리, 캐시, 로그 파일은 저장소에 커밋하지 않습니다.
