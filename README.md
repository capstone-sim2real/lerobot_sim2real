# Capstone Sim2Real

SO-101 로봇팔, Jetson/JetBot 기반 vision-action 실험, 온디바이스 LLM 추론 벤치마크를 정리하는 졸업과제 저장소입니다.

이 저장소는 LeRobot 원본을 직접 수정하지 않고, 팀 장비에 맞춘 실행 스크립트와 문서를 관리합니다.

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
./scripts/setup_lerobot_env.sh
./scripts/so101_scan_motors.sh
```

자세한 설치 절차와 `uv`, `fish`, 권한 문제는 [SO-101 세팅 가이드](docs/guide/SO101_세팅가이드.md)를 봅니다.

## Common Commands

모터 응답 확인:

```bash
./scripts/so101_scan_motors.sh
```

캘리브레이션 초기화 및 재실행:

```bash
./scripts/so101_calibrate.sh all
```

리더-팔로워 텔레오퍼레이션:

```bash
./scripts/so101_teleop.sh 30
./scripts/so101_teleop.sh 60
```

리더 없이 원격 키보드 조작:

```bash
./scripts/so101_keyboard_control.sh --step 1
```

브라우저로 카메라 실시간 확인:

```bash
./scripts/so101_camera_web.sh
```

모방학습 에피소드 1개 기록:

```bash
./scripts/so101_learn.sh record
```

## Repository Layout

```text
.
├── docs/
│   ├── guide/                 SO-101 사용/실험 가이드
│   └── report/                착수보고서 등 제출 문서
├── scripts/                   SO-101 실행 래퍼 스크립트
├── third_party/               외부 참고 자산
└── tools/benchmark/llm/       llama.cpp 기반 LLM 벤치마크
```

LeRobot은 저장소 안에 복사하지 않고 별도 위치에 둡니다.

```text
프로젝트: ~/lerobot_sim2real
LeRobot:  ~/lerobot
venv:     ~/lerobot/.venv
```

프로젝트 스크립트는 내부에서 `~/lerobot/.venv`를 불러옵니다. 다른 위치에 LeRobot을 설치했다면 `LEROBOT_DIR` 환경변수를 지정합니다.

```bash
LEROBOT_DIR=/path/to/lerobot ./scripts/so101_scan_motors.sh
```

## Documents

- [SO-101 세팅 가이드](docs/guide/SO101_세팅가이드.md)
- [SO-101 문제해결](docs/guide/SO101_문제해결.md)
- [SO-101 원격 조작 가이드](docs/guide/SO101_원격조작.md)
- [SO-101 원격 카메라 연결 가이드](docs/guide/SO101_원격카메라.md)
- [SO-101 블록 쌓기 모방학습 로컬 사용설명서](docs/guide/SO101_블록쌓기_모방학습_로컬_사용설명서.md)
- [JetBot Vision-Action 착수보고서](docs/report/착수보고서/JetBot_Vision_Action_착수보고서.pdf)

## Notes

- SO-101 실행 전 로봇팔 전원, 서보 데이지체인 케이블, USB serial 연결을 먼저 확인합니다.
- `/dev/ttyACM0`, `/dev/ttyACM1`은 재부팅이나 재연결 후 바뀔 수 있으므로 가능하면 `/dev/serial/by-id/...` 경로를 사용합니다.
- 원격 키보드 조작 전에는 카메라로 팔 주변에 충돌 위험이 없는지 확인합니다.
- 웹 카메라 서버는 LAN 또는 Tailscale 안에서만 열고, 공인 인터넷에는 직접 노출하지 않습니다.
- 대용량 모델 파일, 빌드 디렉토리, 캐시, 로그 파일은 저장소에 커밋하지 않습니다.
