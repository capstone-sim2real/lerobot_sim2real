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
so101-camera
```

기본값은 `0.0.0.0:8090`이며 비전 오버레이도 함께 활성화됩니다. 오버레이를
완전히 끄려면 `so101-camera --no-overlay`를 사용합니다. 화면의 오버레이는
원본 MJPEG 위에서 브라우저가 합성하므로 영상 스트림을 느리게 만들지 않습니다.

에이전트 검토용으로 장면이 충분히 달라졌을 때만 프레임을 저장하려면 다음처럼 실행함. 2초마다 직전 비교 프레임과 비교하며, 평균 밝기 차이가 8 이상이거나 10초가 지나면 저장함.

```bash
so101-camera \
  --save-dir /tmp/so101-camera --save-interval-s 2 \
  --save-on-change --change-threshold 8 --max-save-interval-s 10
```

CV/IK 파지 flow 실행 (카메라 서버를 먼저 띄워둔 채로):

```bash
so101-run --task 1 --flow pick_lift_lower --color green
```

고정 빨강 테이프 구역은 기존 homography를 보존하는 전용 도구로 한 번 등록함.

```bash
so101-zone-calibrate                         # preview only
so101-zone-calibrate --write                 # zone_polygon_mm 저장
so101-run --task 1 --dry-run                 # 모터 연결 없이 슬롯/IK 확인
so101-run --task 1                           # 외부 블록이 5초간 없을 때까지 수집
```

Task 1의 검출 범위는 주황 부채꼴 안이면서 보라색 구역 밖인 부분임. PLACE 횟수나
색상 개수로 종료하지 않고, 홈 자세에서 fresh frame 기준 외부 검출이 5초 동안
연속 0개일 때만 완료함.

자세한 내용은 [CV+IK 파지·운반 가이드](docs/guide/SO101_CV_IK_파지운반.md).

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
- [SO-101 CV+IK 파지·운반 가이드](docs/guide/SO101_CV_IK_파지운반.md)
- [JetBot Vision-Action 착수보고서](docs/report/착수보고서/JetBot_Vision_Action_착수보고서.pdf)

## Notes

- SO-101 실행 전 로봇팔 전원, 서보 데이지체인 케이블, USB serial 연결을 먼저 확인합니다.
- `/dev/ttyACM0`, `/dev/ttyACM1`은 재부팅이나 재연결 후 바뀔 수 있으므로 가능하면 `/dev/serial/by-id/...` 경로를 사용합니다.
- 원격 키보드 조작 전에는 카메라로 팔 주변에 충돌 위험이 없는지 확인합니다.
- 웹 카메라 서버는 LAN 또는 Tailscale 안에서만 열고, 공인 인터넷에는 직접 노출하지 않습니다.
- 대용량 모델 파일, 빌드 디렉토리, 캐시, 로그 파일은 저장소에 커밋하지 않습니다.
