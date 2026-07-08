# Capstone Sim2Real

SO-101 로봇팔, Jetson/JetBot 기반 vision-action 실험, 그리고 온디바이스 LLM 추론 벤치마크를 정리하는 졸업과제 저장소입니다.

이 저장소는 로봇 세팅과 조작을 반복 가능한 스크립트로 관리하고, 팀원이 같은 환경에서 SO-101 캘리브레이션, 텔레오퍼레이션, 원격 키보드 조작, 원격 카메라 확인을 실행할 수 있도록 문서화합니다. LLM 벤치마크는 `llama.cpp`를 git submodule로 고정해 별도 도구 영역에서 관리합니다.

## 저장소 구조

```text
.
├── docs/
│   ├── SO101_세팅가이드.md
│   ├── SO101_원격조작.md
│   ├── SO101_원격카메라.md
│   └── JetBot_Vision_Action_착수보고서.pdf
├── scripts/
│   ├── so101_env.sh
│   ├── so101_scan_motors.sh
│   ├── so101_calibrate_leader.sh
│   ├── so101_calibrate_follower.sh
│   ├── so101_teleop_30.sh
│   ├── so101_teleop_60.sh
│   ├── so101_keyboard_control.sh
│   └── so101_keyboard_control.py
└── tools/
    └── benchmark/llm/
        ├── app/
        ├── benches/
        └── llama.cpp/
```

## 처음 받기

서브모듈까지 포함해서 받을 때:

```bash
git clone --recurse-submodules <REPOSITORY_URL>
cd lerobot_sim2real
```

이미 clone한 뒤라면:

```bash
git submodule update --init --recursive
```

대용량 모델 파일, 빌드 산출물, 실행 로그는 원칙적으로 저장소에 포함하지 않습니다.

## 팀원 초기세팅

이 프로젝트는 LeRobot을 저장소 안에 복사하지 않고, `~/lerobot`에 별도 clone해서 사용합니다. 팀원 간 환경 차이를 줄이기 위해 LeRobot commit을 고정하고 `uv.lock` 기준으로 가상환경을 맞춥니다.

처음 세팅 순서:

```bash
git clone --recurse-submodules <REPOSITORY_URL> ~/lerobot_sim2real
cd ~/lerobot_sim2real
```

먼저 `uv`가 없다면 설치합니다. 설치 후 `uv --version`이 보여야 합니다.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
uv --version
```

SO-101 USB serial을 일반 사용자로 열 수 있도록 `dialout` 그룹에 현재 사용자를 추가합니다. 이 명령 뒤에는 로그아웃 후 다시 로그인해야 적용됩니다.

```bash
sudo usermod -aG dialout "$USER"
```

다시 로그인한 뒤 `id` 출력에 `dialout`이 있는지 확인합니다.

```bash
id
```

그다음 프로젝트에서 LeRobot 환경을 준비합니다. 이 스크립트는 `~/lerobot`을 준비하고 `~/lerobot/.venv`를 만듭니다.

```bash
cd ~/lerobot_sim2real
./scripts/setup_lerobot_env.sh
```

좌표 기준 키보드 조작까지 쓸 팀원은 kinematics 의존성도 같이 설치합니다.

```bash
INSTALL_KINEMATICS=1 ./scripts/setup_lerobot_env.sh
```

기본 고정 버전:

```text
LeRobot commit: 8a74e0ac6d01706d67fddfed682a09d694d9c8c0
LeRobot path:   ~/lerobot
Python env:     ~/lerobot/.venv
```

세팅 확인:

```bash
./scripts/so101_scan_motors.sh
```

정상이라면 각 팔에서 모터 ID 1-6이 보입니다.

## SO-101 빠른 실행

기본 작업 위치는 아래 경로를 기준으로 합니다.

```text
프로젝트: ~/lerobot_sim2real
LeRobot:  ~/lerobot
```

프로젝트 스크립트는 내부에서 `~/lerobot/.venv`를 불러옵니다. 다른 위치에 LeRobot을 설치했다면 `LEROBOT_DIR` 환경변수를 지정해서 실행합니다.

```bash
LEROBOT_DIR=/path/to/lerobot ./scripts/so101_scan_motors.sh
```

모터 응답 확인:

```bash
cd ~/lerobot_sim2real
./scripts/so101_scan_motors.sh
```

캘리브레이션 초기화 및 재실행:

```bash
./scripts/so101_reset_calibration.sh
./scripts/so101_calibrate_leader.sh
./scripts/so101_calibrate_follower.sh
```

리더-팔로워 텔레오퍼레이션:

```bash
./scripts/so101_teleop_30.sh
./scripts/so101_teleop_60.sh
```

자세한 세팅 절차는 [SO101_세팅가이드.md](docs/SO101_세팅가이드.md)를 참고합니다.

## 원격 조작

리더 팔 없이 SSH 또는 Tailscale SSH로 Jetson에 접속해서 SO-101 follower를 키보드로 조작할 수 있습니다.

```bash
cd ~/lerobot_sim2real
./scripts/so101_keyboard_control.sh
```

처음에는 이동량을 줄여서 확인하는 것을 권장합니다.

```bash
./scripts/so101_keyboard_control.sh --step 1
```

원격 조작 전에는 카메라로 팔 주변 상태를 확인합니다. 키 매핑, 안전 제한, 옵션은 [SO101_원격조작.md](docs/SO101_원격조작.md)에 정리되어 있습니다.

## 원격 카메라

Jetson에 연결된 USB 카메라 영상을 Windows PC에서 GStreamer UDP/RTP로 확인합니다.

기본 사용 노드와 포트:

```text
/dev/video0 -> UDP 5000
/dev/video2 -> UDP 5002
```

네트워크 환경, Tailscale IP 사용, Jetson 송출 명령, Windows 수신 명령은 [SO101_원격카메라.md](docs/SO101_원격카메라.md)를 참고합니다.

## LLM 벤치마크

LLM 추론 벤치마크는 `tools/benchmark/llm/` 아래에서 관리합니다.

```text
tools/benchmark/llm/llama.cpp  upstream llama.cpp submodule
tools/benchmark/llm/app        Orin benchmark integration용 커스텀 앱 소스
tools/benchmark/llm/benches    DGX Spark, Mac M2 Ultra, Nemotron 벤치마크 결과
```

`llama.cpp`는 submodule이므로 업데이트가 필요할 때는 루트 저장소와 submodule 변경을 분리해서 관리합니다.

## 주요 문서

- [SO-101 세팅 가이드](docs/SO101_세팅가이드.md)
- [SO-101 원격 조작 가이드](docs/SO101_원격조작.md)
- [SO-101 원격 카메라 연결 가이드](docs/SO101_원격카메라.md)
- [SO-101 블록 쌓기 모방학습 로컬 사용설명서](docs/SO101_블록쌓기_모방학습_로컬_사용설명서.md)
- [JetBot Vision-Action 착수보고서](docs/JetBot_Vision_Action_착수보고서.pdf)

## 주의사항

- SO-101 실행 전 로봇팔 전원, 서보 데이지체인 케이블, USB serial 연결을 먼저 확인합니다.
- `/dev/ttyACM0`, `/dev/ttyACM1`은 재부팅이나 재연결 후 바뀔 수 있으므로 가능하면 `/dev/serial/by-id/...` 경로를 사용합니다.
- 원격 키보드 조작 전에는 카메라로 팔 주변에 충돌 위험이 없는지 확인합니다.
- 모델 파일, 빌드 디렉토리, 캐시, 로그 파일은 저장소에 커밋하지 않습니다.
