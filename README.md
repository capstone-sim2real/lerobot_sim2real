# Capstone Sim2Real

SO-101 로봇팔로 5개 블록을 지정 영역에 옮기고 적층하는 졸업과제 저장소입니다.
현재 검증·운영 중인 기본 경로는 고정 탑 카메라의 색·형상 검출과 결정론적
IK를 결합한 **Task 1 CV+IK zone gathering**입니다. Task 2는 동일한
PICK/VERIFY/TRANSPORT 흐름에 적층 PLACE 전략만 교체하는 목표 구조를 유지합니다.

과거 ACT/SmolVLA 실험 문서는 CV+IK로 피벗한 근거를 보존하는 기록입니다. 현재
실행법은 이 README와 [현재 아키텍처](docs/architecture.md),
[CV+IK 가이드](docs/guide/SO101_CV_IK_파지운반.md)를 기준으로 합니다.

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
source .venv/bin/activate
so101-scan-motors
```

자세한 설치 절차와 `uv`, `fish`, 권한 문제는 [SO-101 세팅 가이드](docs/guide/SO101_세팅가이드.md)를 봅니다.

## 세션 도구와 실험 기록

- [텔레옵·점 기록·실시간 FK 권장 실행법](docs/guide/SO101_세션도구.md)
- [날짜별 실험 기록](experiments/README.md)

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

블록 하나를 집었다가 같은 위치에 내려놓는 CV/IK smoke flow 실행
(카메라 서버를 먼저 띄워둔 채로):

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

현재 Task 1 runner는 이 완료 판정을 위해 내부 FSM 시간 예산을 적용하지 않습니다.
평가의 180초 제한을 재현할 때는 내부 deadline이 구현될 때까지 외부 supervisor로
시간을 제한하고 실험 로그에 그 명령을 함께 남깁니다.

자세한 내용은 [CV+IK 파지·운반 가이드](docs/guide/SO101_CV_IK_파지운반.md).

### Task 3 — ACT 데이터셋 자동 수집

Task 1의 수집 루프를 그대로 돌리면서 성공한 사이클 하나를 LeRobot 에피소드로
녹화합니다. 사람은 블록 5개를 배치하는 일만 합니다.

```bash
uv pip install --python .venv/bin/python "datasets>=4.7.0,<5.0.0" "av>=15.0.0,<16.0.0"
so101-camera                                 # 손목캠도 쓰면 --wrist-device /dev/videoN
so101-collect --dry-run                      # 카메라·슬롯·features 확인, 팔은 정지
so101-collect                                # Ctrl-C 로 종료
# 동일한 Task 3 진입점 별칭
so101-run --task 3 --dry-run
so101-run --task 3
```

한 에피소드 = `home → 파지 → 운반 → 릴리스 → home 복귀`. 파지는 **1회만**
시도하고 운반까지 성공한 것만 저장합니다. 지정구역 밖이 비면 재배치를 요청하고
계속 수집합니다. Ctrl-C는 미완 에피소드만 버리고 나머지는 보존합니다.

자세한 내용은 [Task 3 데이터 수집 가이드](docs/guide/SO101_TASK3_데이터수집.md).

### LLM 에이전트 — 자연어로 조작 (`so101-agent`)

기존 Task 1/2/3 명령은 그대로 두고, 자연어 요청을 LLM 툴 콜링으로 실행하는 웹
채팅 서버를 따로 띄웁니다. `so101-run`/`so101-collect`와 **동시에 실행하지 않습니다**
(로봇 버스 락으로 막힘).

```bash
# 최초 1회: 의존성 (uv sync 금지 — JetPack torch 휠 보호)
uv pip install --python .venv/bin/python "fastapi>=0.110" "uvicorn>=0.29" \
  "openai>=1.60" "google-genai>=1.0"
uv pip install --python .venv/bin/python -e . --no-deps                       # so101-agent 스크립트 등록

cp .env.example .env && vi .env            # OPENAI_API_KEY와 GEMINI_API_KEY 입력

so101-agent --dry-run                      # 칸·부채꼴 영역 IK, 조그 높이, API 키, 카메라 확인 (팔 정지)
so101-agent --sim --provider fake          # 하드웨어·API 키 없이 UI와 툴 흐름 리허설

so101-camera                               # 실기: 카메라 서버 먼저
so101-agent                                # 기본 OpenAI gpt-5.6-luna, 실패 시 Gemini 폴백
so101-agent --provider gemini              # 명시하면 Gemini만 사용(폴백 없음)
```

예: "노란 블록을 적재 구역 좌상단으로 옮겨줘", "초록 블록 다시 밖으로 꺼내줘",
"5mm만 더 멀리 집어줘", "왼쪽으로 가줘" → "여기 내려놔", "그 블록 20mm만 더 왼쪽으로",
"미션 1 해줘". 여러 브라우저가 같은 세션에 참여할 수 있고, 로봇 명령은 한 번에 하나만
실행됩니다. 명령 하나가 끝날 때까지 입력 잠금, STOP을 누르면
그리퍼를 열고(들고 있던 블록은 놓고) home 복귀 후 그리퍼를 닫는 것까지 자동으로
진행되며 그게 확인돼야 다시 입력할 수 있습니다.

자세한 내용은 [LLM 에이전트 가이드](docs/guide/SO101_LLM_에이전트.md).

## Repository Layout

```text
.
├── docs/
│   ├── calibration/           캘리브레이션 자료 위치 안내
│   ├── eval/                  정량 평가 기록 형식
│   ├── guide/                 현재 운영 가이드와 과거 실험 가이드
│   └── report/                시점별 제출·피벗 기록(역사 자료)
├── experiments/               날짜별 실장비 증거와 실패 기록
├── src/
│   ├── camera/                단일 소유 카메라 서버, 오버레이, 녹화용 프레임 소스
│   ├── data/                  Task 3 LeRobot 에피소드 녹화
│   ├── perception/            homography, 색·형상 검출, 선택
│   ├── control/               IK, 파지, 궤적, 센싱
│   ├── fsm/                   Task 흐름과 상태 구현
│   ├── policy/                보존된 optional ACT PICK 클라이언트
│   ├── runners/               so101-run 조립·실행
│   ├── session/               한 번 연결된 팔 세션, 취소·버스 락, 에이전트 스킬
│   ├── agent/                 so101-agent: LLM 툴·어댑터·대화 루프·웹 UI
│   └── tools/                 캘리브레이션·진단·세션 CLI
├── third_party/               SO-101 자산·LeRobot submodule
└── docker/                    보존된 ACT policy-server 제출 경로
```

`third_party/lerobot`은 Git submodule이고, `.venv/`는 각 팀원이 `uv sync`로 생성하는 로컬 환경입니다. `uv.lock`과 submodule 커밋을 함께 추적해 같은 드라이버·기구학 버전을 재현합니다.

## Documents

- [현재 아키텍처와 구현 상태](docs/architecture.md)
- [SO-101 세팅 가이드](docs/guide/SO101_세팅가이드.md)
- [SO-101 문제해결](docs/guide/SO101_문제해결.md)
- [SO-101 원격 조작 가이드](docs/guide/SO101_원격조작.md)
- [SO-101 원격 카메라 연결 가이드](docs/guide/SO101_원격카메라.md)
- [SO-101 CV+IK 파지·운반 가이드](docs/guide/SO101_CV_IK_파지운반.md)
- [Task 3 ACT 데이터셋 자동 수집 가이드](docs/guide/SO101_TASK3_데이터수집.md)
- [캘리브레이션 자료 위치](docs/calibration/README.md)
- [정량 평가 기록](docs/eval/README.md)
- [과거 ACT/SmolVLA 데이터 수집 기록](docs/guide/SO101_데이터수집_관리.md)
- [과거 ACT/SmolVLA 학습·추론 기록](docs/guide/SO101_학습_추론.md)
- [JetBot Vision-Action 착수보고서](docs/report/착수보고서/JetBot_Vision_Action_착수보고서.pdf)

## Notes

- SO-101 실행 전 로봇팔 전원, 서보 데이지체인 케이블, USB serial 연결을 먼저 확인합니다.
- `/dev/ttyACM0`, `/dev/ttyACM1`은 재부팅이나 재연결 후 바뀔 수 있으므로 가능하면 `/dev/serial/by-id/...` 경로를 사용합니다.
- 원격 키보드 조작 전에는 카메라로 팔 주변에 충돌 위험이 없는지 확인합니다.
- 웹 카메라 서버는 LAN 또는 Tailscale 안에서만 열고, 공인 인터넷에는 직접 노출하지 않습니다.
- 대용량 모델 파일, 빌드 디렉토리, 캐시, 로그 파일은 저장소에 커밋하지 않습니다.
