# Sim2Real: SO-101 Pick & Stack

<p align="center">
  <img src="docs/report/최종보고서/figures/demo_060.jpg" alt="SO-101이 체스판 작업대에서 블록을 지정 영역으로 옮기는 실제 시연 장면" width="760">
</p>

<p align="center">
  <a href="#5-설치-및-실행-방법"><img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12"></a>
  <a href="docs/architecture.md"><img src="https://img.shields.io/badge/control-CV%20%2B%20IK-6A5ACD" alt="CV와 IK 제어"></a>
  <a href="docs/guide/SO101_LLM_에이전트.md"><img src="https://img.shields.io/badge/interface-Web%20%2B%20LLM-00897B" alt="웹 및 LLM 인터페이스"></a>
</p>

고정 카메라로 블록을 찾고 좌표 보정과 역기구학(IK)으로 SO-101 로봇팔을 움직이는 블록 이동 및 적재 프로젝트입니다. **기본 제어 경로는 LLM 호출 없이 동작하는 CV+IK**입니다. 웹의 자연어 에이전트는 같은 제어 기능을 도구로 호출하는 별도 인터페이스입니다.

> **주요 성과** — 블록 모으기 성공률 **93.3%**, 5블록 적층 성공률 **67%**, 단일 블록 파지 성공률 **99%**. 웹 대시보드에서 미션 실행부터 로봇 상태 확인, 시연 데이터 수집까지 연결합니다.

[빠른 실행](#5-설치-및-실행-방법) / [아키텍처](docs/architecture.md) / [실험 기록](experiments/README.md) / [최종보고서](docs/report/최종보고서/final_report.pdf)

## 1. 프로젝트 배경

### 1.1. 국내외 시장 현황 및 문제점

국제로봇연맹(IFR)의 *World Robotics 2025*에 따르면 2024년 전 세계 산업용 로봇 신규 설치는 **542,076대**, 한국은 약 **30,600대**였습니다. 출처: [IFR 산업용 로봇 보고서](https://ifr.org/img/worldrobotics/Executive_Summary_WR_2025_Industrial_Robots.pdf), [IFR 국가별 발표](https://ifr.org/news/global-robot-demand-in-factories-doubles-over-10-years/1st-quarterly-newsletter-2010).

소형 로봇팔에서 단순한 `목표 픽셀 → 목표 관절` 연결만으로 안정적인 파지를 얻기는 어렵습니다. 카메라 위치 변화, 블록 윗면과 캘리브레이션 평면의 높이 차이, 실제 턱 중심과 URDF 기준 프레임의 차이, 팔의 처짐과 그리퍼 정렬 오차가 누적됩니다. 우리 팀의 초기 ACT와 SmolVLA 실험도 시작 위치와 카메라 조건이 바뀌면 데이터 재수집과 재학습 비용이 커지는 문제가 있었습니다. 이 문제 분석은 [CV+IK 전환 기록](docs/report/CV_IK_전환_정리.md)에 남겨 두었습니다.

### 1.2. 필요성과 기대효과

이 프로젝트는 **인식 → 접근 → 파지 확인 → 운반 → 배치 → 재관찰**을 분리해 실패 지점을 찾고 실험을 재현할 기반을 만듭니다. 파지 실패의 원인을 추적하고 좌표와 경로 보정을 반복 적용합니다. 성공한 동작은 학습용 데이터셋으로 자동 기록해 실험과 데이터 수집을 하나의 흐름으로 연결합니다.

## 2. 개발 목표

### 2.1. 목표 및 세부 내용

| 미션 | 목표 | 과제 제한 | 주요 기능 |
|---|---|---:|---|
| Task 1 | 무작위로 놓인 블록 5개를 20 × 10 cm 지정 영역에 배치 | 180초 | 고정 탑 카메라 CV + 동적 IK 슬롯, 파지 검증, 재시도, fresh frame 완료 판정 |
| Task 2 | 블록을 쌓고 5초 이상 유지 | 300초 | 공통 파지, 검증, 운반 흐름을 재사용하는 적층 제어 |
| Task 3 | 성공한 Task 1 사이클을 학습용 에피소드로 기록 | 별도 수집 기능 | 로봇 I/O와 카메라 스트림을 LeRobot 데이터셋 형식으로 기록 |

설계 목표는 상위 계획 방식을 바꿔도 **실제 동작은 동일한 도달성, 관절 제한, 파지 확인 게이트**를 거치게 하는 것입니다. 기본 미션에는 신경망이 필요하지 않으며 ACT PICK과 LLM 도구 호출은 선택 경로입니다.

### 2.2. 기존 서비스 대비 차별성

| 접근 | 장점 | 이 프로젝트에서 보완한 지점 |
|---|---|---|
| [LeRobot 기본 워크플로](https://huggingface.co/docs/lerobot/main/index) | SO-101 제어, 시연 기록, 정책 학습의 공통 기반 | 대회 규격의 영역 판정, 동적 슬롯, 파지 확인, 실패 복구를 통합 |
| 엔드투엔드 모방학습 | 시연에서 행동을 학습 | 카메라나 초기 배치 변화 때의 데이터 재수집 부담을 줄이기 위해 CV+IK를 기본 경로로 채택 |
| **Sim2Real CV+IK** | 단계별 입출력과 실패 사유를 확인 가능 | 픽셀→베이스 mm 보정, 블록 방향 정렬, IK 후보 탐색, 센서 VERIFY, 재관찰 및 실행 기록을 하나의 흐름으로 연결 |

### 2.3. 사회적 가치 도입 계획

SO-101과 오픈소스 소프트웨어를 활용해 로봇 실습과 연구의 진입 장벽을 낮추는 것을 목표로 합니다. 설계 문서, 실행 도구, 실험 기록을 함께 제공해 교육 현장에서 제어 과정을 따라가고 실험을 재현할 수 있도록 구성했습니다. Task 3의 자동 기록 기능은 반복적인 수동 시연 부담을 줄이고, 수집한 데이터를 후속 학습 연구에 활용할 기반을 제공합니다.

## 3. 시스템 설계

### 3.1. 시스템 구성도

```mermaid
flowchart LR
    C[고정 탑 카메라] --> CS[카메라 서버<br/>MJPEG / snapshot :8090]
    CS --> P[색상과 형상 검출<br/>homography → 베이스 mm]
    P --> S[Task 1/2 대상 선택과 계획]
    S --> IK[IK 후보 및 경로 검사]
    IK --> V[SO-101 로봇팔과 그리퍼]
    V --> G[위치 및 부하 센서<br/>파지 VERIFY]
    G --> S
    CS --> W[웹 관제 오버레이]
    A[선택: 웹 기반 LLM 도구 호출<br/>기본 :8099] --> S
    V --> D[Task 3 에피소드 기록]
    CS --> D
```

카메라 장치는 카메라 서버 한 프로세스가 소유합니다. `so101-run`, `so101-collect`, `so101-agent` 중 **로봇 시리얼 버스 소유자는 동시에 하나**입니다. 웹 오버레이는 관찰용이며 그 표시 결과가 기본 FSM 제어 입력으로 되돌아가지는 않습니다. 자세한 경계는 [현재 아키텍처](docs/architecture.md)를 참고하세요.

### 3.2. 사용 기술

| 영역 | 기술 |
|---|---|
| 하드웨어 | SO-101 팔로워, Feetech 서보, 고정 탑 카메라, NVIDIA Jetson Orin 개발 환경 |
| 비전 | Python, OpenCV, HSV 색상 필터와 형상 필터, homography, MJPEG |
| 제어 | PlacO/URDF 기반 IK, 관절 궤적 보간, LeRobot 로봇 I/O, 위치 및 부하 센싱 |
| 서버 및 UI | FastAPI, Uvicorn, 브라우저 HTML/CSS/JavaScript, SSE, 선택적 OpenAI/Gemini/Anthropic 어댑터 |
| 데이터 및 품질 관리 | LeRobotDataset, NumPy, PyYAML, `uv`, pytest |

실제 설치 패키지와 버전 범위는 [pyproject.toml](pyproject.toml), 실행 구성은 [기본 설정](src/configs/default.yaml)이 기준입니다.

## 4. 개발 결과

### 4.1. 전체 시스템 흐름도

```mermaid
flowchart TD
    O[HOME에서 새 카메라 프레임 관찰] --> S[지정 영역 밖 블록 선택]
    S --> P[접근 및 파지 후보 IK 계산]
    P --> K{IK와 간섭 검사 통과?}
    K -- 아니오 --> S
    K -- 예 --> G[그리퍼 닫기]
    G --> V{위치 + 부하로 파지 확인?}
    V -- 실패 --> R[재시도 또는 다른 블록 선택]
    R --> O
    V -- 성공 --> T[블록 상승 및 운반]
    T --> L[Task 1 슬롯 배치 / Task 2 적층 배치]
    L --> H[그리퍼 해제 후 홈 복귀]
    H --> O
    O --> D{영역 밖 블록이 새 프레임에서 5초간 0개?}
    D -- 예, Task 1 --> E[완료]
    D -- 아니오 --> S
```

Task 1의 완료는 배치 명령 횟수가 아니라 **홈 위치에서 다시 본 장면**으로 판정합니다. 파지가 확인되지 않으면 운반하지 않습니다. Task 2는 적층 위치로 운반한 뒤 층별 해제 동작을 수행합니다.

### 4.2. 기능 설명 및 주요 기능 명세서

<p align="center">
  <img src="docs/assets/dashboard.png" alt="SO-101 웹 대시보드: 실시간 카메라와 도달 범위 오버레이, Task 1/2 실행 버튼, 수동 조작 및 관절 진단" width="1000">
</p>

**웹 대시보드** — 왼쪽에서 실시간 카메라와 도달 범위를 보고 오른쪽에서 Task 1/2 실행과 수동 조작 또는 LLM 대화를 선택합니다. 하단 진단 탭에서는 관절과 모터 상태, 검출 결과와 보정 정보와 에피소드를 확인할 수 있습니다. 위 이미지는 Jetson Orin에서 실행한 대시보드의 다크 모드 화면입니다.

| 기능 | 입력 | 출력 및 판정 | 주요 구현 |
|---|---|---|---|
| 카메라 기반 블록 관찰 | MJPEG/snapshot | 색상, 중심, 방향, 영역 안팎 | `src/camera/`, `src/perception/` |
| 좌표 변환 | 블록 픽셀 좌표, 캘리브레이션 H | 로봇 베이스 프레임의 mm 좌표 | `src/perception/homography.py` |
| 파지 계획 | 대상 블록, 그리퍼 방향 | 도달 가능한 IK 후보 또는 실패 이유 | `src/control/ik.py`, `src/control/grasp.py` |
| 파지 검증 | 그리퍼 위치와 부하 | 물체 파지 여부, 재시도 결정 | `src/control/sensing.py`, `src/fsm/` |
| 운반 및 배치 | 검증된 파지, 슬롯/층 목표 | 궤적 실행, 해제, 재관찰 | `src/control/task1_transport.py`, `src/session/` |
| 웹 기반 자연어 조작 | 사용자 명령 또는 수동 패널 | 한도가 있는 primitive 실행 결과와 상태 | `src/agent/` |
| 시연 수집 | 성공한 한 사이클의 영상, 상태, 행동 | LeRobot 에피소드 | `src/data/`, `so101-collect` |

**실기 평가 결과**

![Task 1 완료율 93.3%, Task 2 5블록 적층 성공률 67%, 4블록 적층 성공률 100%, 단일 블록 파지 성공률 99%](docs/assets/evaluation-bar-chart.png)

[그래프 PDF](docs/assets/evaluation-bar-chart.pdf) / [벡터 SVG](docs/assets/evaluation-bar-chart.svg)

| 평가 항목 | 결과 | 집계 기준 |
|---|---:|---|
| Task 1 미션 완료율 | **93.3% (28/30회)** | 180초 안에 블록 5개를 지정 구역에 배치; 최종보고서 집계 |
| Task 2: 5블록 적층 성공률 | **67%** | 30회 시행 |
| Task 2: 4블록 적층 성공률 | **100%** | 30회 시행 |
| 단일 블록 파지 성공률 | **99% (99/100회)** | Task 구분 없이 집계 |

Task 1은 최종보고서의 30회 평가, Task 2와 단일 블록 파지는 최신 팀 집계를 기준으로 정리했습니다. 평가 항목별 정의와 출처는 [평가 자료 안내](docs/assets/README.md#평가-자료-안내)에서 확인할 수 있습니다.

[최신 팀 집계](docs/assets/evaluation-update.json) / [Task 1 보고서 원자료](docs/report/최종보고서/evidence/20260914/team_results.json) / [보고서 집계 근거](docs/report/최종보고서/SOURCES.md)

### 4.3. 디렉토리 구조

```text
.
├── src/
│   ├── camera/       카메라 단독 소유 서버, 프레임 소스
│   ├── perception/   검출, 좌표 변환, 장면 분석
│   ├── control/      IK, 궤적, 파지, 센싱
│   ├── fsm/          미션 상태 전이
│   ├── session/      로봇 세션, 버스 잠금, 스킬
│   ├── agent/        웹 UI, LLM 도구 호출, 수동 조작
│   ├── data/         에피소드 기록
│   ├── runners/      Task CLI
│   ├── policy/       보존된 선택적 ACT 경로
│   └── tools/        캘리브레이션, 진단 CLI
├── frontend/         웹 프런트엔드 빌드 소스
├── docs/             가이드, 아키텍처, 보고서, 발표 자료
├── experiments/      실험 기록과 결과
├── tests/            하드웨어 없는 회귀 테스트
├── third_party/      SO-101 자산과 LeRobot submodule
└── pyproject.toml     패키지, CLI, 의존성
```

### 4.4. 산업체 멘토링 의견 및 반영 사항

2026년 8월 3일 삼성중공업 **이재민 프로**의 서면 자문을 받았습니다. [자문의견서](docs/report/25_%28Sim2Real%29삼성중공업_이재민_자문의견서.pdf)와 [최종보고서 반영 표](docs/report/최종보고서/final_report.pdf)에 상세 내용이 있습니다.

| 자문 요지 | 반영 사항 |
|---|---|
| 설계 가정, 구현, 검증 결과를 구분 | 미션 완료율, 적층 성공률, 파지 성공률을 구분해 평가 결과 정리 |
| 후보 접근법의 선정 기준 필요 | ACT/SmolVLA → 하이브리드 → CV+IK 전환 이유와 재수집 부담 기록 |
| 성공률 외 데이터, 구축, 유지 비용 고려 | 학습 데이터 수집과 캘리브레이션 유지에 필요한 작업을 비교해 제어 방식 선정 |
| 반복 평가와 실패 단계 기록 | 파지, 운반, 배치 단계를 나누어 실패 원인과 재시도를 기록 |

## 5. 설치 및 실행 방법

### 5.1. 설치절차 및 실행 방법

**준비물:** Linux, Python 3.12, `uv`, SO-101 팔로워와 전원, USB 시리얼, 고정 탑 카메라. 실제 동작 전 [장비 세팅 가이드](docs/guide/SO101_세팅가이드.md)에 따라 서보 ID, 캘리브레이션, 전원, 카메라 고정을 확인하세요.

```bash
git clone --recurse-submodules https://github.com/pnucse-capstone2026/capstone-2026-team-25.git ~/lerobot_sim2real
cd ~/lerobot_sim2real
# 이미 clone했다면: git submodule update --init --recursive
sudo usermod -aG dialout "$USER"  # 로그아웃, 재로그인 후 적용
uv sync --python 3.12 --extra hardware --extra dev
uv run so101-scan-motors
```

> **Jetson Orin 기존 환경:** JetPack용 PyTorch 휠을 보존해야 하는 장비에서는 무조건 `uv sync`로 기존 가상환경을 갈아엎지 않습니다. 현재 인터프리터 경로를 확인하고 필요한 선택 의존성만 `uv pip install --python /path/to/existing-venv/bin/python ...`으로 설치하세요. 이 저장소의 Orin worktree는 가상환경이 worktree 밖에 있으므로 아래 `.venv/bin/...` 명령을 그대로 실행할 수 없습니다. [설치 가이드](docs/guide/SO101_세팅가이드.md)와 [데이터 수집 가이드](docs/guide/SO101_TASK3_데이터수집.md)를 참고하세요.

카메라 서버는 별도 터미널에서 먼저 실행하거나 runner가 기존 서버를 재사용하게 둡니다. **같은 카메라 장치를 두 프로세스에서 열지 않습니다.**

```bash
uv run so101-camera                    # 기본 :8090, MJPEG, snapshot, 오버레이
uv run so101-run --task 1 --dry-run    # 팔을 움직이지 않고 슬롯, IK 확인
uv run so101-run --task 1              # Task 1 실기
uv run so101-run --task 2 --dry-run    # 층별 IK 확인
uv run so101-run --task 2              # Task 2 실기
```

시연 수집과 웹 조작은 아래 명령으로 실행합니다. `so101-run`, `so101-collect`, `so101-agent`를 **동시에 실행하지 마세요.**

```bash
uv run so101-collect --dry-run         # Task 3 입력, 슬롯 확인
uv run so101-collect                   # 성공 사이클을 에피소드로 기록

cp .env.example .env                   # API 키는 .env에만 저장
uv pip install --python .venv/bin/python 'fastapi>=0.110' 'uvicorn>=0.29' \
  'httpx>=0.27' 'ruckig==0.19.4' 'openai>=1.60' 'google-genai>=1.0'
uv pip install --python .venv/bin/python -e . --no-deps
so101-agent --dry-run                  # 팔 정지, 설정, 카메라, IK 확인
so101-agent --sim --provider fake      # 장비, API 키 없는 웹 리허설
so101-agent                            # 웹 UI 기본 :8099
```

Task 1/2 실행과 primitive 제어를 제공하는 웹 패널은 별도 진입점 `tools.agent_server`를 사용합니다. 이 서버도 로봇 버스를 단독 소유하므로 `so101-agent`와 함께 실행하지 않습니다. 기존 Orin 환경에서는 `SO101_PY`에 **이미 설치된** Python 경로를 지정하세요.

```bash
SO101_PY=/absolute/path/to/existing-venv/bin/python
PYTHONPATH="$PWD/src" "$SO101_PY" -m tools.agent_server \
  --env-file .env --output experiments/llm_free_missions/live --port 8109
```

| 서비스 | 기본 포트 | 역할 |
|---|---:|---|
| `so101-camera` | 8090 | 카메라 영상, 스냅샷, 오버레이 |
| `so101-agent` | 8099 | 기본 채팅, 수동 조작, 상태 API; `--port`로 변경 가능 |
| `tools.agent_server` | 8109 | Task 1/2 실행, primitive 제어 패널 |

Jetson이나 다른 원격 장비에서 실행한다면 브라우저는 해당 장비의 LAN/Tailscale 주소로 접속합니다. 로봇 제어 서버를 공인 인터넷에 직접 공개하지 마세요. 기본 카메라, 로봇 설정은 [default.yaml](src/configs/default.yaml)을 확인하고 실험값은 `--set key.path=value`로 덮을 수 있습니다. 180초 제한을 적용하는 정식 평가는 외부 supervisor로 실행 시간을 관리합니다.

### 5.2. 오류 발생 시 해결 방법

| 증상 | 먼저 확인할 것 |
|---|---|
| `RobotBusBusy` | 다른 `so101-run`, `so101-collect`, `so101-agent`가 시리얼 버스를 점유했는지 확인하고 한 실행만 남기기 |
| `Missing motor IDs` | 로봇 전원, 서보 데이지체인 케이블, `/dev/serial/by-id/…`, 모터 ID 스캔 확인 |
| 카메라 프레임 지연, 누락 | `:8090/health`에서 프레임 age와 장치 상태 확인; 중복 카메라 서버를 열지 않기 |
| `ik_gate`, 도달 실패 | `--dry-run`과 목표 좌표, 캘리브레이션, 작업 영역 확인; 무리하게 제한값만 높이지 않기 |
| `ModuleNotFoundError: tools` | 프로젝트 루트에서 설치된 CLI로 실행하거나 `PYTHONPATH="$PWD/src" .venv/bin/python -m tools.agent_server`처럼 실행 |
| Orin에서 Torch/CUDA 오류 | JetPack 호환 휠을 확인하고 일반 `uv sync`로 교체하지 않기 |

더 자세한 진단은 [문제해결 가이드](docs/guide/SO101_문제해결.md)에 있습니다.

## 6. 소개 자료 및 시연 영상

### 6.1. 프로젝트 소개 자료

- [최종보고서 PDF](docs/report/최종보고서/final_report.pdf) / [최종보고서 근거, 해석 범위](docs/report/최종보고서/SOURCES.md)
- [세미나 발표자료 PPTX](docs/report/졸과%20세미나%20발표자료.pptx)
- [시스템 아키텍처](docs/architecture.md) / [CV+IK 파지, 운반 가이드](docs/guide/SO101_CV_IK_파지운반.md)

실제 시연에서 촬영한 시작, 이동, 배치 장면입니다.

| 시작 장면 | 이동 중 | 배치 후 |
|:---:|:---:|:---:|
| ![작업대와 블록의 시작 배치](docs/report/최종보고서/figures/demo_000.jpg) | ![로봇팔의 블록 이동 장면](docs/report/최종보고서/figures/demo_060.jpg) | ![배치가 진행된 작업대](docs/report/최종보고서/figures/demo_104.jpg) |

### 6.2. 시연 영상

<table>
  <tr>
    <td align="center" width="50%">
      <a href="https://www.youtube.com/shorts/AEWSGNCYYoI">
        <img src="https://img.youtube.com/vi/AEWSGNCYYoI/hqdefault.jpg" alt="Task 1 블록 모으기 시연 영상 재생" width="480" height="270">
      </a><br>
      <a href="https://www.youtube.com/shorts/AEWSGNCYYoI"><strong>Task 1: 블록 모으기</strong></a>
    </td>
    <td align="center" width="50%">
      <a href="https://www.youtube.com/shorts/N8D6KveiDgM">
        <img src="https://img.youtube.com/vi/N8D6KveiDgM/hqdefault.jpg" alt="Task 2 블록 탑쌓기 시연 영상 재생" width="480" height="270">
      </a><br>
      <a href="https://www.youtube.com/watch?v=N8D6KveiDgM"><strong>Task 2: 블록 탑쌓기</strong></a>
    </td>
  </tr>
</table>

썸네일을 누르면 각 영상으로 이동합니다.

## 7. 팀 구성

### 7.1. 팀원별 소개 및 역할 분담

| 팀원 | 주요 담당 |
|---|---|
| **윤민석** | 블록 검출, 영역 필터, 파지, 적재 보정, 그리퍼 정렬, 재시도, Task 1/2 제어 통합, Task 3 수집 |
| **이동근** | FSM 파지, 대체 후보 보완, 카메라 웹 뷰어, 오버레이, 캘리브레이션 도구, LeRobot/CLI 환경, 보고서 |
| **김주환** | ACT와 SmolVLA 분석과 CV+IK 전환 근거, 평가 체계, 실험 기록, 통합 시험, 코드 검토, 발표 자료 |
| **공동** | 실장비 통합, 데이터 수집, 실패 관찰과 결과 검토 |

역할 범위는 [최종보고서](docs/report/최종보고서/final_report.pdf)의 팀 분담 표를 요약했습니다. 지도교수: **김종덕**.

### 7.2. 팀원 별 참여 후기

<!-- 담당 업무 기반 편집 초안. 작성 경위는 docs/team-reflections.md에 보존. -->

<details open>
<summary><strong>윤민석</strong> — 로봇 제어와 파지 보정</summary>

블록을 검출하는 것과 실제로 안정적으로 집어 옮기는 것 사이에 생각보다 많은 차이가 있었습니다. 파지와 적재 실패를 관찰하며 보정과 재시도 흐름을 다듬었고, 한 번의 성공보다 반복 가능한 동작을 만드는 일이 중요하다고 느꼈습니다. 각 기능을 Task 1/2와 데이터 수집까지 연결하면서 제어 코드의 재사용성과 실험 기록의 필요성을 배웠습니다.

</details>

<details open>
<summary><strong>이동근</strong> — 비전 도구와 웹 대시보드</summary>

카메라 화면에서 맞아 보이는 좌표도 실제 로봇에서는 어긋날 수 있어, 좌표계와 캘리브레이션을 차근차근 확인하는 과정이 중요했습니다. 웹 뷰어와 오버레이에 관찰 정보를 드러내면서 팀원들이 같은 장면을 보고 문제를 논의하기 쉬워졌습니다. 실행 환경과 도구, 문서를 함께 정리하는 일이 실험을 이어가는 데 큰 도움이 된다는 점을 배웠습니다.

</details>

<details open>
<summary><strong>김주환</strong> — 학습 방식 분석과 실험 평가</summary>

ACT와 SmolVLA를 검토하고 CV+IK로 전환하는 과정을 통해, 접근법의 복잡함보다 주어진 장비와 데이터에 맞는 선택이 중요하다고 느꼈습니다. 평가를 정리하면서 파지 성공, 미션 완료, 적층 높이가 서로 다른 지표라는 점을 분명히 구분하려고 했습니다. 통합 시험과 코드 검토에서 발견한 문제를 팀원들과 공유하며, 실패 기록도 다음 설계를 위한 근거가 된다는 것을 배웠습니다.

</details>

## 8. 참고 문헌 및 출처

1. International Federation of Robotics, [*World Robotics 2025: Industrial Robots — Executive Summary*](https://ifr.org/img/worldrobotics/Executive_Summary_WR_2025_Industrial_Robots.pdf). 2024년 세계 설치 수치와 한국 시장 맥락.
2. International Federation of Robotics, [*World Robotics 2025 발표: 국가별 산업용 로봇 설치*](https://ifr.org/news/global-robot-demand-in-factories-doubles-over-10-years/1st-quarterly-newsletter-2010). 한국 2024년 설치 수치.
3. Hugging Face, [LeRobot 문서](https://huggingface.co/docs/lerobot/main/index) 및 [SO-101 가이드](https://huggingface.co/docs/lerobot/so101). 기본 플랫폼, 장비 정보.
4. 프로젝트 [설계 규칙](AGENTS.md), [현재 아키텍처](docs/architecture.md), [실험 기록](experiments/README.md).
5. 프로젝트 [최종보고서](docs/report/최종보고서/final_report.pdf)와 [집계, 증거 해석 범위](docs/report/최종보고서/SOURCES.md). 팀 역할, 멘토링 및 평가의 근거.
6. 삼성중공업 이재민, [서면 자문의견서](docs/report/25_%28Sim2Real%29삼성중공업_이재민_자문의견서.pdf), 2026-08-03.

과거 ACT와 SmolVLA 실험은 [당시 데이터 수집 기록](docs/guide/SO101_데이터수집_관리.md), [학습, 추론 기록](docs/guide/SO101_학습_추론.md)에 보존되어 있습니다. 현재 실행 방법은 위 설치 절과 운영 가이드를 참고하세요.
