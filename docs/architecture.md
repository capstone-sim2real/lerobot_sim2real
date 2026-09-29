# SO-101 아키텍처

현재 코드와 실행 경로를 설명한다. 각 결정의 근거는 [design.md](design.md)에 있다.
과거 ACT/SmolVLA 실험은 [legacy/](legacy/)와 [report/](report/)에 당시 기록으로 보존한다.

## 실행 진입점

| 명령 | 모듈 | 역할 |
|---|---|---|
| `so101-camera` | `camera/server.py` | `/dev/video*` 단독 소유, MJPEG·snapshot·오버레이 (:8090) |
| `so101-run --task 1\|2` | `runners/run_task.py` | CV+IK FSM으로 Task 1 구역 수집 / Task 2 적층 |
| `so101-collect` | `runners/run_task3.py` | Task 1 루프를 돌며 성공 사이클을 LeRobot 에피소드로 기록 |
| `so101-agent` | `agent/server.py` | LLM 툴 콜링 웹 서버 (:8099) |
| `so101-panel` | `agent/panel.py` | Task 1/2 버튼·수동 조작·보정 도구 웹 패널 (:8109) |

로봇 시리얼 버스를 쓰는 `so101-run`/`so101-collect`/`so101-agent`/`so101-panel`은 한 번에
하나만 실행한다. 캘리브레이션·진단 CLI는 `tools/calibration/`, `tools/hardware/`에 있다.

## Task 1: 구역 수집

```text
camera.server (/dev/video0 단독 소유)
  ├─ HTTP snapshot ───────────────▶ perception.detect_blocks
  ├─ MJPEG over HTTP ─────────────▶ 브라우저 원본 영상
  └─ 최신 프레임 1개 ─▶ 저우선순위 vision worker ─▶ SSE JSON ─▶ Canvas overlay
                                                        (표시 전용)
                                      │
                                      ▼
Task1SelectState
  ├─ HOME에서 fresh frame·frame age 확인
  ├─ 도달 부채꼴 ∩ target zone 바깥만 검출
  ├─ nearest-first 선택, 색별 슬롯 예약
  └─ 원거리 pick 좌표·tilt 보정
                                      ▼
CvIkPickState → VERIFY
  ├─ 중심+90도 회전 재시도 grasp attempt를 미리 IK 계산
  ├─ 도달 가능한 attempt만 실행
  └─ Present_Position+Present_Load로 파지 확인
                                      ▼
Task1TransportState
  └─ pick apex → place apex → 동적 IK slot hover
                                      ▼
Task1PlaceState → HOME/SELECT
  └─ slot drop → release → hover
```

Task 1은 PLACE 횟수로 완료하지 않는다. HOME 자세에서 받은 fresh frame에 외부 블록이
연속 5초 동안 없을 때만 DONE이다. 카메라 오류, stale frame, 반복 frame sequence는
이 5초에 포함하지 않는다. 재시도 한도를 넘은 색은 그 순회에서만 뒤로 미루고 다음
순회에서 다시 시도한다. 러너는 내부 시간 예산을 적용하지 않으므로 180초 평가는
외부 supervisor로 제한한다.

## Task 2: 한 좌표 적층

```text
SELECT → PICK → VERIFY → TRANSPORT → PLACE
(Task1)  (동일)  (동일)   Task2StackPlanner   Task2PlaceState
                          층 = placed_count   층별 해제 자세에서 그리퍼 열기
```

- `Task2SelectState`/`Task2TransportState`는 `fsm/task1.py`의 상태를 상속한다. Task 1
  쪽에는 클래스 속성 두 개(`index_extra_key`, `plan_extra_key`)와
  `_active_detections()` 훅만 있다.
- 층 번호는 `ctx.placed_count`에서만 나온다. 색깔별로 기억하면, 떨어진 블록을 다시
  집었을 때 이미 높아진 타워에 옛 층 번호를 배정하게 된다.
- 타워는 zone 안에 선다. 검출기가 zone 안 검출을 버리므로 쌓인 블록이 SELECT에
  보이지 않고, Task 1의 "바깥이 5초간 비면 완료" 판정이 그대로 쓰인다.
- 두 실행 경로 모두 접촉 탐색을 쓰지 않는다. `so101-run --task 2`는 운반 후 층별
  해제 자세에서 그리퍼를 연다. 웹의 `stack_block_to_floor(color, floor)`는 요청한
  0..4층의 명목 높이보다 `task2.drop_clearance_mm` 위에서 놓고, home에서 다시 관찰한다.
  위치 재관찰만으로 5초 안정성이나 실제 타워 높이가 확인되지는 않는다.
- `so101-run --task 2 --dry-run`이 층별 도달 가능 여부를 실제 IK로 출력한다. 탑다운
  리프트는 리치에 강하게 반비례하므로 **층수는 설계 입력이 아니라 이 명령의
  출력**이다. 실장비를 돌리기 전에 확인한다.

## 모듈 구성

| 모듈 | 역할 |
|---|---|
| `config/`, `configs/` | 설정 dataclass와 기본값 YAML, 캘리브레이션 JSON, 에이전트 프롬프트 |
| `camera/` | 카메라 단독 소유 서버, MJPEG/snapshot, 표시용 overlay, 학습용 프레임 소스 |
| `perception/` | homography, zone 제외, 색·형상 블록 검출, 대상 선택, 에이전트용 장면 |
| `control/ik.py`, `control/grasp.py` | 탑다운 IK, 위치·yaw 보정, 파지 attempt 계획과 실행 |
| `control/trajectory.py`, `control/sensing.py` | 관절 보간 재생, 파지 확인·접촉 감지 |
| `control/task1_transport.py`, `control/task2_stack.py` | 슬롯·운반 경로, 타워 층 계획 |
| `fsm/` | 상태 머신, 공통 상태, Task 1/2/3 상태와 flow 조립 |
| `runners/` | `so101-run`, `so101-collect` CLI |
| `data/` | `RecordingRobotIO`와 LeRobot 에피소드 기록 |
| `session/` | `ArmSession`(연결·IK·플래너 1회 생성, STOP, 버스 락), 러너 공용 팩토리 |
| `session/skills/`, `session/primitives/` | 에이전트가 호출하는 스킬과 primitive (mixin 단위로 분리) |
| `session/grid.py` | 체스판 칸 좌표 ↔ 로봇 베이스 mm (표시·주소 지정 전용, [design.md §3](design.md)) |
| `session/place_correction.py` | 놓은 뒤 카메라로 잰 차이로 다음 배치 명령을 보정 ([design.md §11](design.md)) |
| `agent/` | 툴 스키마, LLM 어댑터, 대화 루프, 제어 게이트, FastAPI + 웹 UI, 조작 패널 |
| `tools/` | `calibration/`(점 기록·적합), `hardware/`(모터·FK·그리퍼 진단), `yoloe/`(실험) |

`so101-camera`와 에이전트 화면은 같은 MJPEG 위에 같은 부채꼴을 그린다. 호 좌표는
`perception/detector.py`의 `workspace_sector_points_mm`이 만들고 검출기 게이트와 같은
반경 프로파일을 쓰므로, 화면의 경계가 곧 검출 경계다. 에이전트 화면은 그 위에 적재 구역
칸·이름 붙은 자리·체스판 칸 격자를 SVG로 얹는다.

## 실행 규칙

1. 카메라 서버만 `/dev/video0`을 열고, 러너와 도구는 HTTP snapshot을 쓴다.
2. 기본 카메라는 shoulder 1대다. wrist는 `so101-camera --wrist-device ...`로 명시한
   경우에만 제공된다.
3. 파지 성공은 Present_Position과 Present_Load를 함께 확인해야 인정한다.
4. IK gate를 넘는 grasp/slot waypoint는 실행하지 않는다.
5. 모든 조정값은 `src/configs/default.yaml`에 두고 `--set`으로 덮는다.
6. 관제 overlay는 표시 전용이며 SELECT나 IK의 입력으로 되돌아가지 않는다.
7. 에이전트 서버는 `agent.lock_path` 락을 잡고 시작한다.

## 웹 에이전트 구조

```text
so101-camera :8090 ── /dev/video* 단독 소유
so101-agent  :8099 ── 시리얼 버스 단독 소유 (버스 락)
  웹 이벤트 루프   : HTTP/SSE만 처리, 로봇 미접촉. /api/stop 은 즉시 반환
  so101-robot 스레드: ArmSession 생성·사용·종료, 모든 스킬 실행 (단일 스레드)
  요청 스레드       : LLM 대화 루프 / 조그 / home 복귀, robot 스레드에 작업을 넘기고 대기
브라우저 ── 채팅(SSE) + Web Speech STT + <img src=":8090/video/shoulder.mjpg">
```

`agent/server.py`는 프로세스 수명주기, 정적 웹 자산과 SSE 연결을 조립하고,
`agent/manual_api.py`는 인증된 수동 명령의 HTTP 검증을 맡는다. 연속 키보드 입력은
`agent/keyboard_jog.py`(deadman 상태), `agent/jog_planner.py`(IK 경로),
`agent/jog_executor.py`(고정 주기 재생)가 나눠 맡으며 모두 같은 RobotWorker에서 돈다.
`agent/panel.py`는 같은 서버에 보정 도구와 수동 조작 도구 집합을 얹은 진입점이다.

데이터 수집은 에이전트 도구(에피소드 시작/저장/폐기/현황/마무리)의 조합이다.
`session/collection.py`가 기존 `RecordingRobotIO`와 `EpisodeRecorder`를 연결하고,
같은 RobotWorker가 LLM 대기 중 hold tick을 기록한다. 긴 호출로 기록 시간축이 깨지면
시연을 폐기하고, 사용자 턴이 끝나면 미해결 버퍼도 폐기한다. 자세한 내용은
[guide/web-agent.md](guide/web-agent.md).

## 캘리브레이션

현재 적용본은 `src/configs/calib/venue_lab.json`(2026-09-13 9점 fit)이다. 수치는 항상
파일의 `meta`를 본다. 목표 gate(RMS 5mm, LOO 8mm)를 완전히 충족하지 못했고 물리
그리퍼 기준점과 URDF `gripper_frame_link`의 대응도 검증되지 않았으므로, 재시도와
실기 확인을 전제로 쓴다([design.md §3](design.md)).

## 검증

```bash
uv run --extra dev pytest tests -q
```

테스트는 하드웨어·lerobot·placo 없이 돈다. 하드웨어 동작 성공을 뜻하지 않으며,
실장비 결과는 [실험 기록](../experiments/README.md)에 둔다. 실행 중 IPC와 관절 로그는
Git에서 제외한 `var/so101/`, FSM transition과 summary는 `logs/pick_stack/`에 저장한다.
