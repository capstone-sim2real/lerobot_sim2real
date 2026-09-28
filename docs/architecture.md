# SO-101 현재 아키텍처

이 문서는 **현재 코드와 운영 경로**를 설명한다. 과거 ACT/SmolVLA 실험은
`guide/SO101_데이터수집_관리.md`, `guide/SO101_학습_추론.md`와 `report/`에
당시 기록으로 보존한다. 미래 Task 2는 PICK/VERIFY/TRANSPORT를 공유하고
PLACE만 접촉 기반 적층 전략으로 교체한다는 설계 목표를 유지한다.

## 현재 기본 실행 경로

현재 검증·운영 중인 production 경로는 `so101-run --task 1`의 CV+IK zone
gathering이다. ACT PICK은 `--pick-mode act`로 선택 가능한 보존 경로지만 현재
기본값이 아니다.

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
  ├─ 중심+재시도 grasp attempt를 미리 IK 계산
  ├─ 도달 가능한 attempt만 실행
  └─ Present_Position+Present_Load로 파지 확인
                                      ▼
Task1TransportState
  └─ pick apex → place apex → 동적 IK slot hover
                                      ▼
Task1PlaceState → HOME/SELECT
  └─ slot drop → release → hover
```

Task 1은 PLACE 횟수나 `placed_count==5`로 완료하지 않는다. HOME 자세에서 받은
fresh frame에 외부 블록이 연속 5초 동안 없을 때만 DONE이다. 카메라 오류,
stale frame과 반복 frame sequence는 이 5초에 포함하지 않는다. 재시도 한도를
넘은 색은 그 순회에서만 뒤로 미루고 다음 순회에서 다시 시도한다.

현재 Task 1 runner는 내부 FSM 시간 예산을 적용하지 않는다. 180초 평가 제한이
필요한 실행에서는 내부 deadline이 구현될 때까지 외부 supervisor로 제한해야 한다.

## Task 2: 한 좌표 적층

Task 2는 독립 파이프라인이 아니다. SELECT/PICK/VERIFY/TRANSPORT는 Task 1의
상태를 그대로 상속하고, 바뀌는 것은 목적지가 색깔별 슬롯이 아니라 좌표 하나라는
점과 PLACE가 접촉으로 멈춘다는 점뿐이다.

```text
SELECT → PICK → VERIFY → TRANSPORT → PLACE
(Task1)  (동일)  (동일)   Task2StackPlanner   Task2PlaceState
                          층 = placed_count   DESCEND → RELEASE → RETREAT
```

- `Task2SelectState`/`Task2TransportState`는 `fsm/task1.py`의 상태를 상속한다.
  Task 1 쪽에는 클래스 속성 두 개(`index_extra_key`, `plan_extra_key`)와
  `_active_detections()` 훅만 추가했고 동작은 그대로다.
- 층 번호는 **`ctx.placed_count`에서만** 나온다. Task 1처럼 색깔별로 기억하면,
  실패해 zone 밖에 떨어진 블록을 다시 집었을 때 이미 높아진 타워에 옛 층
  번호를 배정하게 된다.
- 타워는 zone 안에 선다. 검출기가 zone 안 검출을 버리므로 쌓인 블록이 SELECT에
  보이지 않고, 덕분에 Task 1의 "바깥이 5초간 비면 완료" 판정이 수정 없이 산다.
  `Task2SelectState`의 반경 가드는 그 위의 2차 안전망이다.
- **현재 Task 2의 두 실행 경로는 모두 접촉 탐색을 쓰지 않는다.**
  `so101-run --task 2` FSM은 운반 후 hover 자세에서 그리퍼를 연다.
  웹 에이전트의 `stack_block_to_floor(color, floor)`는 요청한 0..4층에
  대해 hover에서 층별 높이 근처로
  내려간 뒤 놓는다. 따라서 같은 적층 목표를 쓰지만 릴리스 높이는 다르다.

- **현재 primitive 적층은 접촉을 판정하지 않는다.** `stack_block_to_floor`는
  층별 명목 높이보다 `task2.drop_clearance_mm` 위의 자세로 이동해 그리퍼를
  열고, home에서 카메라로 블록 위치를 다시 본다. 이 여유는 실기 조정 전
  가정값이다. 부하 스파이크 오탐 때문에 블록을 든 채 멈추는 문제를 피한다.
  위치 재관찰만으로 5초 안정성이나 실제 타워 높이가 확인되는 것은 아니다.
  무너진 블록이 구역 안에 있어도 다시 집으며, 받침 검출/이전 층 기록은
  지정된 floor 동작의 사전 거부 조건이 아니다.

`--dry-run --task 2`가 층별 도달 가능 여부를 실제 IK로 뽑는다. 탑다운 리프트는
리치에 강하게 반비례하므로 **층수는 설계 입력이 아니라 이 명령의 출력**이다.
실장비를 돌리기 전에 반드시 읽는다.

레거시 `build_task2_states`/`StackPlaceStrategy`/`MotionController.stack_place`는
녹화 포즈 레지스트리 세대이며 그대로 남아 있다. `poses.yaml`에 `home`만 기록되어
있어 실행되지 않고, CLI에서도 더는 도달하지 않는다. CV+IK Task 2는 `home` 외의
포즈를 요구하지 않는다.

최종 5초 안정성 판정은 아직 구현하지 않았다.

## 모듈 구성

| 모듈 | 역할 |
|---|---|
| `camera/` | shoulder 카메라 단독 소유, MJPEG/snapshot, 표시용 overlay와 선택적 wrist stream |
| `perception/` | homography, zone 제외, 색·형상 블록 검출, 대상 선택 |
| `control/ik.py`, `control/grasp.py` | 탑다운 IK, 위치·yaw 보정, 재시도 attempt 계획 |
| `control/task1_transport.py` | zone 3+2 슬롯과 pick/place apex의 동적 IK 계획 |
| `control/sensing.py` | 파지 확인과 Task 2 contact 감지 공통 센싱 |
| `fsm/task1.py` | Task 1 전용 fresh-frame SELECT, 동적 TRANSPORT/PLACE |
| `fsm/handlers.py`, `fsm/flows.py` | 공통 상태와 미래 Task 2 PLACE 전략 seam |
| `policy/` | optional ACT PICK 클라이언트와 gRPC transport 보존 경로 |
| `runners/run_task.py` | Task/flow/pick-mode 조립과 실행 CLI |
| `tools/` | 캘리브레이션, 모터 진단, 수동 조작, 세션 CLI |
| `session/` | `ArmSession`(연결·IK·플래너 1회 생성, STOP 취소, 버스 락), 러너 공용 팩토리, 에이전트 스킬 |
| `perception/scene.py` | 에이전트 전용 zone 안/밖 장면과 칸 점유 (검출기 기본 동작은 불변) |
| `session/grid.py` | 체스판 칸 좌표 ↔ 로봇 베이스 mm (표시·주소 지정 전용, AGENTS.md §6) |
| `session/place_correction.py` | 놓은 뒤 카메라로 잰 차이로 다음 배치 명령을 보정 (AGENTS.md §16.4) |
| `agent/` | `so101-agent`: 툴 스키마, Claude/GPT/Gemini 어댑터, 대화 루프, 제어 게이트, FastAPI + 웹 UI |

`so101-camera`와 `so101-agent` 화면은 같은 MJPEG 위에 같은 부채꼴을 그린다. 호 좌표는
양쪽 모두 `perception/detector.py`의 `workspace_sector_points_mm`이 만들고, 검출기 게이트와
같은 반경 프로파일을 쓰므로 화면에 보이는 경계가 곧 검출 경계다. 에이전트 화면은 그 위에
적재 구역 칸·이름 붙은 15개 자리·체스판 칸 격자를 SVG로 얹고, 누르면 입력창에 이름이나
좌표가 들어간다.

## 실행 규칙

1. 카메라 서버만 `/dev/video0`을 열고 runner와 도구는 HTTP snapshot을 사용한다.
2. 기본 카메라는 shoulder 1대다. wrist는 `so101-camera --wrist-device ...`로
   명시한 경우에만 제공된다.
3. 파지 성공은 Present_Position과 Present_Load를 함께 확인해야 인정한다.
4. IK gate를 넘는 grasp/slot waypoint는 실행하지 않는다.
5. 모든 조정값은 `config.py`와 `configs/default.yaml`에 두고 `--set`으로 덮는다.
6. 관제 overlay는 표시 전용이며 SELECT나 IK의 입력으로 되돌아가지 않는다.
7. 로봇 시리얼 버스 소유자는 `so101-run`/`so101-collect`/`so101-agent` 중 하나뿐이다.
   `so101-agent`는 `agent.lock_path` 락을 잡고 시작한다.

### LLM 에이전트 토폴로지

```text
so101-camera :8090 ── /dev/video* 단독 소유 (변경 없음)
so101-agent  :8099 ── 시리얼 버스 단독 소유 (버스 락)
  웹 이벤트 루프   : HTTP/SSE만 처리, 로봇 미접촉. /api/stop 은 즉시 반환
  so101-robot 스레드: ArmSession 생성·사용·종료, 모든 스킬 실행 (단일 스레드)
  요청 스레드       : LLM 대화 루프 / 조그 / home 복귀, robot 스레드에 작업을 넘기고 대기
브라우저 ── 채팅(SSE) + Web Speech STT + <img src=":8090/video/shoulder.mjpg">
```

에이전트 서버의 책임은 다음 모듈로 나뉜다. `agent/server.py`는 프로세스 수명주기,
정적 웹 자산과 SSE 연결만 조립하고, `agent/manual_api.py`는 인증된 수동 명령의 HTTP
검증을 담당한다. 연속 키보드 입력의 deadman 상태는 `agent/keyboard_jog.py`, IK 경로
생성은 `agent/jog_planner.py`, 고정 주기 재생과 다음 경로 선계산은
`agent/jog_executor.py`가 맡는다. 세 모듈 모두 같은 RobotWorker에서 실행되므로 버스
단독 소유 계약은 바뀌지 않는다.

에이전트 도구는 관찰·동작 primitive 및 에피소드 시작/저장/폐기/현황/마무리로
통일한다. 데이터 수집은 이 도구들의 조합이며 Task 3 FSM을 도구로 실행하지 않는다.
`session/collection.py`가 기존 `RecordingRobotIO`와 `EpisodeRecorder`를 연결하고,
동일 RobotWorker가 LLM 대기 중 hold tick을 기록한다. 긴 호출로 기록 시간축이
깨지면 시연을 폐기한다. 한 사용자 턴이 끝나면 미해결 버퍼도 폐기해 IDLE에서
녹화 명령이 계속 나가지 않게 한다. 저장 조건과 한계는
[primitive 가이드](guide/SO101_AGENT_PRIMITIVES.md)를 따른다.

## 현재 캘리브레이션

`src/configs/calib/venue_lab.json`은 2026-09-11에 다시 잡은 9점 fit이다.

- RMS: 10.70 mm
- maximum leave-one-out error: 38.65 mm
- grasp-z mean/std: 1.48 / 1.04 mm
- zone: 빨간 테이프 내부 contour, `zone_polygon_mm` 등록 완료

### Task 2 사다리 (2026-09-11, 이 캘리브레이션 기준 dry-run 측정값)

기본 설정에서 `--dry-run --task 2`는 **5층 중 4층 도달**을 보고한다. 5층은
hover가 6.7mm 모자란다. 기울임 램프가 없으면 2층까지밖에 못 가므로, 3·4층은
램프가 벌어준 것이다.

같은 날 dry-run으로 훑은 5층 도달 조합 (전부 미실측 — 실장비 검증 전이다):

| 조합 | 층 | 최악 tilt err | 타워 통과 여유 | zone 여유 |
|---|---|---|---|---|
| 기본값 | 4 | 4.57° | 15.0mm | 14.0mm |
| `hover_min_clearance_mm=10` | 5 | 4.52° | 10.0mm | 14.0mm |
| `stack_uv=[0.5,0.90]` + `hover_min_clearance_mm=12` | 5 | 4.53° | 12.0mm | 10.0mm |
| `level_tilt_per_level_deg=2.0` + `level_tilt_max_deg=6.0` | 5 | **6.02°** | 15.0mm | 14.0mm |

마지막 조합은 5층의 achieved tilt가 공유 게이트 `ik.max_tilt_error_deg=6.0`을
넘어선다. Task 1이 `pick_tilt_max_deg=5.0`으로 1도 여유를 두는 관례와 어긋나므로
기본값으로 채택하지 않았다. 5층이 필요하면 위 표를 근거로 실장비에서 고른다.

정상 gate(RMS 5mm, LOO 8mm)는 충족하지 못했다. 물리 그리퍼 기준점과 URDF
`gripper_frame_link`의 대응도 아직 검증되지 않았으므로 재시도와 실기 검증을
전제로 사용한다. 이전 15점 fit과 원인 분석은 보고서의 역사 기록을 참조한다.

## 검증

```bash
uv run --extra hardware --extra dev pytest -q
```

현재 필수 회귀 테스트는 120개가 통과하고 2개가 선택 의존성 없이 건너뛴다. 하드웨어 동작 성공을
뜻하지 않으며, 당시 실장비 결과는 [실험 기록](../experiments/README.md)에 둔다.
실행 중 IPC와 관절 로그는 Git에서 제외한 `var/so101/`, FSM transition과 summary는
기본적으로 `logs/pick_stack/`에 저장한다.
