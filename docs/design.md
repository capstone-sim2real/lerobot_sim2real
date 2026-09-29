# 설계 근거

코드가 왜 지금 모양인지를 정리한 문서다. 실행 흐름과 모듈 구성은
[architecture.md](architecture.md)를 본다. 과거 ACT/SmolVLA 실험 기록은
[legacy/](legacy/)와 [report/](report/)에 당시 상태 그대로 보존한다.

## 1. 미션 규격

| | 1차 미션 | 2차 미션 |
|---|---|---|
| 내용 | 20cm×10cm 지정영역에 블록 배치 | 블록 적재 후 5초 유지 |
| 제한시간 | 180초 | 300초 |
| 블록 | 5개, 영역 밖 무작위, 서로 맞닿을 수 있음 | 동일 |
| 인정 | 경계 2cm 걸침·세로 세움·일부 겹침 OK, 적재 불인정 | 5초 이상 유지 |
| 블록당 예산 | 36초 | 60초 |

지정영역은 로봇 베이스 기준 약 25cm 거리, 블록 높이는 20mm다. 과제 규정상
신경망은 하나까지 쓸 수 있으며, 임계값·기하 계산·IK 같은 비학습 알고리즘은 모델로
치지 않는다. 따라서 CV+IK 단독 경로는 규정에 부합한다.

## 2. 태스크 흐름 (FSM)

```
SELECT → PICK → VERIFY → TRANSPORT → PLACE → (블록 남음) → SELECT
           ↑______실패(재시도 / 스킵)______|
```

- **VERIFY는 파지가 확인되지 않으면 절대 TRANSPORT로 진행하지 않는다.**
- 재시도는 타겟별로 세고(`RunContext.record_attempt`), `max_retries_per_block`을
  넘으면 그 블록을 건너뛴다. 한 블록이 시간 전체를 먹지 않게 하기 위해서다.
- Task 1의 완료는 배치 횟수가 아니라 "HOME에서 받은 새 프레임에서 구역 밖 블록이
  `task1.empty_timeout_s` 동안 보이지 않음"으로 판정한다.
- Task 2는 Task 1의 SELECT/PICK/VERIFY/TRANSPORT를 상속해 목적지만 탑 한 곳으로
  바꾸고, PLACE는 TRANSPORT가 멈춘 층별 해제 자세에서 그리퍼만 연다.
- 상태 핸들러는 `BaseRobotIO`와 설정을 주입받을 뿐 내부에서 객체를 만들지 않는다.
  그래서 `MockRobotIO`로 하드웨어 없이 테스트할 수 있다.
- 하드웨어 정리(홈 복귀, 연결 해제)는 러너의 `finally`에 둔다. 예외가 나도 실행된다.
- Task 1 러너는 내부 시간 예산을 끈다. 180초 제한은 외부 supervisor가 강제한다.
- 일부 PICK/TRANSPORT step은 bounded `move_to()`나 하강 전체를 한 번에 수행하므로,
  머신은 그 호출 중간에는 시간 예산을 검사하지 못한다.

## 3. 좌표계와 캘리브레이션

**모든 미터법 좌표는 로봇 베이스 프레임(mm)이다.** 체스판 원점 프레임을 쓰면
board-mm → 로봇 좌표 변환이라는 미지수가 하나 더 생긴다.

```
H : 픽셀 (u,v)  →  로봇 베이스 프레임 (x_mm, y_mm)
```

- **캘리브레이션 평면은 블록 윗면 높이다.** 빈 테이블 평면으로 잡으면 검출기가 보는
  블록 윗면(테이블+20mm)과 어긋나 시차 오차가 생긴다(카메라 60cm·측방 25cm에서
  약 8mm). 각 점에 블록을 놓고 윗면 중앙에 그리퍼 기준점을 대고 FK를 기록한다.
- **캘리브레이션 중 `wrist_roll`은 중립(0 근처)으로 유지한다.** `gripper_frame_link`가
  roll 축에서 약 8mm 벗어나 있어, 손목 각도에 따라 같은 자리의 기록 좌표가 최대
  13.7mm 달라진다. 런타임 IK도 중립 손목으로 잡으므로 두 자세가 일치해야 한다.
- **관절 각도도 함께 기록한다.** 좌표만 남기면 위 오프셋을 나중에 보정할 수 없다.
- **카메라 마운트가 바뀌면 캘리브레이션은 무효다.**

현재 적용본은 2026-09-13의 9점 `fk_direct_pairs` fit이다(RMS 5.15mm, 최악 LOO
13.53mm, grasp-z 평균 4.06mm·표준편차 0.73mm). 목표 gate(RMS 5mm, LOO 8mm)를 완전히
통과하지는 못했다. 순수 체스판 코너만으로 맞춘 homography의 잔차가 0.4mm이므로
카메라와 계산식이 아니라 그리퍼 기준점 대응이 오차의 주원인이다. 재캘리브레이션을
반복하는 대신 **이 오차 규모를 전제로 설계한다**: 그리퍼를 넉넉히 열고, 실패는
90도 회전 재시도와 FSM 재시도로 흡수한다. 정확한 값은 항상
`src/configs/calib/venue_lab.json`의 `meta`를 본다. 이전 fit은
`venue_lab.pre-recalibration-20260911.json`에 남아 있다.

`zone_polygon_mm`은 2026-09-19 빨간 테이프 내부 윤곽 edge-fit으로 등록했다.

**체스판 격자(`board_grid`)는 표시·주소 지정 전용이다.** 정수 칸 좌표 (x, y)는
주소일 뿐이고, 모션 계획 전에 한 번 로봇 mm로 해석된다. 축은 학생이 보는 화면
기준(x+ = 화면 오른쪽, y+ = 로봇에서 멀어지는 쪽)이다. 격자는 H 위에 얹혀 있으므로
카메라가 움직이면 함께 무효다. 적합 RMS가 한 칸의 10%를 넘으면 도구가 저장을
거부한다.

칸의 안쪽 경계는 반경 하나로 표현되지 않는다. 전 칸에 place IK를 돌려 보니
실패는 베이스 정면의 좁은 통로(|y| ≤ 25mm, x ≤ 72mm)뿐이었고, 50mm만 옆으로
비켜도 반경 55mm부터 통과했다(그리퍼의 27mm 측방 오프셋 때문). 그래서
`agent.board_grid`는 반경 대신 그 통로를 제외한다.

## 4. 역기구학 (IK)

측정으로 확인한 SO-101 기구학 특성(`third_party/so101/so101.urdf`, `gripper_frame_link`):

- 회전행렬 3번째 열 `z_g`가 접근축이다. 탑다운 파지 = `z_g ≈ (0,0,-1)`.
- `shoulder_pan` 부호는 atan2와 반대다(양의 pan → 음의 y).
- 그리퍼가 pan 축에서 약 27mm 측방 오프셋되어 있어 방위각 ≠ -pan이다. 시드 pan에
  ±6°, ±12° 후보를 두고 재시도해 흡수한다.
- 탑다운 도달 범위는 반경 0.03 ~ 0.32m다. 그 밖은 `wrist_flex` 한계 때문에 수직
  탑다운이 불가능하지만 팔은 더 멀리 닿으므로 기울인 접근으로 폴백한다.
- placo IK는 시드에 매우 민감하다. 나쁜 시드에서는 200~350mm 오차로 수렴하고,
  FK 그리드로 미리 계산한 탑다운 시드 테이블에서 시작하면 URDF 모델 안에서 0.00mm로
  수렴한다. 실제 팔에는 3절의 캘리브레이션 오차가 더해진다.
- 목표 회전은 시드 자세의 FK 회전을 쓴다. 현재 자세의 회전을 복사하면 측방 이동 시
  5-DOF로 불가능한 자세를 요구하게 된다.
- placo는 메시를 cwd 기준으로 찾으므로 URDF 부모 디렉토리로 cwd를 바꾼 뒤 로드한다.

IK는 소수의 카테시안 웨이포인트에서만 풀고, 사이는 `interpolate()` +
`TrajectoryPlayer`로 관절공간 보간한다. 매 틱 IK를 푸는 폐루프는 느리다.

**그리퍼 yaw는 중립 기준으로 잡는다.** yaw를 베이스 기준 고정값으로 두면 pan이 도는
만큼 `wrist_roll`이 반대로 비틀려 작업영역을 가로지르며 약 80도를 휘두르고
한계각(-125도)까지 간다. 2026-08-31 wrist_roll 서보 과열 정지가 이것 때문이었다.
`TopDownIK.solve(yaw_deg=None)`은 그 위치에서 `wrist_roll`이 0 근처가 되는 yaw를
자동으로 잡고(작업영역 전체 ±3.7도), 블록 각도를 맞춰야 할 때는
`grasp_yaw_deg()`가 정사각형의 90도 대칭으로 중립에 가장 가까운 정렬을 고른다.

PICK은 최초 새 프레임에서 만든 중심·재시도 후보를 실행하고, VERIFY 실패 뒤
HOME→SELECT에서 새 프레임으로 다시 검출한다. 파지 직전 근접 재검출은 없다.

## 5. 배치와 적층

- Task 1은 `task1.release_clearance_mm` 높이에서 놓는다. 웹 primitive의 슬롯/칸
  배치도 접촉 탐색 없이 같은 높이로 내려가 해제한다.
- Task 2는 블록 높이 20mm와 층 번호로 목표 높이를 계산하고, 접촉 감지 없이 그 높이
  위에서 그리퍼를 연다. 웹 primitive `stack_block_to_floor(color, floor)`는 층을
  명시적으로 받고 `task2.drop_clearance_mm` 위에서 놓는다. 이 높이는 실기 조정 전
  가정값이며, 이 동작만으로 적층 높이와 5초 안정성을 검증했다고 보지 않는다.

## 6. 카메라와 검출

- 탑다운 카메라 1대만 제어에 쓴다. `camera.server` 한 프로세스만 `/dev/video*`를 연다.
- **카메라는 강체로 고정되어야 한다.** 12px 드리프트는 작업영역에서 10~20mm
  위치 오차이며 블록 폭(40mm) 대비 파지 실패를 직접 유발한다.
  `tools/calibration/camera_drift_check.py`로 10분간 p95 드리프트 < 2px을 확인한다.
  코너 하나가 튀는 것은 검출 노이즈이므로 `max`로 판정하지 않는다.
- **색만으로 판단하지 않는다.** 빨간 블록과 빨간 경계 테이프는 색이 같지만 형상이
  다르다. 면적/종횡비/solidity/fill 필터가 그 차이를 인코딩한다.
- 검출은 homography로 정사영한 미터법 뷰에서 하므로 모든 임계값이 mm 단위다.
- green/blue와 색 prototype은 실장비 프레임에서 조정했지만 red/yellow/wood의 HSV
  gate 일부는 합성 기준이다. 조명이 바뀌면 `tools/hardware/view_detect.py`로 확인한다.
- 맞닿은 동색 블록은 컨투어가 병합될 수 있다. FSM이 매 사이클 재검출하므로 하나를
  치우면 자연히 분리된다.

## 7. 파지 검증

파지는 `Present_Position`(완전히 닫혔는가)과 `Present_Load`(물고 있는가)를 함께
본다. 빈 손으로 닫히면 위치가 `gripper_empty_closed_max` 아래로 내려간다. 임계값은
장비마다 다르므로 `tools/hardware/tune_gripper_load.py`로 분포를 찍어 정한다.

## 8. 궤적과 안전

- 안전장치는 겹쳐 건다. `interpolate()`가 틱당 관절 변화량을 `max_step_per_tick`으로
  제한하고, lerobot `send_action`의 `max_relative_target` 클램프도 항상 켜 둔다.
- **궤적은 항상 측정된 현재 자세에서 시작한다.** 명령된 자세에서 시작하면 팔이
  목표에서 약간 벗어나 멈췄을 때 다음 동작이 튄다.
- `disable_torque_on_disconnect`는 false다. 종료 후에도 팔이 자세를 유지해야 하고,
  토크 해제는 명시적인 수동 조작이어야 한다.
- 하드웨어를 움직이는 명령은 `--dry-run`을 먼저 지원한다.

## 9. 설정과 테스트

- 모든 튜너블은 `src/config/` dataclass와 `src/configs/*.yaml`에 둔다. 알 수 없는
  YAML 키는 로드 시점에 거부한다. 런타임 오버라이드는 `--set fsm.time_budget_s=180`.
- `import config`는 lerobot·placo·torch 없이 성공해야 한다. 하드웨어 의존 모듈은
  함수 안에서 지연 import한다.
- `uv run --extra dev pytest tests -q`는 하드웨어·lerobot·placo 없이 통과해야 한다.
- 실측하지 않은 값은 코드와 문서에 가정값이라고 적는다.

## 10. Task 3 — ACT 데이터셋 자동 수집

운영 방법은 [guide/task3-data-collection.md](guide/task3-data-collection.md).

- Task 3은 Task 1의 수집 루프 그대로다. 다른 점은 파지 시도 1회
  (`task3.max_grasp_attempts`), 구역이 비면 종료 대신 재배치 요청, 그리고 녹화뿐이다.
- **녹화는 `BaseRobotIO` 데코레이터(`RecordingRobotIO`)에서만 한다.** 모든 팔 명령이
  `send_joints()`를 지나므로 제어 코드에 녹화 훅을 넣지 않아도 된다.
- 에피소드 경계는 `Task1SelectState.enter()`의 홈 복귀 뒤다. 양 끝이 같은 홈 자세라
  시연이 닫힌 사이클이 된다. 카메라 폴링 대기는 에피소드 밖이다.
- 파지 성공과 운반 완료(PLACE 종료)가 모두 있어야 저장한다. 실패한 시연을 저장하면
  정책이 그 실패를 학습한다.
- **데이터셋 fps는 실제 틱 주기와 같아야 한다.** LeRobotDataset은 timestamp를
  `frame_index/fps`로 합성하므로 어긋나면 정책이 다른 속도로 재생된다.
  `RecordingRobotIO`가 절대 데드라인으로 페이싱한다(Orin 실측: 목표 30Hz에 30.06Hz,
  p95 36.1ms).
- 이미지는 camera.server의 MJPEG 스트림에서 받아 백그라운드에서 RGB로 변환한다.
  추론도 같은 파이프라인(`MjpegFrameSource`)을 써야 distribution shift가 없다.
- ACT는 모든 `observation.images.*`의 shape이 같아야 하므로 스트림을
  `task3.image_width/height`로 맞추고, 한 데이터셋 안에서 카메라 구성을 바꾸지 않는다.
- Ctrl-C는 틱 경계에서만 처리한다. 두 번째 Ctrl-C가 parquet/비디오 finalize를 깨면
  데이터셋 전체를 잃는다.

## 11. 웹 에이전트

운영 방법은 [guide/web-agent.md](guide/web-agent.md).

- `so101-run`/`so101-collect`는 에이전트 코드를 import하지 않는다. 에이전트는
  `session/`·`agent/` 코드와 동작이 같은 공용 함수(`control/task1_transport.py`,
  `session/factories.py`)만 공유한다.
- 로봇 버스 소유자는 하나다. `ArmSession.open()`은 `agent.lock_path` 락을 잡고,
  로봇은 `RobotWorker` 한 스레드에서만 구동한다(`TopDownIK`가 cwd를 바꾸므로 IK도
  그 스레드에서 한 번만 만든다).
- LLM은 원시 관절 명령과 자유 좌표를 받지 않는다. 칸 좌표는 유일한 절대 주소이며,
  다른 목표와 똑같이 부채꼴 게이트와 IK 게이트를 통과한다.
- **배치 정확도는 캘리브레이션보다 릴리즈 동작이 정한다.** `release_at`은
  `motion.arrival_tol`(4°)에서 그리퍼를 여는데, 이 장비에서 3°는 283mm 리치 기준
  약 29mm다. 블록을 든 상태의 처짐은 더 기다려도 닫히지 않는다. 그래서 보정은
  상수가 아니라 측정으로 닫는다: 모든 배치는 홈 복귀 후 재관찰로 (명령, 실제) 쌍을
  얻고, `session/place_correction.py`가 팔 기준 프레임에서 학습한다.
- STOP은 `session/cancel.py`의 `CancellableRobotIO`에서만 구현한다. `Cancelled`는
  RuntimeError/OSError/ValueError가 아니어야 FSM이 삼키지 않는다.
- 명령 수락 여부는 서버(`agent/control.py`)가 정한다. IDLE에서만 명령을 받고,
  STOP·로봇 고장 뒤에는 실측으로 홈 도착이 확인돼야 IDLE로 돌아간다.
- LLM SDK와 웹 프레임워크는 지연 import하며 optional extra로 설치한다
  (`uv sync`가 JetPack torch 휠을 바꾸지 않도록 `uv pip install`을 쓴다).
- 웹 검출 backend는 기본 CV이고, 운영자가 IDLE에서 YOLOE를 고를 수 있다. 어느 쪽이든
  작업영역·IK·VERIFY 게이트는 우회하지 않는다.
- `agent.relative.*`, `agent.table_regions.*`, `agent.place_clear_radius_mm`는 실측 전
  가정값이다.
