# AGENTS.md — SO-101 Pick & Stack 설계 규칙

이 문서는 `src/` 코드가 주석에서 참조하는 설계 규칙의 원본이다
(`AGENTS.md §3`, `§4`, `§5`, `§9`, `§11` 등). 코드를 수정하는 에이전트는
먼저 이 문서를 읽고, 여기 규칙과 충돌하는 변경을 하지 않는다.

관련 문서: 현재 아키텍처는 [docs/architecture.md](docs/architecture.md).

### 문서 시점 구분

- **현재 운영 경로:** Task 1 CV+IK zone gathering. 고정 탑 카메라와 동적 IK
  슬롯을 사용하며 `so101-run --task 1`로 실행한다.
- **유지할 목표 구조:** Task 2는 PICK/VERIFY/TRANSPORT를 공유하고 PLACE 전략만
  접촉 기반 적층으로 교체한다. 이 seam은 현재 Task 1 구현 형태와 관계없이
  앞으로도 설계 계약으로 유지한다.
- **과거 기록:** `docs/guide/SO101_데이터수집_관리.md`,
  `docs/guide/SO101_학습_추론.md`, `docs/report/*`의 ACT/SmolVLA 실험과 피벗
  근거는 당시 상태를 보존한 자료다. 현재 실행법으로 해석하거나 현행화하지 않는다.

---

## §1 미션 규격

| | 1차 미션 | 2차 미션 |
|---|---|---|
| 내용 | 20cm×10cm 지정영역에 블록 배치 | 블록 적재 후 5초 유지 |
| 제한시간 | **180초** | **300초** |
| 블록 | 5개, 영역 밖 무작위, 서로 맞닿을 수 있음 | 동일 |
| 인정 | 경계 2cm 걸침·세로 세움·일부 겹침 OK, **적재 불인정** | 5초 이상 유지 |
| 블록당 예산 | **36초** | 60초 |

지정영역은 로봇 베이스 기준 약 25cm 거리. 블록 높이 h=20mm.

과제 공지상 "모델"은 신경망을 뜻하며 **신경망은 하나만 사용**한다(상한이지 필수 아님).
임계값·기하 계산·IK 같은 비학습 알고리즘은 모델로 치지 않는다. 따라서 CV+IK
단독 경로는 규정에 부합한다.

## §2 저장소 구조와 환경

```
~/lerobot_sim2real              이 저장소 (프로젝트)
~/lerobot_sim2real/third_party/lerobot  LeRobot submodule — 직접 수정하지 않는다
~/lerobot_sim2real/.venv        공용 실행 환경 (placo 포함)
```

`src/`가 본체다. LeRobot은 `third_party/lerobot` submodule로 커밋을 고정하고,
루트 `uv` 환경에 editable 설치한다. 실행은 `uv run` 또는 설치된 `so101-*` CLI를 사용한다.

**`import config`는 lerobot·placo·torch 없이도 성공해야 한다.** 하드웨어
의존 모듈은 전부 함수 안에서 lazy import 한다. CI/단위 테스트가 이에 의존한다.

## §3 태스크 흐름 (FSM)

```
SELECT → PICK → VERIFY → TRANSPORT → PLACE → (블록 남음 && 시간 남음) → SELECT
           ↑______실패(재시도 / 스킵)______|
```

- **목표 계약:** 1차와 2차 미션은 PLACE 전략만 교체하고 나머지 경로를 공유한다.
  현재 production Task 1은 fresh-frame 완료 판정과 동적 IK 슬롯 때문에
  `Task1SelectState`/`Task1TransportState`/`Task1PlaceState`를 먼저 사용하고 있다.
  Task 2를 구현할 때 이 차이를 확대하지 말고 PLACE seam으로 수렴시킨다.
- 상태 핸들러는 `BaseRobotIO` + config + 인터페이스만 **주입받는다.** 핸들러 안에서
  객체를 생성하지 않는다. `MockRobotIO`로 하드웨어 없이 단위 테스트가 가능해야 한다.
- **목표 계약:** `step()`은 제한된 동작 단위만 수행하고 반환해야 한다. 현재 일부
  PICK/TRANSPORT/PLACE step은 bounded `move_to()` 또는 descent 전체를 수행하므로
  머신은 그 호출 중간에 예산을 검사하지 못한다. 새 코드는 이 블로킹 범위를
  더 키우지 않는다.
- **HARD RULE: VERIFY는 파지가 확인되지 않으면 절대 TRANSPORT로 진행하지 않는다.**
- 재시도는 타겟별로 센다(`RunContext.record_attempt`). 한 블록이 예산 전체를
  먹지 못하게 `max_retries_per_block` 초과 시 스킵한다.
- DONE은 종단 상태다. 하드웨어 정리(홈 복귀, 연결 해제)는 러너의 `finally`에
  둔다 — 예외 발생 시에도 실행되어야 하기 때문이다.
- 현재 Task 1 zone-gather runner는 내부 FSM 시간 예산을 끈다. 평가의 180초 제한은
  외부 supervisor가 강제해야 하며, 내부 deadline을 구현하기 전까지 이를 숨기지 않는다.
- **Task 3은 Task 1의 수집 루프 그 자체다.** SELECT/PICK/VERIFY/TRANSPORT 핸들러를
  그대로 주입하고 `Task1TransportPlanner`의 슬롯 5개를 쓴다. 다른 것은 세 가지뿐이다:
  파지 시도 1회(`task3.max_grasp_attempts`), 구역이 비면 DONE 대신 재배치 프롬프트,
  그리고 녹화. 이 차이를 더 늘리지 않는다 — §15를 본다.

## §4 PLACE 전략 seam

PLACE의 목표 경계는 `PlaceStrategy` 인터페이스다. 현재 공통 handler에는
`SlotPlaceStrategy`와 `StackPlaceStrategy`가 있고, production Task 1은 동적 IK
슬롯용 `Task1PlaceState`를 사용한다. Task 2는 PICK/VERIFY/TRANSPORT를 복제하지
말고 이 PLACE seam에 접촉 기반 적층을 주입한다. 나중에 정책 기반 정렬로
교체하더라도 같은 경계를 사용한다.

**PICK도 같은 방식으로 교체 가능하다.** `fsm/act_handler.py`의 `ActPickState`는 `PickClient`
Protocol(`ping()`, `run_pick(retreat_pose)`)에만 의존한다. CV+IK 경로는
`fsm/ik_handler.py`의 `CvIkPickState`로 PICK만 교체하고 나머지 상태는
`handlers.py`에서 import해 재사용한다.

## §5 적층: 접촉 기반 하강

**타워 높이를 블록 수로 추측하지 않는다.** 현재 `StackPlaceStrategy`는 기록된
`tower_descent_<n>` 관절 키프레임 사다리를 천천히 따라가며 `ContactMonitor`의
부하 스파이크에서 정지하고 백오프 후 해제한다. 접촉 없이 사다리 바닥에 닿으면
경고 후 해제한다. Task 2를 완성할 때도 이 contact-first 원칙을 유지하며,
`z_release(n)` 같은 층수 기반 dead reckoning을 주 제어로 추가하지 않는다.

## §6 좌표계와 캘리브레이션

**모든 미터법 좌표는 로봇 베이스 프레임(mm)이다.** 임의의 체스판 원점 프레임을
쓰지 않는다 — board-mm → 로봇-m 변환이라는 미지수가 생기기 때문이다.

```
H : 픽셀 (u,v)  →  로봇 베이스 프레임 (x_mm, y_mm)
```

- `PlaneCalibration.base_xy_mm = (0.0, 0.0)` (베이스가 곧 원점). `select.py`의
  nearest-first가 이 값을 그대로 쓴다.
- `zone_polygon_mm`도 로봇 베이스 mm.
- **캘리브레이션 평면은 블록 윗면 높이다.** 빈 테이블 평면에서 잡으면 검출기가
  보는 블록 윗면(테이블+20mm)과 평면이 어긋나 시차 오차가 생긴다(카메라 60cm·측방
  25cm에서 약 8mm — 파지를 깨뜨리기 충분). 각 캘리브레이션 점에 블록을 놓고
  **윗면 중앙**에 그리퍼 기준점을 대고 FK를 기록한다.
- **캘리브레이션 중 `wrist_roll`은 항상 중립(0 근처)으로 유지한다.**
  `gripper_frame_link`는 wrist_roll 회전축에서 약 8mm 벗어나 있어서, 같은
  자리에 턱을 물려도 손목 각도에 따라 기록되는 좌표가 최대 13.7mm 달라진다.
  손목 각도가 점마다 다르면 그 오프셋이 점마다 다르게 섞여 적합이 망가진다.
  런타임 IK도 중립 손목으로 잡으므로(§7) 캘리브레이션 자세와 런타임 자세가
  일치해야 한다. `so101-fk`가 `wrist_roll`을 실시간으로
  표시하니 기록 전에 확인한다.
- **관절 각도도 함께 기록한다.** 좌표만 남기면 위 오프셋을 사후에 역산·보정할
  수 없다 (2026-08-31에 실제로 시도했다가 실패했다). `points.csv`는 5축
  관절값 컬럼을 포함한다.
- **카메라 마운트가 바뀌면 캘리브레이션은 무효다.** 반드시 재실행한다.

**현재 적용 결과 (2026-09-13 재캘리브레이션, `venue_lab.json`의 `meta`) — 9점
`fk_direct_pairs` fit이다.** 수치를 옮겨 적기 전에 항상 그 파일의 `meta`를 본다.

- 순수 체스판 코너(카메라 광학만, FK 없음)로 맞춘 homography는 잔차 RMS
  0.4mm — 카메라·계산식은 문제없다는 뜻.
- 활성 fit은 **RMS 5.15mm, 최악 LOO 13.53mm, grasp-z 평균 4.06mm·표준편차
  0.73mm**다. 정상 gate(RMS 5mm, LOO 8mm)를 통과한 것은 아니다 — RMS는 경계에
  걸치고 LOO는 아직 초과다. 물리 그리퍼 기준점과 URDF `gripper_frame_link`의
  대응은 여전히 검증되지 않았다.
- `zone_polygon_mm`은 2026-09-19에 빨간 테이프 내부 윤곽 edge-fit으로 다시
  등록했다(`zone_source: red_tape_inner_contour_edge_fit_refined`).
- 과거 기록: 2026-09-08의 RMS 8.18mm·LOO 26.64mm·grasp-z 표준편차 6.82mm fit은
  `venue_lab.pre-recalibration-20260911.json`에 남아 있다. 2026-08-31의 15점 fit과
  원인 가설은 `docs/report/CV_IK_전환_정리.md`에 당시 기록으로 보존한다. 이
  숫자들을 현재 값으로 인용하지 않는다.
- **결론: 이 오차 규모(RMS ~5mm, 최악 ~14mm)를 전제로 설계한다.** 재캘리브레이션에
  시간을 더 쓰지 않는다. §7의 IK/파지 설계가 이 오차를 흡수해야 한다
  (그리퍼를 넉넉히 열기, FSM의 기존 재시도 로직으로 실패 흡수 — 매번 완벽할
  필요는 없다).

**체스판 격자(`board_grid`)는 표시·주소 지정 전용이다.** `tools/calibrate_board_grid.py`가
체스판 코너에서 격자의 원점·방향·피치를 재서 캘리브레이션 파일에 넣고,
`session/grid.py`가 정수 칸 좌표 (x, y)를 로봇 베이스 mm로 바꾼다. 위의 "임의
체스판 원점 프레임을 쓰지 않는다"는 규칙은 그대로다 — 칸은 **주소**이고, 모션
계획에 들어가기 전에 한 번 mm로 해석된다. 좌표축은 로봇이 아니라 학생이 보는
화면 기준이다(x+ = 화면 오른쪽, y+ = 로봇에서 멀어지는 쪽). 격자는 H 위에 얹혀
있으므로 **카메라가 움직이면 H와 함께 무효**다. 적합 RMS가 한 칸의 10%를 넘으면
도구가 저장을 거부한다.

**칸의 안쪽 경계는 가정하지 말고 잰다.** `agent.table_regions.min_radius_mm`(150mm)은
자유 배치용 가정값이라 격자에 그대로 쓰면 로봇 좌우의 멀쩡한 자리가 통째로 빈다.
2026-09-19 place IK 게이트로 전 칸을 돌려 보니 실패는 베이스 정면의 좁은 통로
(|y| ≤ 25mm, x ≤ 72mm)뿐이었고, 50mm만 옆으로 비키면 반경 55mm부터 통과했다 —
§7의 그리퍼 27mm 측방 오프셋이 만드는 모양이라 단일 반경으로는 표현되지 않는다.
그래서 `agent.board_grid`는 반경 대신 그 통로를 제외한다. 칸당 IK는 약 215ms라
세션 시작 때 전수 검사하지 않는다. 대신 `--dry-run`이 테두리 칸을 검사하고,
남은 실패 칸은 실행 시 `ik_gate`로 거부된다.

## §7 역기구학 (IK)

측정으로 확인된 SO-101 기구학 특성 (`third_party/so101/so101.urdf`, `gripper_frame_link`):

- **`z_g`(회전행렬 3번째 열)가 접근축이다.** 탑다운 파지 = `z_g ≈ (0,0,-1)`.
- **`shoulder_pan` 부호는 atan2와 반대다.** 양의 pan → 음의 y.
  시드는 `pan ≈ -degrees(atan2(y, x))`.
- **그리퍼는 pan 축에서 약 27mm 측방 오프셋**되어 있어 방위각 ≠ -pan이다.
  시드 pan에 ±6°, ±12° 후보를 두고 재시도해 흡수한다.
- **탑다운 도달 범위: 반경 0.03 ~ 0.32 m** (테이블 높이 기준). 0.32m를 넘으면
  `wrist_flex`가 한계에 걸려 수직 탑다운이 불가능하다. 다만 팔 자체는 더 멀리
  닿으므로(실측 0.44m에서 FK 기록됨) 그 밖은 기울인 접근으로 폴백한다 —
  0.32m를 하드 리밋으로 두지 않는다.
- **placo IK는 시드에 매우 민감하다.** 나쁜 시드에서는 200~350mm 오차로 수렴한다.
  탑다운 자세 시드 테이블(FK 그리드로 사전 계산)에서 시작하면 **0.00mm**로 수렴한다.
- **목표 자세의 회전은 시드 자세의 FK 회전을 쓴다.** 현재 자세의 회전을 복사하면
  (`pose_target = pose_actual.copy()`) 측방 이동 시 5-DOF로 불가능한 자세를
  요구하게 되어 실패한다.
- placo는 **메시를 cwd 기준으로 찾는다.** URDF는 절대경로로 주고 cwd를 URDF의
  부모 디렉토리로 바꾼 뒤 로드한다 (`tools/ik_move.py:load_kinematics` 참고).

**IK는 소수의 카테시안 웨이포인트에서만 푼다.** 웨이포인트 사이는
`control/trajectory.py`의 `interpolate()` + `TrajectoryPlayer`로 관절공간
보간한다. 매 틱 IK를 푸는 폐루프는 느리고(40mm/s) Orin에서 비싸다.

**그리퍼 회전(yaw)은 반드시 중립 기준으로 잡는다.** yaw와 `wrist_roll`은
1:1로 움직이는데, yaw를 로봇 베이스 기준 고정값으로 두면 pan이 도는 만큼
`wrist_roll`이 반대로 비틀려 절대 방향을 유지한다 — 작업영역을 가로지르며
약 80도가 휘둘리고 한계각(-125도)까지 간다. **2026-08-31 wrist_roll 서보
과열 정지가 이것 때문에 발생했다.** `TopDownIK.solve(yaw_deg=None)`(기본값)은
그 위치에서 `wrist_roll`이 0 근처가 되는 yaw를 자동으로 잡고(작업영역 전체
±3.7도), 블록 각도를 맞춰야 할 때는 `grasp_yaw_deg()`가 정사각형의 90도
대칭을 이용해 중립에 가장 가까운 정렬을 고른다. 사람이 손으로 잡을 때
자연스럽게 나오는 자세가 바로 이 중립 자세다.

**주의: 위 "0.00mm 수렴"은 URDF 모델 안에서의 시뮬레이션 결과다.** 실제
팔에서는 §6에서 실측한 규모(현재 적용본 기준 RMS ~5mm, 최악 ~14mm)의 FK 오차가
있다 — 정확한 값은 §6이 아니라 `venue_lab.json`의 `meta`가 기준이다. 따라서 IK 하나만으로 정확히 파지 지점에 도달한다고 가정하지 않는다.
현재 production PICK은 최초 fresh frame에서 만든 중심·재시도 후보를 실행하고,
VERIFY 실패 뒤 HOME→SELECT에서 새 프레임으로 다시 검출한다. 파지 직전 근접
재검출은 아직 구현되지 않았으므로 구현된 것처럼 문서화하지 않는다.

## §8 카메라

- 탑다운 카메라 1대만 사용한다. 손목캠은 시야를 가리므로 분리 상태를 유지한다.
- **카메라는 강체로 고정되어야 한다.** 이것이 전체 파이프라인의 선행 조건이다.
  12px 드리프트는 작업영역에서 10~20mm 위치 오차이며 블록 폭(40mm) 대비
  파지 실패를 직접 유발한다.
- 게이트: `tools/camera_drift_check.py`로 **10분간 p95 드리프트 < 2px.**
  실제 드리프트는 강체 운동이라 모든 코너를 함께 움직인다. 코너 하나가 튀는 것은
  검출 노이즈이므로 `max`로 판정하지 않는다(400여 개 코너에서 헛실패가 난다).
  매 세션 시작 시 재검사한다. 미달이면 캘리브레이션을 신뢰하지 않는다.

## §9 블록 검출: 색 + 형상

**색만으로 판단하지 않는다.** 빨간 블록과 빨간 경계 테이프는 색상은 같지만
형상이 다르다 — 테이프는 얇고 길쭉하며 모서리가 비어 있고, 블록은 채워진
정사각형이다. 면적/종횡비/solidity/fill 필터가 이 차이를 인코딩한다.

- 검출은 homography로 정사영한 **미터법 뷰**에서 수행한다. 따라서 모든 임계값이
  mm 단위이며 카메라 위치와 무관하다.
- `configs/default.yaml`의 green/blue와 색 prototype은 실장비 프레임에서
  조정했지만 red/yellow/wood HSV gate 일부는 합성 기준이 남아 있다. 세션 조명이
  바뀌면 `tools/view_detect.py`로 다시 확인한다.
- 맞닿은 동색 블록은 컨투어가 병합된다. 면적이 단일 블록의 ~2배인 블롭은
  분할하거나, 다음 사이클 재검출에 맡긴다(FSM이 매 사이클 재검출하므로 한 개를
  치우면 자연 분리된다).

## §10 파지 검증과 센싱

- 파지 판정은 `Present_Position`(그리퍼가 완전히 닫혔는가)과 `Present_Load`
  (물체를 물고 있는가)를 함께 본다. 빈 손으로 닫히면 위치가 `gripper_empty_closed_max`
  아래로 내려간다.
- 임계값은 장비마다 다르다. `tools/tune_gripper_load.py`로 분포를 찍어 실측한다.
  `default.yaml`의 값은 가정값이다.

## §11 궤적과 안전

안전장치는 **겹쳐서** 건다.

1. `interpolate()`가 틱당 관절 델타를 `max_step_per_tick`으로 제한한다.
2. lerobot `send_action` 안의 `max_relative_target` 클램프는 항상 켜둔다.
   규칙이든 정책이든 `BaseRobotIO.send_joints()`를 거치면 자동 적용된다.

**궤적은 항상 측정된 현재 자세에서 시작한다.** 명령된 자세가 아니다. 그래야
정책이 목표에서 약간 벗어나 멈춰도 규칙 기반 동작으로 넘어갈 때 점프가 없다.

`disable_torque_on_disconnect`는 **false**다. 정상 종료 후에도 팔이 안전 자세를
유지해야 하며, 토크 해제는 명시적 수동 조작이어야 한다.

## §12 설정

- **모든 튜너블은 `config.py` dataclass + `configs/*.yaml`에 둔다. 코드에 매직 넘버 금지.**
- 알 수 없는 YAML 키는 로드 시점에 거부된다(오타 조기 발견).
- 런타임 오버라이드: `--set fsm.time_budget_s=180`.

## §13 테스트

```bash
cd ~/lerobot_sim2real
uv run --extra dev pytest tests -q
```

하드웨어·lerobot·placo 없이 전부 통과해야 한다. 새 상태 핸들러는 `MockRobotIO`
기반 테스트를 함께 추가한다.

## §14 에이전트 작업 규칙

1. **기존 보고서와 모방학습 코드는 수정·삭제하지 않는다.**
   `src/policy/*`, `src/fsm/handlers.py`, `docs/report/*`는
   읽기 전용으로 취급한다. CV+IK 작업은 전부 **추가(additive)** 로 한다.
2. 기존 함수를 먼저 찾아 재사용한다. 특히 `calibrate_from_pairs()`,
   `interpolate()`, `TrajectoryPlayer`, `check_grasp()`, `ContactMonitor`,
   `PoseRegistry`는 이미 구현되어 있다.
3. 실측으로 확인되지 않은 수치를 문서나 주석에 단정적으로 쓰지 않는다.
   가정값이면 가정값이라고 명시한다.
4. 하드웨어를 움직이는 코드를 처음 실행할 때는 `--dry-run`을 먼저 지원하고 사용한다.

## §15 Task 3 — ACT 데이터셋 자동 수집

운영 가이드는 [docs/guide/SO101_TASK3_데이터수집.md](docs/guide/SO101_TASK3_데이터수집.md).

1. **녹화는 `BaseRobotIO` 데코레이터에서만 한다.** 모든 팔 명령은 예외 없이
   `send_joints()`를 지난다(`TrajectoryPlayer`의 네 메서드 모두). 그래서
   `data/episode_recorder.RecordingRobotIO`가 런 전체를 녹화할 수 있는 단일
   지점이고, FSM·모션·그래스프 코드는 Task 1/2가 이미 검증한 코드 그대로 남는다.
   **녹화를 위해 제어 경로에 훅을 심지 않는다.**
2. **에피소드 경계는 Task 1 구조에서 그대로 떨어진다.** `Task1SelectState.enter()`가
   home 복귀 지점이므로, 그 호출 *뒤에* 에피소드를 닫으면 모든 에피소드가 home
   복귀 동작으로 끝나고 다음 에피소드가 정착된 home에서 시작한다. 양 끝이 같은
   자세라 시연이 닫힌 사이클이 된다. 카메라 폴링 대기는 에피소드 밖이다.
3. **저장 규칙: 파지 성공 + 운반 완료.** PLACE가 끝나야 `task3_episode_ok`가 선다.
   실패한 시연을 저장하면 정책이 그 실패를 학습한다. 파기는 무조건
   `clear_episode_buffer()`이며, 파기 사유는 요약 JSON에 남긴다.
4. **데이터셋 fps는 실제 틱 주기와 같아야 한다.** `_tick_sleep()`은 방금 한 작업과
   무관하게 `1/fps`를 자므로 그 자체로는 주기를 못 만든다. LeRobotDataset은
   `timestamp`를 `frame_index/fps`로 합성하므로 어긋나면 정책이 다른 속도로
   재생된다. Task 3은 `motion.fps`를 올려 그 슬립을 무시할 수준으로 만들고
   `RecordingRobotIO`가 절대 데드라인으로 페이싱한다. 실측(Orin, 카메라 2대):
   목표 30 Hz에 30.06 Hz, 평균 33.3ms, p95 36.1ms.
5. **이미지는 camera.server의 MJPEG 스트림에서 받는다** — §8의 단독 소유 규칙을
   깨지 않기 위해서다. `camera/frame_source.py`가 백그라운드에서 디코드·리사이즈·
   **BGR→RGB 변환**까지 마치므로 제어 스레드는 배열 복사만 한다. RGB 변환은
   선택이 아니다: lerobot 카메라는 `ColorMode.RGB`가 기본이고 ACT 백본도 RGB
   통계로 프리트레인돼 있다.
6. **추론도 같은 파이프라인을 써야 한다.** 기록이 서버 MJPEG(JPEG q80, 다운스케일)
   경유인데 추론에서 `robot.cameras`로 직접 열면 압축 아티팩트와 센서 크롭 FOV가
   달라져 distribution shift가 생긴다. `MjpegFrameSource`를 재사용 모듈로 둔 이유다.
7. **ACT는 모든 `observation.images.*`의 shape이 같기를 요구한다.** 모든 스트림을
   `task3.image_width/height`로 리사이즈하고 `validate_task3`가 이를 강제한다.
   한 데이터셋 안에서 카메라 구성을 바꾸지 않는다 — 바꾸려면 새 데이터셋이다.
8. **Ctrl-C는 틱 경계에서만 푼다.** 시그널 핸들러는 플래그만 세우고 자기 자신을
   `SIG_IGN`으로 교체한다. 두 번째 Ctrl-C가 parquet/비디오 finalize를 깨면
   데이터셋 전체를 잃기 때문이다. 전체 런은 `VideoEncodingManager`로 감싼다.
9. **가정값은 가정값이라고 쓴다**(§14.3). `min_episode_frames`(60),
   `max_episode_frames`(3000)은 실측 전 가정값이다. 첫 라운드 후 교체한다.

## §16 LLM 툴 콜링 에이전트 (`so101-agent`)

운영 가이드는 [docs/guide/SO101_LLM_에이전트.md](docs/guide/SO101_LLM_에이전트.md).

1. **기존 미션 경로는 불변이다.** `so101-run`/`so101-collect`는 에이전트 코드를
   import하지 않는다. 에이전트는 `session/`·`agent/` 추가 코드와, 동작이 같은
   추출(`control/task1_transport.py`의 `solve_place_point`/`fly_carry`/`release_at`,
   `session/factories.py`)만 공유한다.
2. **로봇 버스 소유자는 하나.** 카메라 서버가 `/dev/video*`, 세 러너 중 하나가
   시리얼 버스를 갖는다. `ArmSession.open()`은 `agent.lock_path` 락을 잡는다.
3. **LLM은 원시 관절 명령과 자유로운 절대 좌표를 받지 않는다.** 칸/영역 이름,
   한도가 걸린 상대 mm 벡터, 그리고 체스판 **칸 좌표**(정수 x, y)만 받는다.
   칸 좌표는 예외가 아니라 이산 주소다: 부채꼴 게이트와 IK 게이트를 다른 목표와
   똑같이 통과해야 하고, mm는 여전히 LLM에서 오지 않는다. 화면에서 가리킨 곳을
   이름 없이 말해야 해서 열어둔 유일한 절대 주소이며, 새 툴도 이 선을 지킨다.
4. **배치 정확도는 캘리브레이션이 아니라 릴리즈 동작이 정한다.** `release_at`은
   `motion.arrival_tol`(3°)에서 그리퍼를 열고, 이 장비 실측으로 3°는 283mm 리치에서
   약 29mm다(`MotionConfig.grasp_hover_arrival_tol` 주석). 블록을 든 상태의 정상상태
   처짐이 더해지며, `TrajectoryPlayer.move_to`가 적어 둔 대로 더 기다린다고 닫히지 않는다.
   §6의 캘리브레이션 잔차(RMS ~5mm)를 이 오차의 원인으로 지목하지 않는다.
   보정은 상수로 박지 말고 **측정으로 닫는다**: 모든 배치는 이미 home 복귀 후 재관찰로
   스스로를 검증하므로, 그 (명령, 실제) 쌍이 곧 보정값이다
   (`session/place_correction.py`, 팔 기준 프레임 — 처짐은 뻗은 정도와 회전각의 함수라
   베이스 x/y로 배운 값은 다른 방위각으로 옮겨가지 않는다).
5. **LLM은 primitive만 조합한다.** 복합 pick/place 및 `run_task1/2/3`는 도구로
   노출하지 않는다. 기존 미션 FSM의 재시도 정책은 유지한다. 에이전트의 각 호출은
   제한된 동작 하나이며 파지 검증·접촉·좌표 제한은 코드에서 강제한다.
   데이터 수집은 `begin_episode` → 같은 동작 primitive들 → 홈·재관찰 →
   `save_episode` 조합이다. 성공 플래그는 LLM에서 받지 않는다. 실패·STOP·시간축
   불량은 recorder 버퍼 폐기와 사유 기록으로 처리하며 기존 Task 3 러너는 유지한다.
6. **STOP은 `session/cancel.py`의 `CancellableRobotIO`에서만 구현한다.** 제어
   경로(control/, fsm/)에 훅을 심지 않는다. `Cancelled`는 RuntimeError/OSError/
   ValueError가 아니어야 한다(`fsm/task1.py`가 삼킨다).
7. **잠금은 서버가 결정한다(`agent/control.py`).** IDLE에서만 명령 수락, STOP·로봇
   고장 후에는 실측으로 home 도착이 확인돼야 IDLE로 돌아간다. STOP에는 권한이
   필요 없다.
8. **로봇은 한 스레드에서만 구동한다(`agent/worker.py`).** `TopDownIK`의 `os.chdir`
   때문에 IK도 그 스레드에서 한 번만 만든다. 웹 이벤트 루프는 로봇을 만지지 않는다.
9. **LLM SDK와 웹 프레임워크는 지연 import.** `agent.runner`/`agent.tools`/
   `agent.control`/`session.skills`는 SDK·FastAPI·placo·lerobot 없이 import돼야 하며
   `tests/test_agent_tools.py`가 이를 검사한다. 의존성은 optional extra이고
   `uv sync`가 아닌 `uv pip install`로 설치한다(JetPack torch 휠 보호).
10. **가정값은 가정값이라고 쓴다**(§14.3). `agent.relative.*` 한도, `agent.table_regions.*`,
   `agent.place_clear_radius_mm`는 실측 전 가정값이다. 시연 전 `so101-agent --dry-run`
   으로 확인하고 실측 후 교체한다.

11. **LLM 대기 중 녹화도 로봇 소유 스레드에서 한다.** `RobotWorker`의 idle tick은
    활성 에피소드에서만 기존 `RecordingRobotIO`로 마지막 명령을 유지·기록한다.
    IK/관찰 등 긴 호출의 누락 시간은 보간하거나 30 Hz로 가장하지 않는다. 시간 간격과
    평균 주기 게이트를 넘으면 폐기한다. 홈 도착 뒤 녹화를 멈추고 재관찰은 밖에서 한다.
    종료·STOP 시 진행 중 버퍼를 폐기하고 dataset/video writer를 마무리한다.
