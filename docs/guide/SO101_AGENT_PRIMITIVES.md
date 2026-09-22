# SO-101 LLM primitive 조합과 데이터 수집

에이전트는 primitive 도구 하나로 통일했다. `tool_mode` 옵션이나 복합
`pick_block`/`move_block_to_slot`/`run_task1/2/3` 도구는 없다. 기존 `so101-run`과
`so101-collect` CLI 및 미션 FSM은 유지한다. LLM은 동작 순서를 결정하고,
서버는 좌표·파지·접촉·저장 조건을 강제한다.

## 이름으로 요청하는 미션

시스템 프롬프트 `src/configs/agent_primitives_prompt.md`에 Task 1·2 지침을 둔다.
“Task 1/미션 1”은 180초 규격의 5개 블록 구역 수집, “Task 2/미션 2”는
300초 규격의 5개 블록 적층·5초 유지를 뜻한다. LLM이 같은 primitive를 조합하며,
구체적으로 지정한 블록·개수·목적지는 사용자의 요청이 우선한다.
미션 이름만으로 데이터셋 녹화를 시작하지 않는다.

현재 도구에는 미션별 deadline 강제, 다층 높이 추정 및 5초 연속 적층 검증이 없다.
특히 Task 2 지침은 목표와 접촉 기반 절차를 보존한 것이며 5단 적층의 구현·실측
완료를 뜻하지 않는다. 관측/접근 여유가 부족하면 추가 적층을 멈추고 미확인으로 보고한다.

## 동작 도구

| 도구 | 범위 |
|---|---|
| `observe_scene` | 팔 이동 없이 fresh frame, CV 객체, JPEG 취득 |
| `get_state` | 관절 피드백, URDF FK, 파지·접촉·수집 상태 |
| `describe_places` | 칸/슬롯 주소 안내 |
| `move_to_target` | 객체/칸/슬롯 접근: `hover`, `pregrasp`, `grasp`, `preplace` |
| `move_relative` | 한도 내 상대 이동. 물체를 들고 있으면 하강 거부 |
| `align_gripper` | 빈 그리퍼를 블록 각도에 맞춰 중립에 가까운 방향으로 정렬 |
| `close_gripper` | 현재 위치에서 닫고 위치·부하 센싱으로 파지 확인 |
| `descend_until_contact` | 거리·시간 한도 내 접촉 탐색·백오프. 자동 해제 없음 |
| `open_gripper` | 현재 위치에서 열기. 물체를 들었으면 직전 접촉 확인 필요 |
| `return_to_home` | 빈 그리퍼 홈 복귀 |

객체 주소는 `target_type="object", object_id="yellow_1", observation_id=1`,
칸은 `target_type="cell", x=3, y=4`, 슬롯은
`target_type="slot", slot="top-left"` 형식이다. 관찰 결과의 ID만 사용한다.
`grasp`는 같은 객체의 `pregrasp` 뒤 현재 XY에서 내려가므로 상대 보정이 유지된다.
재관찰로 ID가 갱신되면 새 ID로 다시 접근한다. `hover`는 잡거나 놓지 않는다.
절대 mm/관절 입력은 도구에 없고, 상대 좌표는 설정된 arm/base 프레임이다.

## 수집 도구도 같은 목록에 있다

| 도구 | 범위 |
|---|---|
| `begin_episode(object_id, observation_id)` | 홈·빈 그리퍼에서 외부 블록 하나의 녹화 시작 |
| `save_episode()` | 서버가 성공 증거와 기록 품질을 검사한 뒤 저장 |
| `discard_episode(reason)` | 현재 버퍼만 폐기하고 이유 기록. 기존 저장분 유지 |
| `collection_status()` | 저장 개수·색상별 집계·현재 녹화·폐기 사유·경로 |
| `finish_dataset()` | 열린 에피소드 해결 후 영상·데이터셋 writer 마무리 |

예를 들어 “블록을 모으면서 성공 시연 20개를 수집해”는 다음 순서를 LLM이
반복해서 달성한다. 별도 Task 3 실행 도구나 숨겨진 수집 FSM은 호출하지 않는다.

1. 현황·장면 확인 → 구역 밖 블록과 비어 있는 구역 슬롯 선택.
2. 홈에서 `begin_episode` → 열기 → 수직 리프트 → 접근·정렬·하강 → 닫기·파지 확인.
3. 리프트 → 구역 슬롯/칸 `preplace` → 접촉 하강 → 열기 → 홈 복귀.
4. 재관찰 → `save_episode`. 실패는 폐기하고 저장 개수에 포함하지 않는다.
5. 다음 블록으로 반복. 구역 밖 블록이 없으면 사람에게 재배치를 요청한다.
6. 마지막 에피소드를 저장/폐기한 뒤 `finish_dataset`.

한 에피소드당 파지 시도는 1회다. 실패 후 새 에피소드는 홈에서 시작한다.
저장 인자에 `success=true` 같은 자기 선언은 없다. 선택한 색의 파지, 구역 안
슬롯/칸으로 운반, 접촉 뒤 해제, 빈 그리퍼 홈 도착, 홈 복귀 후 해당 색의 구역 내
검출이 모두 필요하다. 적층은 이 수집 성공 조건이 아니다. 현재 task label은
`task3.task_templates`의 구역 수집 문장이다. 임의 작업이나 적층을 이 label로
저장하면 안 된다. 학습 실행·업로드는 수행하지 않는다.

LLM 응답당 도구 하나만 실행한다. 여러 호출을 함께 반환하면 전부 거부한다.
`agent.max_tool_turns`(기본 50)는 사용자 메시지 하나의 왕복 한도이며, 목표 개수가
크면 여러 메시지에 걸쳐 이어가야 한다. 저장 개수는 `collection_status`로 확인한다. 한 에피소드는 같은 사용자 턴 안에서
저장/폐기까지 해결해야 한다. 응답 종료·API 오류·왕복 한도 초과 시 미완료 버퍼는
`turn_ended`로 폐기하며, 저장된 개수와 writer는 다음 메시지에서 이어 사용할 수 있다.

## 녹화와 시간축

녹화는 기존 `EpisodeRecorder`와 `RecordingRobotIO`를 재사용한다. 제어/FSM
코드에 녹화 훅을 넣지 않는다. 동일한 로봇 소유 스레드가 LLM 대기 동안 마지막
실제 명령을 유지하며 상태·RGB 카메라 프레임을 기록한다. 다른 스레드에서
로봇 버스를 읽거나 쓰지 않는다. MJPEG 리더만 백그라운드에서 동작한다.

홈 도착 후 최종 상태 프레임까지 녹화하고 멈춘다. 그 뒤 재관찰·저장은 에피소드
밖이다. 긴 IK/관찰/IO 호출 때문에 시간축이 끊기면 누락 프레임을 만들어 채우지
않으며, `agent.collection.max_tick_gap_s` 및 `max_mean_period_error` 위반 시 폐기한다.
기본 0.1초/10%는 실측 전 가정값이다. **현재 IK 지연으로 이 게이트에 걸릴 수 있다.**
실측 없이 게이트를 풀거나 30 Hz 수집이 검증됐다고 주장하지 않는다.

STOP·실패·서버 종료는 진행 중 버퍼를 폐기한다. 저장분은 보존한다. `finish_dataset`
및 종료 경로에서 `VideoEncodingManager`가 영상을 마무리한다. 요약 파일은 데이터셋의
`agent_collection_summary.json`이며 폐기 사유와 마지막 에피소드의 도구 이벤트를
포함한다. 파일 경로나 데이터셋 스키마는 LLM 인자가 아니라 운영 설정으로 결정한다.

## Orin 실행과 검증

```bash
cd ~/lerobot_sim2real/worktrees/agent-primitives
PYTHONPATH="$PWD/src" ../../main/.venv/bin/python -m agent.server \
  --dry-run --provider fake --set camera.auto_start=false

PYTHONPATH="$PWD/src:$PWD/tests" ../../main/.venv/bin/python -m pytest tests -q
```

2026-09-20 현재 `main/.venv`에 lerobot/placo가 없어 하드웨어 dry-run은
`No module named 'lerobot'`로 중단된다. 모의 IO와 dataset sink 테스트는 실제 IK,
카메라·부하 임계값, LLM API, parquet/video 호환성, 물리 성공률을 검증하지 않는다.
공용 환경을 `uv sync`로 바꾸지 않는다. 환경 준비 후 dry-run을 먼저 통과시킨다.

실제 실행은 같은 Python/source 선택에서 `--dry-run`을 빼고 vision 지원 공급자를
지정한다. `fake`는 단일 primitive 인터페이스 확인용이며 복합 계획을 만들지 않는다.
`agent.lock_path`는 다른 운영 러너와 같은 절대경로, 서버 포트는 충돌 없는 값으로
지정한다. 카메라는 기존 서버를 공유하고 `camera.auto_start=false`를 유지한다.
수집은 첫 `begin_episode`에서 지연 초기화하며 `datasets/agent/` 아래 새 timestamp/UUID
디렉터리를 만든다. 기존 데이터셋에 자동 append하거나 덮어쓰지 않는다.

Task 3와 같은 `task3.cameras`, image width/height, record fps, frame bounds, task label을
사용한다. 카메라 구성이나 영상 크기를 한 데이터셋 도중 바꾸지 않는다.

## 관찰과 복구 한계

JPEG는 CV에 사용한 프레임을 OpenAI/Anthropic/Gemini에 이미지로 전달한다.
대화에는 최신 이미지만 유지하고 이전 수치 결과는 남긴다. transcript가 켜져 있으면
전달 JPEG와 프레임 번호·시각도 저장한다. 검출기는 색당 하나이며 `_1`은 관찰 내
주소다. 가려진 물체의 부재나 적층 높이를 확정하지 않는다. TCP는 관절 피드백과
URDF 기반 FK이고 영상으로 측정한 그리퍼 위치가 아니다.

접촉 스파이크는 올바른 지지면이나 안정적인 적층의 증명이 아니다. 충돌 회피
플래너·시각 TCP 추적기는 포함하지 않는다. 물리 동작과 기록 속도는 미검증이다.

STOP 후 자동 복귀는 없다. 수동 복구 버튼은 **그리퍼를 열고 홈으로 이동**하므로
팔과 물체를 확인한 뒤 사용한다. 복구 동작은 데이터셋에 저장하지 않는다.
정상 서버 종료에는 기존 ArmSession의 그리퍼 유지·홈 복귀 정리 동작이 남아 있다.


### 보정된 파지 primitive

`agent.primitives.calibrated_pick: true`일 때 기존 도구 이름으로 실험 보정 경로를
사용한다. `observe_scene` → `open_gripper` → `move_to_target(pregrasp)` →
`align_gripper` → `move_to_target(grasp)` → `close_gripper` 순서다.

- pregrasp는 관측 ID에 해당하는 장면으로 기존 CV 위치 보정, 파지 bias/후보,
  양쪽 턱의 URDF 간섭 검사를 실행한다. 기존 접근 계획에 정렬이 포함되므로
  align_gripper는 해당 정렬 상태를 확인하고 중복 회전하지 않는다.
- hover 정착 실패에만 기존 최대 3도 관절 피드포워드를 한 번 적용한다.
  측정 관절과 명목 목표의 차이를 사용하며, 영상으로 잰 TCP 보정은 아니다.
  블록 위 높이, 위쪽 보정 방향, 간섭, 부하, 추종 오차 조건은 그대로 유지한다.
- grasp는 기존 부하 증가 감시 하강만 수행한다. 실패하면 close_gripper를 막고,
  성공 후 별도 close_gripper에서 센서로 파지를 검증한다. 자동 운반은 없다.
- hover에서 수평 move_relative는 명목 보정 계획에 대한 bounded residual을 쓴다.
  다른 이동·키보드 조작·새 관측·홈 복귀는 이전 하강 승인을 무효화한다.
- `session/calibration_motion.py`를 기존 보정 서버와 primitive가 공유한다.
  버스와 녹화 IO는 원래 세션 하나를 사용한다. 설정은 기존
  `agent.calibration_clearance`, motion 및 task1 값을 재사용한다.

이 통합의 모의 테스트는 코드 경로·중단 조건 확인이다. 실제 primitive 조합의
새로운 성공률을 측정한 결과는 아니며, 기존 실험 결과를 성공률로 전용하지 않는다.

### 중간 판단용 미세 조정 도구

- `inspect_motion()`: 현재 관절·부하·모델 FK, 명목 hover 관절 오차와 참조 유효성을
  반환한다. 관측이나 이동을 하지 않아 준비된 파지 경로를 보존한다.
- `correct_hover(joint="elbow_flex", gain=0.5, dry_run=true)`: 준비된 hover에서
  선택한 관절의 측정 오차에 대한 보정량을 미리 본다. `dry_run=false`로 실행한다.
  gain은 0 초과 1 이하이며 기존 최대 3도 한계 안에서 적용된다. 임의 각도 명령이
  아니며 기존 높이·간섭·손목·부하·추종 제한을 유지한다. 전체 축은 `joint="all"`.
- `descend_step(down_mm=2)`: 기존 파지 경로를 따라 제한 거리만 내려간 뒤 현재
  자세를 유지한다. 부하 증가나 추종 실패 시 중단하고 닫기를 금지한다. 중간
  높이에서는 성공 응답이어도 닫을 수 없고, 최종 파지 깊이에 도달해야 한다.

도구 호출 사이에 결과를 판단한다. 같은 실험을 위해 임시 모터 코드를 만들 필요가
없다. 이 도구들의 통합 검사는 모의 장비 기준이며 실물 성공률 측정은 별도다.
