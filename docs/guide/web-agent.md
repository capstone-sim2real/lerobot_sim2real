# 웹 에이전트와 조작 패널

로봇을 웹에서 다루는 진입점은 두 가지다. 둘 다 로봇 시리얼 버스를 단독으로
소유하므로 `so101-run`, `so101-collect`와 동시에 실행하지 않는다.

| 명령 | 기본 포트 | 용도 |
|---|---:|---|
| `so101-agent` | 8099 | LLM 채팅으로 primitive 도구를 조합한다 |
| `so101-panel` | 8109 | Task 1/2 버튼, 수동 조작(조그·칸·슬롯·픽셀 배치), 보정 도구 |

```bash
cp .env.example .env                   # API 키는 .env에만 둔다
so101-agent --dry-run                  # 팔 없이 캘리브레이션·IK·공급자·카메라 점검
so101-agent --sim --provider fake      # 장비·API 키 없는 리허설
so101-agent --provider openai          # anthropic | openai | gemini | fake
so101-panel --env-file .env --output experiments/llm_free_missions/live
```

공급자 키는 `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`/`GOOGLE_API_KEY`
중 선택한 것만 있으면 된다. 모델은 `--model` 또는 `agent.models`로 바꾼다.

## 안전 경계

- LLM은 관절 값이나 자유 mm 좌표를 받지 않는다. 관찰 결과의 객체 ID, 칸 좌표(정수),
  슬롯 이름, 한도가 걸린 상대 mm 벡터, 캘리브레이션된 픽셀만 받는다.
- 모든 목표는 작업영역 부채꼴 게이트와 IK 게이트를 통과해야 한다.
- 웹 서버는 운영자 lease를 가진 IDLE 상태에서만 명령을 받는다. STOP은 권한 없이
  누구나 누를 수 있다. STOP 뒤 자동 복귀는 없으며, 수동 복구 버튼은 그리퍼를 열고
  홈으로 이동한다.
- 로봇 버스·IK·녹화는 `RobotWorker` 스레드 하나가 소유한다. 웹 이벤트 루프는
  로봇을 직접 만지지 않는다.
- LLM 응답 하나에 도구 하나만 실행한다. `agent.max_tool_turns`(기본 50)는 사용자
  메시지 하나의 왕복 한도다.

## 도구

### 미션 단위

| 도구 | 동작 |
|---|---|
| `move_block_to_slot(color\|source, slot)` | 관찰→파지→운반→슬롯 해제→홈을 서버가 순차 실행. 요청 슬롯이 IK에 실패하면 다른 빈 슬롯을 시도하고, 안전 복구 뒤 한 번만 재시도한다 |
| `stack_block_to_floor(color\|source, floor, destination?)` | Task 2. 지정한 층(0=테이블, 4=5번째)에 `task2.drop_clearance_mm` 위에서 놓는다. 배치에 실패하면 빈 테이블에 내려놓고 한 번 재시도한다 |

두 도구 모두 해제 후 카메라로 결과를 재확인하지 않는다. 성공 응답이 실제 층 높이나
5초 안정성을 증명하지 않는다.

### 개별 primitive

| 도구 | 동작 |
|---|---|
| `observe_scene` | 팔을 움직이지 않고 새 프레임, CV 객체, JPEG를 가져온다 |
| `get_state`, `describe_places`, `inspect_motion` | 관절·FK·파지 상태, 주소 목록, hover 오차 조회. 이동 없음 |
| `select_pixel_target(u, v, calibration_id)` | 사용자가 찍은 픽셀을 객체 목표로 등록한다. CV가 놓친 블록에도 쓴다 |
| `move_to_target` | 객체/칸/슬롯으로 접근: `hover`, `pregrasp`, `grasp`, `preplace` |
| `move_to_pixel`, `place_at_pixel` | 캘리브레이션된 픽셀 위치로 빈 그리퍼 이동, 또는 들고 있는 블록 배치 |
| `move_relative` | 한도 안의 상대 이동. 블록을 든 채로는 하강을 거부한다 |
| `align_gripper` | 블록 각도나 지정 yaw에 맞춰 손목 정렬 |
| `close_gripper` | 제자리에서 닫고 위치·부하로 파지를 확인한다 |
| `correct_hover`, `descend_step` | hover 오차의 제한된 보정, 제한 거리만큼의 단계 하강 |
| `descend_until_contact`, `drop_at_zone_target` | 접촉 탐색 하강, 또는 구역 슬롯/칸의 설정 높이로 이동. 해제하지 않는다 |
| `open_gripper`, `return_to_home` | 제자리 해제(접촉 또는 검증된 해제 자세 필요), 빈 그리퍼 홈 복귀 |

주소 형식: 객체 `target_type="object", object_id="yellow_1", observation_id=1`,
칸 `target_type="cell", x=3, y=4`, 슬롯 `target_type="slot", slot="top-left"`.
재관찰로 ID가 바뀌면 새 ID를 쓴다.

`agent.primitives.calibrated_pick: true`이면 `pregrasp`가 CV 위치 보정, 파지 bias,
양쪽 턱의 URDF 간섭 검사를 수행하고 `grasp`는 부하 증가 감시 하강만 한다.
hover에서의 수평 `move_relative`는 명목 보정 계획에 대한 잔차로 적용되고, 다른
이동·새 관측·홈 복귀는 이전 하강 승인을 무효화한다.

## 데이터 수집

| 도구 | 동작 |
|---|---|
| `begin_episode(object_id, observation_id, task?)` | 홈·빈 그리퍼에서 한 블록의 녹화를 시작 |
| `record_tool_sequence(task, color, steps)` | 미리 짠 도구 목록을 녹화하며 실행. 첫 실패에서 멈춘다 |
| `save_episode()` | 파지 확인·해제 완료·홈 복귀·프레임 수·시간축을 서버가 검사한 뒤 저장 |
| `discard_episode(reason)` | 현재 버퍼만 폐기하고 사유를 남긴다 |
| `collection_status()`, `finish_dataset()` | 저장 현황 조회, 영상·데이터셋 writer 마무리 |

- 성공 여부를 LLM 인자로 받지 않는다. 한 에피소드의 파지 시도는 1회다.
- 녹화는 `RecordingRobotIO` 한 곳에서만 한다. LLM을 기다리는 동안에도 로봇 스레드가
  마지막 명령을 유지하며 기록하고, 홈 도착 뒤 녹화를 멈춘다.
- 긴 IK/관찰 호출로 시간축이 끊기면 빈 프레임을 채우지 않고
  `agent.collection.max_tick_gap_s`·`max_mean_period_error`(가정값 0.1초/10%)에 걸려
  폐기된다.
- 응답 종료·API 오류·왕복 한도 초과·STOP은 진행 중 버퍼를 폐기한다. 저장분은 남는다.
- 데이터셋은 첫 `begin_episode`에서 `datasets/agent/` 아래 새 디렉터리로 만든다.
  카메라 구성·해상도·fps·task 문장은 Task 3(`task3.*`) 설정을 그대로 쓴다.

## 키보드 연속 조작

웹 패널에서 WASD / J·K를 누르는 동안 `agent.relative.keyboard_speed_mm_s`
(기본 20 mm/s) 경로 속도로 움직인다. Ruckig 속도 제어(가속도 100 mm/s²,
저크 1000 mm/s³)로 감속·방향 전환을 처리한다. 이 값은 튜닝 출발값이다.

- 키 해제: 감속 후 종료. 방향 변경: 기존 방향으로 감속한 뒤 새 방향으로 가속.
- 창 이탈·키보드 끄기·입력 만료·STOP: 감속 없이 기존 정지 경로를 쓴다.
- 다음 목표가 작업 범위나 IK 검사에 실패하면 이미 검사된 구간 안에서 감속한다.
- `ruckig`가 없으면 시작 요청을 503으로 거부하고 팔을 움직이지 않는다.

## 한계

- 검출기는 색당 블록 하나를 가정한다. 가려진 물체의 부재나 적층 높이를 확정하지 않는다.
- 그리퍼 위치(TCP)는 관절 피드백과 URDF FK로 계산한 값이며 영상으로 측정한 값이 아니다.
- 접촉 스파이크는 올바른 지지면이나 안정 적층의 증명이 아니다.
- 모의 IO 테스트는 코드 경로와 중단 조건을 확인할 뿐 실물 성공률을 측정하지 않는다.
