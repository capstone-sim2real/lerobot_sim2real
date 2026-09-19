# SO-101 LLM 툴 콜링 에이전트 운영 가이드

자연어(텍스트·음성) 요청을 LLM이 툴 호출로 바꿔 실제 팔을 움직이는 `so101-agent`의
실행 방법, 구조, 현장 튜닝 순서를 정리한다. 기존 `so101-run --task 1/2/3`,
`so101-collect`의 동작은 바뀌지 않았다.

## 1. 실행

```bash
cd ~/lerobot_sim2real && source .venv/bin/activate

# 최초 1회 (uv sync 금지: JetPack torch 휠이 교체될 수 있음)
uv pip install --python .venv/bin/python "fastapi>=0.110" "uvicorn>=0.29" \
  "openai>=1.60" "google-genai>=1.0"
uv pip install --python .venv/bin/python -e . --no-deps

# API 키는 .env 파일 하나로 관리한다 (export 불필요, git에 안 올라감)
cp .env.example .env
vi .env    # OPENAI_API_KEY와 GEMINI_API_KEY 입력

# 1) 사전 점검 — 로봇 버스를 열지 않는다
so101-camera &
so101-agent --dry-run

# 2) 리허설 — 하드웨어·API 키 없이 (시뮬 팔 + 시뮬 카메라 + 규칙 기반 가짜 LLM)
so101-agent --sim --provider fake

# 3) 실기 — 기본 OpenAI(gpt-5.6-luna), 요청 시작 전 실패 시 Gemini 폴백
so101-agent                               # http://<jetson IP>:8099/
so101-agent --provider gemini             # Gemini만 사용(명시적 선택은 폴백 없음)
so101-agent --provider anthropic          # Anthropic만 사용
so101-agent --model <모델 id>              # agent.models[provider] 덮어쓰기
```

브라우저에서 `http://<jetson IP>:8099/`. 종료는 서버 터미널에서 Ctrl-C
(들고 있는 블록은 놓지 않고 home으로 복귀한 뒤 연결을 끊는다).

### API 키 (.env 파일)

`so101-agent`는 시작할 때 `.env` 파일을 자동으로 읽는다 — 매번 `export`할 필요가 없다.

1. 저장소 루트에서 `cp .env.example .env`
2. 기본 구성은 OpenAI를 먼저 쓰고 Gemini를 폴백으로 쓰므로
   `OPENAI_API_KEY=...`와 `GEMINI_API_KEY=...`를 모두 채운다.
3. `so101-agent`를 실행하면 `.env`를 읽었다는 로그(`loaded API key(s) from ...`)가 뜬다.

기본 호출은 `openai / gpt-5.6-luna`다. OpenAI가 텍스트나 툴 호출을 하나도 내보내기
전에 API·네트워크 오류로 실패한 경우에만 같은 턴을 `gemini / gemini-3.8-flash`로
다시 요청한다. 일부 응답이 이미 나온 뒤에는 중복 응답이나 중복 툴 실행을 막기 위해
폴백하지 않고 원래 오류를 보고한다. `--provider`를 명시하면 해당 제공자만 사용하며
폴백은 꺼진다.

찾는 위치는 순서대로: `--env-file <경로>`로 명시한 파일 → 현재 디렉터리의 `.env` →
저장소 루트의 `.env`. 이미 셸에서 `export`한 값이 있으면 그 값이 항상 우선한다.
`.env`는 `.gitignore`에 들어 있어 커밋되지 않는다 — API 키를 절대 커밋하지 말 것.

- `so101-run`/`so101-collect`와 동시에 실행하면 버스 락(`var/so101/robot.lock`)
  때문에 시작이 거부된다. 락 보유 프로세스의 pid가 에러에 나온다.
- 모델 id는 `src/configs/default.yaml`의 `agent.models`에 있다(기본 OpenAI는
  `gpt-5.6-luna`, 폴백 Gemini는 `gemini-3.8-flash`). 각 벤더의 현재 모델 목록으로
  확인하고 고친다.

### 음성 입력

보내기 버튼 왼쪽의 `🎤`을 누르면 브라우저 내장 Web Speech API(`ko-KR`)가 음성 인식을
시작하고, `■`을 누르면 끝낸다. 중간 인식 결과는 입력창에 바로 표시되며 기본 설정은
인식 종료 후 자동 전송이다. 자동 실행이 싫으면 `음성 인식 끝나면 자동 전송`을 끄고
문장을 확인한 뒤 `보내기`를 누른다. 별도 STT 모델이나 서버 음성 업로드는 사용하지 않는다.

Tailscale HTTPS가 가장 안정적이다. 다음처럼 에이전트용 포트를 한 번 등록한다.

```bash
tailscale serve --bg --https=8443 http://127.0.0.1:8099
```

- Jetson 자체에서 Chrome으로 `http://localhost:8099` 접속 → 동작
- Tailscale 원격 접속 → `https://<tailscale-hostname>:8443/`
- SSH 대안 → 노트북에서 `ssh -L 8099:localhost:8099 -L 8090:localhost:8090 <jetson>` 후
  `http://localhost:8099`

HTTP IP 주소에서도 앱이 음성 인식을 선제 차단하지 않고 브라우저에 직접 요청한다.
브라우저 정책·버전에 따라 HTTP에서는 매번 권한을 묻거나 거부할 수 있으므로, 거부되면
HTTPS 주소를 사용한다. Web Speech API가 없는 브라우저에서도 버튼은 숨기지 않고 미지원
이유를 표시한다. 브라우저 내장 인식 서비스 특성상 인터넷 연결이 필요할 수 있다.

## 2. 조작 규칙 (서버가 강제)

| 규칙 | 동작 |
|---|---|
| 공유 조작 | 여러 브라우저가 같은 세션에 접속해 제어할 수 있다. 로봇 명령은 서버가 한 번에 하나만 실행하며, 실행 중 다른 요청은 `409 busy`로 거부한다. |
| 입력 잠금 | 명령 하나(여러 툴 호출 포함)가 끝날 때까지 입력창·마이크·조그·클릭이 꺼진다. 서버도 409로 거부 |
| 비상정지 | 화면 오른쪽 아래 STOP 또는 Esc. 다음 모터 명령 직전에 멈춘다 |
| 정지 후 | **자동으로** 그리퍼를 열고(들고 있던 블록은 그 자리에 놓고) home 복귀한 뒤 그리퍼를 닫는다. 낮은 위치면 먼저 수직 상승. 실측으로 home 도착이 확인돼야 입력이 풀린다 |
| 자동 복귀 실패 시 | **[다시 시도]** 버튼으로 같은 절차(그리퍼 열기 → home → 그리퍼 닫기)를 다시 시도할 수 있다 |
| 로봇 고장 | 동작 시간 초과·통신 오류·예상 못한 오류도 비상정지와 같은 자동 복귀 절차를 탄다 |
| 공유 세션 종료 | 연결된 브라우저가 하나도 없는 상태가 15초 넘으면 세션 종료(동작 중이었다면 정지로 처리). 대기 상태로 5분 방치해도 종료 |
| 새 접속자 | 실행 중인 공유 세션과 대화에 그대로 참가한다. 다른 접속자의 화면에도 같은 상태와 실행 결과가 표시된다. |

STOP 버튼은 권한 없이 누를 수 있다(잠금 화면에서도). 정지 시 들고 있던 블록을 무조건
떨어뜨리는 이유: 팔과 블록이 작아 위험이 없고, 경기장 안쪽은 학생 손이 닿지 않는
구조라 안전보다 복구 속도와 단순함을 우선했다. 수동(사람이 버튼을 눌러야만 복귀)으로
되돌리려면 `--set agent.stop_auto_home=false`.

공유 세션 강제 종료: Jetson에서 `curl -X POST localhost:8099/api/lease/force-release`.

## 3. 방향과 장소 이름

| 표현 | 위치 이름(칸·부채꼴 영역) | 이동량(mm) |
|---|---|---|
| 앞/멀리 ↔ 뒤/가까이 | 상단(먼 줄) ↔ 하단(가까운 줄) | `forward_mm` +/− (로봇에서 멀어지는 반경 방향) |
| 왼쪽 ↔ 오른쪽 | 카메라 화면 좌/우 | `left_mm` +/− (로봇 기준과 화면 기준이 일치) |
| 위 ↔ 아래 | 화면 위(먼 쪽) ↔ 아래(가까운 쪽) | 팔 조그: `up_mm`(높이) / 테이블 위 블록 이동: `forward_mm` |
| 조금·살짝 / 거리 없음 | — | `agent.relative.small_step_mm`(10) / `default_step_mm`(20) |

- 적재 구역 5칸: 상단 `좌상단 상단중앙 우상단`, 하단 `좌하단 우하단` (slot 0–4, `task1.slot_uv` 그대로).
  카메라 화면 기준 상단 = 로봇에서 먼 줄임을 `venue_lab.json`의 `meta.zone_polygon_px`로 확인했다.
  **카메라를 다시 달면 dry-run의 `image px` 열로 재확인**할 것.
- 부채꼴 테이블 영역: 열 `leftmost left center right rightmost` × 행 `near middle far`.
  검출기·카메라 오버레이의 주황 부채꼴과 같은 경계를 쓴다. `center/middle`, `center/far`는
  적재 구역 정면이라 항상 건너뛴다.
- **체스판 칸 좌표 `(x, y)`**: 부채꼴 안의 체스판 칸마다 정수 좌표가 붙는다(현재 177칸).
  바깥 경계는 이름 붙은 15개 자리와 같다(부채꼴 외곽 − `edge_margin_mm`). 안쪽은 다르다:
  15개 자리의 `min_radius_mm`(150mm)는 자유 배치용 가정값이라 그대로 쓰면 로봇 좌우의
  멀쩡한 자리가 통째로 비어서, 실측한 베이스 제외 영역으로 대체했다(아래 참고).
  - 축은 **화면 기준**이다: `x+` = 화면 오른쪽, `y+` = 화면 위 = 로봇에서 멀어지는 쪽.
  - `(0, 0)`은 부채꼴 아래-가운데 기준 칸(`center/near` 자리)이라 음수 좌표가 정상이다.
    다른 칸을 기준으로 하려면 `agent.board_grid.origin_mm`.
  - 화면에서 칸을 누르면 입력창에 `격자 (3, 4)`가 들어가고, 수동 패널의 `칸 x y` 칸에
    직접 넣고 "이 칸 위로 이동"을 누르면 LLM 없이 바로 간다.
  - 카메라 화면 위 오버레이는 **한 번에 하나만** 나온다. 영상 아래 단추 토글
    `부채꼴 | 격자` 중 하나를 고르며, 기본값은 `부채꼴`(이름 붙은 15개 점)이다
    (`agent.board_grid.default_layer`, 브라우저별 선택은 기억된다). 적재 구역 5칸과
    주황 부채꼴 경계선은 두 경우 모두 그려진다.
  - **베이스 제외 영역** (`agent.board_grid.min_radius_mm`, `base_keepout_depth_mm`,
    `base_keepout_half_width_mm`): 2026-09-19 place IK 게이트로 전 칸을 돌려 잰 결과,
    실패하는 칸은 베이스 정면의 좁은 통로(|y| ≤ 25mm, x ≤ 72mm)뿐이었고 옆으로 50mm만
    비키면 반경 55mm부터 통과했다. 그리퍼가 pan 축에서 27mm 측방 오프셋된 구조
    (AGENTS.md §7) 때문이라 하나의 반경으로는 표현되지 않는다.
  - 카메라 프레임 밖으로 나가는 칸은 제공하지 않는다. 누를 수도, 놓은 뒤 확인할 수도 없다.
  - 격자 자체는 `tools/calibrate_board_grid.py`가 체스판 코너로 잰다. 재지 않으면 로봇 축에
    정렬한 `agent.board_grid.cell_mm`(25mm) 가상 격자로 폴백하며 dry-run이 이를 경고한다.
  - **정확도는 캘리브레이션이 아니라 놓는 동작이 정한다.** `release_at`은 모든 관절이
    명령에서 `motion.arrival_tol`(3°) 안에 들면 그리퍼를 여는데, 이 장비 실측으로 3°는
    283mm 리치에서 약 29mm다(`MotionConfig.grasp_hover_arrival_tol` 주석). 블록을 든
    상태의 정상상태 처짐이 여기에 더해진다 — 캘리브레이션 잔차(RMS 5mm)보다 한 자릿수 큰
    오차의 출처다.
  - 그래서 **놓은 뒤 카메라로 잰 값으로 다음 명령을 보정한다**(`agent.place_correction`,
    `session/place_correction.py`). 배치마다 이미 하던 확인 관찰을 그대로 쓰므로 추가
    동작이 없다. 결과에 `miss_mm`(목표에서 벗어난 거리), `measured_cell`(실제로 놓인 칸),
    `place_correction`(현재 학습값, 팔 기준 forward/left)이 실린다.
  - 세션 중 학습한 값은 `get_state`의 `place_correction`과 로그에 나온다. 수렴한 값을
    `agent.place_correction.forward_mm/left_mm`에 적어 두면 다음 세션은 거기서 시작한다.
    폭주 방지로 `max_mm`(60mm)을 넘지 못하고, `max_sample_mm`(80mm)보다 크게 빗나간
    배치는 편향이 아니라 사고로 보고 학습에서 제외한다.
- 상대 이동 기준(`agent.relative.frame`): `arm`(기본) = 반경/접선, `base` = ±x/±y 고정축.

## 4. 툴

LLM에게는 원시 관절 명령도, 자유로운 mm 좌표도 없다. 이산적인 체스판 칸 좌표만
예외로 열려 있고, 그것도 부채꼴·IK 게이트를 똑같이 통과해야 한다(AGENTS.md §16.3).

| 툴 | 하는 일 | 대표 요청 |
|---|---|---|
| `move_block_to_slot` | 집기 + 칸에 놓기 + 카메라 확인. 칸이 차 있으면 움직이지 않고 거부 | "노란 블록 좌상단으로" |
| `move_block_to_table` | zone 안/밖 블록을 부채꼴 영역(생략 시 가까운 빈 자리)에 놓기. 놓을 곳이 없으면 집기 전에 거부 | "초록 블록 다시 밖으로", "부채꼴 가장 오른쪽 아래에" |
| `shift_block` | 놓인 블록을 상대 이동. 목적지를 집기 전에 검사, 이동 후 **실측 변위** 보고 | "그 블록 20mm만 더 왼쪽으로" |
| `pick_block` | 색으로 집기(zone 안 포함). 파지점 보정·누적 가능 | "5mm만 더 멀리 집어줘" |
| `place_at_slot` / `place_on_table` / `place_here` | 들고 있는 블록 내려놓기 | "좌상단에 10mm 오른쪽으로", "여기 내려놔" |
| `move_arm` | 조그(1회 ≤ 50mm, 높이 창·부채꼴·IK 오차 검사) | "왼쪽으로 가줘", "조금 위로" |
| `move_block_to_cell` | 집기 + 체스판 칸 좌표에 놓기. 칸이 아니면 집기 전에 거부 | "파란 블록 (3, 4)로 옮겨줘" |
| `place_at_cell` | 들고 있는 블록을 칸 좌표에 놓기. 막히면 옆으로 비켜 놓고 보고 | "(−2, 1)에 놔줘" |
| `move_to_cell` | 현재 높이로 그 칸 위까지 한 번에 이동(조그 거리 제한 없음) | "(0, 3) 위로 가줘" |
| `run_task1` / `run_task2` | 기존 FSM 그대로. 미션 1은 이미 칸에 있는 블록을 보존(점유 칸 예약) | "미션 1 해줘" |
| `run_task3` | 데이터 수집 1라운드. `agent.enable_task3_tool=true`일 때만 | — |
| `observe_scene` / `describe_places` / `get_state` | 관찰·이름 목록·상태 | "블록 어디 있어?" |
| `return_to_home` / `open_gripper` | home / 복구용 그리퍼 열기 | — |

모든 툴은 같은 봉투를 돌려준다: `ok`, `reason`(닫힌 어휘), `detail`(한국어 설명),
`retry_advice`(`retry_ok`/`do_not_retry`/`ask_operator`), 툴별 데이터, `state`(들고 있는 블록,
팔 위치, 칸 점유, 최근 장면).

**재시도는 코드가 한다.** `pick_block`은 운영 미션의 `CvIkPickState`를 그대로 구동한다:
중앙 파지 → 그리퍼 90도 회전 재파지(`run_grasp_attempts`), 실패 시 home 복귀 → 재관측 →
재접근을 `fsm.max_retries_per_block`회. VERIFY 규칙(`check_grasp` 위치+부하)도 동일하다.

## 5. 구조

```text
src/agent/
  server.py     so101-agent CLI, FastAPI 라우트, SSE, dry-run
  service.py    웹 프레임워크 없는 서버 로직 (요청 스레드, STOP, home, 조그)
  control.py    제어 상태 머신 IDLE/BUSY/STOPPING/STOPPED/HOMING + 조작 권한
  worker.py     로봇 전용 스레드 (ArmSession 생성·사용·종료)
  runner.py     LLM 대화 루프 (툴 결과는 한 메시지로, 로봇 고장 시 즉시 중단)
  tools.py      툴 스키마(설정에서 enum·한도 생성), 인자 검증, 예외→봉투
  prompt.py     시스템 프롬프트 템플릿 채우기 (src/configs/agent_system_prompt.md)
  provider/     anthropic / openai / gemini 어댑터, fake(스크립트·규칙)
  sim.py        --sim 용 시뮬 팔·카메라 (실제 IK·파지·FSM 코드 사용)
  web/          index.html, app.js, app.css (빌드 없음)
src/session/
  arm_session.py  연결·캘리브레이션·IK·플래너·PICK 상태 1회 생성, 관찰, 안전 home
  skills.py       위 툴들이 부르는 스킬 (모두 SkillResult 반환)
  cancel.py       CancelToken, Cancelled, CancellableRobotIO (send_joints에서만 STOP)
  lock.py         로봇 버스 락
  relative.py     상대 이동·부채꼴 영역·빈 자리 탐색 (순수 함수)
  factories.py    run_task/run_task3 공용 팩토리 (순환 import 제거)
src/perception/scene.py  zone 안/밖 장면 + 칸 점유 (detect_blocks(include_zone=...) 추가)
```

시스템 프롬프트는 `src/configs/agent_system_prompt.md`. 숫자·이름은 `$변수`로 두고
config에서 채운다. 대화 기록은 `logs/agent/agent_*.jsonl`.

## 6. 현장 튜닝 순서 (첫 실기 날)

1. `so101-camera` 실행 후 `so101-agent --dry-run`
   - zone 칸 5개 IK 오차, `image px`가 화면의 해당 칸 위치와 맞는지
   - 부채꼴 영역 15개 중 `IK-GATE`/`OUT-OF-WORKSPACE`가 없는지 (있으면 `agent.table_regions` 조정)
   - `board cells` 줄: 측정된 격자를 쓰는지(`measured board`) 폴백인지, 칸 수와 x/y 범위,
     테두리 칸 `IK-GATE` 실패 수. 실패가 있으면 그 칸은 화면에 보여도 실행 시
     `ik_gate`로 거부된다 — 잦으면 `agent.board_grid`의 제외 영역을 다시 잰다
   - `jog entry from home` IK 오차, API 키 `set` 여부, 카메라 검출 목록
2. `so101-agent --sim --provider fake`로 화면·STOP·home 흐름 리허설
3. 실기, 빈 테이블에서 조그 패드로 짧게 움직여 `agent.relative.jog_min_z_mm`(가정값 40) 확인
4. 블록 1개로 `move_block_to_slot` → `move_block_to_table` → `shift_block` 순서 확인
   - `shift_block` 결과의 `measured_mm`과 요청값 차이를 기록 (캘리브레이션 RMS ≈ 5mm)
5. 블록이 붙어 있을 때 `destination_blocked`가 너무 자주 나면 `agent.place_clear_radius_mm`(가정값 55) 조정
6. 실제 LLM으로 대표 발화 20개 정도를 확인하고, 잘못 고르는 표현은 시스템 프롬프트에 예시로 추가

가정값(실측 전): `agent.relative.*`(오프셋·조그 한도·높이 창·조그 IK 오차 5mm),
`agent.table_regions.*`, `agent.place_clear_radius_mm`, `agent.table_zone_margin_mm`.

## 7. 알려진 한계

- 절대 위치 정확도는 캘리브레이션 수준(RMS ≈ 5mm, LOO 최대 ≈ 13.5mm)이다. 5mm 단위 요청은
  실측 변위가 다를 수 있고, 결과에 그 사실이 함께 보고된다.
- 색마다 블록 1개라는 경기장 규약을 전제한다(같은 색 2개는 구분하지 않음).
- `place_at_slot`의 점유 판단은 직전 카메라 관찰 기준이다. 사람이 그사이 블록을 옮기면 틀릴 수 있다.
- 부채꼴 영역 배치 시 다른 블록과의 거리 검사도 직전 관찰 기준이다.
- `--sim`은 UI와 툴 흐름 리허설용이다. 정확도·충돌에 대해서는 아무것도 보장하지 않는다.
- 미션 3(`run_task3`)은 에이전트에서 한 라운드만 수집하고 끝난다(재배치 프롬프트 대신 툴 종료).
  실기 검증 전이다.

## 카메라 검출 오버레이 통합

에이전트 화면의 **화면 표시** 버튼에서 전체 표시, 구역·격자, 블록 윤곽,
중심점, 파지 방향, 보정·재시도 후보, 제외 검출, FK 기준점, 상세 정보를
각각 켜거나 끈다. 색상 필터도 제공하며 선택은 해당 브라우저에 저장된다.
구역·격자를 숨기면 해당 위치의 클릭 입력도 함께 비활성화된다.

영상은 기존 카메라 서버의 MJPEG, 검출은 SSE를 사용한다. 에이전트의
/api/camera/{video,config,events,reference} 읽기 전용 중계가 같은 출처로
전달하므로 브라우저가 별도 카메라 포트에 접속할 필요가 없다.
카메라 장치와 분석 프로세스는 기존 카메라 서버가 계속 소유한다.
agent extra에 httpx가 필요하다. 실행 중인 카메라 서버를 재시작하지
않고도 기존 HTTP API로 통합 화면을 사용할 수 있다.

검출은 영상과 비동기이며 화면의 갱신 경과 시간은 마지막 새 검출 수신
기준이다. 새 프레임이 오지 않으면 agent.camera_view.stale_s 이후
검출 표시를 숨긴다. 연결 복구 시 자동으로 표시를 재개한다.
검출 관련 토글을 모두 끄거나 페이지를 숨기면 검출 SSE 구독을 닫는다.
poll_s, retry_s, connect_timeout_s, read_timeout_s도
agent.camera_view에서 조정한다. 이 값들은 화면 운영 기본값이지
물리 제어의 실측 한계가 아니다.

보정·재시도 후보는 카메라 분석 설정의 **표시용 후보**이며 실제 실행
명령의 목표와 같다는 보장은 없다. FK 기준점은 관절 기반 위치의 블록
평면 투영으로, 영상에서 실제 그리퍼 턱을 검출한 위치가 아니다.
교차 캘리브레이션의 점 추가·초기화는 기존 카메라 도구에서 수행한다.

### 로봇 상태 패널

통합 화면의 `로봇 상태 · 모터 / FK / 보정`에서 현재/목표 관절값, 차이,
모터 보정 범위, 원시 인코더, 부하, 온도, 전압 원시값, 토크, 보정 파일과
모터 보정 일치 여부, URDF 관절 범위, FK 및 모터 목표 FK 차이를 확인한다.
각도는 도, 그리퍼는 0–100 정규화 값이다. 전압 raw는 V로 변환하지 않은 값이다.
목표는 Goal_Position 레지스터이며 작업의 최종 Cartesian 목표가 아니다.
파지 상태는 세션의 보유 판정 기록이고 현재 실제 물체 보유를 재검증한 값은 아니다.

`GET /api/telemetry`는 기존 로봇 worker에 읽기 작업을 최대 하나만 대기시킨다.
브라우저는 패널이 열리고 탭이 보일 때 1초마다 조회한다. 모션 실행 중에는
worker가 사용 중이므로 갱신이 늦어질 수 있으며 마지막 측정 시각을 확인한다.
실패한 레지스터 값은 0으로 대체하지 않고 빈 값과 오류를 표시한다.
이 패널은 모터 레지스터를 쓰거나 별도 시리얼 연결을 만들지 않는다.

### 헤드캠 픽셀 목표

이 통합 화면은 격자·부채꼴 점·중앙 5개 슬롯 대신 헤드캠 픽셀을 직접 선택한다.
선택만으로는 모터를 움직이지 않는다. 원본 영상 크기로 환산한 픽셀을 서버에서
보정 좌표로 변환하고 작업 부채꼴, 외곽 여유 거리, 베이스 금지 영역을 검사한다.
초록 외곽선은 작업 영역의 개략 범위이며, 선택점의 유효 여부는 서버 응답으로
확인한다. IK 도달 여부는 실행 시 검사한다.

`선택 위치로 이동`은 빈 그리퍼를 안전 높이로 올려 횡이동한 뒤 고정된 블록 윗면
높이로 내린다. 모든 웨이포인트의 IK가 통과해야 첫 모터 명령을 보낸다.
`선택 위치에 놓기`는 기존 단일 블록 운반·릴리즈와 점유 검사를 사용한다.
쌓기나 블록 높이 추정은 하지 않는다. 기존 틱당 제한, STOP, 단일 버스 소유권은 유지한다.

높이 기준은 `grasp_z_mm_mean`이다. 현재 블록 두께 20mm와 로봇 베이스 기준
TCP Z는 다른 값이다. 임의로 Z=20mm를 명령하지 않는다. 보정 ID가 달라지면
이전 선택을 거부한다. 손목캠 영상에는 픽셀 목표 지정 기능을 적용하지 않는다.

검증은 단위·시뮬레이션·별도 미리보기 서버에서 수행했으며 실제 팔의 도달 정확도와
파지 성공은 검증하지 않았다.
