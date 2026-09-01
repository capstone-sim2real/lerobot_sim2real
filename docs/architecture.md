# pick_stack — SO-101 Pick & Stack 하이브리드 FSM

ACT(PICK) + 규칙 기반(SELECT/VERIFY/TRANSPORT/PLACE)을 잇는 상태머신 패키지.
Orin에서 실행되며, PICK 상태에서만 원격 policy_server(gRPC)를 호출한다.

```
SELECT → PICK → VERIFY → TRANSPORT → PLACE → (블록 남음 && 시간 남음) → SELECT
           ↑______실패(재시도/스킵)______|
```

- Task 1(운반)과 Task 2(적층)는 **PLACE 핸들러 주입만 다르고** 나머지 경로는 동일.
- NN은 ACT 하나뿐(채점 규칙). PLACE는 전략 인터페이스 뒤에 있어 나중에
  stack-align 정책으로 교체 가능 (AGENTS.md §4 seam).

## 모듈 구성 (PR 로드맵)

| 모듈 | 내용 | PR |
|---|---|---|
| `config.py`, `configs/` | dataclass 트리 + YAML + `--set` 오버라이드. 모든 튜너블은 여기로 | 1 (this) |
| `control/robot_io.py` | `BaseRobotIO` 인터페이스, 실기(`So101RobotIO`) / 테스트(`MockRobotIO`) | 1 (this) |
| `fsm/` | 상태 계약(enter/step/exit), 머신 루프, 5분 예산, 재시도/스킵, 전이 CSV 로그 | 1 (this) |
| `perception/` | homography 캘리브레이션, 색+형상 블록 검출, nearest-first SELECT | 2 |
| `control/sensing.py` | Present_Load/Position 파지 검증 + 접촉 감지 (VERIFY·STACK 공용) | 3 |
| `control/poses.py`, `trajectory.py`, `motion.py` | 포즈 레지스트리, 관절 보간, TRANSPORT/PLACE/STACK 프리미티브 | 4 |
| `policy/act_client.py` | lerobot async_inference 개조 (robot 주입형), retreat 종료 감지 | 5 |
| `runners/`, `tools/` | task1/task2 엔트리포인트, 캘리브레이션·튜닝 CLI | 6 |

## 설계 규칙

1. **상태 핸들러는 `BaseRobotIO` + config + 인터페이스만 주입받는다.**
   하드웨어·lerobot 없이 `MockRobotIO`로 단위 테스트 가능해야 한다.
   (`import config`은 lerobot 미설치 환경에서도 성공해야 함 — lazy import)
2. **`step()`은 한 틱(제한된 동작 단위)만 수행하고 빨리 리턴한다.**
   머신이 step 사이에 시간 예산을 검사하므로, 수 분씩 블로킹하면 5분 컷이 무력화된다.
3. **모든 파라미터는 `configs/*.yaml`로.** 코드에 매직 넘버 금지.
   런타임 오버라이드: `--set fsm.time_budget_s=240`.
4. **안전 클램프(`max_relative_target`)는 lerobot `send_action` 안에 있다.**
   정책이든 규칙이든 `BaseRobotIO.send_joints()`를 거치면 자동 적용된다.

## 테스트

```bash
cd ~/lerobot_sim2real
PYTHONPATH=src uv run --no-project --python 3.12 \
  --with pytest --with pyyaml --with numpy python -m pytest tests -q
```

또는 lerobot venv에 editable 설치 후:

```bash
uv pip install --python ~/lerobot/.venv/bin/python -e ".[dev]"
~/lerobot/.venv/bin/python -m pytest tests -q
```
