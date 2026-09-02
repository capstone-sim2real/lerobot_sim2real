# SO-101 CV/IK 아키텍처

프로젝트의 기본 실행 경로는 고정 탑 카메라, 색·형상 검출, homography,
결정론적 IK로 구성됨. ACT/VLA 정책 서버와 원격 추론은 현재 실행 경로에
포함하지 않음.

```text
camera.server (/dev/video0 단독 소유)
  ├─ HTTP snapshot ───────────────▶ perception.detect_blocks → SELECT
  ├─ MJPEG over HTTP ─────────────▶ 브라우저 원본 영상
  └─ 최신 프레임 1개 ─▶ 저우선순위 vision worker ─▶ SSE JSON ─▶ Canvas overlay
                                                        (표시 전용)
        │
        ▼
CvIkPickState
  ├─ 보정된 grasp attempt 생성
  ├─ IK 도달 가능 attempt만 실행
  └─ gripper position/load로 파지 검증
        ▼
VERIFY → TRANSPORT → PLACE → HOME/SELECT
```

Task 1의 active perception 영역은 `도달 부채꼴 ∩ 고정 target zone 바깥`임.
구역 안 블록은 색 할당 전에 검출 목록에서 제거되며, PLACE 횟수와 무관하게 fresh
frame에서 외부 검출 0개가 연속 5초 유지될 때 DONE으로 전이함. 각 색은 하나뿐이므로
색 이름이 재시도 identity와 3+2 슬롯 예약 key를 겸함.

## 모듈 구성

| 모듈 | 역할 |
|---|---|
| `camera/` | USB 카메라 단독 소유, MJPEG/snapshot 제공, 별도 저우선순위 워커와 브라우저 Canvas로 표시 전용 오버레이 제공 |
| `perception/` | homography, 색·형상 블록 검출, 대상 선택 |
| `control/ik.py`, `control/grasp.py` | 탑다운 IK, 보정, 재시도 attempt 계획 |
| `fsm/` | SELECT, PICK, VERIFY, TRANSPORT, PLACE 전이와 시간 예산 |
| `runners/run_task.py` | Task/flow 실행 CLI |
| `tools/` | 캘리브레이션, 모터 진단, 수동 조작용 Python CLI |

## 실행 규칙

1. 카메라 서버만 `/dev/video0`을 열고, runner와 도구는 HTTP snapshot을 사용함.
2. 파지 성공은 그리퍼 Present_Position과 Present_Load를 함께 확인한 경우에만 인정함.
3. IK 미도달 attempt는 건너뛰고, 도달 가능한 attempt를 순서대로 시도함.
4. 모든 조정값은 YAML 또는 `--set` 오버라이드로 관리함.
5. 관제 오버레이 좌표는 SELECT/IK로 되돌아가지 않으며 로봇 제어의 입력이 아님.

## 검증

```bash
uv run --extra hardware --extra dev pytest -q
```
