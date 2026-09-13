# 평가 기록 시트 사용법 (`eval_log_template.csv`)

[AGENTS.md](../../AGENTS.md) §13의 정량 검증 원칙에 따라 사용하는 rollout 기록
템플릿입니다. 현재 기본 경로는 CV+IK이므로 정책 체크포인트가 없는 실행도 기록할
수 있습니다.

## 컬럼 설명

| 컬럼 | 설명 |
|---|---|
| `date` | rollout 진행 날짜 |
| `tester` | 실행자 |
| `policy_checkpoint` | ACT이면 체크포인트, CV+IK이면 `none` |
| `mission` | `1차`(이동) / `2차`(적재) |
| `condition_variable` | 이번 세트에서 바꾼 변수 (예: `chunk_size`, `lighting`, `object_position`). 첫 세트는 `baseline` |
| `condition_value` | 그 변수의 값 (예: `chunk_size=16`) |
| `initial_position_set` | 고정된 초기 위치 세트 이름 (예: `pos_set_A`) — **세트 내에서는 항상 동일하게 유지** |
| `rollout_num` | 1~10 (조건당 10회) |
| `grasp_success` | 파지 성공 여부 (1/0) — 공통 지표 |
| `placement_success` | 안착 성공 여부 (1/0) — 1차 미션만 |
| `stack_success` | 적재 성공 여부 (1/0) — 2차 미션만 |
| `hold_5s_success` | 5초 유지 성공 여부 (1/0) — 최종 판정 |
| `overall_success` | 최종 성공 여부 (1/0) — 위 지표 종합 |
| `failure_stage` | 실패했다면 어느 단계에서 실패했는지 (예: `grasp`, `transport`, `place`, `hold`) |
| `time_to_complete_s` | 완료까지 걸린 시간(초). 1차는 180초, 2차는 300초 제한 참고 |
| `notes` | 특이사항 자유 기록 |

현재 CV+IK 실험은 `notes`에 최소한 Git commit, `flow`, `pick_mode`, calibration
파일/시각, `--set` override, 카메라 drift 결과와 외부 timeout 사용 여부를 적습니다.
Task 1은 runner 내부 FSM budget을 끄므로 180초 평가를 재현했다면 외부 supervisor
명령도 함께 남깁니다. Task 2는 PLACE와 5초 유지 판정이 완성된 뒤 같은 형식으로
기록합니다.

## 프로토콜

1. **rollout 10회 / 조건.**
2. **초기 위치 세트는 조건 내내 고정.** (`initial_position_set`으로 어떤 세트인지 기록)
3. **한 번에 한 변수만 바꾼다.** (`condition_variable` 하나만 이전 세트와 달라야 함)
4. 매 rollout마다 성공/실패와 실패 단계를 반드시 기록.

## 성공률 계산 (조건별)

```
성공률 = overall_success 합 / 10
```

스프레드시트(Google Sheets, Excel)로 열어서 조건별로 피벗 테이블 만들면 비교하기 편합니다.
