# 평가 기록 시트 사용법 (`eval_log_template.csv`)

PROJECT_CONTEXT.md §7 우선순위 2 기준으로 만든 rollout 기록 템플릿입니다.

## 컬럼 설명

| 컬럼 | 설명 |
|---|---|
| `date` | rollout 진행 날짜 |
| `tester` | 실행자 |
| `policy_checkpoint` | 사용한 체크포인트 (예: `act_my_task_step30000`) |
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
