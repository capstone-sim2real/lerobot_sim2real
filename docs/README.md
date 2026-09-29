# 문서 안내

## 시스템

- [architecture.md](architecture.md): 현재 실행 경로와 모듈 구성
- [design.md](design.md): 설계 근거 — 좌표계, 캘리브레이션, IK, 안전장치, 데이터 수집

## 운영 가이드 (`guide/`)

| 문서 | 내용 |
|---|---|
| [setup.md](guide/setup.md) | 장비 세팅, 설치, 서보·카메라 점검 |
| [calibration-tools.md](guide/calibration-tools.md) | 캘리브레이션 점 기록, homography·구역·격자 적합 |
| [cv-ik-pick-place.md](guide/cv-ik-pick-place.md) | `so101-run` Task 1/2 CV+IK 파지·운반 |
| [web-agent.md](guide/web-agent.md) | `so101-agent`(LLM), `so101-panel`(수동 조작), 키보드 조작 |
| [task3-data-collection.md](guide/task3-data-collection.md) | `so101-collect` 학습용 에피소드 자동 수집 |
| [teleoperation.md](guide/teleoperation.md) | SSH 키보드 원격 조작 |
| [remote-camera.md](guide/remote-camera.md) | 카메라 서버와 원격 영상 |
| [troubleshooting.md](guide/troubleshooting.md) | 문제 해결 |
| [yoloe-experiment.md](guide/yoloe-experiment.md) | YOLOE 블록 검출 실험(선택) |

## 캘리브레이션 자료

- 현재 적용본: `src/configs/calib/venue_lab.json` (수치는 항상 파일의 `meta`를 본다).
  2026-09-13 9점 fit, 적용 점 목록은 `experiments/NEW_SESSION/calibration/points.csv`.
- 이전 적용본: `src/configs/calib/venue_lab.pre-recalibration-20260911.json`.
- 원본 사진과 점 CSV: [experiments/](../experiments/README.md).
- `board_grid`는 `tools.calibration.calibrate_board_grid`가 같은 파일에 기록한다.
  homography를 다시 맞췄으면 격자도 다시 잰다.

## 평가 기록

[eval/eval_log_template.csv](eval/eval_log_template.csv)에 조건당 10회 rollout을
기록한다. 초기 위치 세트는 조건 내내 고정하고, 한 번에 한 변수만 바꾼다.
`notes`에는 Git commit, `flow`, 캘리브레이션 파일, `--set` override, 카메라 드리프트
결과, 외부 timeout 사용 여부를 적는다. Task 1 러너는 내부 시간 예산을 끄므로 180초
평가는 외부 supervisor 명령도 함께 남긴다. 성공률은 `overall_success 합 / 10`이다.

## 보고서와 발표

- [report/](report/): 착수·중간·최종보고서, 세미나 발표자료, 자문의견서
- [presentation/](presentation/): 시스템 구성도, 발표 준비 노트, 발표자 노트
- [poster/](poster/): 졸업과제 포스터
- [assets/](assets/): README 이미지와 평가 그래프

## 과거 기록 (`legacy/`)

ACT/SmolVLA 모방학습 시도와 손목 카메라 보정 계획은 당시 상태 그대로 보존한다.
현재 실행 방법으로 해석하지 않는다.

- [act-data-collection.md](legacy/act-data-collection.md): 텔레옵 데이터 수집 관리
- [act-smolvla-training.md](legacy/act-smolvla-training.md): 학습·추론 기록
- [wrist-camera-calibration-plan.md](legacy/wrist-camera-calibration-plan.md): 손목 카메라 보정 실험 계획
