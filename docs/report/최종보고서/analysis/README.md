# 최신 결과와 캘리브레이션 재현

최신 미션·파지 합계와 집계 정의는 [`../evidence/20260914/team_results.json`](../evidence/20260914/team_results.json)에 있다. 1차 28/30회와 148/170회, 2차 20회의 수행 중 최대 높이와 94/113회를 보존한다. 시행별 CSV는 제공되지 않아 생성하지 않았다.

저장소 루트에서 프로젝트 환경으로 실행한다. 기존 수치 함수를 재사용하며 로봇·카메라에 연결하거나 활성 프로필을 변경하지 않는다.

```bash
PYTHONPATH=src .venv/bin/python docs/report/최종보고서/analysis/summarize_calibration.py
```

`../evidence/calibration/manifest.json`의 4개 날짜·8개 스냅샷 해시를 확인하고, 저장 H와 재적합 H 및 RMS·최대 LOO를 대조한다. NumPy/OpenCV와 기존 프로젝트 모듈이 필요하다. PDF 빌드 자체는 아래 생성물을 사용하므로 재집계 환경 없이 가능하다.

| 생성물 | 내용 |
|---|---|
| `calibration_profiles_summary.json` | 9월 1·8·11·13일의 RMS, 최대 LOO, FK z 평균·모표준편차와 점별 오차 |
| `calibration_latest_residuals.csv` | 최신 9점 잔차와 LOO |
| `../figures/calibration_history_table.tex` | 초기 팀 집계와 네 프로필의 비교 표 |
| `../figures/calibration_latest_table.tex` | 최신 점별 잔차 표 |
| `../figures/calibration_latest_map.tex` | 최신 FK/평면 변환 좌표, 실제 축척의 잔차와 등록 영역 |

아래 절의 9월 8일 로그 집계와 `calibration_map.tex`는 과거 진단 자료다. 최신 30회/20회와 합치지 않는다.

# 9월 8일 보관 실험의 재집계

보고서 루트에서 `python3 analysis/summarize_evidence.py`를 실행한다. Python 표준 라이브러리만 사용하며 하드웨어, LeRobot과 네트워크에 접근하지 않는다. PDF 빌드는 생성된 TeX 그림을 사용하므로 Python을 실행하지 않아도 가능하다.

입력은 `evidence/20260908/`의 원본 사본이다. 출처는 원본 체크아웃 `/home/ehdrms/workspace/lerobot_sim2real/experiments/2026-09-08/session_20260908_AcC1p4/`이며, 로그/대응점/행렬/적합 결과는 그 아래 `teleop_ehdrms_ox0ydl6s/`, 세션 README는 바로 아래에서 복사했다. 해시는 `source_hashes.json`에 기록한다.

## 산출물과 해석

| 파일 | 내용 |
|---|---|
| `task1_candidates.csv` | 후보별 시작, 센서 결과, 원본 행 번호. 마지막 후보는 접근 중 중단되어 센서 결과 없음 |
| `task1_cycles.csv` | PICK 진입 6회의 구간, 후보 수, 판정과 슬롯 해제 |
| `task1_transitions.csv` | 원본 타임스탬프로 계산한 상태 전이 시각 |
| `task1_summary.json` | 집계 수치와 알려진 종료 범위. 인정 블록 수와 물리적 정지 시각은 `null` |
| `calibration_residuals.csv` | 저장된 H를 픽셀 대응점에 적용한 좌표와 FK 좌표의 차이 |
| `../figures/task1_timeline.tex` | 상태별 시간 흐름, PICK 내부 후보 경계 |
| `../figures/calibration_map.tex` | 작업 평면의 대응점과 실제 크기의 적합 잔차 화살표 |

집계 단위는 후보 실행이다. 계획만 하고 제외된 후보는 세지 않는다. 센서 결과가 없는 중단을 EMPTY로 세지 않으며, VERIFY의 재확인을 새 후보로 중복 집계하지 않는다. 로그의 `placed 0` 값 대신 명시적인 `released_slot` 전이를 세었고, 이것도 물리적인 인정 블록 수를 뜻하지 않는다.

시간축은 `03:10:05.080` 미션 시작 로그가 0이다. 상태 전이 메시지의 반올림된 `elapsed`와 수십 ms 차이가 날 수 있다. 외부 제한은 초기화부터 계산되므로 그래프의 180초 지점을 실제 종료 시각으로 사용하지 않는다. 파랑 막대는 마지막 동작 로그까지이며 점선은 이후 중단 사실만 표현한다. disconnect 로그도 물리적 모터 정지 시각을 증명하지 않는다.

잔차 화살표는 기록 FK 좌표에서 H로 얻은 좌표를 향한다. 새 homography를 적합하거나 실패 점을 제외하지 않았다. 저장 행렬과 대응점에서 재계산한 RMS가 저장값과 일치하는지 검사한다. LOO 값은 `fit_accepted.txt`의 원기록을 유지했고, 그림은 LOO 방향이나 실제 턱 오차를 나타내지 않는다.

커밋 `1fd914b`는 구조 정리 뒤 보관 및 구현 설명의 기준이다. 실기 당시의 정확한 커밋으로 소급하지 않는다. 당시 전체 시험은 30Hz와 슬롯 반경 보정 0mm, 이후 단일 블록 시험은 45Hz였다는 세션 기록을 유지한다.

## 소프트웨어 검증 자료의 범위

원본 체크아웃의 `tests/test_fsm.py`는 빈손 VERIFY의 운반 차단과 도달 가능한 보정 후보 선택 등을 검사한다. `tests/test_task1.py`는 오래되거나 중복된 프레임의 완료 판정 차단, 재검출 시 완료 대기 초기화와 순회 보류를 검사한다. 개발 검증의 기록된 결과는 로컬 94 passed와 IK skip, Orin 99 passed이며 이번 문서 작업에서 다시 실행한 결과가 아니다. 소프트웨어 재현 명령은 해당 구현 체크아웃에서 `uv run --extra dev pytest tests -q`이다. 보고서 브랜치의 코드는 `main` 기반이므로 그 테스트 결과를 동일하게 기대해서는 안 된다.

## 캘리브레이션 절차 그림의 추가 원기록

`evidence/20260908/p4_live.json`은 동일 P4의 관절값과 접촉 자세 사진 이름을 기록한 원본 사본이다. 같은 폴더의 채택 CSV에서 픽셀 및 FK 좌표를, `figures/calibration_p4_contact.jpg`와 `figures/calibration_p4_clear.jpg`에서 두 장면을 확인한다. 사진 해시는 `asset_manifest.json`, 관절 원기록 해시는 `source_hashes.json`에 있다. 집계 스크립트는 이 기록을 수치 분석의 새 시행으로 계산하지 않으며 해시 목록에만 포함한다.

## 지정 영역의 도식 표현

과거 생성물 `figures/calibration_map.tex`의 지정 영역은 규격 200×100mm의 축 정렬 직사각형으로 표현한다. 중심은 등록한 네 꼭짓점의 평균이며 가로축은 Base Y, 세로축은 Base X이다. 이 직사각형은 위치 관계를 보여주는 개략도이며, 원본 `zone_polygon_mm`이나 homography를 다시 보정한 결과가 아니다. 대응점, 잔차 화살표와 모든 수치 분석은 저장 원자료를 그대로 사용한다.
