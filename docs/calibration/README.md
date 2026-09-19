# 캘리브레이션 자료 위치

이 폴더는 안내 문서만 보관한다.

- [이전 캘리브레이션 자료](../../experiments/legacy/calibration/): 날짜가 확인되지 않은 원본 사진, 점 CSV, 카메라 기준 이미지.
- [2026-09-08 실험](../../experiments/2026-09-08/session_20260908_AcC1p4/README.md): 재고정 후 캘리브레이션과 파지 시험.
- [권장 명령과 파일 역할](../guide/SO101_세션도구.md)
- 현재 적용본: `src/configs/calib/venue_lab.json`.

현재 적용본은 **2026-09-13 재캘리브레이션**의 9점 `fk_direct_pairs` fit이다.
RMS 5.15mm, maximum leave-one-out 13.53mm, grasp-z mean/std 4.06/0.73mm이며,
`zone_polygon_mm`은 2026-09-19에 빨간 테이프 내부 윤곽 edge-fit으로 다시 등록했다
(`zone_source: red_tape_inner_contour_edge_fit_refined`). 정상 gate
(RMS 5mm, LOO 8mm)를 통과한 결과로 오해하지 않는다 — RMS는 경계에 걸치고 LOO는
아직 초과다. 수치는 항상 `venue_lab.json`의 `meta`를 직접 본다.

이전 적용본(2026-09-08, RMS 8.18mm·LOO 26.64mm·grasp-z 16.15/6.82mm)은
`src/configs/calib/venue_lab.pre-recalibration-20260911.json`에 남아 있고, 당시
상태는 위 2026-09-08 세션 README가 기준이다. 그 숫자를 현재 값으로 인용하지 않는다.

`board_grid`(체스판 칸 좌표용 격자)는 `tools/calibrate_board_grid.py`가 같은 파일에
기록한다. 프리뷰로 격자선이 실제 칸 위에 얹히는지 확인한 뒤 `--write` 한다.
homography를 다시 맞췄으면 격자도 다시 잰다.

기존 명령 이름과 옵션은 유지한다. 기본 **입력**은
`experiments/legacy/calibration/`이다. 현재 적용본이 쓴 점 목록은
`experiments/NEW_SESSION/calibration/points.csv`다. 과거 명령에서 `--points`나
`--reference`에 `docs/calibration/...`를 명시했다면 새 경로로 바꾼다.
새 실험은 `--output-dir experiments/<세션>/calibration`을 명시하는 것을 권장한다.

테스트에 필요한 이미지는 `tests/fixtures/`에 별도로 두어 실험 폴더 이동과
독립적으로 테스트가 실행되도록 한다. 이전 자료는 삭제하지 않았다.
