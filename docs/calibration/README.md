# 캘리브레이션 자료 위치

이 폴더는 안내 문서만 보관한다.

- [이전 캘리브레이션 자료](../../experiments/legacy/calibration/): 날짜가 확인되지 않은 원본 사진, 점 CSV, 카메라 기준 이미지.
- [2026-09-08 실험](../../experiments/2026-09-08/session_20260908_AcC1p4/README.md): 재고정 후 캘리브레이션과 파지 시험.
- [권장 명령과 파일 역할](../guide/SO101_세션도구.md)
- 현재 적용본: `src/configs/calib/venue_lab.json`.

기존 명령 이름과 옵션은 유지한다. 기본 **입력**은
`experiments/legacy/calibration/`, 새 캡처의 기본 **출력**은
`experiments/current/calibration/`이다. 과거 명령에서 `--points`나
`--reference`에 `docs/calibration/...`를 명시했다면 새 경로로 바꾼다.
새 실험은 `--output-dir experiments/<세션>/calibration`을 명시하는 것을 권장한다.

테스트에 필요한 이미지는 `tests/fixtures/`에 별도로 두어 실험 폴더 이동과
독립적으로 테스트가 실행되도록 한다. 이전 자료는 삭제하지 않았다.
