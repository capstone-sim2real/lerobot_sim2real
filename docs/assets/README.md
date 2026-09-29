# README 이미지

- `dashboard.png`: 2026-09-29 Orin에서 실행한 `tools.agent_server`의 실제 웹 화면.
- 캡처: Firefox 전체 화면, Niri window screenshot, 2880 × 1800 PNG.
- 상태: 다크 모드, 카메라 연결 정상, 로봇 대기. 작업대는 저조도 상태.
- 화면 합성이나 카메라 영상 교체 없이 캡처했으며, 성능 평가 결과를 나타내는 이미지는 아니다.

## 실기 평가 그래프

- `evaluation-results.png` / `.svg` / `.pdf`: Task 1 보고서 집계와 Task 2·단일 파지 최신 팀 집계 시각화.
- Task 1 원자료: `../report/최종보고서/evidence/20260914/team_results.json`.
- Task 2·파지 원자료: `evaluation-update.json` (README 수정 대화에서 사용자가 제공).
- 재생성: 프로젝트 루트에서 `python3 docs/assets/render_evaluation.py` (Matplotlib, Noto Sans CJK 필요).
- Task 2 시행 횟수와 5초 유지 조건, 파지 재시도 포함 여부는 제공되지 않았다.

- Matplotlib 표준 막대그래프, 공통 성공률 축 0–100%, 평가 항목별 3개 패널. 표본·분산 정보가 없는 항목에 오차막대를 추정하지 않는다.
