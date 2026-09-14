# 최종보고서

2026-09-14 갱신. SO-101 블록 이동·적재 시스템의 구현과 평가를 정리했다. 초기 모방학습 실험과 CV+IK 선정 근거는 연구 내용에서 설명하며, 실행 자료의 자동 수집과 향후 정책 재학습을 구분한다.

- [제출 PDF](final_report.pdf) (48쪽), [LaTeX 진입점](final_report.tex), 장별 본문 `sections/`.
- [결과·구현의 근거와 해석 범위](SOURCES.md), [빌드·검토 기록](VALIDATION.md).
- [집계와 캘리브레이션 재현](analysis/README.md), [후속 시행별 기록 양식](evaluation/README.md).
- [팀이 확인한 최신 집계](evidence/20260914/team_results.json), [초기 모방학습 실험 순서와 전환 동기](evidence/20260914/learning_history.json).
- [그림 출처와 SHA-256](asset_manifest.json).

`sections/`를 유일한 본문 원본으로 사용한다. `final_report_revised.tex`와 `sections_revised/`는 같은 원본을 읽는 호환 진입점이며, 빌드 시 `final_report_revised.pdf`도 제출 PDF와 동일하게 갱신한다.

서식 기준은 사용자가 지정한 `docs/report/reference/example/report_ko/`이다. 해당 `pnureport.cls`를 원본 바이트 그대로 사용하여 나눔명조 11pt, 줄간격 1.5배, 사방 3cm 여백과 예시의 표지·장 제목·머리말 형식을 적용한다.

## 최신 결과의 집계 기준

| 항목 | 결과 | 해석 |
|---|---|---|
| 1차 미션 | 30회 중 28회 완료, 93.3% | 180초 내 다섯 블록 배치. 실패 2회는 파지 중 블록 밀림→작업 반경 이탈→시간 초과 |
| 1차 미션 파지 | 170회 중 148회, 87.1% | 재시도를 포함한 시도당 성공률 |
| 2차 미션 | 5단 목표 20회: 3단 이상 20회, 4단 이상 17회, 5단 0회 | 수행 중 최대 높이의 누적 도달 횟수. 종료 시 높이·5초 유지율과 구분 |
| 2차 미션 파지 | 113회 중 94회, 83.2% | 재시도를 포함한 시도당 성공률 |
| 9월 13일 캘리브레이션 | RMS 5.15mm, 최대 LOO 13.53mm | 저장된 9개 대응점으로 재계산. 실제 턱 위치의 독립 측정 오차와 구분 |
| Task3 | 1라운드, 5에피소드, 2,111프레임, 186초 | 자동 수집 구현·초기 실행. 정책 재학습 미수행 |

최종 적재는 계산한 자세에서 직접 해제하였다. ArUco 위치 추정과 손목 카메라 색상 게이트는 실험용 구현으로 실제 제어에 사용했으나 기준 main에는 반영되지 않았다. 최신 미션·파지 수치는 팀이 제공한 합계이며, 시행별 기록을 새로 만들어 채우지 않았다.

## 다시 빌드하기

저장소 루트에서 실행한다. PDF만 빌드할 때는 보관된 그림·표를 사용하므로 Python 재집계가 필요 없다.

```bash
bash docs/report/최종보고서/build.sh
```

호스트에는 Podman이 필요하다. 최초 실행은 Ubuntu 24.04 기반 이미지 `localhost/so101-report-tex:ubuntu24.04`를 만들며 네트워크를 사용한다. 이후 pdfLaTeX·BibTeX·latexmk로 컴파일한다. `build/`는 중간 산출물 디렉터리이며 Git에서 제외한다. `Containerfile`을 변경하면 이미지를 다시 빌드한다.

```bash
podman build -t localhost/so101-report-tex:ubuntu24.04 -f docs/report/최종보고서/Containerfile docs/report/최종보고서
```

5개 본문 장과 최신 실험 내용을 유지하고 `report_ko` 예시 서식을 적용했다. 구현 설명의 기준은 main `0394cd054f415adb67bbb39fc5e7ad23afdb3bc8`이며, 과거 실험의 정확한 실행 버전으로 소급하지 않는다. 코드·설정·실험 원본을 수정하거나 새로운 하드웨어 시험을 수행한 작업은 아니다.
