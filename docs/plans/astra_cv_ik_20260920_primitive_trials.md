# 2026-09-20 primitive 보정 실험

작업 호스트: `ehdrms@orin-1`, worktree `astra-cv-ik-calibration`.
기록: `experiments/astra_cv_ik/live-session/primitive-rounds/`.
아직 5개 scatter/gather 한 라운드를 완주하지 않았다. 계획 거절은 파지 실패와 별도로 센다.

## 확인된 문제와 변경

- 일반 이동은 Cartesian 10mm IK 지점마다 move_to/settle을 호출해 정지했다. 관절 경로를 연속 재생하고 끝에서 정착한다. 틱 델타/IO clamp/취소/시간 제한과 실제 FK 경로 이탈 검사를 유지한다.
- 프리미티브의 파지 후 IK가 기존 파지 기울기를 0도로 지웠다. 같은 좌표 (281.8,68.5,50)에서 모델 위치 오차는 0도 6.02mm, -3도 0.35mm였다. 이제 파지 계획의 기울기를 보존한다. 이는 IK 모델 수치이며 실물 정확도 자체가 아니다.
- 하중을 든 상태의 최종 관절 오차를 한 차례, 축당 최대 3도로 보정한다. 누적 학습값이 아니라 매번 현재 관절 피드백으로 계산한다. 느린 보정 후 wood-slow-tracking에서 명령 높이 약54mm 대비 FK 높이53.3mm를 확인했다.
- 하강 ContactMonitor의 signed 변화량 +132 -> -60을 192 증가로 세어 공중 접촉으로 판정했다. primitive는 부하 크기 증가 모드를 사용하며 높은 곳에서 감지되면 테이블 release를 승인하지 않는다. 기존 task runner의 센싱 기본 동작은 유지한다.
- 간섭 후보 탐색에 (forward15,left0), (forward15,left-10)mm를 추가했다. 25mm 기존 한도 안의 실험 후보이며 전역적으로 학습/확정한 보정치가 아니다.

## 실제 결과와 실패

- blue-outward-grasp: 파지/초기 리프트 성공. 추가 수직 상승 IK 거절. 기존 place_at_cell 도구로 (-6,1)에 배치, 카메라 위치오차10.6mm. 전후 wood 위치 변화가 있어 무간섭 성공으로 세지 않는다.
- wood-tilt-lift: 파지 성공, lift 최종 FK 오차로 정지/복구.
- wood-tracking-corrected: 파지 성공, 추가 보정 부하 감지로 정지/복구.
- wood-slow-tracking: 파지/50mm lift 성공. wood-primitive-place에서 (-6,-2) 이동 성공, 그러나 높이62mm에서 접촉 오판 후 release. 공중 release 실패로 기록한다. 이후 magnitude/높이 gate를 추가했다.
- green-scatter: trial_f+10_l+0 후보가 접근/닫힘 간섭 검사 통과. 실제 close_gripper에서 yellow가 약14mm 이동했다. 사용자 관찰과 전후 프레임 검출이 일치한다. 무간섭 파지 실패다. 이어 lift 보정 시 shoulder_lift load126.4 ->208으로 중단. 복구 완료.

## green-yellow 접촉 원인 진단

`green-neighbour-shift.json`: yellow 검출 (279.8,-74.9) -> (275.0,-87.7)mm. 접근/하강 중에는 거의 변하지 않았으며 닫힘 구간에서 이동했다.

URDF mesh를 기존 TCP에 그대로 붙이면 움직이는 턱이 베이스 +x 쪽으로 개폐한다고 예측하나 영상은 -y 쪽이다. `jaw_mount_yaw_deg=90`으로 mesh 방향을 보정했다. 같은 실패 장면 재생(`green-contact-model-replay.json`)에서 0도는 clear, 90도는 yellow/moving_jaw 충돌을 반환한다.

이 값은 영상 기반 방향 정렬이며 전체 3D 형상/원점/개폐 각도 매핑을 완전히 실측한 것이 아니다. `physical_geometry_verified=false`, 공간15mm 및 각도 불확실성을 유지한다. 단일 실패 재현을 전체 안전성 검증으로 해석하지 않는다. 블록 운반 체적과 다른 링크의 충돌은 이 턱 검사에 포함되지 않는다.

## 방향 보정 적용 후

39개 관련 테스트 통과. 제어 서버에 적용했고 camera8090은 유지했다. green-mount-corrected-approach는 후보 거절로 접근하지 않았다. yellow-mount-corrected는 trial_f+0_l-10 후보로 hover까지 접근했으나 하강/닫힘은 하지 않았다.

새 프레임에서 기존 5개 외의 작은 주황색 물체가 yellow 오른쪽에 보였다. 기존 타겟 검출 목록에 포함되지 않아 이 장면의 장애물 집합은 불완전하다. 이 물체의 footprint/높이 또는 보수적 keepout을 반영하기 전 yellow 하강은 금지한다. 빈손 home으로 복귀했다. 전체 반복 태스크 완료/무간섭 성공률 향상은 아직 입증하지 못했다.

## 다음 세션 재개 지점

- 키보드 연속 이동은 같은 RobotWorker에서 다음 IK 경로를 단계별로 미리 계산한다.
  50mm 구간 경계 자체로 감속하지 않는다. 키 해제·방향 변경·검사 실패·계산 지연 시 감속한다.
  32개 관련 테스트와 Orin의 기존/새 IK 수치 일치를 확인했으며 실물 조작감은 아직 미확인이다.
- 구역 안에서 검출한 블록은 웹에서 주황색으로 표시한다.
- 불을 끈 뒤에는 노란색만 검출됐다. 팔을 비킨 뒤 명도 기준/형태 필터 조합 6개를
  시험했지만 주변 네 블록이 복구되지 않아 파지는 진행하지 않았다. 빈손 홈 복귀 완료.
- 카메라와 제어 서버는 유지했다. 다음에는 조명 복구 후 fresh frame으로 블록 다섯 개와
  주변 물체를 다시 확인하고 primitive 보정 실험을 재개한다. 검출 누락을 장애물 부재로 처리하지 않는다.
- 원본 프레임·호출 결과·저조도 진단은 Orin의 experiments/astra_cv_ik/live-session/에 보존한다.
  대량 이미지·실행 로그·가상환경은 Git에 추가하지 않는다.
