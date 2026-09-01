# SO-101 CV+IK 파지·운반 가이드

탑다운 카메라로 블록을 찾아 역기구학으로 집고, 지정한 좌표로 옮기는 경로의
실행 방법과 설계를 정리합니다. ACT 정책 경로(`fsm/handlers.py`)와는 별개이며,
현재 통합 지점은 `tools/demo_pick_and_place.py` 하나입니다.

## 1. 실행 방법

터미널 2개가 필요합니다. 카메라 서버가 `/dev/video*`를 붙잡고 있어서,
도구는 HTTP 스냅샷으로 프레임을 받아옵니다.

**터미널 1 — 카메라 서버 (먼저 실행)**

```bash
cd ~/lerobot_sim2real
./scripts/so101_camera_web.sh
```

`http://<호스트>:8090` 에서 화면을 확인할 수 있습니다. 이 서버가 없으면
도구는 `URLError: Connection refused` 로 죽습니다.

**터미널 2 — 파지·운반 실행**

```bash
cd ~/lerobot_sim2real
PYTHONPATH=src ~/lerobot/.venv/bin/python -m tools.demo_pick_and_place \
  --color green --to P13
```

- `--color` : `red` `yellow` `green` `blue` `wood`
- `--to` : `docs/calibration/points.csv` 의 점 이름 (`P1`~`P15`)
- `--dry-run` : 계획만 출력. 하드웨어에 연결하지 않고 모터도 움직이지 않음
- `--set a.b=c` : 설정 임시 오버라이드 (여러 번 사용 가능)

계획을 출력한 뒤 확인을 받습니다. **엔터를 누르면 팔이 움직입니다.**
`n` + 엔터면 취소입니다.

```
About to move the real arm. Enter to proceed, 'n' to cancel:
```

**반드시 좌표를 눈으로 확인하고 누르세요.** 오타 방어가 없습니다.

### 실행 전 점검

| 항목 | 확인 방법 |
|---|---|
| 카메라가 흔들리지 않았는가 | `python -m tools.camera_drift_check --watch 600` (p95 < 2px) |
| 팔에 토크가 걸려 있는가 | 손으로 밀리면 `./scripts/so101_torque_off.sh` 의 반대 — 재연결 필요 |
| 작업영역이 비어 있는가 | 운반 경로에 장애물이 없어야 함 |

---

## 2. 전체 흐름

```
스냅샷 1장 → 색으로 블록 검출 → homography로 mm 좌표
                                      ↓
                       그리퍼 로컬 프레임 보정  (3절)
                                      ↓
              파지점 후보 7개 IK 사전 계산 (본점 + 대각 4 + 좌우 2)
                                      ↓
            열기 → hover → 하강 → 닫기 → 파지 확인   (4절)
                     ↑___실패시 다음 후보___|
                                      ↓
                  들어올림 → 팔 접기 → 선회 → 뻗기   (5절)
                                      ↓
                            내려놓기 → 홈 복귀
```

성공하면 종료코드 0, 모든 후보가 실패하면 1입니다.

---

## 3. 파지점 보정 — 그리퍼 로컬 프레임

### 왜 보정이 필요한가

검출기는 블록 **중심**을 알려주지만, 실제로는 그리퍼가 블록의 **로봇 쪽
가까운 모서리**(40mm 중 5~10mm 지점)를 물었습니다. 원하는 건 중심(20mm)이므로
조준점을 바깥으로 밀어야 합니다.

여기에 더해, 작업영역 **좌측 절반의 성공률이 우측보다 낮습니다.** 좌측은
추가 보정이 필요합니다.

### 왜 "그리퍼 기준" 전후좌우인가

`control/ik.py`의 `pan0 = -degrees(atan2(y, x))` 가 말해주듯, 팔은 항상
베이스에서 목표를 향해 조준합니다. 그리고 `solve(yaw_deg=None)` 경로가
`wrist_roll`을 작업영역 전체에서 ±3.7도 안에 묶어둡니다(AGENTS.md §7).

따라서 그리퍼가 보는 방향은 **판 좌표축이 아니라 반경/접선 방향**입니다.

- **반경 방향** `(cos φ, sin φ)` = 그리퍼 기준 "앞 / 멀리"
- **접선 방향** `(-sin φ, cos φ)` = 그리퍼 기준 "좌측"   (φ = atan2(y, x))

판 중앙에서는 두 축이 일치하지만, 좌우로 갈수록 팔과 함께 회전합니다.
**베이스 프레임에 고정한 오프셋을 쓰면 가장자리에서 대각선으로 흘러갑니다.**

실측 예 — 동일한 "좌측 10mm" 명령의 결과:

| 위치 | (x, y) 변화량 |
|---|---|
| 우측 절반 | (+4.0, +9.2) |
| 좌측 절반 | (−4.8, +8.8) |

구현은 `control/ik.py`의 `gripper_frame_offset()` 한 함수이며, placo 없이
동작하는 순수 삼각함수입니다.

### 보정값 합성

`control/grasp.py`의 `biased_grasp_xy()`:

```
반경   = grasp_radial_offset_mm      (+ 좌측이면 left_half_radial_offset_mm)
접선   = grasp_tangential_offset_mm  (+ 좌측이면 left_half_tangential_offset_mm)
```

좌/우 판정은 **보정 전 검출 좌표**로 합니다. 블록이 어느 쪽에 있는지는
블록의 성질이지, 보정의 결과가 아니기 때문입니다.

기본값으로 우측은 12mm(순수 반경), 좌측은 24.2mm = hypot(22, 10) 이동합니다.
**좌측 합계 22mm는 문서화된 허용오차 ±15mm를 넘습니다** — 7절의 순서대로
단계적으로 튜닝하세요.

---

## 4. 재시도 로직

### 원칙: 닫아봐야 안다

**하강 결과와 무관하게 항상 그리퍼를 닫고 `check_grasp`로 판정합니다.**
하강이 몇 mm 짧게 끝나도 닫으면 잡힐 수 있고, 닫지 않고 실패로 처리하면
실제로는 시도한 적 없는 위치를 재시도하게 됩니다.

각 후보마다 완전한 사이클을 돕니다:

```
열기 → hover 이동 → 하강 → 닫기 → check_grasp → (실패) 열기 → hover 복귀 → 다음 후보
```

파지 판정은 `control/sensing.py`의 `check_grasp` — 그리퍼 `Present_Position`과
`Present_Load`를 함께 봅니다. 빈손이면 끝까지 닫히고(위치 ~3.4), 블록을 물면
중간에 멈춥니다(위치 ~44, 부하 포화).

### 후보 순서

`control/grasp.py`의 `plan_grasp_attempts()`가 **팔이 움직이기 전에** 모든
후보의 IK를 미리 풉니다. 모터 제어 중에 IK를 푸는 지연을 피하고, `--dry-run`
으로 전 후보의 도달 가능 여부를 미리 볼 수 있습니다.

| 순서 | 이름 | (반경, 접선) mm |
|---|---|---|
| 1 | centre | (0, 0) |
| 2 | front-left | (+10, +10) |
| 3 | front-right | (+10, −10) |
| 4 | back-left | (−10, +10) |
| 5 | back-right | (−10, −10) |
| 예비 | lateral-left / lateral-right | (0, ±10) |

**하강이 짧게 끝나면**(`blocked`) 좌우 후보가 대각 후보들보다 **앞으로**
끼어듭니다. 높이는 맞았고 좌우만 틀렸다는 뜻이기 때문입니다. 한 번만
삽입되며, 이후에는 원래 순서대로 진행합니다.

**도달 불가 후보는 실행을 중단시키지 않고 큐에서 빠집니다.** IK 게이트를
반드시 통과해야 하는 것은 첫 후보와 내려놓기 웨이포인트뿐입니다.

### 왜 이 방식인가

절대 정확도는 이미 정량화되어 있습니다 — RMS 11.98mm, LOO 최악 28.56mm.
그리고 `docs/report/CV_IK_전환_정리.md` §4에서 **오차가 위치별 계통오차가
아니라 점 단위 무작위**임을 검증했습니다(가설 4 기각). 캘리브레이션을
더 손봐도 줄지 않으므로, **조준을 더 정확히 하는 대신 허용오차(±15mm) 안에서
여러 점을 시도하는 것**이 이 오차 성격에 맞는 대응입니다.

### 하강 막힘 판정

`control/trajectory.py`의 `descend()`가 담당합니다. `move_to`와 두 가지가
다릅니다.

1. **보간 루프 안에서 센서를 읽지 않습니다.** fps=30에서 매 틱 버스를
   읽으면 시리얼 왕복 때문에 루프가 느려져 `move_timeout_s` 예산을 다
   먹고 팔이 중간에 멈춥니다. 명령 스트림은 `move_to`와 완전히 동일합니다.
2. **짧게 끝난 것을 예외가 아니라 반환값으로 알립니다.** 그리퍼가 블록
   위에 얹히는 것은 재시도할 정상 상황이지 에러가 아닙니다.

판정은 하강 후 **목표 대비 관절 미달량**(`descent_blocked_tol`)입니다.
정착 예산(`descent_settle_s`)은 tol에 들어오는 즉시 빠져나오므로 정상
하강에는 비용이 없고, 진짜 막혔을 때만 서보가 버티는 시간을 제한합니다.

---

## 5. 파지 후 들어올리기와 운반

### 핵심 제약: 높이는 반경으로 산다

`wrist_flex`가 **작업영역 전 구간에서 URDF 한계(±95도)에 붙어 있습니다.**
그리퍼를 수직으로 유지한 채로 올릴 수 있는 높이는 이것으로 결정되며,
반경에 따라 급격히 변합니다.

| 반경 | 탑다운 최대 높이 |
|---|---|
| 195mm | **90mm** |
| 240mm | 80mm |
| 260mm | 70mm |
| 285mm | **50mm** |
| 300mm | 20mm |
| 320mm | 도달 불가 |

**`hover_clearance_mm: 120` 은 작업영역 어디서도 달성할 수 없는 값입니다.**
`highest_reachable_hover()`가 이 값에서 시작해 `hover_search_step_mm`씩
내려가며 실제로 풀리는 높이를 찾습니다.

### 팔을 접어서 높이를 번다

멀리 있는 블록을 집은 반경 그대로 선회하면 50mm 높이로 끌고 갑니다.
그래서 운반 전에 **같은 방위각을 유지한 채 반경만 줄입니다**
(`pull_in()` → `transit_apex_radius_mm`).

```
파지 → 들어올림 → apex_pick(접기) → apex_place(선회) → place_hover(뻗기) → 내려놓기
```

P7→P13 실측 경로를 FK로 계산한 결과:

| 구간 | 개선 전 | 개선 후 |
|---|---|---|
| 선회 중 높이 | 49~58mm | **89~90mm** |

49mm로 내려가는 것은 이제 **목표점 바로 위에서 수직으로만** 일어납니다
(`shoulder_pan` 4.5 → 4.1, 방위각 이동 없음).

> 운반 중에도 그리퍼를 수직으로 유지하기 때문에 90mm가 천장입니다.
> 파지와 릴리스에서만 수직이 필요하므로, 더 높이 들려면 `_topdown_pose()`에
> pitch를 도입해 **운반 중에는 수직 자세를 포기**해야 합니다. 미착수 과제입니다.

---

## 6. 설정값

전부 `src/configs/default.yaml`의 `motion:` 아래에 있고,
`--set motion.<이름>=<값>` 으로 임시 변경할 수 있습니다.
**dataclass(`config.py`)와 yaml을 반드시 함께 수정해야 합니다** —
로더가 모르는 키를 거부합니다.

| 이름 | 기본값 | 의미 |
|---|---|---|
| `grasp_radial_offset_mm` | 12.0 | 전역 반경 보정. **가장 먼저 튜닝할 값** |
| `grasp_tangential_offset_mm` | 0.0 | 전역 좌우 보정 |
| `left_half_y_mm` | 0.0 | 이 값보다 y가 크면 좌측 절반 |
| `left_half_radial_offset_mm` | 10.0 | 좌측 절반 추가 전방 보정 |
| `left_half_tangential_offset_mm` | 10.0 | 좌측 절반 추가 좌측 보정 |
| `grasp_retry_offsets_mm` | 대각 4개 | 재시도 위치 (반경, 접선). `[]` 로 비활성화 |
| `blocked_descent_offsets_mm` | `[10, -10]` | 하강 막힘 시 우선 시도할 좌우 보정 |
| `descent_blocked_tol` | 4.0 | 이만큼 못 미치면 "막힘" 판정 |
| `descent_settle_s` | 5.0 | 하강 정착 예산 |
| `hover_search_step_mm` | 5.0 | 높이 탐색 격자 |
| `transit_apex_radius_mm` | 195.0 | 운반 시 접을 반경. 0이면 비활성화 |

---

## 7. 실기 튜닝 순서

**한 번에 다 켜지 마세요.** 좌측에서는 보정이 합산되어 22mm가 되므로,
전역값을 먼저 확정해야 원인을 분리할 수 있습니다.

**(a) 전역 반경 보정 — 우측 블록으로**

```bash
PYTHONPATH=src ~/lerobot/.venv/bin/python -m tools.demo_pick_and_place \
  --color green --to P8 \
  --set motion.left_half_radial_offset_mm=0 \
  --set motion.left_half_tangential_offset_mm=0 \
  --set motion.grasp_retry_offsets_mm='[]' \
  --set motion.grasp_radial_offset_mm=12
```

8~14mm를 훑으며 "블록 중심을 무는" 값을 찾습니다. 가까운 모서리를 계속
물면 올리고, 반대편 모서리를 밀어내면 내립니다.

**(b) 좌측 보정 — (a)값 고정 후 좌측 블록으로**

```bash
  --set motion.left_half_radial_offset_mm=10 \
  --set motion.left_half_tangential_offset_mm=10
```

좌측 5회 이상. 경계(y≈0) 근처 블록으로 우측 성공률이 떨어지지 않았는지도
확인합니다.

**(c) 재시도·운반 활성화**

일부러 10~15mm 빗나가게 두고 재시도가 붙잡는지 봅니다. 그리고 운반 중에
블록이 바닥에 끌리지 않는지 확인합니다.

**(d) 성공률 측정**

`docs/report/CV_IK_전환_정리.md`의 미완료 항목입니다. 좌/우 각 10회 이상,
변경 전후를 `docs/eval/`에 기록하세요.

---

## 8. 문제 해결

| 증상 | 원인 / 조치 |
|---|---|
| `URLError: Connection refused` | 카메라 서버 미실행. 터미널 1에서 `./scripts/so101_camera_web.sh` |
| `No <color> block found` | 블록이 화면 밖이거나 HSV 범위 밖. `view_detect`로 확인 (아래) |
| `ABORT: a required waypoint exceeds the IK error gate` | 목표가 도달 범위 밖. 5절의 반경-높이 표 확인 |
| 바닥까지 안 내려감 | `--set motion.descent_settle_s=8`. 서보 정착이 느린 것이지 조기 정지가 아님 |
| 닫지 않고 계속 옆으로만 이동 | `descend()`가 매 틱 센서를 읽던 시절의 버그. 재발 시 `tests/test_carry.py`, `tests/test_grasp.py` 회귀 테스트부터 확인 |
| 운반 중 블록이 끌림 | `--set motion.transit_apex_radius_mm=170` 으로 더 접기. 단 90mm가 천장 |
| `Relative goal position magnitude had to be clamped` | lerobot의 `max_relative_target` 안전 클램프. 그리퍼 닫힘 시 정상 동작 |

검출이 의심스러울 때 주석 이미지를 남겨 확인합니다(색 인자는 없고 전 색을
한 번에 처리합니다).

```bash
PYTHONPATH=src ~/lerobot/.venv/bin/python -m tools.view_detect \
    --camera /dev/video0 \
    --calib src/configs/calib/venue_lab.json \
    --out /tmp/detect.png
```

### 테스트

하드웨어·placo 없이 도는 테스트입니다.

```bash
cd ~/lerobot_sim2real
PYTHONPATH=src ~/lerobot/.venv/bin/python -m pytest tests/ -q
```

| 파일 | 범위 |
|---|---|
| `tests/test_grasp.py` | 그리퍼 로컬 오프셋, 좌/우 판정, 후보 순서, 재시도 큐, **항상 닫기** |
| `tests/test_carry.py` | 반경 접기, 높이 탐색, 바닥값 검증 |
| `tests/test_trajectory.py` | `descend()` 도달/막힘 판정, **보간 루프에서 센서 미읽기** |
| `tests/test_ik.py` | 실제 URDF 기반 IK (placo 필요, 없으면 스킵) |

---

## 9. 확장 계획 — 5색 블록을 한 구역에 모으기

빨강·노랑·파랑·초록·나무 블록을 빨간 테이프로 표시한 구역 안에 모으는
작업입니다. 구역은 **가로 20cm × 세로 10cm, 테이프 굵기 2cm**.

**구역의 정확한 좌표는 테이프를 붙인 뒤 측정해야 합니다.** 아래는 붙이기
전에 미리 정해둘 수 있는 것들입니다.

### 9.1 [블로커] 빨간 테이프가 빨간 블록 검출을 망가뜨립니다

현재 도구가 쓰는 `find_block_centroid()`는 **가장 큰** 빨간 덩어리를
고르고 상한이 없습니다. 합성 영상으로 확인한 결과:

| 상황 | `RETR_EXTERNAL` 윤곽 수 | 검출 오차 |
|---|---|---|
| 블록이 **구역 안**에 있을 때 | **1개** (테이프뿐) | 168px |
| 블록이 **구역 밖**에 있을 때 | 2개 (테이프를 선택) | 597px |

구역 안의 빨간 블록은 **아예 보이지 않습니다.** `RETR_EXTERNAL`은 가장
바깥 윤곽만 반환하는데, 테이프 사각형이 블록을 감싸버리기 때문입니다.
2.6px/mm 기준 168px ≈ 65mm 로, 허용오차 ±15mm를 크게 벗어납니다.

**대응 방향** — `perception/detector.py`의 `detect_blocks()`로 갈아타기.
이쪽은 형상 게이트가 있어 테이프를 걸러냅니다.

| 게이트 | 값 | 테이프(≈26,300mm²) |
|---|---|---|
| `area_mm2_min/max` | 900 ~ 2600 | **탈락** |
| `aspect_ratio_max` | 1.6 | 띠 조각이면 탈락 |
| `fill_min`, `solidity_min` | 0.65 / 0.85 | 속 빈 윤곽이면 탈락 |

다만 `detect_blocks()`는 `find_block_centroid()`와 **HSV 범위가 다르고**
(`config.yaml`을 읽음), 내부적으로 `rectify()`를 거칩니다. 교체 시
재캘리브레이션 없이 되는지 먼저 `--dry-run`으로 확인하세요.
`detect_blocks()`는 블록 각도(`minAreaRect`)도 버리고 있으므로,
`TopDownIK.grasp_yaw_deg()`를 쓰려면 `BlockDetection`에 각도 필드 추가가
필요합니다(현재 프로덕션 호출자 없음).

### 9.2 구역 위치 제약 — 도달 범위

구역 네 모서리에 z=20mm로 내려놓을 수 있어야 합니다. 실측 결과:

| 구역 x 범위 (가로 20cm 배치) | 모서리 반경 | 결과 |
|---|---|---|
| 170 ~ 270mm | 197 ~ 288mm | 전부 도달 |
| **190 ~ 290mm** | 215 ~ 307mm | **전부 도달 (권장)** |
| 210 ~ 310mm | 233 ~ 326mm | 먼 모서리 2개 도달 불가 |

세로 20cm로 돌려 배치하면 x 110~310mm까지는 가능하지만, 폭이 10cm로
좁아져 블록 5개를 늘어놓기 어렵습니다.

> 참고: P8은 x=206.6mm, P13은 x=284.9mm. **"P8~P13 높이"라는 감은
> 맞습니다** — 다만 시작을 x≈190mm로 잡고 x≈290mm를 넘기지 마세요.

그리고 5절의 표대로, 구역 바깥쪽(x≈290mm)에서는 들어올릴 수 있는 높이가
**50mm뿐**입니다. 블록을 이미 놓은 위에 다른 블록을 스치지 않게 하려면
내려놓는 순서를 **먼 쪽부터 가까운 쪽으로** 잡는 편이 안전합니다.

### 9.3 테이프를 붙인 뒤 할 일

1. **드리프트 재검사** — 테이프 작업 중 카메라를 건드렸을 수 있음
   ```bash
   python -m tools.camera_drift_check --watch 600
   ```
2. **구역 좌표 등록** — 전용 도구가 이미 있습니다. 저장된 프레임에서 테이프
   안쪽 네 모서리의 픽셀 좌표를 읽어 다시 실행하면 됩니다.
   ```bash
   python -m tools.calibrate_homography \
       --image src/configs/calib/venue_lab.json.frame.png \
       --square-mm 25 --venue lab \
       --base-px <bx,by> --zone-px "x1,y1 x2,y2 x3,y3 x4,y4" \
       --out src/configs/calib/venue_lab.json
   ```
   `PlaneCalibration.zone_polygon_mm`(캘리브레이션 JSON 안, 현재 `null`)에
   mm 좌표로 저장됩니다. `PerceptionConfig`가 아니라 캘리브레이션 파일입니다.
3. **빨강 검출 재확인** — 테이프가 실제로 걸러지는지. `view_detect`는 색을
   고르지 않고 `perception.hsv_ranges`의 전 색을 한 번에 검출해 주석 이미지를
   남깁니다. 빨간 블록만 잡히고 테이프는 빠져야 합니다.
   ```bash
   PYTHONPATH=src ~/lerobot/.venv/bin/python -m tools.view_detect \
       --camera /dev/video0 \
       --calib src/configs/calib/venue_lab.json \
       --out /tmp/detect.png
   ```
   게이트를 바꿔가며 시험할 때는 YAML을 고치지 말고
   `--set perception.area_mm2_max=2600` 처럼 넘깁니다.
4. **내려놓기 슬롯 정의** — 구역 안에 블록 5개 자리를 잡습니다. 블록 40mm에
   그리퍼 조우 70mm이므로 **중심 간격 최소 70~80mm**가 필요합니다.
   가로 20cm(테이프 안쪽 16cm)면 한 줄에 2개가 한계이니, 2×2+1 또는
   2열 배치를 검토하세요.

### 9.4 이미 만들어져 있는 것

구역을 등록해두면 아래가 자동으로 따라옵니다.

- **놓은 블록은 다시 고르지 않습니다.** `perception/select.py`의 `_in_zone()`이
  구역 안(또는 `select.zone_margin_mm=20mm` 이내)의 검출을 후보에서 제외합니다.
  "모으기" 작업에서 이미 옮긴 블록을 다시 집는 것을 막아줍니다.
- **시각 확인.** `view_detect.py`와 `calibrate_homography.py`가 구역 다각형을
  오버레이로 그려줍니다.

### 9.5 아직 없는 것

- **여러 블록을 순차 처리하는 루프.** 현재 도구는 색 하나를 한 번 옮기고
  끝납니다. `fsm/` 상태머신이 이 역할을 하도록 설계되어 있으나
  (`SELECT → PICK → VERIFY → TRANSPORT → PLACE`), CV+IK 경로용
  `fsm/ik_handlers.py`는 **아직 없습니다**(AGENTS.md §14.1에 따라
  `fsm/handlers.py`는 수정 금지, 추가로만 작업).
  `control/grasp.py`는 그때 `IkPickState`가 그대로 재사용할 수 있도록
  도구에서 분리해 둔 것입니다.
- **놓을 자리가 비었는지 확인.** 같은 칸에 두 번 놓는 것을 막는 로직이 없습니다.
- **블록 각도 정렬.** `grasp_yaw_deg()`는 구현되어 있으나 호출자가 없습니다.

---

## 참고 문서

- `docs/report/CV_IK_전환_정리.md` — 전환 배경, 캘리브레이션 정확도 측정,
  배제한 가설 4가지
- `AGENTS.md` §6(좌표계) §7(IK·yaw 중립) §10(파지 검증) §12(설정) §14.1(수정 금지 범위)
- `src/README.md` — 패키지 전체 구조
