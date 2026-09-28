# YOLOE 블록 세그멘테이션 실험

Orin 전용 worktree `~/lerobot_sim2real/worktrees/yoloe-block-segmentation`,
브랜치 `feat/yoloe-block-segmentation`. 기준은
`feat/astra-cv-ik-calibration`의 `dbf742b`; 원본 worktree의 미커밋 변경은 포함하지 않았다.

## 실행

로봇 연결 없이 기존 camera.server HTTP snapshot만 읽는다. 모델 학습은 하지 않는다.
기본 backend는 기존 CV다. 에이전트 웹에서 YOLOE를 명시적으로 선택하면
관찰과 PICK 대상 검출에 연결되며, 다시 CV로 즉시 되돌릴 수 있다.

```bash
cd ~/lerobot_sim2real/worktrees/yoloe-block-segmentation
PYTHONPATH=src .venv/bin/python -m tools.yoloe_blocks \
  --output experiments/yoloe/new-text-run
PYTHONPATH=src .venv/bin/python -m tools.yoloe_blocks \
  --visual-prompts experiments/yoloe/visual-reference.json \
  --output experiments/yoloe/new-visual-run --repeats 4
```

`--image path.jpg`로 저장 프레임을 재처리한다. output은 새 폴더여야 한다.
`--repeats`는 **동일 이미지 반복**이며 비디오/다른 배치 정확도 평가가 아니다.
`--set yoloe.confidence=0.2` 등 기존 config 오버라이드를 쓴다.
초기 confidence=0.1은 탐색용 가정값이다.

Visual prompt JSON은 `{"image": "/absolute/reference.jpg", "bboxes": [[x1,y1,x2,y2], ...]}`.
박스는 원본 해상도의 픽셀 좌표이며 모든 예시를 동일 블록 클래스에 매핑한다.
현재 예시는 사람이 스냅샷에서 지정한 빨강/파랑/노랑/나무색 4개다.
추론 이미지와 같은 배치의 이전 프레임에서 지정했으므로 독립 장면 일반화 검증은 아니다.
`object0`는 색상 이름이 아니다. 색상 분류기는 아직 연결하지 않았다.

## 결과 파일과 좌표 의미

- `input.jpg`, `overlay.jpg`, `mask_*.png`: 원본, 표시 이미지, 인스턴스별 이진 마스크.
- `results.json`: YOLOE 실루엣과 `hybrid_color_geometry`, 색상, robot-base
  `candidate_center_mm` (x,y), 형상/workspace 게이트 결과, 추론 시간, 설정/버전/hash.
- `calibration.json`, visual 모드의 `reference_image.jpg`: 재현용 입력 보존.

첫 번째 좌표는 정사영된 YOLOE 실루엣의 최소 회전 사각형 중심이다.
하이브리드 좌표는 YOLOE 인스턴스 마스크로 주변과 같은 색 영역을 먼저 잘라낸 뒤,
그 안에서 기존 HSV gate, prototype 색상 할당, 형상 gate를 실행한 결과다.
빨간 블록과 연결된 빨간 테이프가 같은 컨투어가 되는 문제를 이 경계에서 끊는다.
기존 `PlaneCalibration`, `_evaluate_contour`, 색 prototype을 재사용한다.
마스크 해상도가 calibration과 다르거나 base_xy_mm이 (0,0)이 아니면 거부한다.
같은 색 블록 여러 개를 하나로 강제하지 않으며 YOLOE miss를 기존 CV로 대체하지 않는다.

**하이브리드 색 컨투어도 윗면만 담는다고 보장하지 않는다.** 비슷한 색의 옆면은
남을 수 있으므로 블록 윗면 높이 평면으로 투영한 추정값일 뿐이다. 팔 가림,
적층 높이, 카메라 드리프트에 따른 오차는 해결하지 않았다.
Z를 추정하지 않으며 `grasp_ready=false`를 기록한다.
형상 게이트 통과는 물리 좌표 정확도나 파지 성공의 증거가 아니다.

## Orin 스냅샷 실측

`experiments/yoloe/{visual-live,text-live}/results.json` 기준.
모델 YOLOE-26s-seg, imgsz=640, CPU 4 threads, 동일 입력 반복 4회.

| 조건 | 검출 | 첫 predict | 이후 predict 3회 |
|---|---:|---:|---:|
| Text: wooden block / toy block / cube | 3개 | 2245 ms | 1381–1580 ms |
| Visual: 4개 블록 예시 | 4개 | 5840 ms | 1439–1485 ms |

Visual 결과는 빨강 (230.6,22.7), 파랑 (285.0,-92.4), 노랑 (136.7,-76.1),
나무색 (156.5,-142.1) mm였다. 색은 사람이 이미지와 대조한 설명이다.
신뢰도는 각각 0.874 / 0.684 / 0.528 / 0.185로 나무색은 약하다.
초록은 팔에 가려져 미검출. 위치 ground truth와 비교하지 않았으므로 정확도 수치가 아니다.
Text는 나무색을 놓쳤다. 시간은 predict 호출만이며 모델 import/load/text encoding,
HTTP 다운로드, 후처리/파일 저장은 제외한다. 실시간 0.3/0.5초 성능을 달성하지 않았다.

추가 fresh snapshot `experiments/yoloe/visual-final`에서도 4개를 분할했으나,
나무색은 aspect 검사에서 탈락했다(신뢰도 0.224). 3개만 형상 게이트를 통과했다.
같은 배치에서도 나무색 마스크 경계가 변하므로 검출/좌표 안정성을 확보한 것은 아니다.
최종 코드의 calibration/reference 사본과 패키지 버전 기록은 이 디렉터리에서 확인한다.

`experiments/yoloe/hybrid-replay-v2`에서 같은 실제 프레임을 하이브리드로 재처리했다.
빨강 `(229.1,21.0)`, 파랑 `(281.7,-89.8)`, 노랑 `(134.2,-73.0)` mm가 색상·형상
gate를 통과했다. 나무색 `(156.7,-141.3)` mm는 YOLOE가 분리하고 색도 `wood`로
맞췄지만 aspect gate에서 탈락했다. 빨강은 테이프와 닿아 있어도 별도 컨투어로
남았다. 이 값들은 영상상 경계 확인 결과이며 위치 ground truth나 물리 파지
검증값이 아니다.
최종 재생 결과는 `experiments/yoloe/hybrid-final`에 있다. 프레임 정사영과 HSV
마스크는 인스턴스 사이에서 공유한다. 4개 인스턴스 하이브리드 후처리는 Orin CPU에서
중앙값 47.4 ms, p95 52.7 ms였다(YOLOE 추론 시간은 별도).

## 환경

별도 `.venv` (Python 3.12). `runtime-readonly.pth`가 기존
`~/lerobot_sim2real/related_repos/lerobot/.venv/lib/python3.12/site-packages`를 참조한다.
추가 패키지는 새 환경에만 설치했다. 기존 torch/torchvision 설치는 변경하지 않았다.
공유 기반 환경이 없어지거나 바뀌면 이 환경도 영향을 받는다.
Ultralytics 8.4.157, torch 2.11.0+cu128, torchvision 0.26.0+cu128.
현재 torch는 Orin SM87 커널이 없어 CPU YOLOE에만 사용한다. GPU 경로는 아래의 직접 TensorRT runtime으로 분리했다.

모델/텍스트 인코더는 `models/`, 결과는 `experiments/yoloe/`에 보존하며 git ignore다.
텍스트 인코더는 프롬프트 준비에 사용되고 매 프레임 호출하지 않는다.
실패한 초기 시험 폴더도 남겨 두었다. `visual-live`, `text-live`는 JSON까지 생성된 완료 시험이다.

공식 API: https://docs.ultralytics.com/models/yoloe/

검증: 통합 브랜치 전체 테스트 `134 passed, 6 skipped`. 하이브리드 단위 테스트는 연결된 빨간
테이프 분리, 해상도 거부, 탈락 컨투어 JSON 직렬화를 포함한다.
이 정적 재생 시험 자체에는 로봇 이동이 포함되지 않았다. 아래 에이전트 통합 절의
실장비 1회 시험에서 별도로 파지와 배치를 검증했다.

## TensorRT FP16 경로 (Orin Nano)

JetPack 6.2.1 계열의 시스템 TensorRT 10.3과 CUDA runtime을 직접 사용한다.
현재 공유 torch 2.11.0+cu128 빌드는 SM87 커널을 포함하지 않아 실제 CUDA tensor
생성이 실패한다. 따라서 torch CUDA/Ultralytics TensorRT backend를 거치지 않고,
`perception.yoloe_tensorrt.TensorRTEngine`이 TensorRT와 `cuda-python`으로 엔진을
실행한다. 일반 앱 import에는 TensorRT/CUDA가 필요 없도록 import는 생성자 안에 있다.

전용 환경과 실행 명령은 다음과 같다. `.venv-trt`는 worktree 안에만 있고 git ignore다.

```bash
/usr/bin/python3.10 -m venv --system-site-packages .venv-trt
.venv-trt/bin/python -m pip install 'cuda-python>=12.6,<13' \
  'opencv-python-headless==4.13.0.92'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  .venv-trt/bin/python -m tools.yoloe_tensorrt \
  --output experiments/yoloe/tensorrt-new --repeats 20
```

엔진은 visual reference의 네 박스를 한 `block` class embedding으로 고정한
`models/yoloe-26s-block-vp-seg.onnx`에서 만들었다. 입력은 float32
`1x3x640x640`, 출력은 `1x300x38` detection과 `1x32x160x160` mask prototype이다.
로컬 산출물은 모델과 함께 git ignore다. 빌드 명령:

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=models/yoloe-26s-block-vp-seg.onnx \
  --saveEngine=models/yoloe-26s-block-vp-seg-fp16.engine \
  --fp16 --memPoolSize=workspace:1024 \
  --timingCacheFile=models/yoloe-26s-block-vp-seg.cache \
  --warmUp=500 --duration=5 --noDataTransfers --useCudaGraph
```

`trtexec`의 GPU-only 결과는 평균 7.73 ms, median 7.59 ms, p95 8.69 ms였다.
이는 전송·전처리·마스크 복원·하이브리드 CV를 제외한 값이다. 실제 도구는 pinned
host memory, H2D/D2H, mask 복원과 기존 색/형상 gate까지 포함한다. 저장 프레임
20회에서 전체 median 208.2 ms, p95 230.2 ms였고 4개 인스턴스 중 평면 블록 3개가
통과했다. 따라서 **한 프레임을 순차 처리하는 조건에서는 0.3초 주기 안에 들어왔다.**
카메라 HTTP 수신 대기와 다른 프로세스의 부하는 이 반복 측정에 포함되지 않는다.

ONNX export 결과는 PyTorch 결과와 박스·좌표는 거의 같았지만 confidence scale이
달랐다. 두 실제 프레임에서 비교해 TensorRT 전용 threshold를 0.04로 두었다
(`yoloe.trt_confidence`, 후보값). 저장 프레임 좌표 차이는 약 0.0–0.6 mm였고,
fresh frame에서는 PyTorch와 TensorRT 모두 빨강·초록·나무색을 찾았다. TensorRT 전체 median은 199.6 ms, p95는 254.2 ms였다. TensorRT는
0.04에서 나무색을 포함해 3개 모두 기존 gate를 통과했다. 이 threshold는 두 장에서만
맞춘 값이라 별도 장면 정밀도/재현율 검증이 필요하다. 팔에 가린 노랑과 현재 놓인
파랑은 두 경로 모두 검출하지 못했다.

실행 전후 `/health`에서 camera.server는 `ok=true`, 30 fps였으며 서버를 재시작하거나
카메라 장치를 직접 열지 않았다. 로봇 동작은 수행하지 않았다. 엔진은 해당 Orin의
TensorRT/CUDA 조합에 묶인 로컬 빌드 산출물이며 다른 JetPack/장치로 복사하지 않는다.

## 에이전트 웹의 CV / YOLOE 선택

통합 브랜치의 카메라 카드에는 `검출` 선택기가 있다.

- `CV`: camera.server가 기존 색상+형상 검출을 분석해 보내는 SSE를 표시한다.
- `YOLOE · TensorRT`: agent 서버가 별도 Python 3.10 worker를 한 번 띄우고, 최신
  HTTP snapshot을 TensorRT+하이브리드 검출한 뒤 별도 SSE로 표시한다.

YOLOE worker는 길이 prefix가 붙은 stdin/stdout 프로토콜을 쓰므로 매 프레임 모델을
다시 로드하지 않는다. agent의 Python 환경에는 TensorRT를 import하지 않으며,
`yoloe.worker_python`과 `yoloe.engine`이 둘 다 있을 때만 UI 옵션을 활성화한다.
worker는 camera.server의 snapshot만 읽고 카메라 장치를 열거나 서버를 재시작하지 않는다.
기본 표시 분석률은 `yoloe.web_analysis_fps=2.0`이다.

선택한 backend는 브라우저 오버레이뿐 아니라 `ArmSession.observe()`와 agent의
Task 1 perception에도 적용된다. 따라서 YOLOE를 선택한 뒤 웹에서 관찰/PICK 명령을
실행하면 YOLOE+색상·형상 결과가 기존 workspace gate, IK gate, 재시도, VERIFY를
그대로 통과한다. 전환은 operator token을 가진 사용자가 로봇 IDLE 상태일 때만 가능하다.
실행 중에는 backend를 바꿀 수 없다. 선택 자체는 팔을 움직이지 않으며 실제 동작은
별도의 웹 명령으로 시작한다.

2026-09-29 실장비 1회 시험에서는 YOLOE를 control backend로 선택하고 노란 블록을
`bottom-left` 슬롯으로 옮겼다. PICK의 위치+부하 VERIFY를 통과했고, 운반·해제·HOME
복귀 뒤 fresh 관찰에서 노란 블록이 슬롯에 확인됐다. 측정 위치 `(223.0, 27.0) mm`,
슬롯 중심 오차 16.0mm, 방향 오차 -4.6°, 전체 동작 19.5초였다. 이는 한 장면의 1/1
성공 증거이며 검출기별 성공률 비교나 반복 재현성을 뜻하지 않는다.
