# Docker — Eval 서버 제출 이미지

> **보관 문서**: ACT policy server 제출용 Docker 경로는 현재 CV/IK 실행
> 범위에서 제외했습니다. 이 문서는 과거 제출 환경 기록입니다.

최종 제출물은 eval 서버(x86_64, `nvidia-driver-580-open`)에 로드되는 Docker
이미지입니다 (AGENTS.md §2.7). 이 이미지는 **policy_server + 학습된 ACT
체크포인트**를 담고, Orin 쪽 FSM(`pick_stack`)이 gRPC로 붙습니다.

```
eval 서버 (이 이미지)                Jetson Orin (별도 세팅)
--------------------                --------------------
lerobot policy_server  <--gRPC-->   pick_stack FSM + robot
+ /models/pretrained_model          (scripts/pickstack_task*.sh)
```

## 설계 결정

- **lerobot 커밋 고정**: `scripts/setup_lerobot_env.sh`의 `LEROBOT_COMMIT`과
  동일한 커밋을 이미지 안에서 체크아웃 — 클라이언트/서버 프로토콜 불일치 차단.
  둘 중 하나를 올리면 반드시 같이 올릴 것
- **CUDA는 torch 휠에 동봉**: base는 plain `python:3.12-slim`. 호스트에서
  필요한 것은 드라이버(`--gpus all`)뿐이라 드라이버 580 환경에서 그대로 동작
- **모델은 빌드 시 베이크**: 런타임 마운트 없이 `docker run` 한 번으로 기동
  (zero manual setup 요건)

## 빌드 & 실행 (WSL/데스크톱 x86 GPU 머신에서)

```bash
cd ~/lerobot_sim2real

# 1) 학습된 체크포인트로 빌드 (models/로 스테이징 후 docker build)
./docker/build_policy_server.sh \
  ~/lerobot/outputs/act_pick_v1/checkpoints/last/pretrained_model

# 2) 실행 (nvidia-container-toolkit 필요)
./docker/run_policy_server.sh

# 3) 다른 셸에서 포트 확인
ss -ltnp | grep :8080
```

Orin 쪽 설정:

```bash
./scripts/pickstack_task1.sh \
  --set policy.server_address=<서버IP>:8080 \
  --set policy.pretrained_name_or_path=/models/pretrained_model
```

`pretrained_name_or_path`는 **컨테이너 내부 경로**(`/models/pretrained_model`)
입니다 — Orin 경로가 아닙니다.

## 조기 스모크 테스트 (필수, AGENTS.md §10)

모델이 나오기 전에도 배관은 검증할 수 있습니다. 아무 ACT 체크포인트(과적합
pilot 모델이면 충분)로 빌드 → 실행 → Orin에서 `nc -vz <서버IP> 8080` →
`pickstack_task1.sh` dry-run으로 `connect()`(모델 로드)까지 확인하세요.
CUDA/드라이버 불일치를 데모 직전에 발견하는 것이 최악의 시나리오입니다.

## 제출 시

```bash
docker save pickstack-policy-server:v1 | gzip > pickstack-policy-server-v1.tar.gz
# eval 서버에서: docker load < pickstack-policy-server-v1.tar.gz
```
