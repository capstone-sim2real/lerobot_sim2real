# SO-101 LLM 에이전트

`so101-agent`는 관찰·접근·정렬·파지·운반·접촉·해제 primitive와 에피소드 수집
primitive를 LLM이 조합하는 러너다. 복합 미션 도구나 skills/primitives 모드 선택은 없다.
`so101-run`과 `so101-collect`는 기존 독립 CLI로 유지한다.

도구 스키마, 수집 순서, 저장 게이트, Orin 실행법과 검증 한계는
[SO101_AGENT_PRIMITIVES.md](SO101_AGENT_PRIMITIVES.md)를 따른다.

공급자는 `--provider openai|anthropic|gemini|fake`, 모델은 `--model`로 지정한다.
SDK와 키는 선택한 공급자에 맞춰 준비한다. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GEMINI_API_KEY`/`GOOGLE_API_KEY`를 환경 또는 `.env`로 전달한다. `--env-file`도 지원한다.
공급자별 이미지 어댑터는 포함되지만 실제 API 검증과 모델별 vision 지원 확인은 별도다.

웹 UI는 운영자 lease와 IDLE에서만 명령을 받으며 STOP은 누구나 요청할 수 있다.
버스·IK·수집은 RobotWorker 하나가 소유한다. 수동 조그/닫기/열기/관찰도 같은
도구 검증 경계를 통과한다. 셀 이동은 `hover`만 수행하며 자동 파지·배치를 하지 않는다.

과거 `agent.tool_mode`, `agent.enable_task3_tool`, `agent.stop_auto_home` 옵션은 사용하지 않는다.
데이터셋 녹화를 제외한 변경은 `so101-agent` 영역에 국한되며, 기존 미션 제어 코드는
에이전트에서 import되는 것 외에 에이전트 로직을 역으로 import하지 않는다.
