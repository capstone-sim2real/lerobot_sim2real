"""One conversation with the LLM: stream a turn, run its tool calls, repeat.

Rules the loop keeps:

- All tool results of one assistant turn go back in ONE user message.
- A robot fault (STOP, motion timeout, bus loss, unexpected error) ends the
  turn at once: remaining calls are answered with a "not executed" result and
  no further LLM request is made, so the model cannot keep commanding an arm
  that may be anywhere.
- ``should_stop`` is checked before every tool call, so STOP pressed while the
  model is still talking also prevents the next motion.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from config import AgentConfig
from session.results import ROBOT_FAULT_REASONS, SkillResult

from .provider.types import Message, Provider, TextDelta, ToolCallEvent, ToolResult, TurnEnd
from .tools import ToolRegistry

logger = logging.getLogger(__name__)

Emit = Callable[[dict[str, Any]], None]


@dataclass
class TurnOutcome:
    robot_fault: bool = False
    stopped: bool = False
    error: str | None = None


class AgentRunner:
    def __init__(
        self,
        provider: Provider,
        registry: ToolRegistry,
        system_prompt: str,
        cfg: AgentConfig,
        *,
        emit: Emit = lambda _event: None,
        should_stop: Callable[[], bool] = lambda: False,
        transcript_dir: str | Path | None = None,
    ):
        self.provider = provider
        self.registry = registry
        self.system_prompt = system_prompt
        self.cfg = cfg
        self.emit = emit
        self.should_stop = should_stop
        self.history: list[Message] = []
        self._transcript: Path | None = None
        if transcript_dir:  # empty disables the transcript
            path = Path(transcript_dir)
            path.mkdir(parents=True, exist_ok=True)
            self._transcript = path / time.strftime("agent_%Y%m%d_%H%M%S.jsonl")

    def reset(self) -> None:
        self.history.clear()
        self._log({"type": "reset"})

    def _log(self, record: dict[str, Any]) -> None:
        if self._transcript is None:
            return
        record = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}
        with self._transcript.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _trim_history(self) -> None:
        limit = self.cfg.max_history_messages
        if len(self.history) <= limit:
            return
        cut = len(self.history) - limit
        # The kept history must open with a plain user message, never with
        # tool results whose calls were cut away.
        while cut < len(self.history) and not (
            self.history[cut].role == "user" and not self.history[cut].tool_results
        ):
            cut += 1
        self.history = self.history[cut:]

    def _not_executed(self, call, reason: str, detail: str) -> ToolResult:
        result = SkillResult(False, call.name, reason, detail, "do_not_retry")
        return ToolResult(call.id, call.name, result.to_envelope(), is_error=True)

    def run_turn(self, user_text: str, *, current_cv: dict[str, Any] | None = None) -> TurnOutcome:
        model_text = user_text
        if current_cv is not None:
            # This is server-produced data, not another user instruction.  A
            # plain text envelope keeps every provider adapter identical while
            # making the observation part of the exact turn that acts on it.
            model_text += (
                "\n\n<current_cv source=\"server_preflight\">\n"
                + json.dumps(current_cv, ensure_ascii=False, default=str)
                + "\n</current_cv>"
            )
        self.history.append(Message("user", text=model_text))
        self._trim_history()
        record: dict[str, Any] = {"type": "user", "text": user_text}
        if current_cv is not None:
            record["current_cv"] = current_cv
        self._log(record)
        outcome = TurnOutcome()
        for _round in range(self.cfg.max_tool_turns):
            text_parts: list[str] = []
            calls = []
            end = TurnEnd("unknown")
            try:
                for event in self.provider.stream_turn(
                    self.system_prompt, self.history, self.registry.specs()
                ):
                    if isinstance(event, TextDelta):
                        text_parts.append(event.text)
                        self.emit({"type": "text_delta", "text": event.text})
                    elif isinstance(event, ToolCallEvent):
                        calls.append(event.call)
                    elif isinstance(event, TurnEnd):
                        end = event
            except Exception as exc:  # noqa: BLE001 - network/API errors end the turn
                logger.exception("LLM request failed")
                outcome.error = f"{type(exc).__name__}: {exc}"
                self.emit({"type": "error", "message": f"LLM 요청 실패: {outcome.error}"})
                self._log({"type": "error", "message": outcome.error})
                if self.history and self.history[-1].role == "user" and not self.history[-1].tool_results:
                    self.history.pop()  # keep the conversation well formed for a retry
                else:
                    self.history.append(Message("assistant", text="(응답을 만들지 못했습니다.)"))
                self.emit({"type": "turn_end", "robot_fault": False})
                return outcome

            text = "".join(text_parts)
            message = Message("assistant", text=text or None, tool_calls=tuple(calls))
            if end.raw is not None:
                message.raw[self.provider.name] = end.raw
            self.history.append(message)
            self._log({"type": "assistant", "text": text, "tool_calls": [c.__dict__ for c in calls],
                       "stop_reason": end.stop_reason, "usage": end.usage})
            if text:
                self.emit({"type": "assistant_text", "text": text})
            if not calls:
                self.emit({"type": "turn_end", "robot_fault": False})
                return outcome

            if len(calls) > 1:
                rejected = tuple(self._not_executed(c, "invalid_arguments", "Primitive mode requires exactly one tool call per response; none executed") for c in calls)
                self.history.append(Message("user", tool_results=rejected))
                self._log({"type": "batch_rejected", "tool_calls": [c.name for c in calls]})
                continue

            results: list[ToolResult] = []
            halted_reason: str | None = None
            for call in calls:
                if halted_reason is not None:
                    results.append(self._not_executed(call, "precondition",
                                                      "앞선 동작이 중단되어 실행하지 않았습니다."))
                    continue
                if self.should_stop():
                    halted_reason = "cancelled"
                    outcome.stopped = True
                    results.append(self._not_executed(call, "cancelled", "비상정지로 실행하지 않았습니다."))
                    continue
                self.emit({"type": "tool_call", "id": call.id, "name": call.name, "arguments": call.arguments})
                result = self.registry.execute(call)
                self.emit({"type": "tool_result", "id": call.id, "name": call.name, "result": result.content})
                self._log({"type": "tool_result", "name": call.name, "arguments": call.arguments,
                           "result": result.content})
                if result.images:
                    # Retain only the newest image batch in API history. Text evidence stays.
                    for previous in self.history:
                        previous.tool_results = tuple(replace(r, images=()) for r in previous.tool_results)
                    if self._transcript is not None:
                        image_dir = self._transcript.parent / "observations"
                        image_dir.mkdir(exist_ok=True)
                        for im in result.images:
                            path = image_dir / (hashlib.sha256(im.jpeg).hexdigest() + ".jpg")
                            path.write_bytes(im.jpeg)
                            self._log({"type": "observation_image", "path": str(path), "frame_seq": im.frame_seq, "captured_at": im.captured_at})
                results.append(result)
                if result.content.get("reason") in ROBOT_FAULT_REASONS:
                    halted_reason = result.content["reason"]
            self.history.append(Message("user", tool_results=tuple(results)))

            if halted_reason is not None:
                outcome.robot_fault = True
                outcome.stopped = outcome.stopped or halted_reason == "cancelled"
                note = "(비상정지되었습니다. 팔 주변을 확인한 뒤 home 복귀를 눌러 주세요.)" if halted_reason == "cancelled" \
                    else "(로봇 동작 중 문제가 생겨 멈췄습니다. home 복귀 후 다시 시도해 주세요.)"
                # close the exchange so the next user message alternates correctly
                self.history.append(Message("assistant", text=note))
                self.emit({"type": "assistant_text", "text": note})
                self.emit({"type": "turn_end", "robot_fault": True})
                return outcome

        outcome.error = "max_tool_turns exceeded"
        note = "(한 번에 너무 많은 단계가 필요해 여기서 멈췄습니다.)"
        self.history.append(Message("assistant", text=note))
        self.emit({"type": "error", "message": outcome.error})
        self.emit({"type": "turn_end", "robot_fault": False})
        return outcome
