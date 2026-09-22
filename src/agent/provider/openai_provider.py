"""GPT via the official ``openai`` SDK (Chat Completions, streaming, tools).

Reads OPENAI_API_KEY from the environment.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Iterator, Sequence

from .schema import to_openai_tools
from .types import Event, Message, TextDelta, ToolCall, ToolCallEvent, ToolSpec, TurnEnd


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str, *, max_tokens: int, client: Any = None):
        if client is None:
            import openai  # lazy: AGENTS.md §2

            client = openai.OpenAI()
        self._client = client
        self.model = model
        self._max_tokens = max_tokens

    def to_messages(self, system: str, messages: Sequence[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for message in messages:
            if message.role == "user":
                for result in message.tool_results:
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": result.call_id,
                            "content": json.dumps(result.content, ensure_ascii=False),
                        }
                    )
                images = [im for result in message.tool_results for im in result.images]
                if images:
                    content = []
                    for im in images:
                        content.extend([
                            {"type": "text", "text": f"Observation camera={im.camera} frame_seq={im.frame_seq} captured_at={im.captured_at}"},
                            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(im.jpeg).decode("ascii")}},
                        ])
                    out.append({"role": "user", "content": content})
                if message.text:
                    out.append({"role": "user", "content": message.text})
                continue
            raw = message.raw.get(self.name)
            if raw is None:
                raw = {"role": "assistant", "content": message.text or None}
                if message.tool_calls:
                    raw["tool_calls"] = [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.name,
                                "arguments": json.dumps(c.arguments, ensure_ascii=False),
                            },
                        }
                        for c in message.tool_calls
                    ]
            out.append(raw)
        return out

    def stream_turn(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> Iterator[Event]:
        request: dict[str, Any] = dict(
            model=self.model,
            messages=self.to_messages(system, messages),
            tools=to_openai_tools(tools),
            max_completion_tokens=self._max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        # Luna's Chat Completions endpoint rejects function tools while its
        # default reasoning mode is enabled.  We use the model strictly as a
        # low-latency tool router, so disable reasoning explicitly.  Keep the
        # workaround model-scoped so overrides to older/non-reasoning models
        # do not receive an unsupported parameter.
        if self.model.startswith("gpt-5.6-luna"):
            request["reasoning_effort"] = "none"
        stream = self._client.chat.completions.create(**request)
        text_parts: list[str] = []
        pending: dict[int, dict[str, str]] = {}
        finish = "stop"
        usage: dict[str, Any] = {}
        for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = {
                    "input_tokens": chunk.usage.prompt_tokens,
                    "output_tokens": chunk.usage.completion_tokens,
                }
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if delta is not None and delta.content:
                text_parts.append(delta.content)
                yield TextDelta(delta.content)
            for tool_delta in (delta.tool_calls or []) if delta is not None else []:
                slot = pending.setdefault(tool_delta.index, {"id": "", "name": "", "arguments": ""})
                if tool_delta.id:
                    slot["id"] = tool_delta.id
                if tool_delta.function is not None:
                    if tool_delta.function.name:
                        slot["name"] += tool_delta.function.name
                    if tool_delta.function.arguments:
                        slot["arguments"] += tool_delta.function.arguments
            if choice.finish_reason:
                finish = choice.finish_reason

        raw: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts) or None}
        raw_calls = []
        for index in sorted(pending):
            slot = pending[index]
            try:
                arguments = json.loads(slot["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {"__unparsed__": slot["arguments"]}
            call_id = slot["id"] or f"call_{index}"
            raw_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": slot["name"], "arguments": slot["arguments"] or "{}"},
                }
            )
            yield ToolCallEvent(ToolCall(call_id, slot["name"], arguments if isinstance(arguments, dict) else {}))
        if raw_calls:
            raw["tool_calls"] = raw_calls
        yield TurnEnd(finish, usage, raw=raw)
