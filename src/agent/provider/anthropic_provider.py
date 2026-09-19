"""Claude via the official ``anthropic`` SDK (Messages API, streaming, tools).

Reads ANTHROPIC_API_KEY from the environment.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Iterator, Sequence

from .schema import to_anthropic_tools
from .types import Event, Message, TextDelta, ToolCall, ToolCallEvent, ToolSpec, TurnEnd


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str, *, max_tokens: int, client: Any = None):
        if client is None:
            import anthropic  # lazy: AGENTS.md §2

            client = anthropic.Anthropic()
        self._client = client
        self.model = model
        self._max_tokens = max_tokens

    def to_messages(self, messages: Sequence[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "user":
                content: list[dict[str, Any]] = [
                    {
                        "type": "tool_result",
                        "tool_use_id": result.call_id,
                        "content": ([{"type": "text", "text": json.dumps(result.content, ensure_ascii=False)}] + [
                            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                             "data": base64.b64encode(im.jpeg).decode("ascii")}} for im in result.images
                        ]) if result.images else json.dumps(result.content, ensure_ascii=False),
                        "is_error": result.is_error,
                    }
                    for result in message.tool_results
                ]
                if message.text:
                    content.append({"type": "text", "text": message.text})
                out.append({"role": "user", "content": content})
                continue
            raw = message.raw.get(self.name)
            if raw is None:
                raw = []
                if message.text:
                    raw.append({"type": "text", "text": message.text})
                raw.extend(
                    {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                    for c in message.tool_calls
                )
            if raw:
                out.append({"role": "assistant", "content": raw})
        return out

    def stream_turn(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> Iterator[Event]:
        with self._client.messages.stream(
            model=self.model,
            max_tokens=self._max_tokens,
            system=system,
            tools=to_anthropic_tools(tools),
            messages=self.to_messages(messages),
        ) as stream:
            for text in stream.text_stream:
                if text:
                    yield TextDelta(text)
            final = stream.get_final_message()
        for block in final.content:
            if block.type == "tool_use":
                yield ToolCallEvent(ToolCall(block.id, block.name, dict(block.input or {})))
        raw = [block.model_dump(exclude_none=True) for block in final.content]
        usage = {
            "input_tokens": getattr(final.usage, "input_tokens", None),
            "output_tokens": getattr(final.usage, "output_tokens", None),
        }
        yield TurnEnd(str(final.stop_reason), usage, raw=raw)
