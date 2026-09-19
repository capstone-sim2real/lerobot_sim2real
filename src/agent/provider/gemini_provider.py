"""Gemini via the official ``google-genai`` SDK (tools, automatic calling off).

Reads GEMINI_API_KEY / GOOGLE_API_KEY from the environment. Non-streaming on
purpose: the model's native ``Content`` (including thought signatures that
newer Gemini models require on function-call replay) is kept whole and sent
back verbatim.
"""

from __future__ import annotations

from typing import Any, Iterator, Sequence

from .schema import to_gemini_function_declarations
from .types import Event, Message, TextDelta, ToolCall, ToolCallEvent, ToolSpec, TurnEnd


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str, *, max_tokens: int, client: Any = None):
        from google.genai import types  # lazy: AGENTS.md §2

        if client is None:
            from google import genai

            client = genai.Client()
        self._types = types
        self._client = client
        self.model = model
        self._max_tokens = max_tokens

    def _declarations(self, tools: Sequence[ToolSpec]) -> list[Any]:
        types = self._types
        fields = getattr(types.FunctionDeclaration, "model_fields", {})
        out = []
        for spec, sanitized in zip(tools, to_gemini_function_declarations(tools)):
            if "parameters_json_schema" in fields and spec.input_schema.get("properties"):
                out.append(
                    types.FunctionDeclaration(
                        name=spec.name,
                        description=spec.description,
                        parameters_json_schema=spec.input_schema,
                    )
                )
            else:
                out.append(types.FunctionDeclaration.model_validate(sanitized))
        return out

    def to_contents(self, messages: Sequence[Message]) -> list[Any]:
        types = self._types
        contents = []
        for message in messages:
            if message.role == "user":
                parts = [
                    types.Part.from_function_response(name=result.name, response=result.content)
                    for result in message.tool_results
                ]
                if message.text:
                    parts.append(types.Part.from_text(text=message.text))
                contents.append(types.Content(role="user", parts=parts))
                continue
            raw = message.raw.get(self.name)
            if raw is None:
                parts = []
                if message.text:
                    parts.append(types.Part.from_text(text=message.text))
                parts.extend(
                    types.Part.from_function_call(name=c.name, args=c.arguments)
                    for c in message.tool_calls
                )
                if not parts:
                    continue
                raw = types.Content(role="model", parts=parts)
            contents.append(raw)
        return contents

    def stream_turn(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> Iterator[Event]:
        types = self._types
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=self._declarations(tools))],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            max_output_tokens=self._max_tokens,
        )
        response = self._client.models.generate_content(
            model=self.model, contents=self.to_contents(messages), config=config
        )
        candidate = response.candidates[0] if response.candidates else None
        content = candidate.content if candidate is not None else None
        parts = list(content.parts or []) if content is not None else []
        count = 0
        for part in parts:
            if getattr(part, "text", None) and not getattr(part, "thought", False):
                yield TextDelta(part.text)
        for part in parts:
            call = getattr(part, "function_call", None)
            if call is None:
                continue
            count += 1
            call_id = getattr(call, "id", None) or f"{call.name}_{count}"
            yield ToolCallEvent(ToolCall(call_id, call.name, dict(call.args or {})))
        usage_meta = getattr(response, "usage_metadata", None)
        usage = {
            "input_tokens": getattr(usage_meta, "prompt_token_count", None),
            "output_tokens": getattr(usage_meta, "candidates_token_count", None),
        }
        finish = str(getattr(candidate, "finish_reason", "STOP")) if candidate is not None else "EMPTY"
        yield TurnEnd(finish, usage, raw=content if parts else None)
