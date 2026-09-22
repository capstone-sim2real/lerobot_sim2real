"""Vendor adapters against stand-in clients: request shape and event parsing."""

import json
from types import SimpleNamespace as NS

import pytest

from agent.provider.anthropic_provider import AnthropicProvider
from agent.provider.fallback import FallbackProvider, ProviderFallbackError
from agent.provider.openai_provider import OpenAIProvider
from agent.provider.types import Message, TextDelta, ToolCall, ToolCallEvent, ToolResult, ToolSpec, TurnEnd

SPEC = ToolSpec("pick_block", "pick", {"type": "object", "properties": {"color": {"type": "string"}},
                                        "required": ["color"], "additionalProperties": False})
HISTORY = [
    Message("user", text="빨간 블록 집어"),
    Message("assistant", text="집을게요", tool_calls=(ToolCall("t1", "pick_block", {"color": "red"}),)),
    Message("user", tool_results=(ToolResult("t1", "pick_block", {"ok": True, "reason": "held"}),)),
]


class _StubProvider:
    def __init__(self, name, model, events):
        self.name = name
        self.model = model
        self.events = events
        self.calls = 0

    def stream_turn(self, _system, _messages, _tools):
        self.calls += 1
        yield from self.events()


def test_fallback_provider_does_not_retry_after_partial_output():
    def fail_after_output():
        yield TextDelta("partial")
        raise ConnectionError("stream interrupted")

    primary = _StubProvider("openai", "gpt-5.6-luna", fail_after_output)
    fallback = _StubProvider("gemini", "gemini-test", lambda: iter([TextDelta("fallback")]))
    provider = FallbackProvider(primary, fallback)

    with pytest.raises(ConnectionError, match="stream interrupted"):
        list(provider.stream_turn("sys", HISTORY, [SPEC]))

    assert (primary.calls, fallback.calls) == (1, 0)
    assert (provider.name, provider.model) == ("openai", "gpt-5.6-luna")


class _Block(NS):
    def model_dump(self, exclude_none=True):
        return {k: v for k, v in vars(self).items() if v is not None}


class _AnthropicStream:
    def __init__(self, final):
        self.text_stream = iter(["안녕", "하세요"])
        self._final = final

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._final


def _chunk(content=None, tool_calls=None, finish=None, usage=None):
    delta = NS(content=content, tool_calls=tool_calls)
    return NS(choices=[NS(delta=delta, finish_reason=finish)], usage=usage)


def test_openai_adapter_accumulates_streamed_tool_arguments():
    chunks = [
        _chunk(content="좋아요"),
        _chunk(tool_calls=[NS(index=0, id="call_1", function=NS(name="pick_block", arguments='{"col'))]),
        _chunk(tool_calls=[NS(index=0, id=None, function=NS(name=None, arguments='or": "red"}'))]),
        _chunk(finish="tool_calls"),
        NS(choices=[], usage=NS(prompt_tokens=7, completion_tokens=3)),
    ]
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return iter(chunks)

    provider = OpenAIProvider("gpt-test", max_tokens=100,
                              client=NS(chat=NS(completions=NS(create=create))))
    events = list(provider.stream_turn("sys", HISTORY, [SPEC]))
    call = next(e.call for e in events if isinstance(e, ToolCallEvent))
    assert (call.id, call.name, call.arguments) == ("call_1", "pick_block", {"color": "red"})
    end = events[-1]
    assert end.stop_reason == "tool_calls" and end.usage == {"input_tokens": 7, "output_tokens": 3}
    assert end.raw["tool_calls"][0]["function"]["arguments"] == '{"color": "red"}'
    messages = captured["messages"]
    assert messages[0] == {"role": "system", "content": "sys"}
    assert messages[2]["tool_calls"][0]["id"] == "t1"
    assert messages[3]["role"] == "tool" and messages[3]["tool_call_id"] == "t1"
    assert captured["max_completion_tokens"] == 100


def test_gemini_adapter_builds_native_contents_and_parses_calls():
    pytest.importorskip("google.genai")
    from agent.provider.gemini_provider import GeminiProvider
    from google.genai import types

    captured = {}
    response_content = types.Content(role="model", parts=[
        types.Part.from_text(text="집을게요"),
        types.Part.from_function_call(name="pick_block", args={"color": "green"}),
    ])

    def generate_content(**kwargs):
        captured.update(kwargs)
        return NS(candidates=[NS(content=response_content, finish_reason="STOP")],
                  usage_metadata=NS(prompt_token_count=4, candidates_token_count=2))

    provider = GeminiProvider("gemini-test", max_tokens=100,
                              client=NS(models=NS(generate_content=generate_content)))
    events = list(provider.stream_turn("sys", HISTORY, [SPEC]))
    call = next(e.call for e in events if isinstance(e, ToolCallEvent))
    assert (call.name, call.arguments, call.id) == ("pick_block", {"color": "green"}, "pick_block_1")
    assert events[-1].raw is response_content
    contents = captured["contents"]
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[2].parts[0].function_response.name == "pick_block"
    declarations = captured["config"].tools[0].function_declarations
    assert declarations[0].name == "pick_block"
