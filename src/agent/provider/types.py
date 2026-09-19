"""Provider-neutral conversation types. No SDK imports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol, Sequence, Union


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    content: dict[str, Any]
    is_error: bool = False


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ToolCallEvent:
    call: ToolCall


@dataclass(frozen=True)
class TurnEnd:
    stop_reason: str
    usage: dict[str, Any] = field(default_factory=dict)
    # The provider's native assistant content for this turn, replayed verbatim
    # on the next request (keeps thinking blocks / thought signatures intact).
    raw: Any = None


Event = Union[TextDelta, ToolCallEvent, TurnEnd]


@dataclass
class Message:
    role: str  # "user" | "assistant"
    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()
    # provider name -> native content (assistant messages only)
    raw: dict[str, Any] = field(default_factory=dict)


class Provider(Protocol):
    name: str
    model: str

    def stream_turn(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> Iterator[Event]: ...
