"""Providers that need no network or API key.

``ScriptedProvider`` replays fixed turns for tests. ``RuleBasedFakeProvider``
maps a handful of Korean phrases to tool calls so the web UI and the whole
tool path can be exercised offline (``so101-agent --provider fake``).
"""

from __future__ import annotations

import copy
import json
import re
from typing import Callable, Iterator, Sequence

from .types import Event, Message, TextDelta, ToolCall, ToolCallEvent, ToolSpec, TurnEnd

Turn = list[Event] | Callable[[Sequence[Message]], list[Event]]


class ScriptedProvider:
    name = "fake"
    model = "scripted"

    def __init__(self, script: list[Turn]):
        self._script = list(script)
        self.seen_messages: list[list[Message]] = []
        self.seen_tools: list[list[ToolSpec]] = []
        self.seen_system: list[str] = []

    def stream_turn(self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> Iterator[Event]:
        self.seen_messages.append(copy.deepcopy(list(messages)))
        self.seen_tools.append(list(tools))
        self.seen_system.append(system)
        if not self._script:
            yield TextDelta("(script exhausted)")
            yield TurnEnd("end_turn")
            return
        turn = self._script.pop(0)
        events = turn(messages) if callable(turn) else turn
        yield from events


_COLORS = {"빨간": "red", "빨강": "red", "노란": "yellow", "노랑": "yellow", "초록": "green",
           "녹색": "green", "파란": "blue", "파랑": "blue", "나무": "wood", "원목": "wood"}
_SLOTS = {"좌상단": "top-left", "상단중앙": "top-center", "우상단": "top-right",
          "좌하단": "bottom-left", "우하단": "bottom-right"}


class RuleBasedFakeProvider:
    """Offline stand-in for an LLM: keyword rules, one tool call, then a summary."""

    name = "fake"
    model = "rules"

    def stream_turn(self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> Iterator[Event]:
        last = messages[-1] if messages else Message("user", text="")
        if last.tool_results:
            for result in last.tool_results:
                yield TextDelta(f"[{result.name}] {result.content.get('detail') or result.content.get('reason')}\n")
            yield TurnEnd("end_turn")
            return
        text = re.sub(r"\s+", "", last.text or "")
        call = self._rule(text)
        if call is None:
            yield TextDelta("(오프라인 fake 모드) 예: '노란 블록 좌상단으로', '미션 1', '상태', '홈', '초록 블록 밖으로'.")
            yield TurnEnd("end_turn")
            return
        yield TextDelta(f"{call.name} 실행: {json.dumps(call.arguments, ensure_ascii=False)}\n")
        yield ToolCallEvent(call)
        yield TurnEnd("tool_use")

    @staticmethod
    def _rule(text: str) -> ToolCall | None:
        color = next((v for k, v in _COLORS.items() if k in text), None)
        slot = next((v for k, v in _SLOTS.items() if k in text), None)
        mm = re.search(r"(\d+)mm", text)
        distance = float(mm.group(1)) if mm else 20.0
        if "미션1" in text or "task1" in text.lower():
            return ToolCall("fake_1", "run_task1", {})
        if "미션2" in text:
            return ToolCall("fake_1", "run_task2", {})
        if "상태" in text:
            return ToolCall("fake_1", "get_state", {})
        if "홈" in text or "home" in text.lower():
            return ToolCall("fake_1", "return_to_home", {})
        if "여기" in text:
            return ToolCall("fake_1", "place_here", {})
        if "왼쪽으로" in text and color is None:
            return ToolCall("fake_1", "move_arm", {"left_mm": min(distance, 50.0)})
        if color and ("멀리" in text or "가까이" in text) and "집" in text:
            step = distance if mm else 5.0
            forward = step if "멀리" in text else -step
            return ToolCall("fake_1", "pick_block", {"color": color, "forward_mm": forward,
                                                     "relative_to_last": "더" in text})
        if color and slot:
            return ToolCall("fake_1", "move_block_to_slot", {"color": color, "slot": slot})
        if color and "밖" in text:
            return ToolCall("fake_1", "move_block_to_table", {"color": color})
        if color and "왼쪽" in text:
            return ToolCall("fake_1", "shift_block", {"color": color, "forward_mm": 0.0, "left_mm": distance})
        if color:
            return ToolCall("fake_1", "pick_block", {"color": color})
        if "보여" in text or "관찰" in text or "블록" in text:
            return ToolCall("fake_1", "observe_scene", {})
        return None
