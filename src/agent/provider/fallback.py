"""Primary/fallback provider composition without vendor SDK imports.

The fallback is used only when the primary fails before yielding any event.
Once text or a tool call has been emitted, retrying through another provider
would duplicate a partial response and could execute ambiguous tool calls.
"""

from __future__ import annotations

import logging
from typing import Iterator, Sequence

from .types import Event, Message, Provider, ToolSpec

logger = logging.getLogger(__name__)


class ProviderFallbackError(RuntimeError):
    """Both the primary provider and its fallback failed before output."""


class FallbackProvider:
    def __init__(self, primary: Provider, fallback: Provider):
        if primary.name == fallback.name:
            raise ValueError("primary and fallback providers must differ")
        self.primary = primary
        self.fallback = fallback
        self.name = primary.name
        self.model = primary.model

    def _select(self, provider: Provider) -> None:
        # AgentRunner reads these after a turn to store native response data
        # under the provider that actually produced it.
        self.name = provider.name
        self.model = provider.model

    def stream_turn(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> Iterator[Event]:
        self._select(self.primary)
        stream = iter(self.primary.stream_turn(system, messages, tools))
        try:
            first = next(stream)
        except StopIteration:
            return
        except Exception as exc:  # noqa: BLE001 - API/network failures trigger fallback
            logger.warning(
                "primary LLM failed before producing output; falling back %s/%s -> %s/%s: %s: %s",
                self.primary.name,
                self.primary.model,
                self.fallback.name,
                self.fallback.model,
                type(exc).__name__,
                exc,
            )
            self._select(self.fallback)
            try:
                yield from self.fallback.stream_turn(system, messages, tools)
            except Exception as fallback_exc:  # noqa: BLE001 - preserve both API failures
                raise ProviderFallbackError(
                    f"primary {self.primary.name}/{self.primary.model} failed: "
                    f"{type(exc).__name__}: {exc}; fallback "
                    f"{self.fallback.name}/{self.fallback.model} failed: "
                    f"{type(fallback_exc).__name__}: {fallback_exc}"
                ) from fallback_exc
            return

        yield first
        # Do not fall back after output has started: duplicating partial text
        # or tool calls is less safe than surfacing the original failure.
        yield from stream
