"""LLM provider adapters behind one ``Provider`` protocol.

``build_provider`` is the only place a concrete adapter is chosen; everything
else receives a ``Provider``.
"""

from __future__ import annotations

from config import AgentConfig

from .types import Provider


def build_provider(cfg: AgentConfig, *, provider: str | None = None, model: str | None = None) -> Provider:
    name = provider or cfg.provider
    model_id = model or cfg.models.get(name, "")
    if name == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(model_id, max_tokens=cfg.max_tokens)
    if name == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(model_id, max_tokens=cfg.max_tokens)
    if name == "gemini":
        from .gemini_provider import GeminiProvider

        return GeminiProvider(model_id, max_tokens=cfg.max_tokens)
    if name == "fake":
        from .fake import RuleBasedFakeProvider

        return RuleBasedFakeProvider()
    raise ValueError(f"unknown provider {name!r}")
