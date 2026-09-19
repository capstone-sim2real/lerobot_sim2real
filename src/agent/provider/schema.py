"""ToolSpec -> each vendor's tool declaration. Pure dict transforms."""

from __future__ import annotations

import copy
from typing import Any, Sequence

from .types import ToolSpec

# JSON-Schema keys Gemini's OpenAPI-subset function declarations reject.
_GEMINI_DROP = {"additionalProperties", "$schema", "default", "const", "examples", "title"}


def to_anthropic_tools(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description, "input_schema": copy.deepcopy(t.input_schema)}
        for t in tools
    ]


def to_openai_tools(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": copy.deepcopy(t.input_schema),
            },
        }
        for t in tools
    ]


def sanitize_for_gemini(schema: Any) -> Any:
    """Drop rejected keys at every depth; omit empty object ``properties``."""
    if isinstance(schema, list):
        return [sanitize_for_gemini(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key in _GEMINI_DROP:
            continue
        if key == "properties":
            if not value:
                continue
            out[key] = {name: sanitize_for_gemini(sub) for name, sub in value.items()}
        elif key == "required" and not value:
            continue
        else:
            out[key] = sanitize_for_gemini(value)
    return out


def to_gemini_function_declarations(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    declarations = []
    for t in tools:
        declaration: dict[str, Any] = {"name": t.name, "description": t.description}
        params = sanitize_for_gemini(t.input_schema)
        if params.get("properties"):
            declaration["parameters"] = params
        declarations.append(declaration)
    return declarations
