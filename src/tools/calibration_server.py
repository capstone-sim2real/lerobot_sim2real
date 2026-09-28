"""Compatibility entry point; use python -m tools.agent_server."""

from tools.agent_server import (
    WEB_MANUAL_TOOLS,
    configure_manual_tools,
    definitions,
    main,
)

__all__ = ["WEB_MANUAL_TOOLS", "configure_manual_tools", "definitions", "main"]


if __name__ == "__main__":
    main()
