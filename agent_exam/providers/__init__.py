from __future__ import annotations

from .base import Provider
from .dummy import DummyProvider


def get_provider(name: str) -> Provider:
    if name == "dummy":
        return DummyProvider()
    if name == "claude_code":
        from .claude_code.provider import ClaudeCodeProvider

        return ClaudeCodeProvider()
    if name == "codex_cli":
        from .codex_cli.provider import CodexCliProvider

        return CodexCliProvider()
    if name == "opencode":
        from .opencode.provider import OpenCodeProvider

        return OpenCodeProvider()
    if name == "copilot_cli":
        from .copilot_cli.provider import CopilotCliProvider

        return CopilotCliProvider()
    raise ValueError(f"unknown provider {name!r}")


__all__ = [
    "ClaudeCodeProvider",
    "CodexCliProvider",
    "CopilotCliProvider",
    "DummyProvider",
    "OpenCodeProvider",
    "Provider",
    "get_provider",
]
