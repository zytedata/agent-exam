"""Per-provider `safe_judge_tools` constants and provider-native permission
translation used by `judge_agent` to expose cwd inspection to the underlying
harness.
"""

from __future__ import annotations

from agent_exam.providers.base import Provider
from agent_exam.providers.claude_code.provider import ClaudeCodeProvider
from agent_exam.providers.codex_cli.provider import CodexCliProvider
from agent_exam.providers.copilot_cli.provider import CopilotCliProvider
from agent_exam.providers.dummy import DummyProvider
from agent_exam.providers.opencode.provider import (
    OpenCodeProvider,
    build_permission_config,
)


def test_base_provider_has_empty_safe_judge_tools():
    """Defaulting empty ensures providers that haven't opted in cause
    judge_agent to fail fast with a clear error rather than silently
    skipping tool exposure."""
    assert Provider.safe_judge_tools == ()


def test_claude_code_safe_judge_tools_uses_capitalized_names():
    assert ClaudeCodeProvider.safe_judge_tools == ("Read", "Glob", "Grep")


def test_opencode_safe_judge_tools_uses_lowercase_names():
    assert OpenCodeProvider.safe_judge_tools == ("read", "glob", "grep")


def test_copilot_cli_safe_judge_tools():
    """Per GitHub Copilot CLI command reference: view (read files/dirs),
    glob (find by pattern), grep (search text)."""
    assert CopilotCliProvider.safe_judge_tools == ("view", "glob", "grep")


def test_codex_cli_safe_judge_tools():
    """Codex exec JSONL currently exposes shell/tool activity as a single
    command_execution item, so judge_agent_options (inherited from the
    base Provider) maps it to allowed_tools=["command_execution"], which
    _build_cmd translates to --sandbox read-only + no network + no web
    search — Codex has no native read/glob/grep-only shell policy."""
    assert CodexCliProvider.safe_judge_tools == ("command_execution",)


def test_dummy_provider_has_no_safe_judge_tools():
    """Dummy doesn't run a real harness; judge_agent against it should
    error rather than pretend to enforce tool restrictions."""
    assert DummyProvider.safe_judge_tools == ()


def test_permission_config_from_explicit_permission_passthrough():
    perm = {"bash": "allow"}
    out = build_permission_config(permission=perm, allowed_tools=None)
    assert out["bash"] == "allow"
    # external_directory is always force-denied.
    assert out["external_directory"] == "deny"


def test_permission_config_translates_allowed_tools_via_wildcard():
    out = build_permission_config(
        permission=None, allowed_tools=("read", "glob", "grep")
    )
    assert out["*"] == "deny"
    assert out["read"] == "allow"
    assert out["glob"] == "allow"
    assert out["grep"] == "allow"
    assert out["external_directory"] == "deny"


def test_permission_config_prefers_explicit_permission_over_allowed_tools():
    """An explicit permission config wins over allowed_tools to avoid
    surprising the caller — the framework should not invent a
    `"*": "deny"` rule when the user has already declared an intent."""
    out = build_permission_config(permission={"bash": "allow"}, allowed_tools=("read",))
    assert "*" not in out
    assert out["bash"] == "allow"


def test_permission_config_preserves_user_external_directory_override():
    """A task that explicitly says external_directory=allow should keep
    that — the auto-deny only kicks in when the key is absent."""
    out = build_permission_config(
        permission={"external_directory": "allow"}, allowed_tools=None
    )
    assert out["external_directory"] == "allow"


def test_permission_config_empty_when_nothing_provided():
    """No permission and no allowed_tools → still get the forced
    external_directory deny."""
    out = build_permission_config(permission=None, allowed_tools=None)
    assert out == {"external_directory": "deny"}
