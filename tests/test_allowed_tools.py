"""Per-task allowed_tools: YAML parses, plumbed to provider_options,
translated to the `--allowed-tools` CLI flag by the claude_code provider.
"""

from __future__ import annotations

from textwrap import dedent
from unittest.mock import patch

import pytest

from agent_exam.errors import UsageError
from agent_exam.tasks import load_task


def test_yaml_parses_allowed_tools_under_provider_section(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(
        dedent(
            """\
            kind: execute
            prompt: x
            claude_code:
              allowed_tools:
                - "Bash(shub*)"
                - "Bash(curl*)"
            assertions: []
            """
        )
    )
    tasks = load_task(p, "s")
    claude = tasks[0].provider_configs["claude_code"]
    assert claude.allowed_tools == ["Bash(shub*)", "Bash(curl*)"]
    assert claude.permission_mode is None


def test_yaml_allowed_tools_defaults_to_empty(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(
        dedent(
            """\
            kind: execute
            prompt: x
            assertions: []
            """
        )
    )
    assert load_task(p, "s")[0].provider_configs == {}


def test_yaml_allowed_tools_rejects_non_list(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(
        dedent(
            """\
            kind: execute
            prompt: x
            claude_code:
              allowed_tools: "Bash(shub*)"
            assertions: []
            """
        )
    )
    with pytest.raises(UsageError, match=r"allowed_tools.*valid list"):
        load_task(p, "s")


def test_yaml_allowed_tools_rejects_non_strings(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(
        dedent(
            """\
            kind: execute
            prompt: x
            claude_code:
              allowed_tools:
                - "Bash(ok*)"
                - 42
            assertions: []
            """
        )
    )
    with pytest.raises(UsageError, match=r"allowed_tools.*valid string"):
        load_task(p, "s")


def test_yaml_rejects_unknown_provider_key(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(
        dedent(
            """\
            kind: execute
            prompt: x
            claude_code:
              allowed_tools: ["Bash(ok*)"]
              allow_tool: "typo"
            assertions: []
            """
        )
    )
    with pytest.raises(UsageError, match=r"claude_code.allow_tool.*Extra inputs"):
        load_task(p, "s")


def test_provider_emits_allowed_tools_flag():
    """Claude Code provider translates allowed_tools to a single
    --allowed-tools comma-joined argv entry.
    """
    from agent_exam.providers.claude_code.provider import ClaudeCodeProvider

    provider = ClaudeCodeProvider()
    # Capture the argv the provider would pass to subprocess.Popen without
    # actually spawning claude. Patch Popen to inspect and then fail fast.
    captured = {}

    class _FakePopenError(Exception):
        pass

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        raise _FakePopenError

    from pathlib import Path

    with (
        patch("subprocess.Popen", side_effect=fake_popen),
        pytest.raises(_FakePopenError),
    ):
        provider._invoke_once(
            prompt="x",
            model="",
            cwd=Path("/tmp"),
            provider_options={
                "allowed_tools": [
                    "Bash(shub deploy*)",
                    "Bash(curl *jobs/list.json*)",
                ],
                "extra_args": ["--add-dir", "/extra"],
            },
            stop_on_first_skill=False,
            timeout_seconds=30,
        )

    cmd = captured["cmd"]
    # --allowed-tools present, followed by one argv entry with comma-joined
    # patterns. Skill is auto-appended (see test_skill_auto_appended).
    idx = cmd.index("--allowed-tools")
    assert cmd[idx + 1] == "Bash(shub deploy*),Bash(curl *jobs/list.json*),Skill"
    # extra_args still land after the allowed-tools argument.
    assert cmd[idx + 2 :] == ["--add-dir", "/extra"]


def _capture_cmd(provider_options):
    """Run _invoke_once with the given options, capture argv via Popen
    short-circuit, return the captured cmd list.
    """
    from agent_exam.providers.claude_code.provider import ClaudeCodeProvider

    captured = {}

    class _FakePopenError(Exception):
        pass

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        raise _FakePopenError

    from pathlib import Path

    with (
        patch("subprocess.Popen", side_effect=fake_popen),
        pytest.raises(_FakePopenError),
    ):
        ClaudeCodeProvider()._invoke_once(
            prompt="x",
            model="",
            cwd=Path("/tmp"),
            provider_options=provider_options,
            stop_on_first_skill=False,
            timeout_seconds=30,
        )
    return captured["cmd"]


def test_skill_auto_appended_when_allowed_tools_set():
    """Skill must be in --allowed-tools whenever the list is non-empty,
    or natural-language skill routing fails in headless `-p` runs (the
    Skill tool's redirect is treated as an error by the agent and it
    falls back to a manual answer).
    """
    cmd = _capture_cmd({"allowed_tools": ["Bash"]})
    idx = cmd.index("--allowed-tools")
    assert cmd[idx + 1] == "Bash,Skill"


def test_skill_not_double_appended_if_user_listed_it():
    cmd = _capture_cmd({"allowed_tools": ["Bash", "Skill"]})
    idx = cmd.index("--allowed-tools")
    assert cmd[idx + 1] == "Bash,Skill"


def test_skill_not_added_when_allowed_tools_empty():
    """Empty allowed_tools means "no allowlist" — Claude Code defaults to
    all tools available, so we must not introduce a flag with just Skill.
    Doing so would silently restrict the run to Skill only.
    """
    cmd = _capture_cmd({})
    assert "--allowed-tools" not in cmd


def test_provider_omits_flag_when_empty():
    from agent_exam.providers.claude_code.provider import ClaudeCodeProvider

    captured = {}

    class _FakePopenError(Exception):
        pass

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        raise _FakePopenError

    from pathlib import Path

    with (
        patch("subprocess.Popen", side_effect=fake_popen),
        pytest.raises(_FakePopenError),
    ):
        ClaudeCodeProvider()._invoke_once(
            prompt="x",
            model="",
            cwd=Path("/tmp"),
            provider_options={},
            stop_on_first_skill=False,
            timeout_seconds=30,
        )
    assert "--allowed-tools" not in captured["cmd"]
