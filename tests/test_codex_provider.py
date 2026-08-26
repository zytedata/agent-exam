from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

from agent_exam.config import ProviderConfig
from agent_exam.errors import FrameworkError, UsageError
from agent_exam.providers.codex_cli.provider import (
    CodexCliProvider,
    CodexCliTaskConfig,
    _render_prefix_rules,
    _start_orphan_watchdog,
)


def test_build_cmd_defaults_to_headless_exec(tmp_path):
    cmd = CodexCliProvider()._build_cmd(
        prompt="x",
        model="gpt-test",
        cwd=tmp_path,
        provider_options={},
    )
    assert cmd[:5] == ["codex", "--ask-for-approval", "never", "--model", "gpt-test"]
    assert "--sandbox" in cmd
    assert cmd[cmd.index("--sandbox") + 1] == "workspace-write"
    assert "exec" in cmd
    assert "--json" in cmd
    assert "--ignore-user-config" in cmd
    assert "--ignore-rules" in cmd
    assert cmd[-1] == "x"


def test_build_cmd_separates_prompt_with_dashdash(tmp_path):
    """A prompt starting with `-` must not be parsed as a codex CLI flag."""
    cmd = CodexCliProvider()._build_cmd(
        prompt="--verbose mode: explain what you would do",
        model="",
        cwd=tmp_path,
        provider_options={},
    )
    assert cmd[-2:] == ["--", "--verbose mode: explain what you would do"]


def test_build_cmd_applies_network_override(tmp_path):
    cmd = CodexCliProvider()._build_cmd(
        prompt="x",
        model="",
        cwd=tmp_path,
        provider_options={"network_access": True},
    )
    assert "-c" in cmd
    assert "sandbox_workspace_write.network_access=true" in cmd


def test_build_cmd_applies_writable_roots(tmp_path):
    cmd = CodexCliProvider()._build_cmd(
        prompt="x",
        model="",
        cwd=tmp_path,
        provider_options={"writable_roots": ["~/.cache/uv", "/opt/data"]},
    )
    assert 'sandbox_workspace_write.writable_roots=["~/.cache/uv", "/opt/data"]' in cmd


def test_permission_profiles_silently_ignore_writable_roots(tmp_path):
    """A run-wide writable_roots default must not break profile tasks."""
    cmd = CodexCliProvider()._build_cmd(
        prompt="x",
        model="",
        cwd=tmp_path,
        provider_options={
            "writable_roots": ["~/.cache/uv"],
            "config_overrides": {"default_permissions": ":workspace"},
        },
    )
    assert not any("writable_roots" in arg for arg in cmd)


def test_judge_allowed_tools_force_readonly_no_network(tmp_path):
    cmd = CodexCliProvider()._build_cmd(
        prompt="x",
        model="",
        cwd=tmp_path,
        provider_options={"allowed_tools": ["command_execution"]},
    )
    assert cmd[cmd.index("--sandbox") + 1] == "read-only"
    assert "sandbox_workspace_write.network_access=false" in cmd
    assert 'web_search="disabled"' in cmd


def test_permission_profile_config_omits_sandbox_flag(tmp_path):
    cmd = CodexCliProvider()._build_cmd(
        prompt="x",
        model="",
        cwd=tmp_path,
        provider_options={
            "config_overrides": {
                "default_permissions": "judge",
                "permissions": {
                    "judge": {
                        "filesystem": {
                            ":minimal": "read",
                            ":workspace_roots": {".": "read"},
                        },
                        "network": {"enabled": False},
                    }
                },
            }
        },
    )

    assert "--sandbox" not in cmd
    assert 'default_permissions="judge"' in cmd
    permissions_arg = next(arg for arg in cmd if arg.startswith("permissions="))
    assert '"judge" = {' in permissions_arg
    assert '":minimal" = "read"' in permissions_arg
    assert '":workspace_roots" = {"." = "read"}' in permissions_arg
    assert '"network" = {"enabled" = false}' in permissions_arg


def test_permission_profiles_reject_network_access_shortcut(tmp_path):
    with pytest.raises(UsageError, match=r"network_access.*permission profiles"):
        CodexCliProvider()._build_cmd(
            prompt="x",
            model="",
            cwd=tmp_path,
            provider_options={
                "network_access": True,
                "config_overrides": {"default_permissions": ":workspace"},
            },
        )


def test_task_options_use_small_codex_surface():
    task_cfg = CodexCliTaskConfig.model_validate(
        {"sandbox": "read-only", "network_access": False}
    )
    framework_cfg = ProviderConfig(extra_args=["--oss"])

    options = CodexCliProvider().task_options(task_cfg, framework_cfg, "execute")

    assert options["sandbox"] == "read-only"
    assert options["network_access"] is False
    assert options["ask_for_approval"] == "never"
    assert options["ignore_user_config"] is True
    assert options["ignore_rules"] is True
    assert options["extra_args"] == ["--oss"]
    assert "config_overrides" not in options


def test_task_options_network_access_task_overrides_framework():
    framework_cfg = ProviderConfig(network_access=True)

    inherited = CodexCliProvider().task_options(None, framework_cfg, "execute")
    assert inherited["network_access"] is True

    task_cfg = CodexCliTaskConfig.model_validate({"network_access": False})
    overridden = CodexCliProvider().task_options(task_cfg, framework_cfg, "execute")
    assert overridden["network_access"] is False

    # No framework default → option absent, codex's own default applies.
    assert "network_access" not in CodexCliProvider().task_options(
        None, ProviderConfig(), "execute"
    )


def test_task_options_writable_roots_task_overrides_framework():
    framework_cfg = ProviderConfig(writable_roots=["~/.cache/uv"])

    inherited = CodexCliProvider().task_options(None, framework_cfg, "execute")
    assert inherited["writable_roots"] == ["~/.cache/uv"]

    task_cfg = CodexCliTaskConfig.model_validate({"writable_roots": ["/opt/data"]})
    overridden = CodexCliProvider().task_options(task_cfg, framework_cfg, "execute")
    assert overridden["writable_roots"] == ["/opt/data"]


def test_judge_agent_options_use_base_allowed_tools():
    """No codex-specific override: the base Provider's allowed_tools bridge
    is used, and _build_cmd's existing allowed_tools handling (tested in
    test_judge_allowed_tools_force_readonly_no_network above) maps it to
    --sandbox read-only + no network + no web search."""
    options = CodexCliProvider().judge_agent_options()

    assert options == {"allowed_tools": ["command_execution"]}


def test_prefix_rules_are_staged_as_project_rules(tmp_path):
    provider = CodexCliProvider()
    options = provider._prepare_prefix_rules(
        tmp_path,
        {
            "prefix_rules": [
                {
                    "pattern": ["shub", ["schedule", "deploy"]],
                    "decision": "allow",
                    "justification": "task needs Scrapy Cloud",
                }
            ]
        },
    )

    rules_path = tmp_path / ".codex" / "rules" / "task.rules"
    assert rules_path.read_text() == (
        "prefix_rule(\n"
        '    pattern = ["shub", ["schedule", "deploy"]],\n'
        '    decision = "allow",\n'
        '    justification = "task needs Scrapy Cloud",\n'
        ")\n"
    )
    assert options["ignore_rules"] is False
    assert options["trust_project_config"] is True


def test_build_cmd_loads_task_local_rules(tmp_path):
    cmd = CodexCliProvider()._build_cmd(
        prompt="x",
        model="",
        cwd=tmp_path,
        provider_options={"ignore_rules": False, "trust_project_config": True},
    )

    assert "--ignore-rules" not in cmd
    assert f'projects."{tmp_path.resolve()}".trust_level="trusted"' in cmd


def test_task_config_accepts_prefix_rules():
    cfg = CodexCliTaskConfig.model_validate(
        {
            "prefix_rules": [
                {
                    "pattern": ["git", ["status", "diff"]],
                    "decision": "prompt",
                }
            ]
        }
    )

    assert cfg.prefix_rules is not None
    assert cfg.prefix_rules[0].pattern == ["git", ["status", "diff"]]


def test_render_prefix_rules_defaults_to_allow():
    assert _render_prefix_rules([{"pattern": ["rg"]}]) == (
        'prefix_rule(\n    pattern = ["rg"],\n    decision = "allow",\n)\n'
    )


def test_stage_run_env_uses_agents_skills(tmp_path):
    skills_dir = tmp_path / "skills"
    (skills_dir / "probe-skill").mkdir(parents=True)
    (skills_dir / "probe-skill" / "SKILL.md").write_text("# skill")

    class Cfg:
        skills_dirs = [skills_dir]

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    CodexCliProvider().stage_run_env(run_dir, Cfg())
    staged = run_dir / ".agents" / "skills" / "probe-skill"
    assert staged.is_dir()
    assert not staged.is_symlink()
    assert (staged / "SKILL.md").is_file()


def test_get_global_skills_uses_codex_home(monkeypatch, tmp_path):
    codex_home = tmp_path / "custom-codex-home"
    skill_dir = codex_home / "skills" / "probe-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# skill")
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "regular-home")

    assert CodexCliProvider().get_global_skills() == ["probe-skill"]


def test_invoke_failure_reports_stream_errors_and_preserves_raw(monkeypatch, tmp_path):
    """A fatal codex exit must quote the JSON-stream error events (the only
    place codex reports why it died) and rescue the raw stream out of the
    ephemeral cwd tree before the runner deletes it."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_codex = bin_dir / "codex"
    fake_codex.write_text(
        "#!/bin/sh\n"
        'echo \'{"type":"thread.started","thread_id":"t1"}\'\n'
        'echo \'{"type":"error","message":"You have hit your usage limit."}\'\n'
        "echo 'Reading additional input from stdin...' >&2\n"
        "exit 1\n"
    )
    fake_codex.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    stable_tmp = tmp_path / "stable-tmp"
    stable_tmp.mkdir()
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(stable_tmp))
    cwd = tmp_path / "run" / "attempt-cwd"
    cwd.mkdir(parents=True)

    with pytest.raises(FrameworkError) as exc_info:
        CodexCliProvider()._invoke_once(
            prompt="x",
            model="",
            cwd=cwd,
            provider_options={},
            stop_on_first_trigger=False,
            timeout_seconds=30,
        )

    message = str(exc_info.value)
    assert "You have hit your usage limit." in message
    assert "raw stream preserved at" in message
    preserved = next(
        line.split(": ", 1)[1]
        for line in message.splitlines()
        if line.startswith("raw stream preserved at")
    )
    assert Path(preserved).is_file()
    assert '"usage limit"' not in preserved  # path, not the message
    assert "usage limit" in Path(preserved).read_text()


def test_orphan_watchdog_kills_codex_when_parent_dies():
    process = subprocess.Popen(["sleep", "30"], start_new_session=True)
    calls = {"n": 0}

    def fake_getppid() -> int:
        calls["n"] += 1
        return 100 if calls["n"] == 1 else 1  # reparented after first check

    stop = _start_orphan_watchdog(process, poll_interval=0.01, getppid=fake_getppid)
    try:
        assert process.wait(timeout=10) != 0
    finally:
        stop.set()
        if process.poll() is None:
            process.kill()


def test_orphan_watchdog_stays_quiet_while_parent_lives():
    process = subprocess.Popen(["sleep", "30"], start_new_session=True)
    stop = _start_orphan_watchdog(process, poll_interval=0.01, getppid=lambda: 100)
    try:
        time.sleep(0.1)
        assert process.poll() is None  # untouched
    finally:
        stop.set()
        process.kill()
        process.wait(timeout=10)


def test_preflight_checks_binary(monkeypatch):
    def fake_run(*a, **kw):
        class _Out:
            returncode = 0
            stdout = "codex-cli 0.133.0"
            stderr = ""

        return _Out()

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(CodexCliProvider, "get_global_skills", lambda self: [])
    results = CodexCliProvider().preflight(cfg=None)
    assert results[0].name == "codex binary"
    assert results[0].status == "OK"
