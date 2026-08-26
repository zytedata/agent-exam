"""Tests for Copilot CLI provider preflight checks."""

from __future__ import annotations

import json

import pytest

from agent_exam.providers.copilot_cli.doctor_probes import (
    _personal_mcp_servers,
    check_personal_mcp_servers,
    personal_mcp_servers,
)
from agent_exam.providers.copilot_cli.provider import CopilotCliProvider


@pytest.fixture(autouse=True)
def _uncached_personal_mcp_servers():
    """`copilot mcp list` is asked once per process, so each test has to ask
    again."""
    _personal_mcp_servers.cache_clear()


def _fake_copilot(monkeypatch, mcp_servers=()):
    """Stand in for both `copilot --version` and `copilot mcp list --json`."""

    def fake_run(cmd, *args, **kwargs):
        class _Out:
            returncode = 0
            stderr = ""
            stdout = (
                json.dumps({"mcpServers": {name: {} for name in mcp_servers}})
                if "mcp" in cmd
                else "copilot 1.0.0"
            )

        return _Out()

    monkeypatch.setattr("subprocess.run", fake_run)


# --- get_global_skills ------------------------------------------------------


def test_global_skills_missing_dir(tmp_path, monkeypatch):
    """When ~/.copilot/skills/ doesn't exist, return empty list."""
    monkeypatch.setenv("HOME", str(tmp_path))
    provider = CopilotCliProvider()
    assert provider.get_global_skills() == []


def test_global_skills_empty_dir(tmp_path, monkeypatch):
    """When ~/.copilot/skills/ exists but is empty, return empty list."""
    home = tmp_path / "home"
    skills_dir = home / ".copilot" / "skills"
    skills_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    provider = CopilotCliProvider()
    assert provider.get_global_skills() == []


def test_global_skills_discovers_skills(tmp_path, monkeypatch):
    """Discover skills with SKILL.md under ~/.copilot/skills/."""
    home = tmp_path / "home"
    skills_dir = home / ".copilot" / "skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "scrape-codegen").mkdir()
    (skills_dir / "scrape-codegen" / "SKILL.md").write_text("# skill")
    (skills_dir / "other-skill").mkdir()
    (skills_dir / "other-skill" / "SKILL.md").write_text("# skill")
    monkeypatch.setenv("HOME", str(home))
    provider = CopilotCliProvider()
    skills = provider.get_global_skills()
    assert sorted(skills) == ["other-skill", "scrape-codegen"]


def test_global_skills_discovers_installed_plugin_skills(tmp_path, monkeypatch):
    """Discover skills installed by current Copilot CLI plugin layout."""
    home = tmp_path / "home"
    skills_dir = (
        home / ".copilot" / "installed-plugins" / "_direct" / "zyte-web-data" / "skills"
    )
    skills_dir.mkdir(parents=True)
    (skills_dir / "scrape-codegen").mkdir()
    (skills_dir / "scrape-codegen" / "SKILL.md").write_text("# skill")
    (skills_dir / "scrape-create-spider").mkdir()
    (skills_dir / "scrape-create-spider" / "SKILL.md").write_text("# skill")
    monkeypatch.setenv("HOME", str(home))
    provider = CopilotCliProvider()
    skills = provider.get_global_skills()
    assert skills == ["scrape-codegen", "scrape-create-spider"]


# --- preflight global-skills check ------------------------------------------


def _run_preflight_check(monkeypatch, tmp_path, global_skills, cfg=None):
    """Helper: set up global skills, mock binary check, run preflight,
    return the personal-skills CheckResult."""

    home = tmp_path / "home"
    skills_dir = home / ".copilot" / "skills"
    skills_dir.mkdir(parents=True)
    for name in global_skills:
        (skills_dir / name).mkdir()
        (skills_dir / name / "SKILL.md").write_text("# skill")
    monkeypatch.setenv("HOME", str(home))

    _fake_copilot(monkeypatch)
    provider = CopilotCliProvider()
    results = provider.preflight(cfg)
    for r in results:
        if r.name == "personal skills":
            return r
    raise AssertionError("personal skills check not found")


def test_preflight_global_skills_without_cfg_warns(monkeypatch, tmp_path):
    """With no cfg, any global skill is reported as a WARN."""
    result = _run_preflight_check(monkeypatch, tmp_path, ["scrape-codegen"], cfg=None)
    assert result.status == "WARN"
    assert "scrape-codegen" in result.hint


def test_preflight_global_skills_without_skills_dirs_warns(monkeypatch, tmp_path):
    """An unset cfg.skills_dirs should warn, not crash."""

    class FakeCfg:
        skills_dirs = None

    result = _run_preflight_check(
        monkeypatch, tmp_path, ["scrape-codegen"], cfg=FakeCfg()
    )
    assert result.status == "WARN"
    assert "scrape-codegen" in result.hint
    assert "skipping staged-skill clash check" in result.hint


def test_preflight_global_skills_clash_with_staged(monkeypatch, tmp_path):
    """When a global skill has the same name as a staged skill → WARN."""
    staged_dir = tmp_path / "staged"
    staged_dir.mkdir()
    (staged_dir / "scrape-codegen").mkdir()
    (staged_dir / "scrape-codegen" / "SKILL.md").write_text("# staged")

    class FakeCfg:
        skills_dirs = [staged_dir]

    result = _run_preflight_check(
        monkeypatch, tmp_path, ["scrape-codegen"], cfg=FakeCfg()
    )
    assert result.status == "WARN"
    assert "scrape-codegen" in result.hint
    assert "clash" in result.hint


def test_preflight_global_skills_no_clash(monkeypatch, tmp_path):
    """Global skills that don't overlap with staged skills → OK."""
    staged_dir = tmp_path / "staged"
    staged_dir.mkdir()
    (staged_dir / "scrape-codegen").mkdir()
    (staged_dir / "scrape-codegen" / "SKILL.md").write_text("# staged")

    class FakeCfg:
        skills_dirs = [staged_dir]

    result = _run_preflight_check(
        monkeypatch, tmp_path, ["personal-note"], cfg=FakeCfg()
    )
    assert result.status == "OK"
    assert "none clash" in result.hint


# --- personal MCP servers check ---------------------------------------------


def test_personal_mcp_servers_none(monkeypatch):
    _fake_copilot(monkeypatch)
    result = check_personal_mcp_servers(None)
    assert result.status == "OK"
    assert result.hint == "none set up"


def test_personal_mcp_servers_are_disabled(monkeypatch):
    _fake_copilot(monkeypatch, ["notes"])

    class FakeCfg:
        mcp_servers = {"files": {}}

    result = check_personal_mcp_servers(FakeCfg())
    assert result.status == "OK"
    assert "notes disabled" in result.hint


def test_personal_mcp_server_sharing_a_name_warns(monkeypatch):
    """A shared name cannot be disabled without disabling the attached
    server, so it stays enabled behind it."""
    _fake_copilot(monkeypatch, ["files"])

    class FakeCfg:
        mcp_servers = {"files": {}}

    result = check_personal_mcp_servers(FakeCfg())
    assert result.status == "WARN"
    assert "files" in result.hint


def test_personal_mcp_servers_survives_an_unusable_copilot(monkeypatch):
    """A `copilot mcp list` that cannot run leaves nothing to disable, and
    the check WARNs rather than reporting a false "none set up"."""

    def fake_run(*args, **kwargs):
        raise OSError("boom")

    monkeypatch.setattr("subprocess.run", fake_run)
    assert personal_mcp_servers() == []
    result = check_personal_mcp_servers(None)
    assert result.status == "WARN"
    assert "copilot mcp list" in result.hint
