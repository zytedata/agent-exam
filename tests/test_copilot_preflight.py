"""Tests for Copilot CLI provider preflight checks."""

from __future__ import annotations

from agent_exam.providers.copilot_cli.provider import CopilotCliProvider

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

    def fake_run(*a, **kw):
        class _Out:
            returncode = 0
            stdout = "copilot 1.0.0"
            stderr = ""

        return _Out()

    monkeypatch.setattr("subprocess.run", fake_run)
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
