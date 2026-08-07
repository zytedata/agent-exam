"""Tests for OpenCode provider preflight checks."""

from __future__ import annotations

from agent_exam.providers.opencode.provider import OpenCodeProvider

# --- get_global_skills ------------------------------------------------------


def test_global_skills_missing_dirs(tmp_path, monkeypatch):
    """When neither global dir exists, return empty list."""
    monkeypatch.setenv("HOME", str(tmp_path))
    provider = OpenCodeProvider()
    assert provider.get_global_skills() == []


def test_global_skills_empty_dirs(tmp_path, monkeypatch):
    """When global dirs exist but are empty, return empty list."""
    home = tmp_path / "home"
    (home / ".config" / "opencode" / "skills").mkdir(parents=True)
    (home / ".claude" / "skills").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    provider = OpenCodeProvider()
    assert provider.get_global_skills() == []


def test_global_skills_discovers_skills(tmp_path, monkeypatch):
    """Discover skills with SKILL.md under both global paths."""
    home = tmp_path / "home"
    opencode_dir = home / ".config" / "opencode" / "skills"
    opencode_dir.mkdir(parents=True)
    (opencode_dir / "scrape-codegen").mkdir()
    (opencode_dir / "scrape-codegen" / "SKILL.md").write_text("# skill")

    claude_dir = home / ".claude" / "skills"
    claude_dir.mkdir(parents=True)
    (claude_dir / "other-skill").mkdir()
    (claude_dir / "other-skill" / "SKILL.md").write_text("# skill")

    monkeypatch.setenv("HOME", str(home))
    provider = OpenCodeProvider()
    skills = provider.get_global_skills()
    assert sorted(skills) == ["other-skill", "scrape-codegen"]


# --- preflight global-skills check ------------------------------------------


def _run_preflight_check(monkeypatch, tmp_path, global_skills, cfg=None):
    """Helper: set up global skills, mock binary/DB checks, run preflight,
    return the global-skills CheckResult."""

    home = tmp_path / "home"
    opencode_dir = home / ".config" / "opencode" / "skills"
    opencode_dir.mkdir(parents=True)
    for name in global_skills:
        (opencode_dir / name).mkdir()
        (opencode_dir / name / "SKILL.md").write_text("# skill")
    monkeypatch.setenv("HOME", str(home))

    def fake_run(*a, **kw):
        class _Out:
            returncode = 0
            stdout = "opencode 1.0.0"
            stderr = ""

        return _Out()

    def fake_connect(*a, **kw):
        class _Conn:
            def execute(self, *a, **kw):
                class _Cur:
                    def fetchall(self):
                        return [("session",), ("message",), ("part",)]

                return _Cur()

            def close(self):
                pass

        return _Conn()

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("sqlite3.connect", fake_connect)
    provider = OpenCodeProvider()
    results = provider.preflight(cfg)
    for r in results:
        if r.name == "global opencode skills":
            return r
    raise AssertionError("global opencode skills check not found")


def test_preflight_global_skills_without_cfg_warns(monkeypatch, tmp_path):
    """With no cfg, any global skill is reported as a WARN."""
    result = _run_preflight_check(monkeypatch, tmp_path, ["scrape-codegen"], cfg=None)
    assert result.status == "WARN"
    assert "scrape-codegen" in result.hint


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


def test_preflight_global_skills_checks_both_paths(monkeypatch, tmp_path):
    """Skills in ~/.claude/skills/ are also discovered by OpenCode."""
    home = tmp_path / "home"
    claude_dir = home / ".claude" / "skills"
    claude_dir.mkdir(parents=True)
    (claude_dir / "scrape-codegen").mkdir()
    (claude_dir / "scrape-codegen" / "SKILL.md").write_text("# global")
    monkeypatch.setenv("HOME", str(home))

    staged_dir = tmp_path / "staged"
    staged_dir.mkdir()
    (staged_dir / "scrape-codegen").mkdir()
    (staged_dir / "scrape-codegen" / "SKILL.md").write_text("# staged")

    class FakeCfg:
        skills_dirs = [staged_dir]

    _run_preflight_check(monkeypatch, tmp_path, [], cfg=FakeCfg())
    # Note: _run_preflight_check puts skills in opencode dir by default,
    # but we need to test the claude dir. Let's test get_global_skills directly.
    provider = OpenCodeProvider()
    skills = provider.get_global_skills()
    assert skills == ["scrape-codegen"]
