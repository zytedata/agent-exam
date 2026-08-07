"""Tests for Claude Code provider preflight checks."""

from __future__ import annotations

import json

from agent_exam.providers.claude_code.provider import ClaudeCodeProvider


class _FakeStdout:
    """File-like iterable returned by _FakeProc.stdout."""

    def __init__(self, lines: list[str]):
        self._lines = lines
        self._idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx < len(self._lines):
            line = self._lines[self._idx]
            self._idx += 1
            return line
        raise StopIteration

    def readline(self):
        try:
            return self.__next__()
        except StopIteration:
            return ""


class _FakeProc:
    """Mock subprocess.Popen that yields JSONL lines on stdout."""

    def __init__(self, lines: list[str]):
        self.stdout = _FakeStdout(lines)
        self.returncode = 0

    def terminate(self):
        pass

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def kill(self):
        pass


def _make_init_event(skills: list[str]) -> str:
    return json.dumps(
        {
            "type": "system",
            "subtype": "init",
            "cwd": "/tmp",
            "session_id": "test-session",
            "skills": skills,
        }
    )


# --- get_global_skills ------------------------------------------------------


def test_global_skills_probe_failure(monkeypatch):
    """When the clean probe can't spawn claude, return None."""

    def fake_popen(*a, **kw):
        raise FileNotFoundError("claude")

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    provider = ClaudeCodeProvider()
    assert provider.get_global_skills() is None


def test_global_skills_returns_probe_skills(monkeypatch):
    """The method returns the skills list from the init event."""

    def fake_popen(*a, **kw):
        return _FakeProc([_make_init_event(["scrape-codegen", "plugin:foo"])])

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    provider = ClaudeCodeProvider()
    assert provider.get_global_skills() == ["scrape-codegen", "plugin:foo"]


# --- preflight global-skills check ------------------------------------------


def _run_preflight_check(monkeypatch, skills, cfg=None):
    """Helper: mock probe, run preflight, return the global-skills CheckResult."""

    def fake_popen(*a, **kw):
        return _FakeProc([_make_init_event(skills)])

    def fake_run(*a, **kw):
        class _Out:
            returncode = 0
            stdout = "claude 0.1.2"
            stderr = ""

        return _Out()

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(
        "agent_exam.providers.claude_code.provider._blocked_plugins_static_check",
        lambda cfg, context="doctor": [],
    )
    provider = ClaudeCodeProvider()
    results = provider.preflight(cfg)
    for r in results:
        if r.name == "global claude skills":
            return r
    raise AssertionError("global claude skills check not found")


def test_preflight_global_skills_without_cfg_warns(monkeypatch):
    """With no cfg, any global skill is reported as a WARN."""
    result = _run_preflight_check(
        monkeypatch, ["scrape-codegen", "plugin:foo"], cfg=None
    )
    assert result.status == "WARN"
    assert "scrape-codegen" in result.hint
    assert "plugin:foo" in result.hint


def test_preflight_global_skills_no_clash_with_staged(monkeypatch, tmp_path):
    """Global skills that don't overlap with staged skills → OK."""
    staged_dir = tmp_path / "staged"
    staged_dir.mkdir()
    (staged_dir / "scrape-codegen").mkdir()
    (staged_dir / "scrape-codegen" / "SKILL.md").write_text("# staged")

    class FakeCfg:
        skills_dirs = [staged_dir]

    result = _run_preflight_check(
        monkeypatch, ["personal-note", "plugin:foo"], cfg=FakeCfg()
    )
    assert result.status == "OK"
    assert "none clash" in result.hint


def test_preflight_global_skills_bare_clash_with_staged_is_warn(monkeypatch, tmp_path):
    """When a global bare skill has the same name as a staged skill → WARN."""
    staged_dir = tmp_path / "staged"
    staged_dir.mkdir()
    (staged_dir / "scrape-codegen").mkdir()
    (staged_dir / "scrape-codegen" / "SKILL.md").write_text("# staged")

    class FakeCfg:
        skills_dirs = [staged_dir]

    result = _run_preflight_check(
        monkeypatch, ["scrape-codegen", "plugin:foo"], cfg=FakeCfg()
    )
    assert result.status == "WARN"
    assert "scrape-codegen" in result.hint
    assert "clash" in result.hint


def test_preflight_global_skills_namespaced_clash_with_staged_is_warn(
    monkeypatch, tmp_path
):
    """When a global namespaced skill strips to a staged bare name → WARN."""
    staged_dir = tmp_path / "staged"
    staged_dir.mkdir()
    (staged_dir / "scrape-codegen").mkdir()
    (staged_dir / "scrape-codegen" / "SKILL.md").write_text("# staged")

    class FakeCfg:
        skills_dirs = [staged_dir]

    result = _run_preflight_check(
        monkeypatch, ["plugin-a:scrape-codegen", "other"], cfg=FakeCfg()
    )
    assert result.status == "WARN"
    # The warning shows the FULL namespaced name so the user knows what to remove.
    assert "plugin-a:scrape-codegen" in result.hint
    assert "clash" in result.hint


def test_preflight_global_skills_ignores_missing_staged_dir(monkeypatch, tmp_path):
    """A missing skills_dir entry is skipped gracefully."""

    class FakeCfg:
        skills_dirs = [tmp_path / "does-not-exist"]

    result = _run_preflight_check(monkeypatch, ["scrape-codegen"], cfg=FakeCfg())
    assert result.status == "OK"
    assert "none clash" in result.hint
