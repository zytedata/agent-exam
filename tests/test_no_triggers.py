"""Tests for `--no-triggers`.

The same trigger-filter `--without-skill` applies implicitly, exposed as
its own flag so the with-skill half of a reality-check comparison covers
exactly the same tasks. Uses the dummy provider — no real subprocess.
"""

from __future__ import annotations

import json
from textwrap import dedent

import pytest

from agent_exam.config import load_config
from agent_exam.errors import UsageError
from agent_exam.runner import RunRequest, run


def _project(root, *, suite: str, trigger_only: bool = False):
    (root / "evals" / "suites" / suite / "tasks").mkdir(parents=True)
    (root / "evals" / "fixtures").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[tool.agent-exam]\nevals_dir = "evals"\n')
    (root / "evals" / "config.yaml").write_text(
        dedent(
            """\
            default_harness: dummy
            skills_dirs: []
            providers:
              dummy:
                judge_model: haiku
            """
        )
    )
    if not trigger_only:
        (root / "evals" / "suites" / suite / "tasks" / "basic.yaml").write_text(
            dedent(
                """\
                kind: execute
                prompt: do the thing
                assertions: []
                """
            )
        )
    (root / "evals" / "suites" / suite / "tasks" / "trigger.yaml").write_text(
        dedent(
            f"""\
            kind: trigger
            skill: {suite}
            positive: [ping]
            """
        )
    )


def test_no_triggers_drops_trigger_tasks_but_keeps_the_skill(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    _project(root, suite="skill-a")
    monkeypatch.chdir(root)

    cfg = load_config(root)
    req = RunRequest(
        specs=[("skill-a", None)],
        provider="dummy",
        model="",
        k=1,
        n_parallel=1,
        without_skill=False,
        no_triggers=True,
    )
    run(cfg, req)

    run_dir = min((root / "evals" / "runs").iterdir())
    rj = json.loads((run_dir / "run.json").read_text())
    # Ordinary run mode — the skill stays in the bundle.
    assert rj["run_mode"] == "run"
    assert rj["config"]["no_triggers"] is True
    assert rj["config"]["without_skill"] is False
    assert rj["config"]["skills_excluded"] == []

    suite_dir = run_dir / "artifacts" / "skill-a"
    assert sorted(p.name for p in suite_dir.iterdir()) == ["basic"]


def test_without_skill_records_no_triggers(tmp_path, monkeypatch):
    """--without-skill implies the filter, so run.json reports it too."""
    root = tmp_path / "proj"
    _project(root, suite="skill-a")
    monkeypatch.chdir(root)

    cfg = load_config(root)
    req = RunRequest(
        specs=[("skill-a", None)],
        provider="dummy",
        model="",
        k=1,
        n_parallel=1,
        without_skill=True,
    )
    run(cfg, req)

    run_dir = min((root / "evals" / "runs").iterdir())
    rj = json.loads((run_dir / "run.json").read_text())
    assert rj["config"]["no_triggers"] is True
    assert rj["config"]["skills_excluded"] == ["skill-a"]


def test_no_triggers_on_a_trigger_only_suite_errors(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    _project(root, suite="trigger-only", trigger_only=True)
    monkeypatch.chdir(root)

    cfg = load_config(root)
    req = RunRequest(
        specs=[("trigger-only", None)],
        provider="dummy",
        model="",
        k=1,
        n_parallel=1,
        without_skill=False,
        no_triggers=True,
    )
    with pytest.raises(UsageError, match=r"no kind: execute.*--no-triggers"):
        run(cfg, req)
