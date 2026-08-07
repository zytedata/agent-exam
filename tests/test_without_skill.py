"""Tests for the `--without-skill` / `--no-skills` reality-check modes.

Covers the two pure-Python pieces — `stage_skills_into(..., exclude=)`
and the runner's trigger-filter + k=1 forcing — without spawning real
claude -p subprocesses.
"""

from __future__ import annotations

import json

from agent_exam.providers.skill_staging import stage_skills_into


def test_stage_excludes_named_skill(tmp_path):
    src = tmp_path / "skills"
    for name in ("scrape-codegen", "scrape-scrapy-cloud", "scrape-explore-site"):
        (src / name).mkdir(parents=True)
        (src / name / "SKILL.md").write_text(f"# {name}")

    run_dir = tmp_path / "run"
    staged = stage_skills_into(
        run_dir, [src], ".claude/skills", exclude=frozenset({"scrape-scrapy-cloud"})
    )
    assert staged == ["scrape-codegen", "scrape-explore-site"]
    # `.claude/skills/` is what Claude Code's walk-up looks for.
    target = run_dir / ".claude" / "skills"
    assert target.is_dir()
    linked = sorted(p.name for p in target.iterdir())
    assert linked == ["scrape-codegen", "scrape-explore-site"]
    assert not (target / "scrape-scrapy-cloud").exists()
    # Entries are real directory copies, not symlinks.
    assert (target / "scrape-codegen").is_dir()
    assert not (target / "scrape-codegen").is_symlink()


def test_stage_no_exclude_keeps_everything(tmp_path):
    src = tmp_path / "skills"
    (src / "a").mkdir(parents=True)
    (src / "a" / "SKILL.md").write_text("")
    (src / "b").mkdir()
    (src / "b" / "SKILL.md").write_text("")

    run_dir = tmp_path / "run"
    staged = stage_skills_into(run_dir, [src], ".claude/skills")
    assert staged == ["a", "b"]


def test_stage_exclude_nonexistent_skill_is_noop(tmp_path):
    src = tmp_path / "skills"
    (src / "a").mkdir(parents=True)
    (src / "a" / "SKILL.md").write_text("")

    run_dir = tmp_path / "run"
    staged = stage_skills_into(
        run_dir, [src], ".claude/skills", exclude=frozenset({"does-not-exist"})
    )
    assert staged == ["a"]


def test_stage_empty_skills_dirs_is_noop(tmp_path):
    run_dir = tmp_path / "run"
    staged = stage_skills_into(run_dir, [], ".claude/skills")
    assert staged == []
    # No `.claude/` dir created when there's nothing to stage.
    assert not (run_dir / ".claude").exists()


def test_runner_without_skill_drops_trigger_tasks(tmp_path, monkeypatch):
    """runner.run() with without_skill=True should skip kind: trigger tasks.

    Tests the filtering logic end-to-end with the dummy provider so no
    real claude -p subprocess is required.
    """
    from textwrap import dedent

    # Minimal project tree.
    root = tmp_path / "proj"
    (root / "evals" / "suites" / "skill-a" / "tasks").mkdir(parents=True)
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
    # One execute task and one trigger task.
    (root / "evals" / "suites" / "skill-a" / "tasks" / "basic.yaml").write_text(
        dedent(
            """\
            kind: execute
            prompt: do the thing
            assertions: []
            """
        )
    )
    (root / "evals" / "suites" / "skill-a" / "tasks" / "trigger.yaml").write_text(
        dedent(
            """\
            kind: trigger
            skill: skill-a
            positive: [ping]
            """
        )
    )
    monkeypatch.chdir(root)

    from agent_exam.config import load_config
    from agent_exam.runner import RunRequest, run

    cfg = load_config(root)
    req = RunRequest(
        specs=[("skill-a", None)],
        provider="dummy",
        model="",
        k=3,
        n_parallel=1,
        without_skill=True,
    )
    exit_code = run(cfg, req)

    # Without-skill exits 0 regardless of assertion failures.
    assert exit_code == 0

    # Inspect the run directory: only one task ran, k=3 attempts per task.
    runs = sorted((root / "evals" / "runs").iterdir())
    assert len(runs) == 1
    run_dir = runs[0]
    rj = json.loads((run_dir / "run.json").read_text())
    assert rj["run_mode"] == "without-skill"
    assert rj["config"]["skills_excluded"] == ["skill-a"]
    assert rj["config"]["k"] == 3

    # Only the execute task landed on disk — trigger was filtered.
    suite_dir = run_dir / "artifacts" / "skill-a"
    tasks_ran = sorted(p.name for p in suite_dir.iterdir())
    assert tasks_ran == ["basic"]
    attempts = sorted(p.name for p in (suite_dir / "basic").iterdir())
    assert attempts == ["attempt-1", "attempt-2", "attempt-3"]


def test_runner_without_skill_uses_evaluated_skills_from_suite_yml(
    tmp_path, monkeypatch
):
    """suite.yml evaluated_skills overrides the default suite-name exclusion."""
    from textwrap import dedent

    root = tmp_path / "proj"
    (root / "evals" / "suites" / "skill-a" / "tasks").mkdir(parents=True)
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
    (root / "evals" / "suites" / "skill-a" / "suite.yml").write_text(
        dedent(
            """\
            evaluated_skills:
              - skill-a
              - skill-a-internal
            """
        )
    )
    (root / "evals" / "suites" / "skill-a" / "tasks" / "basic.yaml").write_text(
        dedent(
            """\
            kind: execute
            prompt: do the thing
            assertions: []
            """
        )
    )
    monkeypatch.chdir(root)

    from agent_exam.config import load_config
    from agent_exam.runner import RunRequest, run

    cfg = load_config(root)
    req = RunRequest(
        specs=[("skill-a", None)],
        provider="dummy",
        model="",
        k=1,
        n_parallel=1,
        without_skill=True,
    )
    exit_code = run(cfg, req)
    assert exit_code == 0

    runs = sorted((root / "evals" / "runs").iterdir())
    assert len(runs) == 1
    run_dir = runs[0]
    rj = json.loads((run_dir / "run.json").read_text())
    assert rj["config"]["skills_excluded"] == ["skill-a", "skill-a-internal"]


def test_load_suite_config_validates_evaluated_skills_type(tmp_path, monkeypatch):
    """Bad evaluated_skills values raise UsageError at load time."""

    import pytest

    from agent_exam.errors import UsageError
    from agent_exam.tasks import load_suite_config

    root = tmp_path / "proj"
    (root / "evals" / "suites" / "bad-suite" / "tasks").mkdir(parents=True)
    (root / "evals" / "suites" / "bad-suite" / "suite.yml").write_text(
        "evaluated_skills: not-a-list\n"
    )
    monkeypatch.chdir(root)

    with pytest.raises(UsageError, match=r"evaluated_skills.*valid list"):
        load_suite_config(root / "evals", "bad-suite")


def test_runner_no_skills_drops_trigger_tasks(tmp_path, monkeypatch):
    """--no-skills is a reality check too: triggers skipped, exit 0."""
    from textwrap import dedent

    root = tmp_path / "proj"
    (root / "evals" / "suites" / "skill-a" / "tasks").mkdir(parents=True)
    (root / "skills" / "skill-a").mkdir(parents=True)
    (root / "skills" / "skill-a" / "SKILL.md").write_text("# skill-a")
    (root / "skills" / "unrelated").mkdir(parents=True)
    (root / "skills" / "unrelated" / "SKILL.md").write_text("# unrelated")
    (root / "pyproject.toml").write_text('[tool.agent-exam]\nevals_dir = "evals"\n')
    (root / "evals" / "config.yaml").write_text(
        dedent(
            """\
            default_harness: dummy
            skills_dirs:
              - ./skills
            providers:
              dummy:
                judge_model: haiku
            """
        )
    )
    (root / "evals" / "suites" / "skill-a" / "tasks" / "basic.yaml").write_text(
        dedent(
            """\
            kind: execute
            prompt: do the thing
            assertions: []
            """
        )
    )
    (root / "evals" / "suites" / "skill-a" / "tasks" / "trigger.yaml").write_text(
        dedent(
            """\
            kind: trigger
            skill: skill-a
            positive: [ping]
            """
        )
    )
    monkeypatch.chdir(root)

    from agent_exam.config import load_config
    from agent_exam.runner import RunRequest, run

    cfg = load_config(root)
    req = RunRequest(
        specs=[("skill-a", None)],
        provider="dummy",
        model="",
        k=1,
        n_parallel=1,
        without_skill=False,
        no_skills=True,
    )
    assert run(cfg, req) == 0

    runs = sorted((root / "evals" / "runs").iterdir())
    assert len(runs) == 1
    rj = json.loads((runs[0] / "run.json").read_text())
    assert rj["run_mode"] == "no-skills"
    assert rj["config"]["no_skills"] is True
    # Skills outside the suite's own are excluded too.
    assert rj["config"]["skills_excluded"] == ["skill-a", "unrelated"]

    suite_dir = runs[0] / "artifacts" / "skill-a"
    assert sorted(p.name for p in suite_dir.iterdir()) == ["basic"]


def test_no_skills_conflicts_with_without_skill():
    """The two reality-check flags are mutually exclusive."""
    import pytest

    from agent_exam.errors import UsageError
    from agent_exam.runner import RunRequest

    with pytest.raises(UsageError, match="mutually exclusive"):
        RunRequest(
            specs=[("skill-a", None)],
            provider="dummy",
            model="",
            k=1,
            n_parallel=1,
            without_skill=True,
            no_skills=True,
        )


def test_runner_without_skill_requires_execute_tasks(tmp_path, monkeypatch):
    """A suite with ONLY trigger tasks errors cleanly in reality-check mode."""
    from textwrap import dedent

    import pytest

    from agent_exam.errors import UsageError

    root = tmp_path / "proj"
    (root / "evals" / "suites" / "trigger-only" / "tasks").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[tool.agent-exam]\nevals_dir = "evals"\n')
    (root / "evals" / "config.yaml").write_text(
        "default_harness: dummy\nskills_dirs: []\nproviders:\n  dummy:\n    judge_model: haiku\n"
    )
    (root / "evals" / "suites" / "trigger-only" / "tasks" / "t.yaml").write_text(
        dedent(
            """\
            kind: trigger
            skill: trigger-only
            positive: [ping]
            """
        )
    )
    monkeypatch.chdir(root)

    from agent_exam.config import load_config
    from agent_exam.runner import RunRequest, run

    cfg = load_config(root)
    req = RunRequest(
        specs=[("trigger-only", None)],
        provider="dummy",
        model="",
        k=1,
        n_parallel=1,
        without_skill=True,
    )
    with pytest.raises(UsageError, match="no kind: execute"):
        run(cfg, req)
