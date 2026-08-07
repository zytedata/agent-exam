"""Integration tests for skill staging placement in the attempt cwd.

These tests run the full ``runner.run()`` pipeline with the dummy provider
(so no real LLM calls) and then inspect the ephemeral tmp root to verify
that skills are discovered at the project root (cwd) rather than in a
parent directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

from agent_exam.config import load_config
from agent_exam.runner import RunRequest, run


def _setup_project(tmp_path: Path) -> Path:
    """Create a minimal project tree with one skill and one execute task."""
    root = tmp_path / "proj"
    (root / "evals" / "suites" / "skill-a" / "tasks").mkdir(parents=True)
    (root / "evals" / "fixtures").mkdir(parents=True)
    (root / "build" / "dummy" / "skills" / "skill-a").mkdir(parents=True)
    (root / "build" / "dummy" / "skills" / "skill-a" / "SKILL.md").write_text(
        "# skill-a"
    )
    (root / "build" / "dummy" / "skills" / "skill-b").mkdir(parents=True)
    (root / "build" / "dummy" / "skills" / "skill-b" / "SKILL.md").write_text(
        "# skill-b"
    )
    (root / "pyproject.toml").write_text('[tool.agent-exam]\nevals_dir = "evals"\n')
    (root / "evals" / "config.yaml").write_text(
        dedent(
            """\
            default_harness: dummy
            skills_dirs:
              - ./build/dummy/skills
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
    return root


def test_skills_staged_in_attempt_cwd(tmp_path):
    """Skills are discovered at cwd level, not in a parent directory."""
    root = _setup_project(tmp_path)

    cfg = load_config(root)
    req = RunRequest(
        specs=[("skill-a", None)],
        provider="dummy",
        model="",
        k=1,
        n_parallel=1,
        without_skill=False,
        cleanup_tmp_root=False,
    )
    exit_code = run(cfg, req)
    assert exit_code == 0

    # Find the run directory and read tmp_root from run.json.
    runs = sorted((root / "evals" / "runs").iterdir())
    assert len(runs) == 1
    run_json = json.loads((runs[0] / "run.json").read_text())
    tmp_root = Path(run_json["config"]["tmp_root"])
    assert tmp_root.exists()

    # Find the runtime cwd that was created inside tmp_root.
    runtime_cwds = [
        d for d in tmp_root.iterdir() if d.is_dir() and d.name != "triggers"
    ]
    assert len(runtime_cwds) == 1
    runtime_cwd = runtime_cwds[0]

    # Skills staged IN the attempt cwd (project root).
    skill_dir = runtime_cwd / ".claude" / "skills" / "skill-a"
    assert skill_dir.is_dir(), f"Expected real directory copy at {skill_dir}"
    assert not skill_dir.is_symlink(), f"Expected real directory copy at {skill_dir}"

    # Skills NOT staged at the run tmp root (parent directory).
    assert not (tmp_root / ".claude").exists(), (
        "Skills should not be staged at the run tmp root"
    )


def test_trigger_with_fixture_gets_per_attempt_cwd(tmp_path):
    """A trigger with `setup.fixture:` is staged like an execute task:
    per-attempt uuid cwd with the fixture contents copied in. No shared
    `triggers/` dir."""
    root = _setup_project(tmp_path)
    # Add a fixture and a negative-only trigger that references it.
    (root / "evals" / "fixtures" / "trig-fix").mkdir()
    (root / "evals" / "fixtures" / "trig-fix" / "marker.txt").write_text("hello")
    (root / "evals" / "suites" / "skill-a" / "tasks" / "trigger.yaml").write_text(
        dedent(
            """\
            kind: trigger
            skill: skill-a
            setup:
              fixture: trig-fix
            negative:
              - say no
            """
        )
    )
    # Remove the basic execute task so the run is trigger-only.
    (root / "evals" / "suites" / "skill-a" / "tasks" / "basic.yaml").unlink()

    cfg = load_config(root)
    req = RunRequest(
        specs=[("skill-a", None)],
        provider="dummy",
        model="",
        k=2,
        n_parallel=1,
        without_skill=False,
        cleanup_tmp_root=False,
    )
    exit_code = run(cfg, req)
    assert exit_code == 0

    runs = sorted((root / "evals" / "runs").iterdir())
    tmp_root = Path(
        json.loads((runs[0] / "run.json").read_text())["config"]["tmp_root"]
    )

    # Fixtured triggers must NOT use the shared `triggers/` cwd.
    assert not (tmp_root / "triggers").exists()
    # Each attempt gets its own uuid cwd with the fixture marker.
    runtime_cwds = [d for d in tmp_root.iterdir() if d.is_dir()]
    assert len(runtime_cwds) == 2
    for cwd in runtime_cwds:
        assert (cwd / "marker.txt").read_text() == "hello"


def test_without_skill_excludes_from_attempt_cwd(tmp_path):
    """--without-skill removes excluded skills from the attempt cwd."""
    root = _setup_project(tmp_path)
    (root / "evals" / "suites" / "skill-a" / "suite.yml").write_text(
        dedent(
            """\
            evaluated_skills:
              - skill-a
            """
        )
    )

    cfg = load_config(root)
    req = RunRequest(
        specs=[("skill-a", None)],
        provider="dummy",
        model="",
        k=1,
        n_parallel=1,
        without_skill=True,
        cleanup_tmp_root=False,
    )
    exit_code = run(cfg, req)
    assert exit_code == 0

    runs = sorted((root / "evals" / "runs").iterdir())
    run_json = json.loads((runs[0] / "run.json").read_text())
    tmp_root = Path(run_json["config"]["tmp_root"])
    assert tmp_root.exists()

    runtime_cwds = [
        d for d in tmp_root.iterdir() if d.is_dir() and d.name != "triggers"
    ]
    assert len(runtime_cwds) == 1
    runtime_cwd = runtime_cwds[0]

    # Excluded skill is absent.
    assert not (runtime_cwd / ".claude" / "skills" / "skill-a").exists()
    # Non-excluded skill is present.
    skill_b = runtime_cwd / ".claude" / "skills" / "skill-b"
    assert skill_b.is_dir()
    assert not skill_b.is_symlink()


def test_no_skills_stages_nothing_in_attempt_cwd(tmp_path):
    """--no-skills stages no skill at all, not just the evaluated one."""
    root = _setup_project(tmp_path)
    (root / "evals" / "suites" / "skill-a" / "suite.yml").write_text(
        dedent(
            """\
            evaluated_skills:
              - skill-a
            """
        )
    )

    cfg = load_config(root)
    req = RunRequest(
        specs=[("skill-a", None)],
        provider="dummy",
        model="",
        k=1,
        n_parallel=1,
        without_skill=False,
        no_skills=True,
        cleanup_tmp_root=False,
    )
    exit_code = run(cfg, req)
    assert exit_code == 0

    runs = sorted((root / "evals" / "runs").iterdir())
    run_json = json.loads((runs[0] / "run.json").read_text())
    assert run_json["run_mode"] == "no-skills"
    # Every discovered skill is recorded as excluded, not just skill-a.
    assert run_json["config"]["skills_excluded"] == ["skill-a", "skill-b"]

    tmp_root = Path(run_json["config"]["tmp_root"])
    runtime_cwds = [
        d for d in tmp_root.iterdir() if d.is_dir() and d.name != "triggers"
    ]
    assert len(runtime_cwds) == 1
    # Nothing staged: no `.claude/` dir is created when the bundle is empty.
    assert not (runtime_cwds[0] / ".claude").exists()
