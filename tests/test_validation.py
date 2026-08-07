"""Tests for validation.py — static suite validation — plus the
load-time assertion-type check it relies on."""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

import pytest

from agent_exam.config import load_config
from agent_exam.errors import UsageError
from agent_exam.tasks import load_suite_config, load_task
from agent_exam.validation import validate_suite

if TYPE_CHECKING:
    from pathlib import Path


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "evals" / "suites" / "skill-a" / "tasks").mkdir(parents=True)
    (root / "evals" / "fixtures").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[tool.agent-exam]\nevals_dir = "evals"\n')
    (root / "evals" / "config.yaml").write_text(
        "default_harness: dummy\nproviders:\n  dummy:\n    judge_model: haiku\n"
    )
    return root


def _task(root: Path, name: str, body: str) -> None:
    (root / "evals" / "suites" / "skill-a" / "tasks" / name).write_text(dedent(body))


# --- load-time assertion-type validation -----------------------------------


def test_unknown_assertion_type_rejected_at_load(tmp_path):
    """A typo'd assertion type fails at task-load time, not after an
    agent run during scoring."""
    p = tmp_path / "t.yaml"
    p.write_text("kind: execute\nprompt: x\nassertions:\n  - judege: typo\n")
    with pytest.raises(UsageError, match="unknown assertion type"):
        load_task(p, "s")


def test_known_assertion_type_accepted_at_load(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("kind: execute\nprompt: x\nassertions:\n  - judge: ok\n")
    [task] = load_task(p, "s")
    assert task.assertions[0].type == "judge"


def test_bad_assertion_config_rejected_at_load(tmp_path):
    """A malformed assertion config fails at task-load time, via the
    assertion's shared `validate` — not silently at scoring time."""
    p = tmp_path / "t.yaml"
    p.write_text(
        dedent(
            """\
            kind: execute
            prompt: x
            assertions:
              - tool_count:
                  name: Bash
            """
        )
    )
    with pytest.raises(UsageError, match="tool_count"):
        load_task(p, "s")


def test_unknown_provider_in_assertion_rejected(tmp_path):
    """A typo'd harness name in an assertion's `providers:` meta-field
    fails at load — otherwise the assertion silently skips on every run."""
    p = tmp_path / "t.yaml"
    p.write_text(
        dedent(
            """\
            kind: execute
            prompt: x
            assertions:
              - tool_called: Bash
                providers: [claude_codeX]
            """
        )
    )
    with pytest.raises(UsageError, match="unknown harness name"):
        load_task(p, "s")


def test_codex_cli_provider_section_and_filter_accepted(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(
        dedent(
            """\
            kind: execute
            prompt: x
            codex_cli:
              sandbox: workspace-write
              network_access: true
            assertions:
              - tool_called: command_execution
                providers: [codex_cli]
            """
        )
    )
    [task] = load_task(p, "s")
    assert task.provider_configs["codex_cli"].network_access is True
    assert task.assertions[0].providers == ["codex_cli"]


def test_unknown_top_level_key_rejected(tmp_path):
    """A misspelled top-level key fails at load — otherwise it's silently
    ignored (a typo'd `assertions:` → a task with zero assertions that
    'passes')."""
    p = tmp_path / "t.yaml"
    p.write_text(
        dedent(
            """\
            kind: execute
            prompt: x
            assertons:        # typo
              - judge: ok
            """
        )
    )
    with pytest.raises(UsageError, match=r"assertons.*Extra inputs"):
        load_task(p, "s")


def test_trigger_only_key_rejected_on_execute_task(tmp_path):
    """The schema is kind-specific (discriminated union on `kind`): a
    trigger-only key on an execute task surfaces as a forbidden extra."""
    p = tmp_path / "t.yaml"
    p.write_text("kind: execute\nprompt: x\nassertions: []\npositive: [a]\n")
    with pytest.raises(UsageError, match=r"positive.*Extra inputs"):
        load_task(p, "s")


def test_valid_but_unused_top_level_key_accepted(tmp_path):
    """The allowlist comes from what the loader reads, not what current
    YAMLs use — `known_issue` at the top level is valid even if no
    shipped task happens to set it."""
    p = tmp_path / "t.yaml"
    p.write_text("kind: execute\nprompt: x\nknown_issue: tracked\nassertions: []\n")
    [task] = load_task(p, "s")
    assert task.known_issue == "tracked"


# --- load-time field validation --------------------------------------------


def test_setup_must_be_mapping(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("kind: execute\nprompt: x\nassertions: []\nsetup: nope\n")
    with pytest.raises(UsageError, match=r"setup.*valid dictionary"):
        load_task(p, "s")


def test_unknown_setup_key_rejected(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(
        dedent(
            """\
            kind: execute
            prompt: x
            assertions: []
            setup:
              fixutre: typo
            """
        )
    )
    with pytest.raises(UsageError, match=r"setup.fixutre.*Extra inputs"):
        load_task(p, "s")


def test_setup_fixture_must_be_string(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(
        dedent(
            """\
            kind: execute
            prompt: x
            assertions: []
            setup:
              fixture: [a, b]
            """
        )
    )
    with pytest.raises(UsageError, match=r"setup.fixture.*valid string"):
        load_task(p, "s")


def test_timeout_seconds_rejects_non_numbers(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text('kind: execute\nprompt: x\nassertions: []\ntimeout_seconds: "60"\n')
    with pytest.raises(UsageError, match=r"timeout_seconds.*must be a number"):
        load_task(p, "s")


def test_timeout_seconds_rejects_non_positive(tmp_path):
    for bad in ("-5", "0"):
        p = tmp_path / "t.yaml"
        p.write_text(
            f"kind: execute\nprompt: x\nassertions: []\ntimeout_seconds: {bad}\n"
        )
        with pytest.raises(UsageError, match="greater than 0"):
            load_task(p, "s")


def test_timeout_seconds_accepts_float(tmp_path):
    """Floats are valid — `subprocess` timeouts accept them, and
    sub-second granularity is meaningful for fast probes."""
    p = tmp_path / "t.yaml"
    p.write_text("kind: execute\nprompt: x\nassertions: []\ntimeout_seconds: 60.5\n")
    [task] = load_task(p, "s")
    assert task.timeout_seconds == 60.5


def test_concurrency_group_must_be_string(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("kind: execute\nprompt: x\nassertions: []\nconcurrency_group: [a]\n")
    with pytest.raises(UsageError, match=r"concurrency_group.*valid string"):
        load_task(p, "s")


def test_unknown_suite_yml_key_rejected(tmp_path):
    root = _project(tmp_path)
    (root / "evals" / "suites" / "skill-a" / "suite.yml").write_text(
        "evaluated_skils: [skill-a]\n"  # typo
    )
    with pytest.raises(UsageError, match="Extra inputs"):
        load_suite_config(root / "evals", "skill-a")


# --- validate_suite --------------------------------------------------------


def test_validate_clean_suite(tmp_path):
    root = _project(tmp_path)
    _task(root, "basic.yaml", "kind: execute\nprompt: x\nassertions: []\n")
    results = validate_suite(load_config(root), "skill-a")
    assert all(r.status == "OK" for r in results)
    assert any(r.name == "skill-a: task files parse" for r in results)


def test_validate_counts_files_not_expanded_tasks(tmp_path):
    """A trigger file fans out into one task per case, but the check
    counts source files — that's the unit it actually validates."""
    root = _project(tmp_path)
    _task(root, "exec.yaml", "kind: execute\nprompt: x\nassertions: []\n")
    _task(
        root,
        "trig.yaml",
        """
        kind: trigger
        skill: skill-a
        positive: [a, b, c]
        negative: [d, e]
        """,
    )
    results = validate_suite(load_config(root), "skill-a")
    parse = next(r for r in results if r.name == "skill-a: task files parse")
    # 2 files, even though trig.yaml expands to 5 tasks.
    assert parse.hint == "2 file(s)"


def test_validate_catches_parse_error(tmp_path):
    root = _project(tmp_path)
    _task(root, "bad.yaml", "kind: execute\nprompt: x\nassertions:\n  - judege: typo\n")
    results = validate_suite(load_config(root), "skill-a")
    assert any(r.status == "FAIL" for r in results)


def test_validate_catches_bad_assertion_config(tmp_path):
    root = _project(tmp_path)
    _task(
        root,
        "bad.yaml",
        """
        kind: execute
        prompt: x
        assertions:
          - file_contains:
              path: a.py
        """,
    )
    results = validate_suite(load_config(root), "skill-a")
    assert any(r.status == "FAIL" for r in results)


def test_validate_catches_missing_fixture(tmp_path):
    root = _project(tmp_path)
    _task(
        root,
        "fix.yaml",
        """
        kind: execute
        prompt: x
        setup:
          fixture: does-not-exist
        assertions: []
        """,
    )
    results = validate_suite(load_config(root), "skill-a")
    fixture_check = [r for r in results if "fixtures exist" in r.name]
    assert fixture_check
    assert fixture_check[0].status == "FAIL"
    assert "does-not-exist" in fixture_check[0].hint


def test_validate_passes_with_existing_fixture(tmp_path):
    root = _project(tmp_path)
    (root / "evals" / "fixtures" / "myfix").mkdir()
    _task(
        root,
        "fix.yaml",
        """
        kind: execute
        prompt: x
        setup:
          fixture: myfix
        assertions: []
        """,
    )
    results = validate_suite(load_config(root), "skill-a")
    assert all(r.status == "OK" for r in results)


def test_validate_task_filter_narrows_scope(tmp_path):
    """task_filter narrows validation — a sibling task's bad fixture
    doesn't fail a single-task scope."""
    root = _project(tmp_path)
    _task(root, "good.yaml", "kind: execute\nprompt: x\nassertions: []\n")
    _task(
        root,
        "bad-fixture.yaml",
        """
        kind: execute
        prompt: x
        setup:
          fixture: missing
        assertions: []
        """,
    )
    results = validate_suite(load_config(root), "skill-a", task_filter="good")
    assert all(r.status == "OK" for r in results)


def test_validate_catches_undeclared_concurrency_group(tmp_path):
    """A `concurrency_group` not declared in config.yaml is caught by
    validate_suite — not just mid-run when the pool builds semaphores."""
    root = _project(tmp_path)
    _task(
        root,
        "t.yaml",
        """
        kind: execute
        prompt: x
        concurrency_group: not-declared
        assertions: []
        """,
    )
    results = validate_suite(load_config(root), "skill-a")
    cg = [r for r in results if "concurrency" in r.name]
    assert cg
    assert cg[0].status == "FAIL"
    assert "not-declared" in cg[0].hint


def test_validate_catches_bad_suite_yml(tmp_path):
    """A malformed suite.yml surfaces in validate_suite, not only when
    the runner loads it."""
    root = _project(tmp_path)
    _task(root, "basic.yaml", "kind: execute\nprompt: x\nassertions: []\n")
    (root / "evals" / "suites" / "skill-a" / "suite.yml").write_text(
        "evaluated_skils: [skill-a]\n"  # typo
    )
    results = validate_suite(load_config(root), "skill-a")
    suite_yml = [r for r in results if "suite.yml" in r.name]
    assert suite_yml
    assert suite_yml[0].status == "FAIL"


# --- skills_dirs default ----------------------------------------------------


def _minimal_project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "evals").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'x'\nversion = '0'\n")
    return root


def test_skills_dir_at_project_root_is_the_default(tmp_path, monkeypatch):
    root = _minimal_project(tmp_path)
    (root / "skills").mkdir()
    monkeypatch.chdir(root)

    assert load_config().skills_dirs == [root / "skills"]


def test_no_skills_dir_leaves_skills_dirs_unset(tmp_path, monkeypatch):
    root = _minimal_project(tmp_path)
    monkeypatch.chdir(root)

    assert load_config().skills_dirs is None


def test_explicit_skills_dirs_wins_over_the_default(tmp_path, monkeypatch):
    root = _minimal_project(tmp_path)
    (root / "skills").mkdir()
    (root / "elsewhere").mkdir()
    (root / "evals" / "config.yaml").write_text("skills_dirs:\n  - ./elsewhere\n")
    monkeypatch.chdir(root)

    assert load_config().skills_dirs == [root / "elsewhere"]


def test_default_does_not_lock_out_the_pre_run_hook(tmp_path, monkeypatch):
    """The default must behave like a config.yaml value, not like a
    config.local.yaml one: a pre-run hook still overrides it."""
    root = _minimal_project(tmp_path)
    (root / "skills").mkdir()
    monkeypatch.chdir(root)

    assert load_config()._skills_dirs_locked is False
