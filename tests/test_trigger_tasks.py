from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from agent_exam.errors import UsageError
from agent_exam.tasks import load_task

if TYPE_CHECKING:
    from pathlib import Path


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "trigger.yaml"
    p.write_text(body)
    return p


def test_trigger_expands_positive_and_negative(tmp_path):
    path = _write(
        tmp_path,
        """
kind: trigger
skill: scrape-codegen
positive:
  - Build the project now.
  - Generate code from my spec.
negative:
  - What is Python?
  - Tell me a joke.
""",
    )
    tasks = load_task(path, suite="scrape-codegen")
    assert len(tasks) == 4
    assert all(t.kind == "trigger" and t.stop_on_first_trigger for t in tasks)
    # Positives first, negatives second (deterministic).
    assert [t.should_trigger for t in tasks] == [True, True, False, False]
    assert tasks[0].assertions[0].type == "first_skill"
    assert tasks[2].assertions[0].type == "skill_not_invoked"
    # Names auto-numbered.
    assert [t.name for t in tasks] == [
        "trigger-0",
        "trigger-1",
        "trigger-2",
        "trigger-3",
    ]


def test_trigger_allows_positive_only(tmp_path):
    path = _write(
        tmp_path,
        """
kind: trigger
skill: x
positive:
  - one
""",
    )
    tasks = load_task(path, suite="s")
    assert len(tasks) == 1
    assert tasks[0].should_trigger is True


def test_trigger_allows_negative_only(tmp_path):
    path = _write(
        tmp_path,
        """
kind: trigger
skill: x
negative:
  - one
""",
    )
    tasks = load_task(path, suite="s")
    assert len(tasks) == 1
    assert tasks[0].should_trigger is False


def test_trigger_accepts_fixture_under_setup(tmp_path):
    """Triggers with `setup.fixture:` get per-attempt cwds in pool.py
    (the same path execute tasks take); each emitted case carries the
    fixture name.
    """
    path = _write(
        tmp_path,
        """
kind: trigger
skill: x
setup:
  fixture: my-fixture
positive: [a]
negative: [b]
""",
    )
    tasks = load_task(path, suite="s")
    assert len(tasks) == 2
    assert all(t.fixture == "my-fixture" for t in tasks)


def test_trigger_no_fixture_when_unset(tmp_path):
    path = _write(
        tmp_path,
        """
kind: trigger
skill: x
positive: [a]
""",
    )
    [task] = load_task(path, suite="s")
    assert task.fixture is None


def test_trigger_rejects_missing_target(tmp_path):
    path = _write(
        tmp_path,
        """
kind: trigger
positive: [hi]
""",
    )
    with pytest.raises(UsageError, match="exactly one of"):
        load_task(path, suite="s")


def test_trigger_rejects_empty_positive_and_negative(tmp_path):
    path = _write(
        tmp_path,
        """
kind: trigger
skill: x
""",
    )
    with pytest.raises(UsageError, match="at least one of"):
        load_task(path, suite="s")


def test_trigger_rejects_non_string_case(tmp_path):
    path = _write(
        tmp_path,
        """
kind: trigger
skill: x
positive:
  - prompt: not-a-string
""",
    )
    with pytest.raises(UsageError, match="non-empty string"):
        load_task(path, suite="s")
