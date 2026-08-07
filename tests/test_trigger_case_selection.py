"""Selecting a single fanned-out trigger case via `<suite>::<task>::<n>`.

The run-spec parser maps `::<n>` to the fan-out case name `<task>-<n>`, and
`load_suite` resolves that name to the one case even though it has no file of
its own.
"""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

import click
import pytest

from agent_exam.cli import _parse_suite_spec
from agent_exam.errors import UsageError
from agent_exam.tasks import load_suite

if TYPE_CHECKING:
    from pathlib import Path

TRIGGER = dedent(
    """\
    kind: trigger
    skill: myskill
    positive:
      - first positive prompt
      - second positive prompt
    negative:
      - a negative prompt
    """
)


def _suite(tmp_path: Path) -> Path:
    evals = tmp_path / "evals"
    (evals / "suites" / "s" / "tasks").mkdir(parents=True)
    (evals / "suites" / "s" / "tasks" / "trigger.yaml").write_text(TRIGGER)
    return evals


# --- parser ----------------------------------------------------------------


def test_parse_two_and_three_segment_specs():
    assert _parse_suite_spec("s") == ("s", None)
    assert _parse_suite_spec("s::trigger") == ("s", "trigger")
    # ::n folds into the fan-out case name the loader understands.
    assert _parse_suite_spec("s::trigger::0") == ("s", "trigger-0")
    assert _parse_suite_spec("s::trigger::2") == ("s", "trigger-2")


def test_parse_rejects_non_integer_case():
    with pytest.raises(click.UsageError, match="must be a non-negative integer"):
        _parse_suite_spec("s::trigger::last")


def test_parse_rejects_four_segments():
    with pytest.raises(click.UsageError, match="bad suite spec"):
        _parse_suite_spec("s::trigger::0::extra")


# --- load_suite selection --------------------------------------------------


def test_whole_trigger_file_still_loads_all_cases(tmp_path):
    evals = _suite(tmp_path)
    tasks = load_suite(evals, "s", task_filter="trigger")
    assert [t.name for t in tasks] == ["trigger-0", "trigger-1", "trigger-2"]


def test_select_single_positive_case(tmp_path):
    evals = _suite(tmp_path)
    [task] = load_suite(evals, "s", task_filter="trigger-1")
    assert task.name == "trigger-1"
    assert task.should_trigger is True
    assert task.prompt == "second positive prompt"


def test_select_single_negative_case(tmp_path):
    evals = _suite(tmp_path)
    [task] = load_suite(evals, "s", task_filter="trigger-2")
    assert task.name == "trigger-2"
    assert task.should_trigger is False


def test_out_of_range_case_errors(tmp_path):
    evals = _suite(tmp_path)
    with pytest.raises(UsageError, match=r"has 3 case\(s\)"):
        load_suite(evals, "s", task_filter="trigger-9")
