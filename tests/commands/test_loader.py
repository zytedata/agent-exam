from __future__ import annotations

import pytest

from agent_exam.commands._loader import parse_run_spec, parse_task_spec
from agent_exam.errors import UsageError


def test_parse_run_spec_all_forms():
    s = parse_run_spec("run-abc")
    assert s.run_id == "run-abc"
    assert s.suite is None
    assert s.task is None

    s = parse_run_spec("run-abc::suite-x")
    assert s.suite == "suite-x"
    assert s.task is None

    s = parse_run_spec("run-abc::suite-x::task-y")
    assert s.task == "task-y"
    assert s.attempt is None

    s = parse_run_spec("run-abc::suite-x::task-y::attempt-3")
    assert s.attempt == 3


def test_parse_run_spec_rejects_bad_attempt():
    with pytest.raises(UsageError):
        parse_run_spec("r::s::t::attempt-abc")


def test_parse_run_spec_rejects_too_many_segments():
    with pytest.raises(UsageError):
        parse_run_spec("r::s::t::attempt-1::extra")


def test_parse_run_spec_rejects_empty_segment():
    with pytest.raises(UsageError):
        parse_run_spec("r::")


def test_parse_task_spec():
    assert parse_task_spec("scrape::basic") == ("scrape", "basic")


def test_parse_task_spec_suite_only():
    """Suite-only is valid (history <suite>)."""
    assert parse_task_spec("scrape") == ("scrape", None)


def test_parse_task_spec_rejects_too_many_segments():
    with pytest.raises(UsageError):
        parse_task_spec("scrape::basic::attempt-1")


def test_parse_task_spec_rejects_empty():
    with pytest.raises(UsageError):
        parse_task_spec("")

    with pytest.raises(UsageError):
        parse_task_spec("scrape::")
