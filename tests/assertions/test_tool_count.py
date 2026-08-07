from __future__ import annotations

from fixtures.canned_run_result import (
    assistant_turn,
    run_result,
    tool_call,
)

from agent_exam.assertions import tool_count
from agent_exam.assertions.tool_count import ToolCountConfig


def _trial_with(*names):
    calls = [tool_call(n, tool_use_id=f"tu_{i}") for i, n in enumerate(names)]
    return run_result([assistant_turn(*calls)])


def _cfg(**kw) -> ToolCountConfig:
    return ToolCountConfig.model_validate({"name": kw.pop("name", "Read"), **kw})


def test_exactly_pass(cwd):
    r = tool_count.check(
        _cfg(name="Read", exactly=2), _trial_with("Read", "Read", "Write"), cwd
    )
    assert r.pass_


def test_exactly_fail(cwd):
    r = tool_count.check(_cfg(name="Read", exactly=1), _trial_with("Read", "Read"), cwd)
    assert not r.pass_


def test_min_pass(cwd):
    r = tool_count.check(
        _cfg(name="Read", min=2), _trial_with("Read", "Read", "Read"), cwd
    )
    assert r.pass_


def test_max_fail(cwd):
    r = tool_count.check(_cfg(name="Read", max=1), _trial_with("Read", "Read"), cwd)
    assert not r.pass_


def test_min_and_max(cwd):
    r = tool_count.check(
        _cfg(name="Read", min=1, max=3), _trial_with("Read", "Read"), cwd
    )
    assert r.pass_
