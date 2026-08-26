from __future__ import annotations

from fixtures.canned_run_result import (
    assistant_turn,
    run_result,
    tool_call,
    user_turn,
)

from agent_exam.assertions import first_tool
from agent_exam.assertions.first_tool import FirstToolConfig

_TARGET = "mcp__files__search"


def _trial_with_tools(*names):
    calls = [tool_call(n, tool_use_id=f"tu_{i}") for i, n in enumerate(names)]
    return run_result([user_turn("go"), assistant_turn(*calls)])


def test_native_calls_before_the_target_pass(cwd):
    r = first_tool.check(
        FirstToolConfig(name=_TARGET),
        _trial_with_tools("Grep", "Read", _TARGET),
        cwd,
    )
    assert r.pass_


def test_another_mcp_tool_first_fails(cwd):
    r = first_tool.check(
        FirstToolConfig(name=_TARGET),
        _trial_with_tools("Read", "mcp__notes__search", _TARGET),
        cwd,
    )
    assert not r.pass_
    assert "mcp__notes__search" in r.reason


def test_target_never_called_fails(cwd):
    r = first_tool.check(FirstToolConfig(name=_TARGET), _trial_with_tools("Read"), cwd)
    assert not r.pass_
    assert "never called" in r.reason
