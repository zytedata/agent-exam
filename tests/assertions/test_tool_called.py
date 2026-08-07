from __future__ import annotations

from fixtures.canned_run_result import (
    assistant_turn,
    run_result,
    tool_call,
    user_turn,
)

from agent_exam.assertions import tool_called, tool_not_called
from agent_exam.assertions.tool_called import ToolCalledConfig
from agent_exam.assertions.tool_not_called import ToolNotCalledConfig


def _trial_with_tools(*names):
    calls = [tool_call(n, tool_use_id=f"tu_{i}") for i, n in enumerate(names)]
    return run_result([user_turn("go"), assistant_turn(*calls)])


def test_called_pass(cwd):
    r = tool_called.check(
        ToolCalledConfig(name="Read"), _trial_with_tools("Read", "Write"), cwd
    )
    assert r.pass_
    assert "called 1 time" in r.reason


def test_called_fail(cwd):
    r = tool_called.check(
        ToolCalledConfig(name="WebFetch"), _trial_with_tools("Read"), cwd
    )
    assert not r.pass_


def test_not_called_pass(cwd):
    r = tool_not_called.check(
        ToolNotCalledConfig(name="WebFetch"), _trial_with_tools("Read"), cwd
    )
    assert r.pass_


def test_not_called_fail(cwd):
    r = tool_not_called.check(
        ToolNotCalledConfig(name="WebFetch"),
        _trial_with_tools("WebFetch", "WebFetch"),
        cwd,
    )
    assert not r.pass_
    assert "2 times" in r.reason


def test_subagent_tool_calls_counted(cwd):
    sub_turn = assistant_turn(tool_call("WebFetch", tool_use_id="tu_sub"))
    parent_call = tool_call("Skill", tool_use_id="tu_parent", subagent=[sub_turn])
    r = tool_not_called.check(
        ToolNotCalledConfig(name="WebFetch"),
        run_result([user_turn("go"), assistant_turn(parent_call)]),
        cwd,
    )
    assert not r.pass_, "subagent-issued WebFetch should be caught"
