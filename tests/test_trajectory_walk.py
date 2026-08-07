from __future__ import annotations

from fixtures.canned_run_result import (
    assistant_turn,
    skill_inv,
    tool_call,
    user_turn,
)

from agent_exam.trajectory_walk import (
    count_tool_calls,
    first_skill_invocation,
    iter_tool_calls,
    walk_turns,
)


def test_walk_order_depth_first_preorder():
    # user, assistant with subagent: [assistant], assistant
    sub_turn = assistant_turn(tool_call("InnerTool", tool_use_id="i"))
    outer = assistant_turn(tool_call("Outer", tool_use_id="o", subagent=[sub_turn]))
    tail = assistant_turn(tool_call("Tail", tool_use_id="t"))
    trajectory = [user_turn("go"), outer, tail]

    walked = list(walk_turns(trajectory))
    assert [t.role for t in walked] == ["user", "assistant", "assistant", "assistant"]
    # Walker should reach the inner subagent turn before the tail.
    tool_names = [
        b.name
        for t in walked
        for b in t.content
        if b.__class__.__name__ == "ToolCallBlock"
    ]
    assert tool_names == ["Outer", "InnerTool", "Tail"]


def test_count_tool_calls_across_subagents():
    sub = assistant_turn(tool_call("WebFetch", tool_use_id="s"))
    parent = assistant_turn(tool_call("Skill", tool_use_id="p", subagent=[sub]))
    n = count_tool_calls([parent], "WebFetch")
    assert n == 1


def test_count_ignores_subagents_when_disabled():
    sub = assistant_turn(tool_call("WebFetch", tool_use_id="s"))
    parent = assistant_turn(tool_call("Skill", tool_use_id="p", subagent=[sub]))
    calls = list(iter_tool_calls([parent], include_subagents=False))
    assert [c.name for c in calls] == ["Skill"]


def test_first_skill_invocation_finds_none_if_empty():
    assert first_skill_invocation([user_turn("x"), assistant_turn()]) is None


def test_first_skill_invocation_returns_earliest():
    first_turn = assistant_turn(skill_invocations=[skill_inv("a"), skill_inv("b")])
    later_turn = assistant_turn(skill_invocations=[skill_inv("c")])
    found = first_skill_invocation([first_turn, later_turn])
    assert found is not None
    assert found.skill_name == "a"
