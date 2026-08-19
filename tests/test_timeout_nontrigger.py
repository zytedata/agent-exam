"""A trigger whose routing decision is only observable on the wall clock
reports it as a timeout, so `_settled_on_timeout` decides when that timeout
is really the answer.
"""

from __future__ import annotations

from pathlib import Path

from agent_exam.pool import _settled_on_timeout, _target_tool_already_called
from agent_exam.schemas import (
    Metrics,
    RunResult,
    SkillInvocation,
    Tokens,
    ToolCallBlock,
    Turn,
)
from agent_exam.tasks import Task


def _run(
    n_tool_calls: int,
    skills: tuple[str, ...] = (),
    tools: tuple[str, ...] = (),
) -> RunResult:
    turn = Turn(
        role="assistant",
        content=[
            ToolCallBlock(
                tool_use_id=f"c{i}", name=name, input={}, status="ok", result=""
            )
            for i, name in enumerate(tools)
        ],
        skill_invocations=[
            SkillInvocation(skill_name=s, trigger_kind="skill_tool") for s in skills
        ],
    )
    return RunResult(
        trajectory=[turn],
        metrics=Metrics(
            wall_time_seconds=60.0,
            tokens=Tokens(),
            cost_usd=None,
            peak_context=0,
            turn_count=1,
            n_tool_calls=n_tool_calls,
        ),
    )


def _task(should_trigger: bool | None, target_tool: str | None = None) -> Task:
    return Task(
        suite="scrapy",
        name="t-0",
        kind="trigger" if should_trigger is not None else "execute",
        prompt="Debug this Scrapy pipeline.",
        description=None,
        assertions=[],
        fixture=None,
        env={},
        timeout_seconds=None,
        concurrency_group=None,
        raw={},
        source_path=Path("/tmp/t.yaml"),
        should_trigger=should_trigger,
        target_tool=target_tool,
    )


def test_positive_that_worked_without_a_skill_is_settled():
    assert _settled_on_timeout(_task(True), _run(n_tool_calls=7))


def test_cold_start_timeout_is_not_settled():
    """No tool ran — the agent never got to route, so the timeout stands."""
    assert not _settled_on_timeout(_task(True), _run(n_tool_calls=0))


def test_skill_fire_is_not_settled():
    assert not _settled_on_timeout(_task(True), _run(7, ("scrapy",)))


def test_negative_skill_trigger_and_execute_tasks_keep_the_timeout():
    assert not _settled_on_timeout(_task(False), _run(n_tool_calls=7))
    assert not _settled_on_timeout(_task(None), _run(n_tool_calls=7))


def test_a_positive_tool_trigger_that_ran_a_tool_is_settled():
    """A positive case is cut on the first MCP call, so an agent that reached
    the wall clock having run anything never made one."""
    task = _task(True, target_tool="mcp__files__search")

    assert _settled_on_timeout(task, _run(n_tool_calls=7))
    assert not _settled_on_timeout(task, _run(n_tool_calls=0))


def test_a_positive_tool_trigger_with_only_native_tool_calls_is_settled():
    """`n_tool_calls` alone isn't proof the target's server went unused — a
    native tool call bumps it too, so this re-checks the trajectory itself
    for an MCP call rather than trusting the counter blindly."""
    task = _task(True, target_tool="mcp__files__search")

    assert _settled_on_timeout(task, _run(7, tools=("Bash",)))


def test_a_positive_tool_trigger_with_an_mcp_call_already_recorded_is_not_settled():
    """A race that lets an MCP call slip past the kill signal leaves real
    evidence in the trajectory — this must not paper over it as a routing
    miss."""
    task = _task(True, target_tool="mcp__files__search")

    assert not _settled_on_timeout(task, _run(7, tools=("mcp__github__search",)))


def test_a_negative_tool_trigger_keeps_the_timeout():
    """Nothing cuts a negative case short of the turn ending, so a timeout
    means the agent was still working — the target being absent so far says
    nothing about the call it was about to make."""
    task = _task(False, target_tool="mcp__files__search")

    assert not _settled_on_timeout(task, _run(n_tool_calls=7))


def test_no_partial_trajectory_keeps_the_timeout():
    assert not _settled_on_timeout(_task(True), None)


def test_target_tool_already_called_survives_a_sibling_server_failing():
    """A sibling MCP server reporting broken after the target was already
    called shouldn't erase a decisive pass."""
    task = _task(True, target_tool="mcp__files__search")

    assert _target_tool_already_called(task, _run(1, tools=("mcp__files__search",)))


def test_target_tool_not_called_is_not_already_called():
    task = _task(True, target_tool="mcp__files__search")

    assert not _target_tool_already_called(
        task, _run(1, tools=("mcp__github__search",))
    )
    assert not _target_tool_already_called(task, _run(0))


def test_no_target_tool_is_never_already_called():
    assert not _target_tool_already_called(_task(True), _run(1, tools=("Bash",)))
