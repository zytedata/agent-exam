"""Helpers for walking a normalized trajectory.

Every trajectory-walking assertion uses these — don't duplicate the
recursion logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .schemas import SkillInvocation, ToolCallBlock, Turn

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


def walk_turns(trajectory: list[Turn]) -> Iterator[Turn]:
    """Yield every Turn in depth-first pre-order, recursing into subagents.

    An assistant turn with a subagent-spawning tool call yields the parent
    turn first, then the subagent's turns, then returns to the parent's
    sibling turn. This matches "what happened, in time order" across the
    whole tree.
    """
    for turn in trajectory:
        yield turn
        for block in turn.content:
            if isinstance(block, ToolCallBlock) and block.subagent:
                yield from walk_turns(block.subagent)


def iter_tool_calls(
    trajectory: list[Turn], include_subagents: bool = True
) -> Iterator[ToolCallBlock]:
    """Yield every ToolCallBlock in trajectory order.

    By default recurses into subagents so assertions like
    `tool_not_called: WebFetch` catch subagent-issued calls.
    """
    for turn in trajectory:
        for block in turn.content:
            if isinstance(block, ToolCallBlock):
                yield block
                if include_subagents and block.subagent:
                    yield from iter_tool_calls(block.subagent, include_subagents)


def count_tool_calls(
    trajectory: list[Turn], name: str, include_subagents: bool = True
) -> int:
    return sum(
        1
        for call in iter_tool_calls(trajectory, include_subagents)
        if call.name == name
    )


def first_skill_invocation(trajectory: list[Turn]) -> SkillInvocation | None:
    """Return the first SkillInvocation across the trajectory tree, or None."""
    for turn in walk_turns(trajectory):
        if turn.skill_invocations:
            return turn.skill_invocations[0]
    return None


def find_skill_invocation(
    trajectory: list[Turn],
    expected: str,
    is_same_skill: Callable[[str, str], bool],
) -> SkillInvocation | None:
    """Return the first SkillInvocation matching `expected`, or None.

    `is_same_skill` is the provider's name comparator — typically
    `context.provider.is_same_skill`. Distinct from
    `first_skill_invocation`, which doesn't filter by target.
    """
    for turn in walk_turns(trajectory):
        for inv in turn.skill_invocations:
            if is_same_skill(inv.skill_name, expected):
                return inv
    return None
