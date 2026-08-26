from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from .._models import _ScalarShorthandModel
from ..mcp import is_mcp_tool
from ..schemas import AssertionResult, RunResult
from ..trajectory_walk import iter_tool_calls

if TYPE_CHECKING:
    from pathlib import Path


class FirstToolConfig(_ScalarShorthandModel):
    """`first_tool: mcp__files__search` or
    `first_tool: {name: mcp__files__search}`."""

    _shorthand_key: ClassVar[str] = "name"
    name: str


def check(config: FirstToolConfig, result: RunResult, cwd: Path) -> AssertionResult:
    """Pass when *name* is the first MCP tool the agent reached for.

    Native tools are ignored: an agent greps and reads before it decides
    which tool the request calls for, so only MCP calls carry the routing
    decision.
    """
    expected = config.name
    for call in iter_tool_calls(result.trajectory):
        if call.name == expected:
            return AssertionResult(
                pass_=True,
                reason=f"{expected} called first",
                details={"expected": expected, "actual": expected},
            )
        if is_mcp_tool(call.name):
            return AssertionResult(
                pass_=False,
                reason=f"{call.name} called before {expected}",
                details={"expected": expected, "actual": call.name},
            )
    return AssertionResult(
        pass_=False,
        reason=f"{expected} never called",
        details={"expected": expected, "actual": None},
    )
