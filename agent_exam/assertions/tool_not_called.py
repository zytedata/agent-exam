from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from .._models import _ScalarShorthandModel
from ..schemas import AssertionResult, RunResult
from ..trajectory_walk import count_tool_calls

if TYPE_CHECKING:
    from pathlib import Path


class ToolNotCalledConfig(_ScalarShorthandModel):
    """`tool_not_called: WebFetch` or `tool_not_called: {name: WebFetch}`."""

    _shorthand_key: ClassVar[str] = "name"
    name: str


def check(config: ToolNotCalledConfig, result: RunResult, cwd: Path) -> AssertionResult:
    count = count_tool_calls(result.trajectory, config.name)
    return AssertionResult(
        pass_=count == 0,
        reason=(
            f"{config.name} not called"
            if count == 0
            else f"{config.name} called {count} time{'s' if count != 1 else ''}"
        ),
        details={"name": config.name, "count": count},
    )
