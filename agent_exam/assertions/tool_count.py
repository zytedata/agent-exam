from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field, StrictInt, model_validator

from .._models import _StrictModel
from ..schemas import AssertionResult, RunResult
from ..trajectory_walk import count_tool_calls

if TYPE_CHECKING:
    from pathlib import Path


class ToolCountConfig(_StrictModel):
    """`tool_count: {name: <tool>, exactly: N}` or
    `tool_count: {name: <tool>, min: N, max: N}`.

    `exactly` and `min`/`max` are mutually exclusive; at least one is
    required. Bounds are non-negative integers, with `min <= max`.
    """

    name: str = Field(min_length=1)
    exactly: StrictInt | None = Field(default=None, ge=0)
    min: StrictInt | None = Field(default=None, ge=0)
    max: StrictInt | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_bounds(self) -> ToolCountConfig:
        if self.exactly is None and self.min is None and self.max is None:
            raise ValueError("needs 'exactly', 'min', or 'max'")
        if self.exactly is not None and (self.min is not None or self.max is not None):
            raise ValueError("use either 'exactly' or 'min'/'max', not both")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"'min' ({self.min}) must be <= 'max' ({self.max})")
        return self


def check(config: ToolCountConfig, result: RunResult, cwd: Path) -> AssertionResult:
    name = config.name
    count = count_tool_calls(result.trajectory, name)

    if config.exactly is not None:
        ok = count == config.exactly
        reason = f"{name} called {count}× (expected exactly {config.exactly})"
    else:
        ok = True
        bits = []
        if config.min is not None:
            ok = ok and count >= config.min
            bits.append(f"min={config.min}")
        if config.max is not None:
            ok = ok and count <= config.max
            bits.append(f"max={config.max}")
        reason = f"{name} called {count}× ({', '.join(bits)})"

    return AssertionResult(
        pass_=ok,
        reason=reason,
        details={"name": name, "count": count},
    )
