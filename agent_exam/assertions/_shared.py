"""Config models shared across multiple assertions."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field, StrictBool, field_validator

from .._models import _ScalarShorthandModel


class JudgeConfigBase(_ScalarShorthandModel):
    """Shared config for `judge` and `judge_agent`: a criterion plus
    optional `include_trajectory` / `pass_on`. Accepts the scalar
    shorthand `judge: <criterion>` and the full mapping form."""

    _shorthand_key: ClassVar[str] = "criterion"
    criterion: str
    include_trajectory: StrictBool = True
    pass_on: list[str] | None = Field(default=None, min_length=1)

    @field_validator("criterion")
    @classmethod
    def _non_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty or whitespace")
        return v

    @field_validator("pass_on")
    @classmethod
    def _no_empty_entries(cls, v: list[str] | None) -> list[str] | None:
        if v is not None and not all(s for s in v):
            raise ValueError("entries must be non-empty strings")
        return v
