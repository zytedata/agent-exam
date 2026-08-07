from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pydantic import StrictBool, model_validator

from .._models import _StrictModel
from ..schemas import AssertionResult, RunResult

if TYPE_CHECKING:
    from pathlib import Path


class FileContainsConfig(_StrictModel):
    """`file_contains: {path: ..., pattern: ..., regex?: false}`."""

    path: str
    pattern: str
    # StrictBool — lax `bool` would coerce YAML-style "yes"/"no" strings.
    regex: StrictBool = False

    @model_validator(mode="after")
    def _validate_regex(self) -> FileContainsConfig:
        if self.regex:
            try:
                re.compile(self.pattern)
            except re.error as exc:
                raise ValueError(
                    f"invalid regex pattern {self.pattern!r}: {exc}"
                ) from exc
        return self


def check(config: FileContainsConfig, result: RunResult, cwd: Path) -> AssertionResult:
    full = cwd / config.path
    if not full.is_file():
        return AssertionResult(
            pass_=False,
            reason=f"missing {config.path}",
            details={"path": str(full)},
        )
    text = full.read_text(errors="replace")
    if config.regex:
        found = re.search(config.pattern, text) is not None
    else:
        found = config.pattern in text
    return AssertionResult(
        pass_=found,
        reason=(
            f"{'matched' if found else 'no match for'} "
            f"{config.pattern!r} in {config.path}"
        ),
        details={
            "path": str(full),
            "pattern": config.pattern,
            "regex": config.regex,
        },
    )
