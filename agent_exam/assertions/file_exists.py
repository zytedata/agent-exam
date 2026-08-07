from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from .._models import _ScalarShorthandModel
from ..schemas import AssertionResult, RunResult

if TYPE_CHECKING:
    from pathlib import Path


class FileExistsConfig(_ScalarShorthandModel):
    """`file_exists: out/x.py` or `file_exists: {path: out/x.py}`."""

    _shorthand_key: ClassVar[str] = "path"
    path: str


def check(config: FileExistsConfig, result: RunResult, cwd: Path) -> AssertionResult:
    full = cwd / config.path
    exists = full.is_file()
    return AssertionResult(
        pass_=exists,
        reason=f"{'found' if exists else 'missing'} {config.path}",
        details={"path": str(full)},
    )
