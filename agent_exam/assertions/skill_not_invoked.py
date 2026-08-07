from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from .._models import _ScalarShorthandModel
from ..schemas import AssertionResult, RunResult
from ..trajectory_walk import find_skill_invocation

if TYPE_CHECKING:
    from pathlib import Path


class SkillNotInvokedConfig(_ScalarShorthandModel):
    """`skill_not_invoked: foo` or `skill_not_invoked: {skill: foo}`."""

    _shorthand_key: ClassVar[str] = "skill"
    skill: str


def check(
    config: SkillNotInvokedConfig, result: RunResult, cwd: Path, context: Any
) -> AssertionResult:
    expected = config.skill
    inv = find_skill_invocation(
        result.trajectory, expected, context.provider.is_same_skill
    )
    if inv is not None:
        return AssertionResult(
            pass_=False,
            reason=f"skill {expected!r} was invoked (trigger_kind={inv.trigger_kind})",
            details={"skill": expected, "trigger_kind": inv.trigger_kind},
        )
    return AssertionResult(
        pass_=True,
        reason=f"skill {expected!r} not invoked",
        details={"skill": expected},
    )
