from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from .._models import _ScalarShorthandModel
from ..schemas import AssertionResult, RunResult
from ..trajectory_walk import first_skill_invocation

if TYPE_CHECKING:
    from pathlib import Path


class FirstSkillConfig(_ScalarShorthandModel):
    """`first_skill: scrape-codegen` or `first_skill: {skill: scrape-codegen}`."""

    _shorthand_key: ClassVar[str] = "skill"
    skill: str


def check(
    config: FirstSkillConfig, result: RunResult, cwd: Path, context: Any
) -> AssertionResult:
    expected = config.skill
    first = first_skill_invocation(result.trajectory)
    if first is None:
        return AssertionResult(
            pass_=False,
            reason=f"no skill invocation in trajectory (expected {expected!r})",
            details={"expected": expected},
        )
    ok = context.provider.is_same_skill(first.skill_name, expected)
    return AssertionResult(
        pass_=ok,
        reason=(
            f"first skill was {first.skill_name!r}"
            if ok
            else f"first skill was {first.skill_name!r}, expected {expected!r}"
        ),
        details={
            "expected": expected,
            "actual": first.skill_name,
            "trigger_kind": first.trigger_kind,
        },
    )
