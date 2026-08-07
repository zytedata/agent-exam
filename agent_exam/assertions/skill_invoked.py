"""Assert a skill was invoked at least once in the trajectory.

Inverse of `skill_not_invoked`. With a twist: reality-check runs
(`--without-skill`, `--no-skills`) intentionally drop skills from the
bundle, so a `skill_invoked: <dropped-skill>` assertion would
always fail and add noise to the reality-check report. When the
asserted skill is in the run's `skills_excluded` set, the assertion
inverts (pass-on-absent, fail-on-present) so the same task YAML stays
meaningful under both normal and reality-check modes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from .._models import _ScalarShorthandModel
from ..schemas import AssertionResult, RunResult
from ..trajectory_walk import find_skill_invocation

if TYPE_CHECKING:
    from pathlib import Path


class SkillInvokedConfig(_ScalarShorthandModel):
    """`skill_invoked: scrape-codegen` or `skill_invoked: {skill: ...}`."""

    _shorthand_key: ClassVar[str] = "skill"
    skill: str


def check(
    config: SkillInvokedConfig, result: RunResult, cwd: Path, context: Any
) -> AssertionResult:
    expected = config.skill
    is_same_skill = context.provider.is_same_skill
    invoked = find_skill_invocation(result.trajectory, expected, is_same_skill)
    excluded = any(is_same_skill(name, expected) for name in context.skills_excluded)

    if excluded:
        # Reality-check inversion: the skill is supposed to be absent.
        if invoked is None:
            return AssertionResult(
                pass_=True,
                reason=f"skill {expected!r} not invoked (excluded from bundle)",
                details={"skill": expected, "excluded": True},
            )
        return AssertionResult(
            pass_=False,
            reason=(
                f"skill {expected!r} was invoked despite being excluded "
                f"from the bundle (trigger_kind={invoked.trigger_kind})"
            ),
            details={
                "skill": expected,
                "excluded": True,
                "trigger_kind": invoked.trigger_kind,
            },
        )

    if invoked is not None:
        return AssertionResult(
            pass_=True,
            reason=f"skill {expected!r} invoked (trigger_kind={invoked.trigger_kind})",
            details={"skill": expected, "trigger_kind": invoked.trigger_kind},
        )
    return AssertionResult(
        pass_=False,
        reason=f"skill {expected!r} not invoked",
        details={"skill": expected},
    )
