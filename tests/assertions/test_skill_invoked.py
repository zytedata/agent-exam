from __future__ import annotations

from fixtures.canned_run_result import (
    assistant_turn,
    run_result,
    skill_inv,
    tool_call,
    user_turn,
)

from agent_exam.assertions import skill_invoked
from agent_exam.assertions.skill_invoked import SkillInvokedConfig
from agent_exam.providers.base import Provider
from agent_exam.scoring_context import ScoringContext


def _cfg(skill="scrape-codegen") -> SkillInvokedConfig:
    return SkillInvokedConfig(skill=skill)


def test_passes_when_skill_present(cwd, ctx):
    turn = assistant_turn(skill_invocations=[skill_inv("scrape-codegen")])
    r = skill_invoked.check(_cfg(), run_result([user_turn("x"), turn]), cwd, ctx)
    assert r.pass_
    assert "invoked" in r.reason


def test_fails_when_skill_absent(cwd, ctx):
    turn = assistant_turn(skill_invocations=[skill_inv("other-skill")])
    r = skill_invoked.check(_cfg(), run_result([user_turn("x"), turn]), cwd, ctx)
    assert not r.pass_
    assert "not invoked" in r.reason


def test_catches_subagent_invocations(cwd, ctx):
    sub = assistant_turn(skill_invocations=[skill_inv("scrape-codegen")])
    parent = assistant_turn(tool_call("Agent", tool_use_id="p", subagent=[sub]))
    r = skill_invoked.check(_cfg(), run_result([user_turn("x"), parent]), cwd, ctx)
    assert r.pass_


def test_inverts_when_skill_excluded_and_absent(cwd):
    """In --without-skill runs, asserting on the dropped skill flips:
    absence becomes a pass."""
    turn = assistant_turn(skill_invocations=[skill_inv("other-skill")])
    ctx = ScoringContext(
        provider=Provider(), skills_excluded=frozenset({"scrape-codegen"})
    )
    r = skill_invoked.check(_cfg(), run_result([user_turn("x"), turn]), cwd, ctx)
    assert r.pass_
    assert "excluded" in r.reason
    assert r.details.get("excluded") is True


def test_inverts_when_skill_excluded_and_present(cwd):
    """In --without-skill runs, the dropped skill firing anyway is a fail."""
    turn = assistant_turn(skill_invocations=[skill_inv("scrape-codegen")])
    ctx = ScoringContext(
        provider=Provider(), skills_excluded=frozenset({"scrape-codegen"})
    )
    r = skill_invoked.check(_cfg(), run_result([user_turn("x"), turn]), cwd, ctx)
    assert not r.pass_
    assert "excluded" in r.reason


def test_other_excluded_skill_does_not_invert(cwd):
    """Asserting on a skill that's NOT in the excluded set works normally
    even when other skills are excluded."""
    turn = assistant_turn(skill_invocations=[skill_inv("scrape-codegen")])
    ctx = ScoringContext(
        provider=Provider(), skills_excluded=frozenset({"some-other-skill"})
    )
    r = skill_invoked.check(_cfg(), run_result([user_turn("x"), turn]), cwd, ctx)
    assert r.pass_


def test_uses_provider_matcher(cwd):
    """The assertion routes through context.provider.is_same_skill, so a
    provider with custom matching rules drives the comparison."""
    calls: list[tuple[str, str]] = []

    class CaseInsensitiveProvider(Provider):
        @staticmethod
        def is_same_skill(detected: str, target: str) -> bool:
            calls.append((detected, target))
            return detected.lower() == target.lower()

    turn = assistant_turn(skill_invocations=[skill_inv("Scrape-Codegen")])
    ctx = ScoringContext(provider=CaseInsensitiveProvider())
    r = skill_invoked.check(_cfg(), run_result([user_turn("x"), turn]), cwd, ctx)
    assert r.pass_
    assert calls, "provider matcher was not used"


def test_missing_provider_fails_fast():
    """ScoringContext requires a provider; constructing without one raises."""
    import pytest

    with pytest.raises(TypeError):
        ScoringContext()  # type: ignore[call-arg]
