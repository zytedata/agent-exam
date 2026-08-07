from __future__ import annotations

from fixtures.canned_run_result import (
    assistant_turn,
    run_result,
    skill_inv,
    tool_call,
    user_turn,
)

from agent_exam.assertions import skill_not_invoked
from agent_exam.assertions.skill_not_invoked import SkillNotInvokedConfig


def _cfg(skill="scrape-codegen") -> SkillNotInvokedConfig:
    return SkillNotInvokedConfig(skill=skill)


def test_passes_when_skill_absent(cwd, ctx):
    turn = assistant_turn(skill_invocations=[skill_inv("other-skill")])
    r = skill_not_invoked.check(_cfg(), run_result([user_turn("x"), turn]), cwd, ctx)
    assert r.pass_


def test_fails_when_skill_present(cwd, ctx):
    turn = assistant_turn(skill_invocations=[skill_inv("scrape-codegen")])
    r = skill_not_invoked.check(_cfg(), run_result([user_turn("x"), turn]), cwd, ctx)
    assert not r.pass_
    assert "was invoked" in r.reason


def test_catches_subagent_invocations(cwd, ctx):
    sub = assistant_turn(skill_invocations=[skill_inv("scrape-codegen")])
    parent = assistant_turn(tool_call("Agent", tool_use_id="p", subagent=[sub]))
    r = skill_not_invoked.check(_cfg(), run_result([user_turn("x"), parent]), cwd, ctx)
    assert not r.pass_
