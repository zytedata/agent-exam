from __future__ import annotations

from fixtures.canned_run_result import (
    assistant_turn,
    run_result,
    skill_inv,
    user_turn,
)

from agent_exam.assertions import first_skill
from agent_exam.assertions.first_skill import FirstSkillConfig


def test_matches_first_skill(cwd, ctx):
    turn = assistant_turn(
        skill_invocations=[skill_inv("scrape-codegen"), skill_inv("scrape")]
    )
    r = first_skill.check(
        FirstSkillConfig(skill="scrape-codegen"),
        run_result([user_turn("x"), turn]),
        cwd,
        ctx,
    )
    assert r.pass_


def test_mismatched_skill(cwd, ctx):
    turn = assistant_turn(skill_invocations=[skill_inv("scrape-explore-site")])
    r = first_skill.check(
        FirstSkillConfig(skill="scrape-codegen"),
        run_result([user_turn("x"), turn]),
        cwd,
        ctx,
    )
    assert not r.pass_
    assert "scrape-explore-site" in r.reason


def test_no_skill_invocation(cwd, ctx):
    r = first_skill.check(
        FirstSkillConfig(skill="scrape-codegen"),
        run_result([user_turn("x"), assistant_turn()]),
        cwd,
        ctx,
    )
    assert not r.pass_
    assert "no skill" in r.reason
