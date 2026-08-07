from __future__ import annotations

from typing import TYPE_CHECKING

from fixtures.canned_run_result import assistant_turn, run_result, text, user_turn

from agent_exam.assertions import judge
from agent_exam.assertions.judge import JudgeConfig
from agent_exam.judge import JudgeCache, JudgeCall
from agent_exam.providers.base import Provider
from agent_exam.schemas import Metrics, RunResult, TextBlock, Tokens, Turn
from agent_exam.scoring_context import ScoringContext

if TYPE_CHECKING:
    from pathlib import Path


class _StubProvider(Provider):
    """Provider stub that returns a canned verdict for `judge:` tests.

    Exercises the real `call_judge` dispatch path; only the harness
    subprocess is faked.
    """

    name = "stub"

    def __init__(self, response: str = "ok.\nVERDICT: YES"):
        self.response = response
        self.invocations: list[dict] = []

    def invoke(
        self,
        prompt: str,
        model: str,
        cwd: Path,
        provider_options: dict,
        stop_on_first_skill: bool,
        timeout_seconds: int,
    ) -> RunResult:
        self.invocations.append(
            {
                "prompt": prompt,
                "model": model,
                "cwd": cwd,
                "provider_options": dict(provider_options),
            }
        )
        return RunResult(
            trajectory=[
                Turn(role="user", content=[TextBlock(text=prompt)]),
                Turn(
                    role="assistant",
                    content=[TextBlock(text=self.response)],
                    model=model,
                    tokens=Tokens(input=0, output=0, cache_read=0),
                    context=0,
                ),
            ],
            metrics=Metrics(
                wall_time_seconds=0.0,
                tokens=Tokens(input=0, output=0, cache_read=0),
                cost_usd=0.0,
                peak_context=0,
                turn_count=1,
                raw={"provider": "stub"},
            ),
        )


def _context(
    cache_path,
    *,
    response: str = "ok.\nVERDICT: YES",
    pass_on=None,
) -> ScoringContext:
    prov = _StubProvider(response=response)
    cache = JudgeCache(cache_path)
    jc = JudgeCall(
        provider=prov,
        judge_model="claude-haiku-4-5-test",
        provider_options={},
        timeout_seconds=60,
    )
    return ScoringContext(
        provider=prov,
        judge_call=jc,
        judge_cache=cache,
        judge_pass_on=pass_on or ["YES"],
    )


def _trajectory(final: str = "all done"):
    return [user_turn("do the thing"), assistant_turn(text(final))]


def _cfg(criterion: str = "c", **kw) -> JudgeConfig:
    return JudgeConfig(criterion=criterion, **kw)


def test_criterion_pass(tmp_path, cwd):
    ctx = _context(tmp_path / "judge-cache.json")
    r = judge.check(
        _cfg("Response mentions the task."),
        run_result(_trajectory()),
        cwd,
        context=ctx,
    )
    assert r.pass_
    assert r.details["verdict"] == "YES"
    assert r.details["cache"] == "miss"


def test_verdict_no_fails_by_default(tmp_path, cwd):
    ctx = _context(tmp_path / "c.json", response="no.\nVERDICT: NO")
    r = judge.check(_cfg(), run_result(_trajectory()), cwd, context=ctx)
    assert not r.pass_
    assert r.details["verdict"] == "NO"


def test_unclear_fails_by_default(tmp_path, cwd):
    ctx = _context(tmp_path / "c.json", response="no marker here")
    r = judge.check(_cfg(), run_result(_trajectory()), cwd, context=ctx)
    assert not r.pass_
    assert r.details["verdict"] == "UNCLEAR"


def test_pass_on_override_in_config(tmp_path, cwd):
    ctx = _context(tmp_path / "c.json", response="ok.\nVERDICT: UNCLEAR")
    r = judge.check(
        _cfg(pass_on=["YES", "UNCLEAR"]),
        run_result(_trajectory()),
        cwd,
        context=ctx,
    )
    assert r.pass_


def test_cache_hit_skips_invoke(tmp_path, cwd):
    ctx = _context(tmp_path / "c.json")
    traj = _trajectory("same output")
    judge.check(_cfg("same criterion"), run_result(traj), cwd, context=ctx)
    prov: _StubProvider = ctx.judge_call.provider  # type: ignore[assignment]
    before = len(prov.invocations)
    r2 = judge.check(_cfg("same criterion"), run_result(traj), cwd, context=ctx)
    after = len(prov.invocations)
    assert before == 1, "provider must not be invoked on cache hit"
    assert after == 1, "provider must not be invoked on cache hit"
    assert r2.details["cache"] == "hit"
    assert r2.details["verdict"] == "YES"


def test_cache_miss_on_different_criterion(tmp_path, cwd):
    ctx = _context(tmp_path / "c.json")
    traj = _trajectory("same output")
    judge.check(_cfg("criterion A"), run_result(traj), cwd, context=ctx)
    judge.check(_cfg("criterion B"), run_result(traj), cwd, context=ctx)
    prov: _StubProvider = ctx.judge_call.provider  # type: ignore[assignment]
    assert len(prov.invocations) == 2


def test_cache_miss_on_different_output(tmp_path, cwd):
    ctx = _context(tmp_path / "c.json")
    judge.check(_cfg(), run_result(_trajectory("out1")), cwd, context=ctx)
    judge.check(_cfg(), run_result(_trajectory("out2")), cwd, context=ctx)
    prov: _StubProvider = ctx.judge_call.provider  # type: ignore[assignment]
    assert len(prov.invocations) == 2


def test_missing_context_fails_fast(cwd):
    r = judge.check(_cfg(), run_result(_trajectory()), cwd, context=None)
    assert not r.pass_
    assert "no judge_call" in r.reason


def test_include_trajectory_false_shortens_prompt(tmp_path, cwd):
    ctx = _context(tmp_path / "c.json")
    judge.check(
        _cfg(include_trajectory=False),
        run_result(_trajectory()),
        cwd,
        context=ctx,
    )
    prov: _StubProvider = ctx.judge_call.provider  # type: ignore[assignment]
    prompt = prov.invocations[0]["prompt"]
    assert "AGENT'S TRAJECTORY:" not in prompt
