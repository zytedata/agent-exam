from __future__ import annotations

from typing import TYPE_CHECKING

from fixtures.canned_run_result import assistant_turn, run_result, text, user_turn

from agent_exam.assertions import judge_agent
from agent_exam.assertions.judge_agent import JudgeAgentConfig
from agent_exam.judge import JudgeCache, JudgeCall
from agent_exam.providers.base import Provider
from agent_exam.schemas import Metrics, RunResult, TextBlock, Tokens, Turn
from agent_exam.scoring_context import ScoringContext

if TYPE_CHECKING:
    from pathlib import Path


class _StubProvider(Provider):
    """Provider that records `.invoke()` calls and returns a canned verdict.

    Exercises the real dispatch path (cwd copy, allowed_tools merging,
    response extraction) — we only intercept the outermost subprocess
    call by faking the harness's response.
    """

    name = "stub"
    safe_judge_tools = ("read_file", "glob")

    def __init__(self, response: str = "ok.\nVERDICT: YES"):
        self.response = response
        self.invocations: list[dict] = []

    def invoke(
        self,
        prompt: str,
        model: str,
        cwd: Path,
        provider_options: dict,
        stop_on_first_trigger: bool,
        timeout_seconds: int,
    ) -> RunResult:
        cwd_files = sorted(p.name for p in cwd.iterdir()) if cwd.is_dir() else []
        self.invocations.append(
            {
                "prompt": prompt,
                "model": model,
                "cwd": cwd,
                "cwd_files": cwd_files,
                "provider_options": dict(provider_options),
                "timeout_seconds": timeout_seconds,
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


class _NoToolsProvider(Provider):
    """Provider with no safe_judge_tools — judge_agent should refuse."""

    name = "no-tools"
    safe_judge_tools: tuple[str, ...] = ()

    def invoke(self, *args, **kwargs):  # pragma: no cover — never called
        raise AssertionError("no-tools provider should not be invoked")


def _context(
    cache_path,
    *,
    response: str = "ok.\nVERDICT: YES",
    pass_on=None,
    provider: Provider | None = None,
    agent_timeout_seconds: int = 300,
) -> ScoringContext:
    prov = provider or _StubProvider(response=response)
    cache = JudgeCache(cache_path)
    jc = JudgeCall(
        provider=prov,
        judge_model="claude-haiku-4-5-test",
        provider_options={"extra_args": ["--foo"]},
        timeout_seconds=60,
        agent_timeout_seconds=agent_timeout_seconds,
    )
    return ScoringContext(
        provider=prov,
        judge_call=jc,
        judge_cache=cache,
        judge_pass_on=pass_on or ["YES"],
    )


def _trajectory(final: str = "all done"):
    return [user_turn("do the thing"), assistant_turn(text(final))]


def _populate_cwd(cwd: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        p = cwd / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


def _cfg(criterion: str = "c", **kw) -> JudgeAgentConfig:
    return JudgeAgentConfig(criterion=criterion, **kw)


def test_criterion_pass(tmp_path, cwd):
    ctx = _context(tmp_path / "judge-cache.json")
    r = judge_agent.check(
        _cfg("Response mentions the task."),
        run_result(_trajectory()),
        cwd,
        context=ctx,
    )
    assert r.pass_
    assert r.details["verdict"] == "YES"
    assert r.details["cache"] == "miss"


def test_verdict_no_fails(tmp_path, cwd):
    ctx = _context(tmp_path / "c.json", response="no.\nVERDICT: NO")
    r = judge_agent.check(_cfg(), run_result(_trajectory()), cwd, context=ctx)
    assert not r.pass_
    assert r.details["verdict"] == "NO"


def test_unclear_fails(tmp_path, cwd):
    ctx = _context(tmp_path / "c.json", response="no marker here")
    r = judge_agent.check(_cfg(), run_result(_trajectory()), cwd, context=ctx)
    assert not r.pass_
    assert r.details["verdict"] == "UNCLEAR"


def test_pass_on_override(tmp_path, cwd):
    ctx = _context(tmp_path / "c.json", response="ok.\nVERDICT: UNCLEAR")
    r = judge_agent.check(
        _cfg(pass_on=["YES", "UNCLEAR"]),
        run_result(_trajectory()),
        cwd,
        context=ctx,
    )
    assert r.pass_


def test_cache_hit_skips_invoke(tmp_path, cwd):
    ctx = _context(tmp_path / "c.json")
    traj = _trajectory("same output")
    judge_agent.check(_cfg("crit"), run_result(traj), cwd, context=ctx)
    prov: _StubProvider = ctx.judge_call.provider  # type: ignore[assignment]
    before = len(prov.invocations)
    r2 = judge_agent.check(_cfg("crit"), run_result(traj), cwd, context=ctx)
    after = len(prov.invocations)
    assert before == 1, "provider must not be invoked on cache hit"
    assert after == 1, "provider must not be invoked on cache hit"
    assert r2.details["cache"] == "hit"


def test_cache_miss_on_different_cwd_content(tmp_path):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    _populate_cwd(cwd, {"a.txt": "version 1"})
    ctx = _context(tmp_path / "c.json")
    judge_agent.check(_cfg("crit"), run_result(_trajectory()), cwd, context=ctx)
    _populate_cwd(cwd, {"a.txt": "version 2"})
    ctx2 = _context(tmp_path / "c.json")
    judge_agent.check(_cfg("crit"), run_result(_trajectory()), cwd, context=ctx2)
    prov2: _StubProvider = ctx2.judge_call.provider  # type: ignore[assignment]
    assert len(prov2.invocations) == 1


def test_cache_miss_on_different_safe_tools(tmp_path, cwd):
    ctx = _context(tmp_path / "c.json")
    judge_agent.check(_cfg("crit"), run_result(_trajectory()), cwd, context=ctx)

    class _OtherProvider(_StubProvider):
        safe_judge_tools = ("read_file", "glob", "grep")

    ctx2 = _context(tmp_path / "c.json", provider=_OtherProvider())
    judge_agent.check(_cfg("crit"), run_result(_trajectory()), cwd, context=ctx2)
    prov2: _StubProvider = ctx2.judge_call.provider  # type: ignore[assignment]
    assert len(prov2.invocations) == 1


def test_cache_miss_on_different_criterion(tmp_path, cwd):
    ctx = _context(tmp_path / "c.json")
    traj = _trajectory("same output")
    judge_agent.check(_cfg("criterion A"), run_result(traj), cwd, context=ctx)
    judge_agent.check(_cfg("criterion B"), run_result(traj), cwd, context=ctx)
    prov: _StubProvider = ctx.judge_call.provider  # type: ignore[assignment]
    assert len(prov.invocations) == 2


def test_missing_context_fails_fast(cwd):
    r = judge_agent.check(_cfg(), run_result(_trajectory()), cwd, context=None)
    assert not r.pass_
    assert "no judge_call" in r.reason


def test_provider_without_safe_tools_fails(tmp_path, cwd):
    ctx = _context(tmp_path / "c.json", provider=_NoToolsProvider())
    r = judge_agent.check(_cfg(), run_result(_trajectory()), cwd, context=ctx)
    assert not r.pass_
    assert "safe_judge_tools" in r.reason


def test_include_trajectory_false_shortens_prompt(tmp_path, cwd):
    ctx = _context(tmp_path / "c.json")
    judge_agent.check(
        _cfg(include_trajectory=False),
        run_result(_trajectory()),
        cwd,
        context=ctx,
    )
    prov: _StubProvider = ctx.judge_call.provider  # type: ignore[assignment]
    prompt = prov.invocations[0]["prompt"]
    assert "AGENT'S TRAJECTORY:" not in prompt


def test_dispatch_sets_allowed_tools_from_safe_judge_tools(tmp_path, cwd):
    """Verify the dispatch wires safe_judge_tools into provider_options."""
    ctx = _context(tmp_path / "c.json")
    judge_agent.check(_cfg(), run_result(_trajectory()), cwd, context=ctx)
    prov: _StubProvider = ctx.judge_call.provider  # type: ignore[assignment]
    opts = prov.invocations[0]["provider_options"]
    assert opts["allowed_tools"] == list(prov.safe_judge_tools)
    assert opts["extra_args"] == ["--foo"]


def test_dispatch_passes_agent_timeout_not_plain_timeout(tmp_path, cwd):
    """judge_agent must use agent_timeout_seconds, not timeout_seconds."""
    ctx = _context(tmp_path / "c.json", agent_timeout_seconds=600)
    judge_agent.check(_cfg(), run_result(_trajectory()), cwd, context=ctx)
    prov: _StubProvider = ctx.judge_call.provider  # type: ignore[assignment]
    assert prov.invocations[0]["timeout_seconds"] == 600


def test_dispatch_copies_archive_into_judge_cwd(tmp_path):
    """The provider should see a copy of the archived cwd, not the path itself."""
    archive = tmp_path / "archive"
    archive.mkdir()
    _populate_cwd(archive, {"a.txt": "x", "nested/b.txt": "y"})
    ctx = _context(tmp_path / "c.json")
    judge_agent.check(_cfg(), run_result(_trajectory()), archive, context=ctx)
    prov: _StubProvider = ctx.judge_call.provider  # type: ignore[assignment]
    inv = prov.invocations[0]
    assert inv["cwd"] != archive
    assert "a.txt" in inv["cwd_files"]


def test_cwd_hash_memoized_within_attempt(tmp_path, cwd):
    """Two judge_agent assertions on the same cwd should hash it once."""
    ctx = _context(tmp_path / "c.json")
    _populate_cwd(cwd, {"f.txt": "x"})
    judge_agent.check(_cfg("crit-a"), run_result(_trajectory("o1")), cwd, context=ctx)
    cached_hash = ctx._cwd_hash_cache[cwd]
    judge_agent.check(_cfg("crit-b"), run_result(_trajectory("o2")), cwd, context=ctx)
    assert ctx._cwd_hash_cache[cwd] == cached_hash
    assert len(ctx._cwd_hash_cache) == 1
