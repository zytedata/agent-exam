from __future__ import annotations

from typing import TYPE_CHECKING

from ..judge import (
    agent_output_hash,
    build_prompt,
    call_judge,
    format_trajectory,
    key_for,
    parse_verdict,
)
from ..judge.format_trajectory import final_output_text
from ..schemas import AssertionResult, RunResult
from ._shared import JudgeConfigBase

if TYPE_CHECKING:
    from pathlib import Path


class JudgeConfig(JudgeConfigBase):
    """`judge: <criterion>` or
    `judge: {criterion: ..., include_trajectory?: ..., pass_on?: [...]}`."""


def check(
    config: JudgeConfig,
    result: RunResult,
    cwd: Path,
    context=None,
) -> AssertionResult:
    if context is None or context.judge_call is None:
        return AssertionResult(
            pass_=False,
            reason="judge: no judge_call in scoring context "
            "(runner did not configure a judge)",
        )

    pass_on = config.pass_on or context.judge_pass_on or ["YES"]
    output_hash = agent_output_hash(result.trajectory)
    cache_key = key_for(config.criterion, output_hash, context.judge_call.judge_model)

    cached = context.judge_cache.get(cache_key) if context.judge_cache else None
    if cached is not None:
        verdict = cached["verdict"]
        reasoning = cached.get("reasoning", "")
        cache_hit = True
    else:
        final_output = final_output_text(result.trajectory)
        traj_text = (
            format_trajectory(result.trajectory) if config.include_trajectory else ""
        )
        prompt = build_prompt(
            config.criterion, final_output, traj_text, config.include_trajectory
        )
        raw_response = call_judge(context.judge_call, prompt)
        verdict, reasoning = parse_verdict(raw_response)
        if context.judge_cache is not None:
            context.judge_cache.put(cache_key, config.criterion, verdict, reasoning)
        cache_hit = False

    passed = verdict in pass_on
    return AssertionResult(
        pass_=passed,
        reason=f"judge said {verdict}",
        details={
            "verdict": verdict,
            "reasoning": reasoning,
            "criterion": config.criterion,
            "cache": "hit" if cache_hit else "miss",
        },
    )
