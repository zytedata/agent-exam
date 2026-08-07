from __future__ import annotations

from typing import TYPE_CHECKING

from ..judge import (
    agent_output_hash,
    build_prompt,
    call_judge_agent,
    format_trajectory,
    key_for_judge_agent,
    parse_verdict,
    tools_signature,
)
from ..judge.format_trajectory import final_output_text
from ..schemas import AssertionResult, RunResult
from ._shared import JudgeConfigBase

if TYPE_CHECKING:
    from pathlib import Path


class JudgeAgentConfig(JudgeConfigBase):
    """`judge_agent: <criterion>` or
    `judge_agent: {criterion: ..., include_trajectory?: ..., pass_on?: [...]}`.

    Same config shape as `judge`; the difference is the runtime — the
    judge runs against the attempt's archived cwd with read-only file
    tools available."""


def check(
    config: JudgeAgentConfig,
    result: RunResult,
    cwd: Path,
    context=None,
) -> AssertionResult:
    """Agentic judge — same config shape as the plain ``judge`` assertion
    but the judge runs against the attempt's archived cwd with read-only
    file tools available, so it can verify criteria that depend on the
    artifacts the skill produced.

    Cache key extends ``judge``'s key with the cwd content hash and the
    safe-tools signature: any change to the criterion, the agent's
    output, the cwd, the toolset, or the judge model invalidates the
    cached verdict.
    """
    if context is None or context.judge_call is None:
        return AssertionResult(
            pass_=False,
            reason="judge_agent: no judge_call in scoring context "
            "(runner did not configure a judge)",
        )

    provider = context.judge_call.provider
    if not provider.safe_judge_tools:
        return AssertionResult(
            pass_=False,
            reason=(
                f"judge_agent: provider {provider.name!r} has no "
                "safe_judge_tools configured"
            ),
        )

    pass_on = config.pass_on or context.judge_pass_on or ["YES"]
    output_hash = agent_output_hash(result.trajectory)
    cwd_h = context.cwd_hash_for(cwd)
    tools_sig = tools_signature(provider.safe_judge_tools)
    cache_key = key_for_judge_agent(
        config.criterion,
        output_hash,
        cwd_h,
        tools_sig,
        context.judge_call.judge_model,
    )

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
            config.criterion,
            final_output,
            traj_text,
            config.include_trajectory,
            inspect_cwd=True,
        )
        raw_response = call_judge_agent(context.judge_call, prompt, cwd)
        verdict, reasoning = parse_verdict(raw_response)
        if context.judge_cache is not None:
            context.judge_cache.put(cache_key, config.criterion, verdict, reasoning)
        cache_hit = False

    passed = verdict in pass_on
    return AssertionResult(
        pass_=passed,
        reason=f"judge_agent said {verdict}",
        details={
            "verdict": verdict,
            "reasoning": reasoning,
            "criterion": config.criterion,
            "cache": "hit" if cache_hit else "miss",
        },
    )
