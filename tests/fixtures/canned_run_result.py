"""Factories for building RunResult / Turn / ToolCallBlock instances in tests."""

from __future__ import annotations

from agent_exam.schemas import (
    Metrics,
    RunResult,
    SkillInvocation,
    TextBlock,
    ThinkingBlock,
    Tokens,
    ToolCallBlock,
    Turn,
)


def tool_call(
    name: str,
    *,
    tool_use_id: str = "toolu_01",
    input_: dict | None = None,
    status: str = "ok",
    result: str = "",
    subagent: list[Turn] | None = None,
) -> ToolCallBlock:
    return ToolCallBlock(
        tool_use_id=tool_use_id,
        name=name,
        input=input_ or {},
        status=status,
        result=result,
        subagent=subagent,
    )


def assistant_turn(
    *blocks,
    model: str = "claude-sonnet-4-6",
    tokens: tuple[int, int, int] = (100, 20, 0),
    context: int = 100,
    skill_invocations: list[SkillInvocation] | None = None,
) -> Turn:
    return Turn(
        role="assistant",
        content=list(blocks),
        model=model,
        tokens=Tokens(input=tokens[0], output=tokens[1], cache_read=tokens[2]),
        context=context,
        skill_invocations=skill_invocations or [],
    )


def user_turn(text: str) -> Turn:
    return Turn(role="user", content=[TextBlock(text=text)])


def run_result(
    trajectory: list[Turn],
    *,
    wall_time_seconds: float = 1.0,
    cost_usd: float = 0.01,
    peak_context: int = 100,
    turn_count: int | None = None,
) -> RunResult:
    if turn_count is None:
        turn_count = sum(1 for t in trajectory if t.role == "assistant")
    metrics = Metrics(
        wall_time_seconds=wall_time_seconds,
        tokens=Tokens(input=0, output=0, cache_read=0),
        cost_usd=cost_usd,
        peak_context=peak_context,
        turn_count=turn_count,
        raw={},
    )
    return RunResult(trajectory=trajectory, metrics=metrics)


def thinking(text: str) -> ThinkingBlock:
    return ThinkingBlock(text=text)


def text(t: str) -> TextBlock:
    return TextBlock(text=t)


def skill_inv(name: str, trigger_kind: str = "first_skill_md_read") -> SkillInvocation:
    return SkillInvocation(skill_name=name, trigger_kind=trigger_kind)
