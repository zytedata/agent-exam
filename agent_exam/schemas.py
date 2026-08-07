from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

BlockType = Literal["text", "thinking", "tool_call"]


@dataclass
class TextBlock:
    text: str
    type: Literal["text"] = "text"


@dataclass
class ThinkingBlock:
    text: str
    type: Literal["thinking"] = "thinking"


@dataclass
class ToolCallBlock:
    tool_use_id: str
    name: str
    input: dict
    status: Literal["ok", "error", "permission_denied", "rejected", "aborted"]
    result: str
    started_at: float | None = None
    duration_ms: int | None = None
    subagent: list[Turn] | None = None
    type: Literal["tool_call"] = "tool_call"


ContentBlock = TextBlock | ThinkingBlock | ToolCallBlock


@dataclass
class SkillInvocation:
    skill_name: str
    trigger_kind: str
    triggered_by_tool_use_id: str | None = None


@dataclass
class Tokens:
    input: int = 0
    output: int = 0
    cache_read: int = 0


@dataclass
class Turn:
    role: Literal["user", "assistant"]
    content: list[ContentBlock]
    model: str | None = None
    tokens: Tokens | None = None
    context: int | None = None
    started_at: float | None = None
    skill_invocations: list[SkillInvocation] = field(default_factory=list)


@dataclass
class Metrics:
    wall_time_seconds: float
    tokens: Tokens
    cost_usd: (
        float | None
    )  # None = cost unknown (e.g. killed early, or provider doesn't report it)
    peak_context: int
    turn_count: int
    # Tool-call health. Counted by walking ToolCallBlock entries in the
    # trajectory; each provider's transcript loader populates these.
    n_tool_calls: int = 0  # total real tool uses (excludes
    # harness-internal `Skill` redirects)
    n_tool_errors: int = 0  # tool ran, returned is_error=true
    n_permission_denied: int = 0  # blocked by the harness's permission
    # system (see no_permission_errors
    # assertion for context)
    n_tool_rejected: int = 0  # pre-run rejection by the harness —
    # bad arguments, unknown tool, schema
    # mismatch. Scaffolding today: no
    # Claude Code detector yet; field is
    # ready for when a real example
    # surfaces. Distinct from both
    # `n_tool_errors` (tool's fault) and
    # `n_permission_denied` (policy block).
    raw: dict = field(default_factory=dict)


@dataclass
class RunResult:
    trajectory: list[Turn]
    metrics: Metrics
    raw_transcript_path: Path | None = None
    # Actual model used by the provider (may differ from the requested model
    # when the provider has its own default, e.g. OpenCode defaulting to
    # kimi-k2.6 when no --model flag is passed).
    model: str | None = None


@dataclass
class AssertionResult:
    pass_: bool
    reason: str
    details: dict = field(default_factory=dict)


@dataclass
class CheckResult:
    name: str
    status: Literal["OK", "WARN", "FAIL"]
    hint: str = ""
