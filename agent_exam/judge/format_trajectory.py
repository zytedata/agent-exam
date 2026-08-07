from __future__ import annotations

import json

from ..schemas import TextBlock, ThinkingBlock, ToolCallBlock, Turn

DEFAULT_MAX_CHARS = 50_000

# Per-block truncation limits inside the judge-facing trajectory.
#
# Text and tool results both carry judge-critical information: text is
# what the agent shows the user (e.g. a 40-field schema review with
# descriptions + examples), tool results carry command output / file
# contents / API responses. Thinking and tool inputs are usually short
# and structural — tighter limits don't lose signal.
TEXT_BLOCK_MAX = 5_000
THINKING_BLOCK_MAX = 200
TOOL_INPUT_MAX = 600
TOOL_RESULT_MAX = 1_000


def format_trajectory(
    trajectory: list[Turn], max_chars: int = DEFAULT_MAX_CHARS
) -> str:
    """Render a trajectory as compact text for the judge prompt.

    Layout per turn:

        [turn 3, assistant]
          thinking: The user wants ...
          text: I'll read the spec file first.
          tool_use Read({"file_path": "/path/spec.json"}) → ok: <result...>

    Subagents render indented under their parent ToolCallBlock. If the total
    exceeds `max_chars`, the middle is truncated with an explicit marker.
    """
    lines = _render(trajectory, depth=0)
    joined = "\n".join(lines)
    if len(joined) <= max_chars:
        return joined
    keep = max_chars // 2
    return f"{joined[:keep]}\n...[trajectory truncated]...\n{joined[-keep:]}"


def _render(trajectory: list[Turn], depth: int) -> list[str]:
    indent = "  " * depth
    out: list[str] = []
    for i, turn in enumerate(trajectory):
        out.append(f"{indent}[turn {i}, {turn.role}]")
        for block in turn.content:
            if isinstance(block, TextBlock):
                if block.text.strip():
                    out.append(f"{indent}  text: {_trunc(block.text, TEXT_BLOCK_MAX)}")
            elif isinstance(block, ThinkingBlock):
                if block.text.strip():
                    out.append(
                        f"{indent}  thinking: {_trunc(block.text, THINKING_BLOCK_MAX)}"
                    )
            elif isinstance(block, ToolCallBlock):
                input_summary = _trunc(
                    json.dumps(block.input, ensure_ascii=False, default=str),
                    TOOL_INPUT_MAX,
                )
                result_summary = _trunc_mid(block.result, TOOL_RESULT_MAX)
                out.append(
                    f"{indent}  tool_use {block.name}({input_summary})"
                    f" → {block.status}: {result_summary}"
                )
                if block.subagent:
                    out.extend(_render(block.subagent, depth + 1))
    return out


def _trunc(s: str, n: int) -> str:
    if s is None:
        return ""
    s = s.replace("\n", " ")
    if len(s) <= n:
        return s
    return s[:n] + "..."


def _trunc_mid(s: str, n: int) -> str:
    """Truncate by keeping the head and tail, showing how many chars were removed."""
    if s is None:
        return ""
    s = s.replace("\n", " ")
    if len(s) <= n:
        return s
    keep = n // 2
    removed = len(s) - 2 * keep
    return s[:keep] + f"...[{removed} chars removed]..." + s[-keep:]


def final_output_text(trajectory: list[Turn]) -> str:
    """Return the text of the last assistant turn (concatenated TextBlocks)."""
    for turn in reversed(trajectory):
        if turn.role != "assistant":
            continue
        texts = [b.text for b in turn.content if isinstance(b, TextBlock)]
        return "\n".join(t for t in texts if t is not None)
    return ""
