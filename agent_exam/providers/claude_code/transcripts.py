"""Build a normalized RunResult from a Claude Code transcript on disk.

Uses `claude_measure_usage.parse` for per-turn metrics and subagent discovery.
The trajectory itself (content blocks, tool results matched by id, nested
subagents) is walked here so we can keep the full blocks — `parse.py`'s
`turns` output is metrics-oriented and drops block content.
"""

from __future__ import annotations

import json
from pathlib import Path

from claude_measure_usage.metrics import model_aware_cost_breakdown
from claude_measure_usage.parse import (
    build_agent_tree,
    find_subagent_transcripts,
    merge_tokens_by_model,
    parse_transcript,
    parse_ts,
    total_from_by_model,
)

from ...schemas import (
    Metrics,
    RunResult,
    SkillInvocation,
    TextBlock,
    ThinkingBlock,
    Tokens,
    ToolCallBlock,
    Turn,
)
from .skill_detect import detect_from_input

_SKIP_ENTRY_TYPES = frozenset(
    {
        "ai-title",
        "last-prompt",
        "attachment",
        "queue-operation",
        "file-history-snapshot",
        "system",
    }
)


def _iter_jsonl(path: str):
    with Path(path).open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _entry_ts(entry: dict) -> float | None:
    ts = entry.get("timestamp")
    if not ts:
        return None
    try:
        return parse_ts(ts)
    except Exception:
        return None


def _content_list(msg: dict) -> list:
    content = msg.get("content")
    if isinstance(content, list):
        return content
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return []


def _is_tool_result_only(content: list) -> bool:
    if not content:
        return False
    return all(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)


def _collect_tool_results(entries: list[dict]) -> dict[str, dict]:
    """Map tool_use_id → {result, is_error, ts}."""
    results: dict[str, dict] = {}
    for entry in entries:
        if entry.get("type") != "user":
            continue
        msg = entry.get("message") or {}
        if not isinstance(msg, dict):
            continue
        for block in _content_list(msg):
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            tool_use_id = block.get("tool_use_id")
            if not tool_use_id:
                continue
            content = block.get("content")
            if isinstance(content, list):
                text_parts = []
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "text":
                        text_parts.append(b.get("text", ""))
                    elif isinstance(b, str):
                        text_parts.append(b)
                result_str = (
                    "\n".join(text_parts) if text_parts else json.dumps(content)
                )
            elif isinstance(content, str):
                result_str = content
            else:
                result_str = ""
            tur = entry.get("toolUseResult") or {}
            is_error = bool(block.get("is_error")) or bool(
                tur.get("is_error") if isinstance(tur, dict) else False
            )
            results[tool_use_id] = {
                "result": result_str,
                "is_error": is_error,
                "ts": _entry_ts(entry),
            }
    return results


_PERMISSION_DENIAL_PATTERNS = (
    # Raised when a Read/Edit/Write/Bash operation is denied outright.
    "requested permissions",
    # Single-command denial: "This command requires approval".
    "requires approval",
    # Compound-command denial: "The following parts require approval: ...".
    # Needed as a separate pattern because neither subsumes the other:
    # "requires approval" has the `s`; "require approval" (base form)
    # doesn't.
    "require approval",
    # Seen when Claude retried and the user kept declining.
    "permission prompt keeps getting declined",
    # Explicit deny messages from approval prompts.
    "haven't granted it yet",
)


def _count_tool_statuses(trajectory) -> tuple[int, int, int, int]:
    """Walk the trajectory and return
    `(total, errors, permission_denied, rejected)` tool-call counts.

    The synthetic `Skill` tool name is Claude Code's internal skill-
    invocation redirect (is_error=true with content `Execute skill: X`);
    we exclude it from all counters since it's not a real tool call.
    Descends into subagent trajectories.
    """
    total = errors = denied = rejected = 0
    stack = [trajectory]
    while stack:
        turns = stack.pop()
        for turn in turns:
            for block in turn.content:
                if not isinstance(block, ToolCallBlock):
                    continue
                if block.name == "Skill":
                    # Harness-internal redirect, not a real tool call.
                    continue
                total += 1
                if block.status == "permission_denied":
                    denied += 1
                elif block.status == "rejected":
                    rejected += 1
                elif block.status == "error":
                    errors += 1
                if block.subagent:
                    stack.append(block.subagent)
    return total, errors, denied, rejected


def _is_permission_denial(result_str: str) -> bool:
    """Claude Code marks permission-gated failures with distinctive strings
    in the tool_result; we lift them to a normalized `permission_denied`
    status so provider-agnostic assertions can flag them.
    """
    if not result_str:
        return False
    low = result_str.lower()
    return any(p in low for p in _PERMISSION_DENIAL_PATTERNS)


# Signals that a tool call was rejected by Claude Code's input
# validation before the tool ran — bad argument types, missing required
# fields, etc. Empirically captured from a probe run:
# `<tool_use_error>InputValidationError: Edit failed due to the
# following issue: The required parameter `new_string` is missing
# </tool_use_error>`
_REJECTION_PATTERNS = (
    "<tool_use_error>",
    "inputvalidationerror",
)


def _is_harness_rejection(result_str: str) -> bool:
    """Claude Code wraps schema-validation failures in a distinctive
    `<tool_use_error>InputValidationError: ...</tool_use_error>`
    envelope. Distinct from both tool-side errors (tool ran and failed)
    and permission denials (policy block).
    """
    if not result_str:
        return False
    low = result_str.lower()
    return any(p in low for p in _REJECTION_PATTERNS)


def _block_from_content(
    block: dict,
    tool_results: dict[str, dict],
    subagent_for_tool_use: dict[str, list[Turn]],
    turn_ts: float | None,
    zero_ts: float | None,
):
    btype = block.get("type")
    if btype == "text":
        return TextBlock(text=block.get("text", ""))
    if btype == "thinking":
        return ThinkingBlock(text=block.get("thinking", block.get("text", "")))
    if btype == "tool_use":
        tool_use_id = block.get("id", "")
        result_info = tool_results.get(tool_use_id)
        if result_info is None:
            status = "aborted"
            result_str = ""
            duration_ms: int | None = None
        else:
            result_str = result_info["result"]
            if result_info["is_error"]:
                if _is_permission_denial(result_str):
                    status = "permission_denied"
                elif _is_harness_rejection(result_str):
                    status = "rejected"
                else:
                    status = "error"
            else:
                status = "ok"
            if turn_ts is not None and result_info["ts"] is not None:
                duration_ms = int(max(0.0, result_info["ts"] - turn_ts) * 1000)
            else:
                duration_ms = None
        started = (
            (turn_ts - zero_ts)
            if (turn_ts is not None and zero_ts is not None)
            else None
        )
        return ToolCallBlock(
            tool_use_id=tool_use_id,
            name=block.get("name", "unknown"),
            input=dict(block.get("input") or {}),
            status=status,
            result=result_str,
            started_at=started,
            duration_ms=duration_ms,
            subagent=subagent_for_tool_use.get(tool_use_id),
        )
    return None


def build_trajectory(
    transcript_path: str,
    zero_ts: float | None = None,
    subagent_for_tool_use: dict[str, list[Turn]] | None = None,
) -> list[Turn]:
    """Produce a normalized list of Turn dataclasses for one transcript.

    `zero_ts` is the reference epoch used for `started_at` values; when None,
    the first-entry timestamp of this transcript is used.
    `subagent_for_tool_use` attaches pre-built subagent trajectories to
    matching tool_use ids.
    """
    entries = list(_iter_jsonl(transcript_path))
    if zero_ts is None:
        for entry in entries:
            ts = _entry_ts(entry)
            if ts is not None:
                zero_ts = ts
                break
    subagent_for_tool_use = subagent_for_tool_use or {}

    tool_results = _collect_tool_results(entries)
    turns: list[Turn] = []

    current_msg_id: str | None = None
    current_turn: Turn | None = None
    current_turn_entry_ts: float | None = None

    def _flush_assistant():
        nonlocal current_turn, current_msg_id, current_turn_entry_ts
        if current_turn is not None:
            turns.append(current_turn)
        current_turn = None
        current_msg_id = None
        current_turn_entry_ts = None

    for entry in entries:
        etype = entry.get("type")
        if etype in _SKIP_ENTRY_TYPES:
            continue
        msg = entry.get("message")
        if not isinstance(msg, dict):
            continue
        if entry.get("isMeta"):
            continue
        content = _content_list(msg)
        ts = _entry_ts(entry)

        if etype == "user":
            if _is_tool_result_only(content):
                continue  # already captured by _collect_tool_results
            _flush_assistant()
            text_parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            user_text = "\n".join(text_parts) if text_parts else ""
            turns.append(
                Turn(
                    role="user",
                    content=[TextBlock(text=user_text)] if user_text else [],
                    started_at=(ts - zero_ts)
                    if (ts is not None and zero_ts is not None)
                    else None,
                )
            )
            continue

        if etype != "assistant":
            continue

        msg_id = msg.get("id")
        if msg_id != current_msg_id:
            _flush_assistant()
            usage = msg.get("usage") or {}
            tokens = Tokens(
                input=usage.get("input_tokens", 0) or 0,
                output=usage.get("output_tokens", 0) or 0,
                cache_read=usage.get("cache_read_input_tokens", 0) or 0,
            )
            ctx = (
                (usage.get("input_tokens", 0) or 0)
                + (usage.get("cache_read_input_tokens", 0) or 0)
                + (usage.get("cache_creation_input_tokens", 0) or 0)
            )
            current_turn = Turn(
                role="assistant",
                content=[],
                model=msg.get("model"),
                tokens=tokens,
                context=ctx,
                started_at=(ts - zero_ts)
                if (ts is not None and zero_ts is not None)
                else None,
            )
            current_msg_id = msg_id
            current_turn_entry_ts = ts

        for block in content:
            if not isinstance(block, dict):
                continue
            built = _block_from_content(
                block,
                tool_results,
                subagent_for_tool_use,
                current_turn_entry_ts,
                zero_ts,
            )
            if built is not None:
                current_turn.content.append(built)
                if isinstance(built, ToolCallBlock):
                    skill, trigger_kind = detect_from_input(built.name, built.input)
                    if skill:
                        current_turn.skill_invocations.append(
                            SkillInvocation(
                                skill_name=skill,
                                trigger_kind=trigger_kind or "unknown",
                                triggered_by_tool_use_id=built.tool_use_id,
                            )
                        )

    _flush_assistant()
    return turns


def _sonnet_equiv_to_usd(total_sonnet_equiv_tokens: float) -> float:
    # Sonnet input is $3 per million tokens; the measure-usage cost breakdown
    # expresses everything in Sonnet-input-equivalent tokens.
    return round(total_sonnet_equiv_tokens * 3.0 / 1_000_000.0, 6)


def load_run_result(
    transcript_path: Path,
    wall_time_seconds: float,
    explicit_cost_usd: float | None = None,
    stream_detected_skill: SkillInvocation | None = None,
) -> RunResult:
    """Parse a transcript on disk into a RunResult (trajectory + metrics).

    `stream_detected_skill` is provided when the run was cut short by
    `stop_on_first_skill`: the transcript may lack the triggering tool_use
    (we killed before the tool ran) so the provider hands us what it
    observed on the stream. We append it to the last assistant turn's
    skill_invocations if the retrospective walk didn't already pick it up.
    """
    main_parsed = parse_transcript(str(transcript_path))
    main_zero = main_parsed.get("first_entry_ts")

    subagent_infos = find_subagent_transcripts(str(transcript_path), main_zero or 0)
    agent_id_to_tool_use = main_parsed.get("agent_id_to_tool_use", {})

    subagent_for_tool_use: dict[str, list[Turn]] = {}
    for info in subagent_infos:
        # Map the subagent's agentId back to the parent tool_use_id.
        agent_id = _agent_id_from_path(info["path"])
        tool_use_id = agent_id_to_tool_use.get(agent_id)
        if not tool_use_id:
            continue
        subagent_for_tool_use[tool_use_id] = build_trajectory(
            info["path"],
            zero_ts=main_zero,
            subagent_for_tool_use=None,
        )

    trajectory = build_trajectory(
        str(transcript_path),
        zero_ts=main_zero,
        subagent_for_tool_use=subagent_for_tool_use,
    )

    # Aggregate metrics across main + subagents.
    all_tokens_by_model = dict(main_parsed["tokens_by_model"])
    peak_context = main_parsed.get("peak_context_tokens", 0)
    tree = build_agent_tree(str(transcript_path), main_parsed, subagent_infos)

    def _walk_tree(nodes):
        for node in nodes:
            yield node
            yield from _walk_tree(node["children"])

    for node in _walk_tree(tree):
        all_tokens_by_model = merge_tokens_by_model(
            all_tokens_by_model, node["tokens_by_model"]
        )
        if node.get("peak_context_tokens", 0) > peak_context:
            peak_context = node["peak_context_tokens"]

    totals = total_from_by_model(all_tokens_by_model)
    tokens = Tokens(
        input=totals.get("input_tokens", 0),
        output=totals.get("output_tokens", 0),
        cache_read=totals.get("cache_read_input_tokens", 0),
    )

    if explicit_cost_usd is not None:
        cost_usd = float(explicit_cost_usd)
    else:
        cost = model_aware_cost_breakdown(all_tokens_by_model)
        cost_usd = _sonnet_equiv_to_usd(cost["total"])

    (
        n_tool_calls,
        n_tool_errors,
        n_permission_denied,
        n_tool_rejected,
    ) = _count_tool_statuses(trajectory)

    metrics = Metrics(
        wall_time_seconds=round(wall_time_seconds, 3),
        tokens=tokens,
        cost_usd=cost_usd,
        peak_context=peak_context,
        turn_count=main_parsed.get("turn_count", 0),
        n_tool_calls=n_tool_calls,
        n_tool_errors=n_tool_errors,
        n_permission_denied=n_permission_denied,
        n_tool_rejected=n_tool_rejected,
        raw={
            "tokens_by_model": all_tokens_by_model,
            "tokens_breakdown": {
                k: totals.get(k, 0)
                for k in (
                    "cache_creation_input_tokens",
                    "ephemeral_5m_input_tokens",
                    "ephemeral_1h_input_tokens",
                )
            },
            "main_turn_count": main_parsed.get("turn_count", 0),
            "subagent_count": sum(1 for _ in _walk_tree(tree)),
        },
    )

    if stream_detected_skill is not None:
        already = any(
            si.skill_name == stream_detected_skill.skill_name
            for turn in trajectory
            for si in turn.skill_invocations
        )
        if not already:
            # Attach to the last assistant turn; if there are no assistant
            # turns (kill happened before the first usage-carrying event),
            # synthesize one carrying only the invocation.
            for turn in reversed(trajectory):
                if turn.role == "assistant":
                    turn.skill_invocations.append(stream_detected_skill)
                    break
            else:
                trajectory.append(
                    Turn(
                        role="assistant",
                        content=[],
                        skill_invocations=[stream_detected_skill],
                    )
                )

    return RunResult(
        trajectory=trajectory,
        metrics=metrics,
        raw_transcript_path=transcript_path,
    )


def _agent_id_from_path(path: str) -> str:
    name = Path(path).name
    if name.startswith("agent-") and name.endswith(".jsonl"):
        return name[len("agent-") : -len(".jsonl")]
    return ""
