from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from ..base import Provider
    from .stream_parser import StreamState

_PERMISSION_DENIED_PATTERNS = (
    "the user rejected permission",
    "the user has specified a rule which prevents",
)


def _db_path() -> Path:
    """Return the OpenCode SQLite DB path.

    Checks OPENCODE_DATA_DIR first (explicit override), then XDG_DATA_HOME,
    then falls back to ~/.local/share/opencode/opencode.db.
    """
    data_dir_env = os.environ.get("OPENCODE_DATA_DIR")
    if data_dir_env:
        return Path(data_dir_env) / "opencode.db"
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "opencode" / "opencode.db"
    return Path.home() / ".local" / "share" / "opencode" / "opencode.db"


def build_run_result(
    state: StreamState,
    wall_time_seconds: float,
    stream_detected_skill: SkillInvocation | None = None,
    raw_transcript_path: Path | None = None,
    user_prompt: str | None = None,
) -> RunResult:
    """Build a RunResult from the opencode SQLite DB for the session.

    The DB is the authoritative source: it includes reasoning events
    (never emitted to stdout) and is more complete for timed-out runs.
    """
    events = (
        _load_session_events(state.session_id, assistant_only=True)
        if state.session_id
        else []
    )
    trajectory, subagent_session_map = _build_trajectory(
        events, user_prompt=user_prompt
    )
    subagent_metrics = _attach_subagents(trajectory, subagent_session_map)
    skill_invocations = _extract_skill_invocations(
        trajectory, stream_detected_skill, state.provider
    )
    db_metrics = _aggregate_session_metrics(events) if events else None
    metrics = _build_metrics(
        wall_time_seconds, trajectory, subagent_metrics, db_metrics
    )
    _attach_skill_invocations(trajectory, skill_invocations)
    actual_model = next(
        (t.model for t in trajectory if t.role == "assistant" and t.model),
        None,
    )
    return RunResult(
        trajectory=trajectory,
        metrics=metrics,
        raw_transcript_path=raw_transcript_path,
        model=actual_model,
    )


def _build_trajectory(
    events: list[dict], user_prompt: str | None = None
) -> tuple[list[Turn], dict[str, str]]:
    """Group events into Turns by messageID.

    Returns (turns, subagent_session_map) where subagent_session_map
    maps task tool callID → subagent sessionID from metadata.
    """
    user_prompt_text = ""
    current_assistant: _AssistantTurnBuilder | None = None
    turns: list[Turn] = []
    subagent_session_map: dict[str, str] = {}

    for event in events:
        event_type = event.get("type")
        part = event.get("part") or {}
        message_id = part.get("messageID", "")

        if event_type == "step_start":
            if (
                current_assistant is not None
                and current_assistant.message_id != message_id
            ):
                turn = current_assistant.build()
                if turn is not None:
                    turns.append(turn)
            if (
                message_id
                and part.get("type") == "step-start"
                and (
                    current_assistant is None
                    or current_assistant.message_id != message_id
                )
            ):
                current_assistant = _AssistantTurnBuilder(message_id)
                ts = event.get("timestamp")
                if isinstance(ts, (int, float)):
                    current_assistant.set_started_at(ts / 1000.0)
                model = part.get("model", "")
                if model:
                    current_assistant.set_model(model)

        elif event_type == "text":
            text = part.get("text", "")
            if current_assistant is not None:
                current_assistant.add_text(text)
            else:
                user_prompt_text = text

        elif event_type == "reasoning":
            text = part.get("text") or part.get("part", {}).get("text", "")
            if current_assistant is not None and text:
                current_assistant.add_thinking(text)

        elif event_type == "tool_use":
            block, sa_session_id = _tool_call_from_event(part)
            if block is not None and current_assistant is not None:
                current_assistant.add_tool_call(block)
            if sa_session_id and block is not None:
                subagent_session_map[block.tool_use_id] = sa_session_id

        elif event_type == "step_finish":
            tokens_data = part.get("tokens") or {}
            ctx = (
                (tokens_data.get("input", 0) or 0)
                + (tokens_data.get("output", 0) or 0)
                + (tokens_data.get("reasoning", 0) or 0)
                + (tokens_data.get("cache", {}).get("read", 0) or 0)
                + (tokens_data.get("cache", {}).get("write", 0) or 0)
            )
            if current_assistant is not None:
                current_assistant.set_context(ctx)
                inp = tokens_data.get("input", 0) or 0
                out = tokens_data.get("output", 0) or 0
                cache_read = (tokens_data.get("cache") or {}).get("read", 0) or 0
                if inp or out or cache_read:
                    current_assistant.set_tokens(
                        Tokens(input=inp, output=out, cache_read=cache_read)
                    )

    if current_assistant is not None:
        turn = current_assistant.build()
        if turn is not None:
            turns.append(turn)

    effective_prompt = user_prompt_text or user_prompt
    if effective_prompt:
        turns.insert(
            0,
            Turn(
                role="user",
                content=[TextBlock(text=effective_prompt)],
                started_at=0.0,
            ),
        )

    return turns, subagent_session_map


class _AssistantTurnBuilder:
    def __init__(self, message_id: str):
        self.message_id = message_id
        self.content: list[Any] = []
        self.context: int | None = None
        self.started_at: float | None = None
        self.tokens: Tokens | None = None
        self.model: str | None = None

    def add_text(self, text: str) -> None:
        if text:
            self.content.append(TextBlock(text=text))

    def add_thinking(self, text: str) -> None:
        if text:
            self.content.append(ThinkingBlock(text=text))

    def add_tool_call(self, block: ToolCallBlock) -> None:
        self.content.append(block)

    def set_context(self, ctx: int) -> None:
        self.context = ctx

    def set_started_at(self, ts: float) -> None:
        self.started_at = ts

    def set_tokens(self, tokens: Tokens) -> None:
        self.tokens = tokens

    def set_model(self, model: str) -> None:
        self.model = model or None

    def build(self) -> Turn | None:
        if not self.content:
            return None
        return Turn(
            role="assistant",
            content=self.content,
            context=self.context,
            started_at=self.started_at,
            tokens=self.tokens,
            model=self.model,
        )


def _tool_call_from_event(part: dict) -> tuple[ToolCallBlock | None, str | None]:
    """Returns (ToolCallBlock, subagent_session_id or None)."""
    tool_state = part.get("state") or {}
    tool = part.get("tool")
    call_id = part.get("callID", "")
    if not tool:
        return None, None

    status_raw = tool_state.get("status", "completed")
    inp = tool_state.get("input") or {}
    if not isinstance(inp, dict):
        inp = {}
    output = tool_state.get("output", "")
    error = tool_state.get("error", "")

    if status_raw == "error":
        status = _classify_error(error)
        result = error
    elif status_raw == "running":
        status = "aborted"
        result = ""
    else:
        status = "ok"
        result = output if isinstance(output, str) else json.dumps(output, default=str)

    time_info = tool_state.get("time") or {}
    started_at = time_info.get("start")
    duration_ms = None
    if time_info.get("start") and time_info.get("end"):
        duration_ms = int(time_info["end"] - time_info["start"])

    subagent_session_id = None
    if tool == "task":
        metadata = tool_state.get("metadata") or {}
        if isinstance(metadata, dict):
            subagent_session_id = metadata.get("sessionId")

    return ToolCallBlock(
        tool_use_id=call_id,
        name=tool,
        input=dict(inp),
        status=status,
        result=result,
        started_at=started_at,
        duration_ms=duration_ms,
    ), subagent_session_id


def _classify_error(error: str) -> str:
    if not error:
        return "error"
    low = error.lower()
    for pattern in _PERMISSION_DENIED_PATTERNS:
        if pattern in low:
            return "permission_denied"
    return "error"


def _extract_skill_invocations(
    trajectory: list[Turn],
    stream_detected_skill: SkillInvocation | None,
    provider: Provider,
) -> list[SkillInvocation]:
    invocations: list[SkillInvocation] = []
    for turn in trajectory:
        if turn.role != "assistant":
            continue
        for block in turn.content:
            if not isinstance(block, ToolCallBlock):
                continue
            if block.name == "skill":
                name = block.input.get("name")
                if name:
                    invocations.append(
                        SkillInvocation(
                            skill_name=str(name),
                            trigger_kind="skill_tool",
                            triggered_by_tool_use_id=block.tool_use_id,
                        )
                    )

    if stream_detected_skill is not None:
        already = any(
            provider.is_same_skill(si.skill_name, stream_detected_skill.skill_name)
            for si in invocations
        )
        if not already:
            invocations.append(stream_detected_skill)

    return invocations


def _attach_skill_invocations(
    trajectory: list[Turn], invocations: list[SkillInvocation]
) -> None:
    if not invocations:
        return
    for inv in invocations:
        if inv.triggered_by_tool_use_id:
            for turn in trajectory:
                if turn.role != "assistant":
                    continue
                for block in turn.content:
                    if (
                        isinstance(block, ToolCallBlock)
                        and block.tool_use_id == inv.triggered_by_tool_use_id
                    ):
                        turn.skill_invocations.append(inv)
                        break
                else:
                    continue
                break
            else:
                _append_to_last_assistant(trajectory, inv)
        else:
            _append_to_last_assistant(trajectory, inv)


def _append_to_last_assistant(trajectory: list[Turn], inv: SkillInvocation) -> None:
    for turn in reversed(trajectory):
        if turn.role == "assistant":
            turn.skill_invocations.append(inv)
            return
    trajectory.append(
        Turn(
            role="assistant",
            content=[],
            skill_invocations=[inv],
        )
    )


def _count_tool_statuses(
    trajectory: list[Turn],
) -> tuple[int, int, int, int]:
    total = errors = denied = rejected = 0
    stack = [trajectory]
    while stack:
        turns = stack.pop()
        for turn in turns:
            for block in turn.content:
                if not isinstance(block, ToolCallBlock):
                    continue
                if block.name == "skill":
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


def _build_metrics(
    wall_time_seconds: float,
    trajectory: list[Turn],
    subagent_metrics: dict[str, Any] | None = None,
    db_metrics: dict[str, Any] | None = None,
) -> Metrics:
    db_metrics = db_metrics or {}
    tokens = Tokens(
        input=db_metrics.get("input", 0),
        output=db_metrics.get("output", 0),
        cache_read=db_metrics.get("cache_read", 0),
    )
    peak_context = db_metrics.get("peak_context", 0)
    # cost is None when no step_finish events reported cost (e.g. killed early).
    raw_cost = db_metrics.get("cost")
    cost_usd = round(raw_cost, 6) if isinstance(raw_cost, (int, float)) else None

    if subagent_metrics:
        tokens.input += subagent_metrics.get("input", 0)
        tokens.output += subagent_metrics.get("output", 0)
        tokens.cache_read += subagent_metrics.get("cache_read", 0)
        sa_peak = subagent_metrics.get("peak_context", 0)
        peak_context = max(peak_context, sa_peak)
        sa_cost = subagent_metrics.get("cost")
        if isinstance(sa_cost, (int, float)):
            cost_usd = round((cost_usd or 0.0) + sa_cost, 6)

    n_tool_calls, n_tool_errors, n_permission_denied, n_tool_rejected = (
        _count_tool_statuses(trajectory)
    )

    return Metrics(
        wall_time_seconds=round(wall_time_seconds, 3),
        tokens=tokens,
        cost_usd=cost_usd,
        peak_context=peak_context,
        turn_count=sum(1 for t in trajectory if t.role == "assistant"),
        n_tool_calls=n_tool_calls,
        n_tool_errors=n_tool_errors,
        n_permission_denied=n_permission_denied,
        n_tool_rejected=n_tool_rejected,
        raw={
            "cache_write": db_metrics.get("cache_write", 0),
            "reasoning": db_metrics.get("reasoning", 0),
        },
    )


def _attach_subagents(
    trajectory: list[Turn], subagent_session_map: dict[str, str]
) -> dict[str, Any] | None:
    """Query SQLite DB for subagent data, attach to task tool calls.

    Returns aggregated subagent metrics, or None if DB is inaccessible.
    """
    if not subagent_session_map:
        return None

    task_blocks: list[tuple[ToolCallBlock, str]] = []
    for turn in trajectory:
        for block in turn.content:
            if isinstance(block, ToolCallBlock) and block.name == "task":
                sa_id = subagent_session_map.get(block.tool_use_id)
                if sa_id:
                    task_blocks.append((block, sa_id))

    if not task_blocks:
        return None

    total_metrics: dict[str, Any] = {
        "input": 0,
        "output": 0,
        "cache_read": 0,
        "peak_context": 0,
        "cost": None,
    }

    for block, sa_session_id in task_blocks:
        sa_events = _load_session_events(sa_session_id)
        if sa_events:
            sa_trajectory, _ = _build_trajectory(sa_events)
            block.subagent = sa_trajectory
            sa_metrics = _aggregate_session_metrics(sa_events)
            for key in ("input", "output", "cache_read"):
                total_metrics[key] += sa_metrics.get(key, 0)
            if sa_metrics.get("peak_context", 0) > total_metrics["peak_context"]:
                total_metrics["peak_context"] = sa_metrics["peak_context"]
            sa_cost = sa_metrics.get("cost")
            if isinstance(sa_cost, (int, float)):
                total_metrics["cost"] = (total_metrics["cost"] or 0.0) + sa_cost

    return (
        total_metrics
        if total_metrics["input"] > 0 or total_metrics["output"] > 0
        else None
    )


def _normalize_db_events(raw_events: list[dict]) -> list[dict]:
    """Convert DB-stored events into the NDJSON stream format.

    When events carry `_msg_id` (injected by _load_session_events), the
    actual DB message_id is used as messageID so turns group correctly.
    Without it (legacy path) synthetic IDs are generated on step-start
    boundaries. The `_model` field is embedded in step-start parts so
    _build_trajectory can set it on each Turn. The `_time_created` DB column
    timestamp is used instead of the (absent) JSON timestamp field.
    """
    events: list[dict] = []
    current_msg_id = ""
    msg_counter = 0

    for db_row in raw_events:
        et = db_row.get("type", "")
        session_id = db_row.get("sessionID", "")
        # Prefer DB column timestamp; fall back to event-embedded timestamp.
        timestamp = db_row.get("_time_created") or db_row.get("timestamp", 0)
        model = db_row.get("_model", "")

        # Determine messageID: use actual DB message_id when available,
        # otherwise synthesize one on step-start boundaries.
        actual_msg_id = db_row.get("_msg_id", "")
        if actual_msg_id:
            current_msg_id = actual_msg_id
        elif et == "step-start":
            msg_counter += 1
            current_msg_id = f"msg_db_{msg_counter}"

        if et == "text":
            events.append(
                {
                    "type": "text",
                    "sessionID": session_id,
                    "timestamp": timestamp,
                    "part": {
                        "type": "text",
                        "text": db_row.get("text", ""),
                        "messageID": current_msg_id,
                        "sessionID": session_id,
                    },
                }
            )

        elif et == "step-start":
            events.append(
                {
                    "type": "step_start",
                    "sessionID": session_id,
                    "timestamp": timestamp,
                    "part": {
                        "type": "step-start",
                        "messageID": current_msg_id,
                        "sessionID": session_id,
                        "model": model,
                    },
                }
            )

        elif et == "reasoning":
            text = db_row.get("text", "")
            if text:
                events.append(
                    {
                        "type": "reasoning",
                        "sessionID": session_id,
                        "timestamp": timestamp,
                        "part": {
                            "type": "reasoning",
                            "text": text,
                            "messageID": current_msg_id,
                            "sessionID": session_id,
                        },
                    }
                )

        elif et == "tool":
            state = db_row.get("state") or {}
            events.append(
                {
                    "type": "tool_use",
                    "sessionID": session_id,
                    "timestamp": timestamp,
                    "part": {
                        "type": "tool_use",
                        "tool": db_row.get("tool", ""),
                        "callID": db_row.get("callID", ""),
                        "state": state,
                        "messageID": current_msg_id,
                        "sessionID": session_id,
                    },
                }
            )

        elif et == "step-finish":
            events.append(
                {
                    "type": "step_finish",
                    "sessionID": session_id,
                    "timestamp": timestamp,
                    "part": {
                        "type": "step-finish",
                        "tokens": db_row.get("tokens"),
                        "cost": db_row.get("cost"),
                        "reason": db_row.get("reason"),
                        "messageID": current_msg_id,
                        "sessionID": session_id,
                    },
                }
            )

        elif et == "error":
            events.append(
                {
                    "type": "error",
                    "sessionID": session_id,
                    "timestamp": timestamp,
                    "error": db_row.get("error"),
                }
            )

    return events


def _load_session_events(session_id: str, assistant_only: bool = False) -> list[dict]:
    """Load and normalize DB events for a session.

    assistant_only=True filters to assistant-message parts only — correct for
    building the main-session transcript (user prompt comes from the explicit
    user_prompt parameter instead). False (default) includes all parts,
    which is the right choice for subagents where we want the full turn set.
    """
    db = _db_path()
    if not db.exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            # Build message_id → model string from assistant messages.
            msg_rows = conn.execute(
                "SELECT id, data FROM message WHERE session_id = ? ORDER BY time_created",
                (session_id,),
            ).fetchall()
            model_map: dict[str, str] = {}
            for msg_row in msg_rows:
                md = json.loads(msg_row["data"]) if msg_row["data"] else {}
                model_id = md.get("modelID", "")
                provider_id = md.get("providerID", "")
                if model_id:
                    model_map[msg_row["id"]] = (
                        f"{provider_id}/{model_id}" if provider_id else model_id
                    )

            if assistant_only:
                query = (
                    "SELECT p.data, p.message_id, p.time_created "
                    "FROM part p "
                    "JOIN message m ON p.message_id = m.id "
                    "WHERE p.session_id = ? "
                    "  AND json_extract(m.data, '$.role') = 'assistant' "
                    "ORDER BY p.time_created, p.id"
                )
            else:
                query = (
                    "SELECT data, message_id, time_created "
                    "FROM part WHERE session_id = ? ORDER BY time_created, id"
                )

            rows = conn.execute(query, (session_id,)).fetchall()
            events: list[dict] = []
            for row in rows:
                try:
                    data = json.loads(row["data"])
                    if isinstance(data, dict):
                        data["sessionID"] = session_id
                        data["_msg_id"] = row["message_id"]
                        data["_model"] = model_map.get(row["message_id"], "")
                        data["_time_created"] = row["time_created"]
                        events.append(data)
                except (json.JSONDecodeError, KeyError):
                    continue
            return _normalize_db_events(events)
        finally:
            conn.close()
    except sqlite3.Error:
        return []


def _aggregate_session_metrics(events: list[dict]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "input": 0,
        "output": 0,
        "cache_read": 0,
        "cache_write": 0,
        "reasoning": 0,
        "peak_context": 0,
        "cost": None,  # None until at least one step_finish reports cost data
    }
    for event in events:
        if event.get("type") != "step_finish":
            continue
        part = event.get("part") or {}
        tokens = part.get("tokens") or {}
        metrics["input"] += tokens.get("input", 0) or 0
        metrics["output"] += tokens.get("output", 0) or 0
        metrics["reasoning"] += tokens.get("reasoning", 0) or 0
        cache = tokens.get("cache") or {}
        metrics["cache_read"] += cache.get("read", 0) or 0
        metrics["cache_write"] += cache.get("write", 0) or 0
        total = tokens.get("total", 0) or 0
        metrics["peak_context"] = max(metrics["peak_context"], total)
        cost = part.get("cost")
        if isinstance(cost, (int, float)):
            metrics["cost"] = (metrics["cost"] or 0.0) + cost
    return metrics
