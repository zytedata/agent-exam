from __future__ import annotations

from typing import TYPE_CHECKING

from ...mcp import join_canonical_tool_name
from ...schemas import (
    Metrics,
    RunResult,
    SkillInvocation,
    TextBlock,
    Tokens,
    ToolCallBlock,
    Turn,
)

if TYPE_CHECKING:
    from pathlib import Path

    from .stream_parser import StreamState


def _request_tool_name(req: dict) -> str:
    """Name a requested tool, spelling an MCP call `mcp__<server>__<tool>`.

    A request for an MCP tool carries its server and tool in fields of
    their own, alongside the joined name Copilot shows the model.
    """
    server, tool = req.get("mcpServerName"), req.get("mcpToolName")
    if isinstance(server, str) and isinstance(tool, str) and server and tool:
        return join_canonical_tool_name(server, tool)
    return req.get("name", "")


def build_run_result(
    state: StreamState,
    wall_time_seconds: float,
    stream_detected_skill: SkillInvocation | None = None,
    raw_transcript_path: Path | None = None,
    user_prompt: str | None = None,
) -> RunResult:
    """Build a RunResult from the captured StreamState.

    The stdout JSONL stream is the authoritative record for Copilot CLI —
    there is no separate transcript file or database.
    """
    trajectory = _build_trajectory(state.events, user_prompt=user_prompt)
    skill_invocations = _extract_skill_invocations(trajectory, stream_detected_skill)
    metrics = _build_metrics(wall_time_seconds, trajectory, state.premium_requests)
    _attach_skill_invocations(trajectory, skill_invocations)
    return RunResult(
        trajectory=trajectory,
        metrics=metrics,
        raw_transcript_path=raw_transcript_path,
        model=state.model,
        mcp_server_status=state.mcp_server_status,
    )


def _build_trajectory(events: list[dict], user_prompt: str | None = None) -> list[Turn]:
    """Convert the flat event list into a list of Turn objects.

    Structure:
    - `user.message` events (without a `source` indicating a skill injection)
      become user Turns.
    - Groups of events between `assistant.turn_start` and `assistant.turn_end`
      become assistant Turns, with content assembled from `assistant.message`
      and `tool.execution_complete` events.
    """
    turns: list[Turn] = []

    # Emit an implicit user Turn for the original prompt if we have it and
    # no user.message event appears at the start of the stream.
    first_user_seen = False

    # Pending tool call metadata keyed by toolCallId:
    # {toolCallId: {"name": str, "arguments": dict, "call_id": str, "started_at": float|None}}
    pending_tool_calls: dict[str, dict] = {}

    current_assistant: _AssistantTurnBuilder | None = None

    for event in events:
        event_type = event.get("type", "")
        data = event.get("data") or {}

        if event_type == "user.message":
            content = data.get("content") or ""
            if content.lstrip().startswith("<skill-context"):
                # Skill context injection — included as a user turn so judges
                # can see what skill content was provided to the model.
                turns.append(_make_user_turn(content))
                continue
            # Real user message.
            if not first_user_seen and user_prompt and not data.get("content"):
                turns.append(_make_user_turn(user_prompt))
            elif data.get("content"):
                turns.append(_make_user_turn(data["content"]))
            first_user_seen = True

        elif event_type == "assistant.turn_start":
            if current_assistant is not None:
                built = current_assistant.build()
                if built is not None:
                    turns.append(built)
            current_assistant = _AssistantTurnBuilder(
                turn_id=data.get("turnId", ""),
                started_at=_ts(event),
            )
            pending_tool_calls = {}

        elif event_type == "assistant.message":
            if event.get("agentId"):
                continue
            if current_assistant is not None:
                current_assistant.add_message(data, _ts(event))
            # Register pending tool calls so we can match completions.
            for req in data.get("toolRequests") or []:
                call_id = req.get("toolCallId", "")
                if call_id:
                    pending_tool_calls[call_id] = {
                        "name": _request_tool_name(req),
                        "arguments": req.get("arguments") or {},
                        "call_id": call_id,
                        "started_at": None,
                    }

        elif event_type == "tool.execution_start":
            if event.get("agentId"):
                continue
            call_id = data.get("toolCallId", "")
            if call_id in pending_tool_calls:
                pending_tool_calls[call_id]["started_at"] = _ts(event)

        elif event_type == "tool.execution_complete":
            if event.get("agentId"):
                continue
            call_id = data.get("toolCallId", "")
            if call_id in pending_tool_calls and current_assistant is not None:
                meta = pending_tool_calls.pop(call_id)
                current_assistant.add_tool_result(
                    call_id=call_id,
                    tool_name=meta["name"] or data.get("toolName", ""),
                    arguments=meta["arguments"],
                    started_at=meta.get("started_at"),
                    completed_at=_ts(event),
                    success=data.get("success", True),
                    result_content=(data.get("result") or {}).get("content", ""),
                )

        elif event_type == "assistant.turn_end":
            if current_assistant is not None:
                built = current_assistant.build()
                if built is not None:
                    turns.append(built)
                current_assistant = None

    # Flush any in-progress turn (e.g. killed early by stop_on_first_skill).
    if current_assistant is not None:
        built = current_assistant.build()
        if built is not None:
            turns.append(built)

    # If no user turn was emitted at all, prepend the prompt.
    if not first_user_seen and user_prompt:
        turns.insert(0, _make_user_turn(user_prompt))

    return turns


def _make_user_turn(content: str) -> Turn:
    return Turn(role="user", content=[TextBlock(text=content)])


def _ts(event: dict) -> float | None:
    ts = event.get("timestamp")
    if isinstance(ts, str):
        # ISO-8601 strings: convert to epoch seconds for consistency.
        try:
            from datetime import datetime

            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt.timestamp()
        except Exception:
            return None
    if isinstance(ts, (int, float)):
        # Copilot CLI emits millisecond epoch in some events; normalise to seconds.
        return ts / 1000.0 if ts > 1e10 else ts
    return None


class _AssistantTurnBuilder:
    """Accumulates events within one assistant turn and builds a Turn."""

    def __init__(self, turn_id: str, started_at: float | None) -> None:
        self.turn_id = turn_id
        self.started_at = started_at
        self._text_parts: list[str] = []
        self._output_tokens: int = 0
        self._tool_calls: list[ToolCallBlock] = []
        # Pending tool requests before their completions arrive.
        self._pending: dict[str, dict] = {}  # call_id -> {name, arguments, started_at}

    def add_message(self, data: dict, ts: float | None) -> None:
        content = data.get("content") or ""
        if content:
            self._text_parts.append(content)
        self._output_tokens += data.get("outputTokens") or 0
        # Register all toolRequests as pending.
        for req in data.get("toolRequests") or []:
            call_id = req.get("toolCallId", "")
            if call_id:
                self._pending[call_id] = {
                    "name": _request_tool_name(req),
                    "arguments": req.get("arguments") or {},
                    "started_at": ts,
                }

    def add_tool_result(
        self,
        call_id: str,
        tool_name: str,
        arguments: dict,
        started_at: float | None,
        completed_at: float | None,
        success: bool,
        result_content: str,
    ) -> None:
        duration_ms: int | None = None
        if started_at is not None and completed_at is not None:
            duration_ms = int((completed_at - started_at) * 1000)

        status: str
        if not success:
            content_lower = result_content.lower()
            if any(
                p in content_lower
                for p in ("permission denied", "not permitted", "rejected")
            ):
                status = "permission_denied"
            else:
                status = "error"
        else:
            status = "ok"

        self._tool_calls.append(
            ToolCallBlock(
                tool_use_id=call_id,
                name=tool_name,
                input=arguments,
                status=status,
                result=result_content,
                started_at=started_at,
                duration_ms=duration_ms,
            )
        )
        self._pending.pop(call_id, None)

    def build(self) -> Turn | None:
        content = []
        if self._text_parts:
            content.append(TextBlock(text="".join(self._text_parts)))
        content.extend(self._tool_calls)
        # Include tool calls that were requested but never completed (e.g.
        # the process was killed by stop_on_first_skill before execution_complete).
        for call_id, meta in self._pending.items():
            content.append(
                ToolCallBlock(
                    tool_use_id=call_id,
                    name=meta["name"],
                    input=meta["arguments"],
                    status="ok",
                    result="",
                    started_at=meta.get("started_at"),
                )
            )
        if not content:
            return None
        tokens = Tokens(input=0, output=self._output_tokens, cache_read=0)
        return Turn(
            role="assistant",
            content=content,
            tokens=tokens,
            started_at=self.started_at,
        )


def _extract_skill_invocations(
    trajectory: list[Turn],
    stream_detected_skill: SkillInvocation | None,
) -> list[SkillInvocation]:
    """Return skill invocations from stream detection or trajectory scan."""
    if stream_detected_skill is not None:
        return [stream_detected_skill]
    invocations: list[SkillInvocation] = []
    for turn in trajectory:
        if turn.role != "assistant":
            continue
        for block in turn.content:
            if isinstance(block, ToolCallBlock) and block.name == "skill":
                skill_name = block.input.get("skill") or block.input.get("name") or ""
                if skill_name:
                    invocations.append(
                        SkillInvocation(
                            skill_name=skill_name,
                            trigger_kind="skill_tool",
                            triggered_by_tool_use_id=block.tool_use_id,
                        )
                    )
    return invocations


def _attach_skill_invocations(
    trajectory: list[Turn], invocations: list[SkillInvocation]
) -> None:
    """Attach skill invocations to the assistant turn that triggered them."""
    if not invocations:
        return
    inv_by_id = {
        i.triggered_by_tool_use_id: i for i in invocations if i.triggered_by_tool_use_id
    }
    unattached = [i for i in invocations if not i.triggered_by_tool_use_id]
    for turn in trajectory:
        if turn.role != "assistant":
            continue
        for block in turn.content:
            if isinstance(block, ToolCallBlock) and block.tool_use_id in inv_by_id:
                turn.skill_invocations.append(inv_by_id.pop(block.tool_use_id))
    # Any id-keyed invocations not matched above (e.g. killed before
    # tool.execution_complete so no ToolCallBlock was created) fall back to
    # unattached so they still get attributed to an assistant turn.
    unattached.extend(inv_by_id.values())
    # Attach any unattached invocations to the first assistant turn.
    if unattached:
        for turn in trajectory:
            if turn.role == "assistant":
                turn.skill_invocations.extend(unattached)
                break


def _build_metrics(
    wall_time_seconds: float,
    trajectory: list[Turn],
    premium_requests: int,
) -> Metrics:
    total_output = 0
    n_tool_calls = n_tool_errors = n_permission_denied = n_tool_rejected = 0
    peak_context = 0

    for turn in trajectory:
        if turn.role != "assistant":
            continue
        if turn.tokens:
            total_output += turn.tokens.output
        for block in turn.content:
            if not isinstance(block, ToolCallBlock):
                continue
            # Exclude internal framework tools from the tool-call health counts.
            if block.name in ("skill", "report_intent"):
                continue
            n_tool_calls += 1
            if block.status == "error":
                n_tool_errors += 1
            elif block.status == "permission_denied":
                n_permission_denied += 1
            elif block.status == "rejected":
                n_tool_rejected += 1

    turn_count = sum(1 for t in trajectory if t.role == "assistant")
    tokens = Tokens(input=0, output=total_output, cache_read=0)

    return Metrics(
        wall_time_seconds=wall_time_seconds,
        tokens=tokens,
        cost_usd=None,  # Copilot CLI does not report cost
        peak_context=peak_context,
        turn_count=turn_count,
        n_tool_calls=n_tool_calls,
        n_tool_errors=n_tool_errors,
        n_permission_denied=n_permission_denied,
        n_tool_rejected=n_tool_rejected,
        raw={"premium_requests": premium_requests},
    )
