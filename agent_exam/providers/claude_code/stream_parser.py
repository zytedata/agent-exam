from __future__ import annotations

import io
import json
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import IO

from ...mcp import server_status_map, settles_tool_trigger
from ...schemas import SkillInvocation
from .skill_detect import detect_from_partial


@dataclass
class StreamState:
    """Mutable shared state that a reader thread fills as events arrive."""

    session_id: str | None = None
    total_cost_usd: float | None = None
    result_is_error: bool = False
    result_subtype: str | None = None
    api_error_status: int | None = None
    result_error: str | None = None
    stderr_tail: deque = field(default_factory=lambda: deque(maxlen=8192))

    # MCP server name -> connection status, from the `system/init` event.
    # A server that fails to start leaves the agent without its tools and
    # nothing else says so; doctor reports it (see `mcp.connection_check`).
    mcp_server_status: dict[str, str] | None = None

    # Skill-detection fields (driven by drain_stream when
    # skill_detection_enabled=True). `kill_signal` fires on a match so the
    # provider's main thread can terminate the subprocess early.
    skill_detection_enabled: bool = False
    target_skill: str | None = None  # None = fire on any skill
    detected_skill: SkillInvocation | None = None
    # Set for a trigger task whose target is a tool rather than a skill:
    # the run is cut on that tool's `tool_use` instead of on a skill fire,
    # and `detected_tool` records it, since the kill lands before Claude
    # Code writes the call to the transcript.
    target_tool: str | None = None
    detected_tool: str | None = None
    kill_signal: threading.Event = field(default_factory=threading.Event)

    # Negative-trigger mode: for cases where the skill is expected NOT
    # to fire, the routing decision is settled as soon as the agent
    # either (a) starts a non-Skill, non-Read tool use, or (b) finishes
    # its first assistant turn without firing a skill. Both are
    # sufficient signals that the skill isn't being invoked here, so we
    # kill early rather than let the agent keep working. Shaves tens of
    # seconds (and tool-call cost) off negative-trigger attempts.
    negative_trigger_mode: bool = False

    # Internal cursor for the current tool_use being streamed.
    _cur_tool_name: str | None = None
    _cur_tool_use_id: str | None = None
    _cur_accumulated: str = ""


def drain_stream(stdout: IO[bytes], state: StreamState) -> None:
    """Read NDJSON lines from `stdout`, extracting fields we care about.

    Runs in a reader thread. Claude Code emits one JSON object per line; we
    only care about a few fields (`session_id` early on, `total_cost_usd`
    and `is_error` from the final `result` event). Everything else is
    discarded so the pipe can't block the subprocess.
    """
    for raw_line in io.TextIOWrapper(stdout, encoding="utf-8", errors="replace"):
        line = raw_line.strip()
        if line:
            _dispatch(line, state)


def drain_stderr(stderr: IO[bytes], state: StreamState) -> None:
    while True:
        chunk = stderr.read(4096)
        if not chunk:
            break
        state.stderr_tail.extend(chunk)


def _dispatch(line: str, state: StreamState) -> None:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return
    if not isinstance(event, dict):
        return
    sid = event.get("session_id")
    if sid and state.session_id is None:
        state.session_id = sid
    if event.get("type") == "system" and event.get("subtype") == "init":
        statuses = server_status_map(event.get("mcp_servers"))
        if statuses is not None:
            state.mcp_server_status = statuses
    elif event.get("type") == "result":
        state.total_cost_usd = event.get("total_cost_usd")
        state.result_is_error = bool(event.get("is_error"))
        state.result_subtype = event.get("subtype")
        status = event.get("api_error_status")
        if isinstance(status, int):
            state.api_error_status = status
        err = event.get("result") if state.result_is_error else None
        if isinstance(err, str):
            state.result_error = err
    elif state.skill_detection_enabled and event.get("type") == "stream_event":
        _dispatch_skill_detection(event.get("event") or {}, state)


def _dispatch_skill_detection(se: dict, state: StreamState) -> None:
    se_type = se.get("type")
    if se_type == "content_block_start":
        cb = se.get("content_block") or {}
        if cb.get("type") == "tool_use":
            name = cb.get("name", "")
            if state.target_tool:
                if settles_tool_trigger(
                    name, state.target_tool, state.negative_trigger_mode
                ):
                    state.detected_tool = name
                    state.kill_signal.set()
            elif name in ("Skill", "Read"):
                state._cur_tool_name = name
                state._cur_tool_use_id = cb.get("id", "")
                state._cur_accumulated = ""
            else:
                state._cur_tool_name = None
                state._cur_accumulated = ""
                # Negative-trigger mode: any non-Skill/Read tool use
                # means the agent routed elsewhere — settle now.
                if state.negative_trigger_mode:
                    state.kill_signal.set()
    elif se_type == "content_block_delta" and state._cur_tool_name:
        delta = se.get("delta") or {}
        if delta.get("type") == "input_json_delta":
            state._cur_accumulated += delta.get("partial_json", "")
            skill = detect_from_partial(state._cur_tool_name, state._cur_accumulated)
            if skill is not None:
                # Kill on ANY skill invocation, not just the target. Which
                # skill fired is a property of the run; the `first_skill` /
                # `skill_not_invoked` assertions decide pass/fail against
                # the target. Early-killing wrong-skill invocations avoids
                # burning the rest of the wall-clock budget after the
                # eval's answer is already determined.
                state.detected_skill = SkillInvocation(
                    skill_name=skill,
                    trigger_kind="skill_tool"
                    if state._cur_tool_name == "Skill"
                    else "skill_md_read",
                    triggered_by_tool_use_id=state._cur_tool_use_id,
                )
                state.kill_signal.set()
    elif se_type in ("content_block_stop", "message_stop"):
        state._cur_tool_name = None
        state._cur_tool_use_id = None
        state._cur_accumulated = ""
        # Negative-trigger mode: end of the first assistant turn with
        # no skill fire is a decisive "skill didn't route here" signal.
        # A tool target settles on the target call or on the turn ending
        # of its own accord, so message_stop — which also ends a message
        # that only announced tool calls — decides nothing there.
        if (
            state.negative_trigger_mode
            and state.target_tool is None
            and se_type == "message_stop"
            and state.detected_skill is None
        ):
            state.kill_signal.set()
