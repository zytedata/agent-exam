from __future__ import annotations

import io
import json
import re
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import IO

from ...mcp import canonical_tool_name, settles_tool_trigger
from ...schemas import SkillInvocation
from ..base import Provider


@dataclass
class StreamState:
    """Mutable shared state that a reader thread fills as events arrive."""

    session_id: str | None = None
    stderr_tail: deque = field(default_factory=lambda: deque(maxlen=8192))

    # MCP server name -> connection status, read off the `--print-logs`
    # output on stderr. A server that fails to start leaves the agent
    # without its tools and nothing else says so; the attempt is failed
    # over it (see `mcp.connection_check`).
    mcp_server_status: dict[str, str] | None = None

    # Skill-detection fields (driven by drain_stream when
    # skill_detection_enabled=True). `kill_signal` fires on a match so the
    # provider's main thread can terminate the subprocess early.
    skill_detection_enabled: bool = False
    target_skill: str | None = None
    detected_skill: SkillInvocation | None = None
    kill_signal: threading.Event = field(default_factory=threading.Event)

    # Tool-targeted trigger: the canonical name of the tool the run is cut
    # on, and the configured MCP server names its harness spelling is
    # matched through. `detected_tool` holds the spelling OpenCode used.
    target_tool: str | None = None
    detected_tool: str | None = None
    mcp_server_names: tuple[str, ...] = ()

    # Negative-trigger mode: for cases where the skill is expected NOT
    # to fire, the routing decision is settled as soon as the agent
    # either (a) starts a non-skill tool use, or (b) finishes its
    # first assistant turn without firing a skill.
    negative_trigger_mode: bool = False

    # Provider instance — used by the transcript builder's dedupe check
    # and any other harness-specific hooks. Required, no default: the
    # real provider passes `self`, and tests that build StreamState
    # directly must pass one explicitly so a forgotten wiring fails
    # loudly.
    provider: Provider = field(kw_only=True)


def drain_stream(
    stdout: IO[bytes], state: StreamState, raw_out: IO[bytes] | None = None
) -> None:
    """Read NDJSON lines from `stdout`, extracting fields we care about.

    If `raw_out` is provided, each raw line is written there verbatim as it
    arrives (the caller owns the file handle and closes it). Writing happens
    before any parsing so the file is faithful to the original byte stream.
    """
    for raw_line in io.TextIOWrapper(stdout, encoding="utf-8", errors="replace"):
        line = raw_line.rstrip("\n")
        if raw_out is not None:
            raw_out.write((line + "\n").encode())
        if line.strip():
            _dispatch(line, state)


_MCP_LOG = re.compile(r"\bservice=mcp\b.*?\bkey=(\S+)")
_MCP_CONNECTED = "create() successfully created client"
_ROUTINE_LOG_LEVELS = ("INFO ", "DEBUG ")
_ROUTINE_PREFIX_LEN = max(len(level) for level in _ROUTINE_LOG_LEVELS)


def drain_stderr(stderr: IO[bytes], state: StreamState) -> None:
    """Read raw chunks, not text lines, dispatching each complete log line
    to `_dispatch_log` and folding routine ones out of the tail.

    A line with no trailing newline yet — e.g. a hung MCP server that wrote
    a diagnostic mid-line — would otherwise sit invisible until a newline
    or EOF a hung child may never produce. It is flushed into the tail as
    soon as enough of it has arrived to rule out a routine log level.
    """
    buf = bytearray()
    flushed = 0
    while True:
        chunk = stderr.read1(4096)
        if not chunk:
            break
        buf.extend(chunk)
        while True:
            newline_idx = buf.find(b"\n")
            if newline_idx == -1:
                break
            raw_line = bytes(buf[: newline_idx + 1])
            del buf[: newline_idx + 1]
            line = raw_line.decode("utf-8", errors="replace")
            _dispatch_log(line, state)
            if not line.startswith(_ROUTINE_LOG_LEVELS):
                state.stderr_tail.extend(raw_line[flushed:])
            flushed = 0
        pending = bytes(buf)
        if len(pending) >= _ROUTINE_PREFIX_LEN and len(pending) > flushed:
            pending_text = pending.decode("utf-8", errors="replace")
            if not pending_text.startswith(_ROUTINE_LOG_LEVELS):
                state.stderr_tail.extend(pending[flushed:])
                flushed = len(pending)
    pending = bytes(buf)
    if len(pending) > flushed:
        pending_text = pending.decode("utf-8", errors="replace")
        if not pending_text.startswith(_ROUTINE_LOG_LEVELS):
            state.stderr_tail.extend(pending[flushed:])


def _dispatch_log(line: str, state: StreamState) -> None:
    """Track the MCP connection statuses opencode logs to stderr.

    A server is announced as ``found`` when its config is read, and again
    once its client exists. Whatever can go wrong in between — a command
    that is not on `PATH`, a URL that does not answer — leaves it at the
    first line with no further mention, so ``found`` starts a server off as
    failed and only the client line clears it. A status never regresses
    from ``connected`` back to ``failed``, so a stray repeat of the
    ``found`` line (a second config read racing the first) can't undo an
    already-successful connection.
    """
    match = _MCP_LOG.search(line)
    if match is None:
        return
    if _MCP_CONNECTED in line:
        status = "connected"
    elif line.rstrip().endswith(" found"):
        status = "failed"
    else:
        return
    if state.mcp_server_status is None:
        state.mcp_server_status = {}
    key = match.group(1)
    if status == "failed" and state.mcp_server_status.get(key) == "connected":
        return
    state.mcp_server_status[key] = status


def _dispatch(line: str, state: StreamState) -> None:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return
    if not isinstance(event, dict):
        return

    sid = event.get("sessionID")
    if sid and state.session_id is None:
        state.session_id = sid

    event_type = event.get("type")
    part = event.get("part") or {}

    if event_type == "step_finish":
        if (
            state.skill_detection_enabled
            and state.negative_trigger_mode
            and state.target_tool is None
        ):
            reason = part.get("reason")
            if reason == "stop" and state.detected_skill is None:
                state.kill_signal.set()

    elif event_type == "tool_use" and state.skill_detection_enabled:
        _dispatch_skill_detection(part, state)


def _dispatch_skill_detection(part: dict, state: StreamState) -> None:
    tool = part.get("tool")
    call_id = part.get("callID", "")
    tool_state = part.get("state") or {}
    tool_status = tool_state.get("status")

    if state.target_tool:
        # OpenCode publishes a tool part on stdout once the call is over, so
        # the kill lands on the finished call and saves the rest of the turn
        # rather than the call itself. `running` is not decisive — the same
        # part repeats once the call actually finishes.
        if (
            isinstance(tool, str)
            and tool_status in ("completed", "error")
            and settles_tool_trigger(
                canonical_tool_name(tool, state.mcp_server_names),
                state.target_tool,
                state.negative_trigger_mode,
            )
        ):
            state.detected_tool = tool
            state.kill_signal.set()
        return

    if tool == "skill" and tool_status in ("completed", "error"):
        skill_name = (tool_state.get("input") or {}).get("name")
        if skill_name:
            state.detected_skill = SkillInvocation(
                skill_name=skill_name,
                trigger_kind="skill_tool",
                triggered_by_tool_use_id=call_id,
            )
            state.kill_signal.set()
            return

    # For negative triggers: any completed tool that isn't the skill dispatcher
    # means routing went elsewhere — the decision is settled.
    if (
        state.negative_trigger_mode
        and tool != "skill"
        and tool_status
        in (
            "completed",
            "error",
        )
    ):
        state.kill_signal.set()
