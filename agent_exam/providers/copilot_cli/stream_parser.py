from __future__ import annotations

import io
import json
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import IO

from ...schemas import SkillInvocation
from ..base import Provider


@dataclass
class StreamState:
    """Mutable shared state filled by the reader thread as JSONL events arrive."""

    session_id: str | None = None
    model: str | None = None
    exit_code: int | None = None
    premium_requests: int = 0
    stderr_tail: deque = field(default_factory=lambda: deque(maxlen=8192))

    # All non-ephemeral events, in arrival order, for trajectory building.
    events: list = field(default_factory=list)

    # Skill-detection fields (populated when skill_detection_enabled=True).
    # `kill_signal` fires on a match so the provider's main thread can
    # terminate the subprocess early.
    skill_detection_enabled: bool = False
    target_skill: str | None = None
    detected_skill: SkillInvocation | None = None
    kill_signal: threading.Event = field(default_factory=threading.Event)

    # Negative-trigger mode: kill as soon as the routing decision is clear.
    # For Copilot CLI this is always at assistant.message time (tool calls are
    # not streamed incrementally, only the final accumulated message carries
    # toolRequests).
    negative_trigger_mode: bool = False

    # Provider instance — used for skill-name matching and any other
    # harness-specific hooks. Required, no default: the real provider
    # passes `self`, and tests that build StreamState directly must pass
    # one explicitly so a forgotten wiring fails loudly.
    provider: Provider = field(kw_only=True)


def drain_stream(
    stdout: IO[bytes],
    state: StreamState,
    raw_out: IO[bytes] | None = None,
) -> None:
    """Read NDJSON lines from `stdout`, extracting fields we care about.

    If `raw_out` is provided, each raw line is written there verbatim as it
    arrives (the caller owns the file handle and closes it).
    """
    for raw_line in io.TextIOWrapper(stdout, encoding="utf-8", errors="replace"):
        line = raw_line.rstrip("\n")
        if raw_out is not None:
            raw_out.write((line + "\n").encode())
        if line.strip():
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

    event_type = event.get("type", "")
    is_ephemeral = event.get("ephemeral", False)

    # Always collect non-ephemeral events for trajectory building.
    if not is_ephemeral:
        state.events.append(event)

    if event_type == "session.tools_updated":
        model = (event.get("data") or {}).get("model")
        if model and state.model is None:
            state.model = model

    elif event_type == "assistant.message":
        if state.skill_detection_enabled:
            _check_skill_in_message(event, state)
            # For negative trigger mode: once the model's tool requests are
            # known (i.e. this message has arrived), if no skill was requested
            # the routing decision is settled — kill immediately rather than
            # waiting for tool.execution_start or assistant.turn_end.
            if (
                state.negative_trigger_mode
                and not state.kill_signal.is_set()
                and state.detected_skill is None
            ):
                state.kill_signal.set()

    elif event_type == "assistant.turn_end":
        # Belt-and-suspenders: if kill_signal wasn't already set by
        # assistant.message (e.g. the stream is missing that event),
        # fire here as a fallback.
        if (
            state.skill_detection_enabled
            and state.negative_trigger_mode
            and not state.kill_signal.is_set()
            and state.detected_skill is None
        ):
            state.kill_signal.set()

    elif event_type == "result":
        data = event.get("data") or {}
        sid = data.get("sessionId")
        if sid:
            state.session_id = sid
        exit_code = data.get("exitCode")
        if exit_code is not None:
            state.exit_code = exit_code
        usage = data.get("usage") or {}
        state.premium_requests = usage.get("premiumRequests", 0)


def _check_skill_in_message(event: dict, state: StreamState) -> None:
    """Check assistant.message toolRequests for a skill invocation (positive trigger)."""
    data = event.get("data") or {}
    tool_requests = data.get("toolRequests") or []
    for req in tool_requests:
        if req.get("name") != "skill":
            continue
        skill_name = (req.get("arguments") or {}).get("skill")
        if skill_name is None:
            continue
        if state.target_skill and not state.provider.is_same_skill(
            skill_name, state.target_skill
        ):
            continue
        state.detected_skill = SkillInvocation(
            skill_name=skill_name,
            trigger_kind="skill_tool",
            triggered_by_tool_use_id=req.get("toolCallId"),
        )
        state.kill_signal.set()
        return
