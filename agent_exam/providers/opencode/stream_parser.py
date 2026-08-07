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
    """Mutable shared state that a reader thread fills as events arrive."""

    session_id: str | None = None
    stderr_tail: deque = field(default_factory=lambda: deque(maxlen=8192))

    # Skill-detection fields (driven by drain_stream when
    # skill_detection_enabled=True). `kill_signal` fires on a match so the
    # provider's main thread can terminate the subprocess early.
    skill_detection_enabled: bool = False
    target_skill: str | None = None
    detected_skill: SkillInvocation | None = None
    kill_signal: threading.Event = field(default_factory=threading.Event)

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

    sid = event.get("sessionID")
    if sid and state.session_id is None:
        state.session_id = sid

    event_type = event.get("type")
    part = event.get("part") or {}

    if event_type == "step_finish":
        if state.skill_detection_enabled and state.negative_trigger_mode:
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
