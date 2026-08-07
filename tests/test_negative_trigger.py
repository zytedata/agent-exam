"""Negative-trigger stream logic: the kill_signal fires early when the
routing decision is already evident — either a non-Skill tool was
used, or the first assistant turn ended without a skill fire. Lets
negative-trigger attempts settle in seconds rather than burning the
full 30s wall-clock.
"""

from __future__ import annotations

import json

from agent_exam.providers.claude_code.stream_parser import (
    StreamState,
    _dispatch,
)


def _emit(state: StreamState, event: dict) -> None:
    _dispatch(json.dumps(event).encode("utf-8"), state)


def _state() -> StreamState:
    s = StreamState()
    s.skill_detection_enabled = True
    s.negative_trigger_mode = True
    return s


def test_non_skill_tool_fires_kill_signal():
    s = _state()
    _emit(
        s,
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_start",
                "content_block": {"type": "tool_use", "name": "Bash", "id": "t1"},
            },
        },
    )
    assert s.kill_signal.is_set()
    assert s.detected_skill is None


def test_skill_tool_delays_kill_until_delta():
    """Starting a Skill tool_use shouldn't fire kill in negative mode —
    the routing decision is still in flight. Kill fires only if the
    streamed input names an actual skill (normal detection path)."""
    s = _state()
    _emit(
        s,
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_start",
                "content_block": {"type": "tool_use", "name": "Skill", "id": "t1"},
            },
        },
    )
    assert not s.kill_signal.is_set()


def test_message_stop_without_skill_fires_kill_signal():
    """First assistant turn ends with no skill fire → decisive negative."""
    s = _state()
    _emit(
        s,
        {
            "type": "stream_event",
            "event": {"type": "message_stop"},
        },
    )
    assert s.kill_signal.is_set()
    assert s.detected_skill is None


def test_message_stop_not_fired_when_positive():
    """Non-negative mode: message_stop alone shouldn't kill."""
    s = StreamState()
    s.skill_detection_enabled = True  # positive-trigger defaults
    _emit(
        s,
        {
            "type": "stream_event",
            "event": {"type": "message_stop"},
        },
    )
    assert not s.kill_signal.is_set()


def test_skill_detection_still_works_in_negative_mode():
    """Even in negative mode, a detected skill still sets detected_skill
    — so the skill_not_invoked assertion has the signal to fail on.
    """
    s = _state()
    _emit(
        s,
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_start",
                "content_block": {"type": "tool_use", "name": "Skill", "id": "t1"},
            },
        },
    )
    _emit(
        s,
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"skill": "scrape-scrapy-cloud"',
                },
            },
        },
    )
    assert s.kill_signal.is_set()
    assert s.detected_skill is not None
    assert s.detected_skill.skill_name == "scrape-scrapy-cloud"
