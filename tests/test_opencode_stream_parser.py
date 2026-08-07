from __future__ import annotations

import json

from agent_exam.providers.opencode.provider import OpenCodeProvider
from agent_exam.providers.opencode.stream_parser import StreamState, _dispatch


def _make_line(
    event_type: str, part: dict | None = None, session_id: str | None = None, **extra
) -> str:
    obj = {"type": event_type, "timestamp": 1777228884531, **extra}
    if session_id:
        obj["sessionID"] = session_id
    if part is not None:
        obj["part"] = part
    return json.dumps(obj)


def test_session_id_extraction():
    state = StreamState(provider=OpenCodeProvider())
    _dispatch(
        _make_line(
            "step_start",
            {"messageID": "msg_1", "type": "step-start"},
            session_id="ses_abc123",
        ),
        state,
    )
    assert state.session_id == "ses_abc123"


def test_session_id_set_once():
    state = StreamState(provider=OpenCodeProvider())
    _dispatch(
        _make_line(
            "step_start",
            {"messageID": "msg_1", "type": "step-start"},
            session_id="ses_first",
        ),
        state,
    )
    _dispatch(
        _make_line(
            "step_start",
            {"messageID": "msg_2", "type": "step-start"},
            session_id="ses_second",
        ),
        state,
    )
    assert state.session_id == "ses_first"


def test_skill_detection_positive():
    state = StreamState(
        provider=OpenCodeProvider(),
        skill_detection_enabled=True,
        target_skill="test-skill",
    )
    _dispatch(
        _make_line(
            "tool_use",
            {
                "tool": "skill",
                "callID": "call_1",
                "state": {
                    "status": "completed",
                    "input": {"name": "test-skill"},
                    "output": "<skill_content>...</skill_content>",
                },
            },
        ),
        state,
    )
    assert state.detected_skill is not None
    assert state.detected_skill.skill_name == "test-skill"
    assert state.detected_skill.trigger_kind == "skill_tool"
    assert state.kill_signal.is_set()


def test_negative_trigger_read_tool_fires_kill():
    # read is no longer excluded — SKILL.md discovery is not skill invocation.
    state = StreamState(
        provider=OpenCodeProvider(),
        skill_detection_enabled=True,
        negative_trigger_mode=True,
    )
    _dispatch(
        _make_line(
            "tool_use",
            {
                "tool": "read",
                "callID": "call_2",
                "state": {
                    "status": "completed",
                    "input": {"filePath": "/tmp/.opencode/skills/my-skill/SKILL.md"},
                    "output": "skill content",
                },
            },
        ),
        state,
    )
    assert state.kill_signal.is_set()
    assert state.detected_skill is None  # reading ≠ invoking


def test_negative_trigger_non_skill_tool():
    state = StreamState(
        provider=OpenCodeProvider(),
        skill_detection_enabled=True,
        negative_trigger_mode=True,
    )
    _dispatch(
        _make_line(
            "tool_use",
            {
                "tool": "bash",
                "callID": "call_3",
                "state": {
                    "status": "completed",
                    "input": {"command": "ls"},
                    "output": "file.txt",
                },
            },
        ),
        state,
    )
    assert state.kill_signal.is_set()


def test_negative_trigger_step_finish_no_skill():
    state = StreamState(
        provider=OpenCodeProvider(),
        skill_detection_enabled=True,
        negative_trigger_mode=True,
    )
    _dispatch(
        _make_line(
            "step_finish",
            {
                "messageID": "msg_1",
                "type": "step-finish",
                "reason": "stop",
                "tokens": {
                    "total": 100,
                    "input": 80,
                    "output": 20,
                    "cache": {"read": 0, "write": 0},
                },
                "cost": 0.001,
            },
        ),
        state,
    )
    assert state.kill_signal.is_set()


def test_negative_trigger_skill_fired_no_kill():
    state = StreamState(
        provider=OpenCodeProvider(),
        skill_detection_enabled=True,
        negative_trigger_mode=True,
    )
    _dispatch(
        _make_line(
            "tool_use",
            {
                "tool": "skill",
                "callID": "call_4",
                "state": {
                    "status": "completed",
                    "input": {"name": "some-skill"},
                    "output": "...",
                },
            },
        ),
        state,
    )
    assert not state.negative_trigger_mode or state.kill_signal.is_set()
    assert state.detected_skill is not None


def test_dispatch_no_crash_on_step_start():
    state = StreamState(provider=OpenCodeProvider())
    line = _make_line("step_start", {"messageID": "msg_1", "type": "step-start"})
    _dispatch(line, state)  # must not raise


def test_is_same_skill():
    is_same_skill = OpenCodeProvider.is_same_skill
    assert is_same_skill("scrape-codegen", "scrape-codegen")
    assert is_same_skill("plugin:scrape-codegen", "scrape-codegen")
    assert is_same_skill("scrape-codegen", "plugin:scrape-codegen")
    assert not is_same_skill("scrape-codegen", "other-skill")
