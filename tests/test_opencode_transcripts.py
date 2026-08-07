from __future__ import annotations

from agent_exam.providers.opencode.provider import OpenCodeProvider
from agent_exam.providers.opencode.stream_parser import StreamState
from agent_exam.providers.opencode.transcripts import (
    _attach_skill_invocations,
    _build_trajectory,
    _classify_error,
    _extract_skill_invocations,
    _tool_call_from_event,
    build_run_result,
)
from agent_exam.schemas import SkillInvocation, TextBlock, ToolCallBlock


def _event(
    type_: str, part: dict | None = None, timestamp: int = 1777228884531, **extra
) -> dict:
    obj = {"type": type_, "timestamp": timestamp, **extra}
    if part is not None:
        obj["part"] = part
    return obj


def _step_start(message_id: str, timestamp: int = 1777228884531) -> dict:
    return _event(
        "step_start",
        {"id": "prt_1", "messageID": message_id, "type": "step-start"},
        timestamp=timestamp,
    )


def _text(message_id: str, text: str) -> dict:
    return _event(
        "text", {"id": "prt_2", "messageID": message_id, "type": "text", "text": text}
    )


def _tool_use(
    message_id: str,
    tool: str,
    call_id: str,
    status: str = "completed",
    output: str = "",
    error: str = "",
    input_: dict | None = None,
    metadata: dict | None = None,
) -> dict:
    state: dict = {
        "status": status,
        "input": input_ or {},
        "time": {"start": 1777228884532, "end": 1777228884534},
    }
    if output:
        state["output"] = output
    if error:
        state["error"] = error
    if metadata:
        state["metadata"] = metadata
    return _event(
        "tool_use",
        {
            "type": "tool",
            "tool": tool,
            "callID": call_id,
            "state": state,
            "messageID": message_id,
        },
    )


def _step_finish(
    message_id: str,
    reason: str = "stop",
    tokens: dict | None = None,
    cost: float = 0.001,
) -> dict:
    return _event(
        "step_finish",
        {
            "id": "prt_3",
            "messageID": message_id,
            "type": "step-finish",
            "reason": reason,
            "tokens": tokens
            or {
                "total": 100,
                "input": 80,
                "output": 20,
                "reasoning": 0,
                "cache": {"read": 0, "write": 0},
            },
            "cost": cost,
        },
    )


def test_simple_text_response():
    events = [
        _step_start("msg_1"),
        _text("msg_1", "4"),
        _step_finish("msg_1"),
    ]
    turns, _ = _build_trajectory(events)
    assert len(turns) == 1
    assert turns[0].role == "assistant"
    assert len(turns[0].content) == 1
    assert isinstance(turns[0].content[0], TextBlock)
    assert turns[0].content[0].text == "4"


def test_tool_call_then_text():
    events = [
        _step_start("msg_1"),
        _tool_use("msg_1", "bash", "call_1", output="hello\n"),
        _step_finish("msg_1", reason="tool-calls"),
        _step_start("msg_2"),
        _text("msg_2", "result"),
        _step_finish("msg_2"),
    ]
    turns, _ = _build_trajectory(events)
    assert len(turns) == 2
    assert turns[0].content[0].name == "bash"
    assert turns[0].content[0].status == "ok"
    assert turns[1].content[0].text == "result"


def test_tool_status_completed_ok():
    block, _ = _tool_call_from_event(
        _tool_use("msg_1", "bash", "call_1", output="done")["part"]
    )
    assert block is not None
    assert block.status == "ok"


def test_tool_status_error():
    block, _ = _tool_call_from_event(
        _tool_use("msg_1", "bash", "call_1", status="error", error="command not found")[
            "part"
        ]
    )
    assert block is not None
    assert block.status == "error"


def test_tool_status_running_is_aborted():
    block, _ = _tool_call_from_event(
        _tool_use("msg_1", "bash", "call_1", status="running")["part"]
    )
    assert block is not None
    assert block.status == "aborted"
    assert block.result == ""


def test_tool_status_permission_denied_ask():
    block, _ = _tool_call_from_event(
        _tool_use(
            "msg_1",
            "bash",
            "call_1",
            status="error",
            error="The user rejected permission to use this specific tool call.",
        )["part"]
    )
    assert block is not None
    assert block.status == "permission_denied"


def test_tool_status_permission_denied_deny():
    block, _ = _tool_call_from_event(
        _tool_use(
            "msg_1",
            "bash",
            "call_1",
            status="error",
            error="The user has specified a rule which prevents you from using this specific tool call.",
        )["part"]
    )
    assert block is not None
    assert block.status == "permission_denied"


def test_classify_error():
    assert (
        _classify_error("The user rejected permission to use this specific tool call.")
        == "permission_denied"
    )
    assert (
        _classify_error(
            "The user has specified a rule which prevents you from using this specific tool call."
        )
        == "permission_denied"
    )
    assert _classify_error("something went wrong") == "error"
    assert _classify_error("") == "error"


def test_skill_invocation_extraction():
    events = [
        _step_start("msg_1"),
        _tool_use(
            "msg_1",
            "skill",
            "call_sk1",
            input_={"name": "scrape-codegen"},
            output="...",
        ),
        _step_finish("msg_1", reason="tool-calls"),
        _step_start("msg_2"),
        _text("msg_2", "done"),
        _step_finish("msg_2"),
    ]
    turns, _ = _build_trajectory(events)
    invocations = _extract_skill_invocations(turns, None, OpenCodeProvider())
    _attach_skill_invocations(turns, invocations)
    all_inv = [si for t in turns for si in t.skill_invocations]
    assert len(all_inv) == 1
    assert all_inv[0].skill_name == "scrape-codegen"
    assert all_inv[0].trigger_kind == "skill_tool"


def test_skill_md_read_not_counted_as_invocation():
    # Reading SKILL.md is skill discovery, not invocation — should not
    # appear as a SkillInvocation in the trajectory.
    events = [
        _step_start("msg_1"),
        _tool_use(
            "msg_1",
            "read",
            "call_rd1",
            input_={"filePath": "/tmp/.opencode/skills/my-skill/SKILL.md"},
            output="...",
        ),
        _step_finish("msg_1", reason="tool-calls"),
        _step_start("msg_2"),
        _text("msg_2", "done"),
        _step_finish("msg_2"),
    ]
    turns, _ = _build_trajectory(events)
    invocations = _extract_skill_invocations(turns, None, OpenCodeProvider())
    assert len(invocations) == 0


def test_metrics_aggregation():
    from agent_exam.providers.opencode.transcripts import _build_metrics

    db_metrics = {
        "input": 2300,
        "output": 150,
        "cache_read": 400,
        "cache_write": 120,
        "reasoning": 30,
        "peak_context": 2000,
        "cost": 0.015,
    }
    metrics = _build_metrics(
        wall_time_seconds=2.5, trajectory=[], db_metrics=db_metrics
    )
    assert metrics.tokens.input == 2300
    assert metrics.tokens.output == 150
    assert metrics.tokens.cache_read == 400
    assert metrics.cost_usd == 0.015
    assert metrics.peak_context == 2000
    assert metrics.wall_time_seconds == 2.5
    assert metrics.raw["cache_write"] == 120
    assert metrics.raw["reasoning"] == 30


def test_subagent_session_id_extracted():
    block, sa_id = _tool_call_from_event(
        _tool_use(
            "msg_1",
            "task",
            "call_t1",
            input_={
                "description": "test",
                "prompt": "do stuff",
                "subagent_type": "general",
            },
            output="result",
            metadata={
                "sessionId": "ses_sub123",
                "model": {"modelID": "test", "providerID": "test"},
            },
        )["part"]
    )
    assert block is not None
    assert block.name == "task"
    assert sa_id == "ses_sub123"
    _turns, subagent_map = _build_trajectory(
        [
            _step_start("msg_1"),
            _tool_use(
                "msg_1",
                "task",
                "call_t1",
                input_={"description": "test"},
                output="result",
                metadata={"sessionId": "ses_sub123"},
            ),
            _step_finish("msg_1"),
        ]
    )
    assert "call_t1" in subagent_map
    assert subagent_map["call_t1"] == "ses_sub123"


def test_bash_nonzero_exit_is_ok():
    block, _ = _tool_call_from_event(
        _tool_use("msg_1", "bash", "call_b1", output="error: file not found")["part"]
    )
    assert block is not None
    assert block.status == "ok"


def test_skill_tool_not_counted_as_tool_call():
    events = [
        _step_start("msg_1"),
        _tool_use("msg_1", "skill", "call_sk1", input_={"name": "test"}, output="..."),
        _step_finish("msg_1"),
    ]
    turns, _ = _build_trajectory(events)
    tool_blocks = [
        b
        for t in turns
        for b in t.content
        if isinstance(b, ToolCallBlock) and b.name != "skill"
    ]
    assert len(tool_blocks) == 0


def test_stream_detected_skill_appended():
    # When we kill early and the trajectory is empty, stream_detected_skill
    # should still be surfaced (build_run_result creates a synthetic turn).
    stream_skill = SkillInvocation(
        skill_name="my-skill",
        trigger_kind="skill_tool",
        triggered_by_tool_use_id="call_1",
    )
    state = StreamState(
        provider=OpenCodeProvider()
    )  # session_id=None → empty trajectory from DB
    result = build_run_result(
        state, wall_time_seconds=1.0, stream_detected_skill=stream_skill
    )
    invocations = [si for t in result.trajectory for si in t.skill_invocations]
    assert len(invocations) == 1
    assert invocations[0].skill_name == "my-skill"


def test_multiple_tool_calls_in_turn():
    events = [
        _step_start("msg_1"),
        _tool_use("msg_1", "bash", "call_1", output="out1"),
        _tool_use(
            "msg_1",
            "read",
            "call_2",
            input_={"filePath": "/tmp/test.py"},
            output="out2",
        ),
        _step_finish("msg_1", reason="tool-calls"),
    ]
    turns, _ = _build_trajectory(events)
    assert len(turns) == 1
    assert len(turns[0].content) == 2
    assert turns[0].content[0].name == "bash"
    assert turns[0].content[1].name == "read"


def test_permission_denied_counted():
    events = [
        _step_start("msg_1"),
        _tool_use(
            "msg_1",
            "bash",
            "call_1",
            status="error",
            error="The user rejected permission to use this specific tool call.",
        ),
        _step_finish("msg_1", reason="tool-calls"),
    ]
    turns, _ = _build_trajectory(events)
    tool_blocks = [b for t in turns for b in t.content if isinstance(b, ToolCallBlock)]
    assert len(tool_blocks) == 1
    assert tool_blocks[0].status == "permission_denied"


def test_turn_started_at_from_step_start_timestamp():
    events = [
        _step_start("msg_1", timestamp=1777228884531),
        _text("msg_1", "hi"),
        _step_finish("msg_1"),
    ]
    turns, _ = _build_trajectory(events)
    assert len(turns) == 1
    assert turns[0].started_at == 1777228884531 / 1000.0


def test_turn_tokens_from_step_finish():
    events = [
        _step_start("msg_1"),
        _text("msg_1", "hi"),
        _step_finish(
            "msg_1",
            tokens={
                "total": 500,
                "input": 400,
                "output": 50,
                "reasoning": 10,
                "cache": {"read": 30, "write": 10},
            },
        ),
    ]
    turns, _ = _build_trajectory(events)
    assert len(turns) == 1
    assert turns[0].tokens is not None
    assert turns[0].tokens.input == 400
    assert turns[0].tokens.output == 50
    assert turns[0].tokens.cache_read == 30


def test_user_prompt_injected():
    events = [
        _step_start("msg_1"),
        _text("msg_1", "answer"),
        _step_finish("msg_1"),
    ]
    turns, _ = _build_trajectory(events, user_prompt="What is 2+2?")
    assert len(turns) == 2
    assert turns[0].role == "user"
    assert isinstance(turns[0].content[0], TextBlock)
    assert turns[0].content[0].text == "What is 2+2?"
    assert turns[1].role == "assistant"


def test_user_prompt_not_duplicated_when_text_event_present():
    events = [
        _step_start("msg_1"),
        _text("msg_1", "answer"),
        _step_finish("msg_1"),
    ]
    turns, _ = _build_trajectory(events, user_prompt="What is 2+2?")
    user_turns = [t for t in turns if t.role == "user"]
    assert len(user_turns) == 1
