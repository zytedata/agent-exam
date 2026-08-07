from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_exam.errors import FrameworkError
from agent_exam.providers.codex_cli.stream_parser import (
    StreamState,
    _dispatch,
    stream_error_messages,
)
from agent_exam.providers.codex_cli.transcripts import (
    build_run_result,
    find_session_explicit_skill_invocation,
)
from agent_exam.schemas import TextBlock, ThinkingBlock, ToolCallBlock


@pytest.fixture(autouse=True)
def _isolated_codex_home(monkeypatch):
    """codex_home() checks $CODEX_HOME before Path.home(); on a machine
    where it's set (the exact audience for this provider), tests that only
    patch Path.home() would silently read the developer's real Codex
    session tree instead of the tmp_path fixture."""
    monkeypatch.delenv("CODEX_HOME", raising=False)


def _line(event_type: str, **extra) -> str:
    return json.dumps({"type": event_type, **extra})


def _write_codex_session(home: Path, thread_id: str, events: list[dict]) -> Path:
    return _write_codex_session_under(home / ".codex", thread_id, events)


def _write_codex_session_under(
    codex_home: Path, thread_id: str, events: list[dict]
) -> Path:
    path = (
        codex_home
        / "sessions"
        / "2026"
        / "06"
        / "15"
        / f"rollout-2026-06-15T00-00-00-{thread_id}.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    return path


def test_thread_id_extraction():
    state = StreamState()
    _dispatch(_line("thread.started", thread_id="thread_123"), state)
    assert state.thread_id == "thread_123"


def test_usage_extraction():
    state = StreamState()
    _dispatch(
        _line(
            "turn.completed",
            usage={
                "input_tokens": 10,
                "cached_input_tokens": 4,
                "output_tokens": 3,
                "reasoning_output_tokens": 2,
            },
        ),
        state,
    )
    assert state.input_tokens == 10
    assert state.cached_input_tokens == 4
    assert state.output_tokens == 3
    assert state.reasoning_output_tokens == 2


def test_skill_detection_from_skill_md_read_command():
    state = StreamState(skill_detection_enabled=True)
    _dispatch(
        _line(
            "item.completed",
            item={
                "id": "item_1",
                "type": "command_execution",
                "command": "sed -n '1,160p' .agents/skills/probe-skill/SKILL.md",
                "aggregated_output": "# skill",
                "exit_code": 0,
                "status": "completed",
            },
        ),
        state,
    )
    assert state.detected_skill is not None
    assert state.detected_skill.skill_name == "probe-skill"
    assert state.detected_skill.trigger_kind == "skill_md_read"
    assert state.kill_signal.is_set()


def test_skill_detection_from_codex_shared_read_parser_command():
    state = StreamState(skill_detection_enabled=True)
    _dispatch(
        _line(
            "item.completed",
            item={
                "id": "item_1",
                "type": "command_execution",
                "command": "nl -ba .agents/skills/probe-skill/SKILL.md",
                "aggregated_output": "# skill",
                "exit_code": 0,
                "status": "completed",
            },
        ),
        state,
    )
    assert state.detected_skill is not None
    assert state.detected_skill.skill_name == "probe-skill"
    assert state.detected_skill.trigger_kind == "skill_md_read"
    assert state.kill_signal.is_set()


def test_skill_detection_from_shell_wrapped_read_command():
    """Codex runs every command through a login shell, e.g.
    `/bin/zsh -lc 'cat SKILL.md'` — argv[0] is the shell, not the reader."""
    state = StreamState(skill_detection_enabled=True)
    _dispatch(
        _line(
            "item.completed",
            item={
                "id": "item_1",
                "type": "command_execution",
                "command": "/bin/zsh -lc 'cat .agents/skills/probe-skill/SKILL.md'",
                "aggregated_output": "# skill",
                "exit_code": 0,
                "status": "completed",
            },
        ),
        state,
    )
    assert state.detected_skill is not None
    assert state.detected_skill.skill_name == "probe-skill"
    assert state.detected_skill.trigger_kind == "skill_md_read"
    assert state.kill_signal.is_set()


@pytest.mark.parametrize(
    ("command", "workdir"),
    [
        ("python3 -u .agents/skills/probe-skill/scripts/run.py", None),
        ("python3 scripts/run.py", "/repo/.agents/skills/probe-skill"),
        (
            "uv run --no-project .agents/skills/probe-skill/scripts/run.py --check-key",
            None,
        ),
        (
            (
                "uv run --project . --with libcst "
                ".agents/skills/probe-skill/scripts/run.py"
            ),
            None,
        ),
        ("uv run python -u scripts/run.py", "/repo/.agents/skills/probe-skill"),
    ],
)
def test_skill_detection_ignores_skill_script_commands(command, workdir):
    state = StreamState(skill_detection_enabled=True)
    item = {
        "id": "item_1",
        "type": "command_execution",
        "command": command,
        "aggregated_output": "ok",
        "exit_code": 0,
        "status": "completed",
    }
    if workdir:
        item["workdir"] = workdir
    _dispatch(
        _line(
            "item.completed",
            item=item,
        ),
        state,
    )
    assert state.detected_skill is None
    assert not state.kill_signal.is_set()


def test_wrong_skill_md_read_still_kills_and_records_actual_skill():
    state = StreamState(
        skill_detection_enabled=True,
    )
    _dispatch(
        _line(
            "item.completed",
            item={
                "id": "item_1",
                "type": "command_execution",
                "command": "sed -n '1,160p' .agents/skills/other-skill/SKILL.md",
                "aggregated_output": "# skill",
                "exit_code": 0,
                "status": "completed",
            },
        ),
        state,
    )

    assert state.detected_skill is not None
    assert state.detected_skill.skill_name == "other-skill"
    assert state.detected_skill.trigger_kind == "skill_md_read"
    assert state.kill_signal.is_set()


def test_skill_detection_from_resolved_build_skill_path():
    state = StreamState(skill_detection_enabled=True)
    _dispatch(
        _line(
            "item.completed",
            item={
                "id": "item_1",
                "type": "command_execution",
                "command": "sed -n '1,220p' /repo/build/codex_cli/skills/scrape-zyte-login/SKILL.md",
                "aggregated_output": "# skill",
                "exit_code": 0,
                "status": "completed",
            },
        ),
        state,
    )
    assert state.detected_skill is not None
    assert state.detected_skill.skill_name == "scrape-zyte-login"


def test_negative_trigger_kills_on_non_skill_command():
    state = StreamState(
        skill_detection_enabled=True,
        negative_trigger_mode=True,
    )
    _dispatch(
        _line(
            "item.completed",
            item={
                "id": "item_2",
                "type": "command_execution",
                "command": "ls",
                "aggregated_output": "x\n",
                "exit_code": 0,
                "status": "completed",
            },
        ),
        state,
    )
    assert state.detected_skill is None
    assert state.kill_signal.is_set()


def test_negative_trigger_does_not_kill_on_reader_command():
    """A negative-trigger agent legitimately inspects files (even
    shell-wrapped, as Codex always wraps commands) before deciding whether
    to invoke a skill — mirrors Claude Code's Read exemption."""
    state = StreamState(
        skill_detection_enabled=True,
        negative_trigger_mode=True,
    )
    _dispatch(
        _line(
            "item.completed",
            item={
                "id": "item_2",
                "type": "command_execution",
                "command": "/bin/zsh -lc 'cat README.md'",
                "aggregated_output": "hello\n",
                "exit_code": 0,
                "status": "completed",
            },
        ),
        state,
    )
    assert state.detected_skill is None
    assert not state.kill_signal.is_set()


@pytest.mark.parametrize("item_type", ["web_search", "file_change", "mcp_tool_call"])
def test_negative_trigger_kills_on_dedicated_tool_items(item_type):
    """Codex's dedicated tools are as decisive as a shell command — an
    agent fetching a live URL has routed away from the skill; previously
    these burned to the trigger timeout."""
    state = StreamState(
        skill_detection_enabled=True,
        negative_trigger_mode=True,
    )
    _dispatch(
        _line(
            "item.started",
            item={"id": "item_1", "type": item_type},
        ),
        state,
    )
    assert state.detected_skill is None
    assert state.kill_signal.is_set()


def test_negative_trigger_ignores_message_and_reasoning_items():
    state = StreamState(
        skill_detection_enabled=True,
        negative_trigger_mode=True,
    )
    for item_type in ("agent_message", "reasoning", "todo_list", "other"):
        _dispatch(
            _line("item.completed", item={"id": "i", "type": item_type}),
            state,
        )
    assert not state.kill_signal.is_set()


def test_negative_trigger_kills_on_turn_completed_without_skill():
    state = StreamState(
        skill_detection_enabled=True,
        negative_trigger_mode=True,
    )
    _dispatch(
        _line(
            "turn.completed",
            usage={
                "input_tokens": 10,
                "cached_input_tokens": 4,
                "output_tokens": 3,
                "reasoning_output_tokens": 2,
            },
        ),
        state,
    )

    assert state.detected_skill is None
    assert state.kill_signal.is_set()


def test_build_run_result_requires_codex_session_for_completed_run(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    state = StreamState()
    _dispatch(_line("thread.started", thread_id="missing-thread"), state)

    with pytest.raises(FrameworkError, match="persisted session transcript"):
        build_run_result(state, wall_time_seconds=1.0, user_prompt="x")


def test_build_run_result_allows_minimal_trigger_kill_result():
    state = StreamState()
    for line in [
        _line("thread.started", thread_id="missing-thread"),
        _line(
            "item.completed",
            item={
                "id": "item_0",
                "type": "command_execution",
                "command": "sed -n '1,160p' .agents/skills/probe-skill/SKILL.md",
                "aggregated_output": "# skill",
                "exit_code": 0,
                "status": "completed",
            },
        ),
        _line(
            "turn.completed",
            usage={
                "input_tokens": 10,
                "cached_input_tokens": 4,
                "output_tokens": 3,
                "reasoning_output_tokens": 1,
            },
        ),
    ]:
        _dispatch(line, state)

    result = build_run_result(
        state,
        wall_time_seconds=1.5,
        user_prompt="x",
        allow_minimal_trigger_result=True,
    )

    assert len(result.trajectory) == 2
    assert result.trajectory[1].content == []
    assert result.trajectory[1].skill_invocations[0].skill_name == "probe-skill"
    assert result.metrics.n_tool_calls == 0
    assert result.metrics.raw["minimal_trigger_result"] is True


def test_build_run_result_attaches_codex_subagent_session(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    parent_path = _write_codex_session(
        tmp_path,
        "parent-thread",
        [
            {
                "type": "turn_context",
                "timestamp": "2026-06-15T00:00:00.000Z",
                "payload": {"model": "gpt-real"},
            },
            {
                "type": "event_msg",
                "timestamp": "2026-06-15T00:00:01.000Z",
                "payload": {"type": "task_started", "started_at": 90.0},
            },
            {
                "type": "event_msg",
                "timestamp": "2026-06-15T00:00:01.500Z",
                "payload": {"type": "user_message", "message": "parent prompt"},
            },
            {
                "type": "response_item",
                "timestamp": "2026-06-15T00:00:02.000Z",
                "payload": {
                    "type": "function_call",
                    "call_id": "call_spawn",
                    "name": "spawn_agent",
                    "arguments": json.dumps({"message": "child prompt"}),
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-06-15T00:00:03.000Z",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call_spawn",
                    "output": json.dumps({"agent_id": "child-thread"}),
                },
            },
            {
                "type": "event_msg",
                "timestamp": "2026-06-15T00:00:04.000Z",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": {
                            "input_tokens": 10,
                            "cached_input_tokens": 1,
                            "output_tokens": 2,
                            "reasoning_output_tokens": 3,
                        }
                    },
                },
            },
        ],
    )
    child_path = _write_codex_session(
        tmp_path,
        "child-thread",
        [
            {
                "type": "session_meta",
                "timestamp": "2026-06-15T00:00:01.000Z",
                "payload": {
                    "id": "child-thread",
                    "thread_source": "subagent",
                    "source": {
                        "subagent": {
                            "thread_spawn": {"parent_thread_id": "parent-thread"}
                        }
                    },
                },
            },
            {
                "type": "event_msg",
                "timestamp": "2026-06-15T00:00:01.000Z",
                "payload": {"type": "task_started", "started_at": 100.0},
            },
            {
                "type": "event_msg",
                "timestamp": "2026-06-15T00:00:02.000Z",
                "payload": {"type": "user_message", "message": "child prompt"},
            },
            {
                "type": "response_item",
                "timestamp": "2026-06-15T00:00:03.000Z",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "child answer"}],
                },
            },
            {
                "type": "event_msg",
                "timestamp": "2026-06-15T00:00:04.000Z",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": {
                            "input_tokens": 5,
                            "cached_input_tokens": 2,
                            "output_tokens": 1,
                            "reasoning_output_tokens": 1,
                        }
                    },
                },
            },
        ],
    )

    state = StreamState()
    _dispatch(_line("thread.started", thread_id="parent-thread"), state)

    result = build_run_result(state, wall_time_seconds=1.0, user_prompt="x")

    assert result.model == "gpt-real"
    tool = result.trajectory[1].content[0]
    assert isinstance(tool, ToolCallBlock)
    assert tool.subagent is not None
    assert len(tool.subagent) == 2
    assert tool.subagent[0].role == "user"
    assert isinstance(tool.subagent[0].content[0], TextBlock)
    assert tool.subagent[0].content[0].text == "child prompt"
    assert isinstance(tool.subagent[1].content[0], TextBlock)
    assert tool.subagent[1].content[0].text == "child answer"
    assert result.metrics.tokens.input == 15
    assert result.metrics.tokens.output == 3
    assert result.metrics.tokens.cache_read == 3
    assert result.metrics.raw["reasoning_output_tokens"] == 4
    assert result.metrics.raw["subagent_count"] == 1
    assert result.metrics.raw["codex_session_paths"] == [
        str(parent_path),
        str(child_path),
    ]


def test_build_run_result_fails_on_missing_codex_subagent_session(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _write_codex_session(
        tmp_path,
        "parent-thread",
        [
            {
                "type": "event_msg",
                "timestamp": "2026-06-15T00:00:01.000Z",
                "payload": {"type": "user_message", "message": "parent prompt"},
            },
            {
                "type": "response_item",
                "timestamp": "2026-06-15T00:00:02.000Z",
                "payload": {
                    "type": "function_call",
                    "call_id": "call_spawn",
                    "name": "spawn_agent",
                    "arguments": json.dumps({"message": "child prompt"}),
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-06-15T00:00:03.000Z",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call_spawn",
                    "output": json.dumps({"agent_id": "missing-child"}),
                },
            },
        ],
    )

    state = StreamState()
    _dispatch(_line("thread.started", thread_id="parent-thread"), state)

    with pytest.raises(FrameworkError, match="subagent session transcript"):
        build_run_result(state, wall_time_seconds=1.0, user_prompt="x")


def test_build_run_result_allows_minimal_result_on_missing_codex_subagent_session(
    monkeypatch, tmp_path
):
    """A killed-on-skill trigger run can leave a spawned subagent's session
    unflushed. allow_minimal_trigger_result must fall back to the minimal
    stream-derived result instead of propagating the subagent's
    FrameworkError — the trigger was already detected via the stream."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _write_codex_session(
        tmp_path,
        "parent-thread",
        [
            {
                "type": "event_msg",
                "timestamp": "2026-06-15T00:00:01.000Z",
                "payload": {"type": "user_message", "message": "parent prompt"},
            },
            {
                "type": "response_item",
                "timestamp": "2026-06-15T00:00:02.000Z",
                "payload": {
                    "type": "function_call",
                    "call_id": "call_spawn",
                    "name": "spawn_agent",
                    "arguments": json.dumps({"message": "child prompt"}),
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-06-15T00:00:03.000Z",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call_spawn",
                    "output": json.dumps({"agent_id": "missing-child"}),
                },
            },
        ],
    )

    state = StreamState()
    _dispatch(_line("thread.started", thread_id="parent-thread"), state)

    result = build_run_result(
        state,
        wall_time_seconds=1.0,
        user_prompt="x",
        allow_minimal_trigger_result=True,
    )
    assert result.metrics.raw["minimal_trigger_result"] is True


def test_build_run_result_detects_skill_from_codex_session_when_stream_omits_it(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _write_codex_session(
        tmp_path,
        "parent-thread",
        [
            {
                "type": "event_msg",
                "timestamp": "2026-06-15T00:00:01.000Z",
                "payload": {"type": "user_message", "message": "use a skill"},
            },
            {
                "type": "response_item",
                "timestamp": "2026-06-15T00:00:02.000Z",
                "payload": {
                    "type": "function_call",
                    "call_id": "call_skill",
                    "name": "exec_command",
                    "arguments": json.dumps(
                        {"cmd": "sed -n '1,160p' .agents/skills/probe-skill/SKILL.md"}
                    ),
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-06-15T00:00:03.000Z",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call_skill",
                    "output": "# skill",
                },
            },
        ],
    )

    state = StreamState()
    _dispatch(_line("thread.started", thread_id="parent-thread"), state)

    result = build_run_result(state, wall_time_seconds=1.0, user_prompt="x")

    assert result.trajectory[1].skill_invocations[0].skill_name == "probe-skill"
    assert result.trajectory[1].skill_invocations[0].trigger_kind == "skill_md_read"
    assert result.trajectory[1].skill_invocations[0].triggered_by_tool_use_id == (
        "call_skill"
    )


def test_build_run_result_keeps_skill_script_command_as_plain_tool_call(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _write_codex_session(
        tmp_path,
        "parent-thread",
        [
            {
                "type": "event_msg",
                "timestamp": "2026-06-15T00:00:01.000Z",
                "payload": {"type": "user_message", "message": "use a skill script"},
            },
            {
                "type": "response_item",
                "timestamp": "2026-06-15T00:00:02.000Z",
                "payload": {
                    "type": "function_call",
                    "call_id": "call_skill_script",
                    "name": "exec_command",
                    "arguments": json.dumps(
                        {
                            "cmd": (
                                "uv run --no-project "
                                ".agents/skills/probe-skill/scripts/run.py"
                            ),
                        }
                    ),
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-06-15T00:00:03.000Z",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call_skill_script",
                    "output": "Process exited with code 0\nOutput:\nok\n",
                },
            },
        ],
    )

    state = StreamState()
    _dispatch(_line("thread.started", thread_id="parent-thread"), state)

    result = build_run_result(state, wall_time_seconds=1.0, user_prompt="x")

    assert result.trajectory[1].skill_invocations == []
    assert isinstance(result.trajectory[1].content[0], ToolCallBlock)
    assert result.trajectory[1].content[0].tool_use_id == "call_skill_script"
    assert result.trajectory[1].content[0].name == "command_execution"
    assert result.trajectory[1].content[0].status == "ok"


def test_build_run_result_dedups_skill_read_seen_via_stream_and_session(
    monkeypatch, tmp_path
):
    """The same SKILL.md read is visible both as a stream item (id
    "item_1") and as a persisted session function_call (id "call_skill") —
    two different id namespaces for the same underlying tool call. Must
    not be recorded as two separate invocations."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _write_codex_session(
        tmp_path,
        "parent-thread",
        [
            {
                "type": "event_msg",
                "timestamp": "2026-06-15T00:00:01.000Z",
                "payload": {"type": "user_message", "message": "use a skill"},
            },
            {
                "type": "response_item",
                "timestamp": "2026-06-15T00:00:02.000Z",
                "payload": {
                    "type": "function_call",
                    "call_id": "call_skill",
                    "name": "exec_command",
                    "arguments": json.dumps(
                        {"cmd": "sed -n '1,160p' .agents/skills/probe-skill/SKILL.md"}
                    ),
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-06-15T00:00:03.000Z",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call_skill",
                    "output": "# skill",
                },
            },
        ],
    )

    state = StreamState()
    _dispatch(_line("thread.started", thread_id="parent-thread"), state)
    _dispatch(
        _line(
            "item.completed",
            item={
                "id": "item_1",
                "type": "command_execution",
                "command": "sed -n '1,160p' .agents/skills/probe-skill/SKILL.md",
                "aggregated_output": "# skill",
                "exit_code": 0,
                "status": "completed",
            },
        ),
        state,
    )

    result = build_run_result(state, wall_time_seconds=1.0, user_prompt="x")

    assert len(result.trajectory[1].skill_invocations) == 1
    assert result.trajectory[1].skill_invocations[0].skill_name == "probe-skill"


def test_build_run_result_does_not_record_sibling_script_as_skill_invocation(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _write_codex_session(
        tmp_path,
        "parent-thread",
        [
            {
                "type": "event_msg",
                "timestamp": "2026-06-15T00:00:01.000Z",
                "payload": {"type": "user_message", "message": "use $scrape-define"},
            },
            {
                "type": "response_item",
                "timestamp": "2026-06-15T00:00:02.000Z",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "<skill>\n"
                                "<name>scrape-define</name>\n"
                                "<path>/repo/.agents/skills/scrape-define/SKILL.md</path>\n"
                                "# define\n"
                                "</skill>"
                            ),
                        }
                    ],
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-06-15T00:00:03.000Z",
                "payload": {
                    "type": "function_call",
                    "call_id": "call_sibling_script",
                    "name": "exec_command",
                    "arguments": json.dumps(
                        {
                            "cmd": (
                                "uv run "
                                "${CLAUDE_SKILL_DIR}/../scrape-explore-site/"
                                "scripts/download.py"
                            ),
                        }
                    ),
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-06-15T00:00:04.000Z",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call_sibling_script",
                    "output": "Process exited with code 0\nOutput:\nok\n",
                },
            },
        ],
    )

    state = StreamState()
    _dispatch(_line("thread.started", thread_id="parent-thread"), state)

    result = build_run_result(state, wall_time_seconds=1.0, user_prompt="x")

    invocations = result.trajectory[1].skill_invocations
    assert [inv.skill_name for inv in invocations] == ["scrape-define"]
    assert [inv.trigger_kind for inv in invocations] == ["explicit_skill"]
    assert isinstance(result.trajectory[1].content[0], ToolCallBlock)
    assert result.trajectory[1].content[0].tool_use_id == "call_sibling_script"


def test_build_run_result_preserves_codex_reasoning_summaries(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _write_codex_session(
        tmp_path,
        "parent-thread",
        [
            {
                "type": "event_msg",
                "timestamp": "2026-06-15T00:00:01.000Z",
                "payload": {"type": "user_message", "message": "think then answer"},
            },
            {
                "type": "response_item",
                "timestamp": "2026-06-15T00:00:02.000Z",
                "payload": {
                    "type": "reasoning",
                    "summary": [
                        {"type": "summary_text", "text": "Checked the fixture."},
                        {"summary": "Chose the direct answer."},
                    ],
                    "encrypted_content": "opaque",
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-06-15T00:00:03.000Z",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "done"}],
                },
            },
        ],
    )

    state = StreamState()
    _dispatch(_line("thread.started", thread_id="parent-thread"), state)

    result = build_run_result(state, wall_time_seconds=1.0, user_prompt="x")

    assert isinstance(result.trajectory[1].content[0], ThinkingBlock)
    assert result.trajectory[1].content[0].text == "Checked the fixture."
    assert isinstance(result.trajectory[1].content[1], ThinkingBlock)
    assert result.trajectory[1].content[1].text == "Chose the direct answer."
    assert isinstance(result.trajectory[1].content[2], TextBlock)
    assert result.trajectory[1].content[2].text == "done"


def test_build_run_result_detects_explicit_codex_skill_injection(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _write_codex_session(
        tmp_path,
        "parent-thread",
        [
            {
                "type": "event_msg",
                "timestamp": "2026-06-15T00:00:01.000Z",
                "payload": {
                    "type": "user_message",
                    "message": "Please use $probe-skill.",
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-06-15T00:00:02.000Z",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "<skill>\n"
                                "<name>probe-skill</name>\n"
                                "<path>/tmp/.agents/skills/probe-skill/SKILL.md</path>\n"
                                "# Probe skill\n"
                                "</skill>"
                            ),
                        }
                    ],
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-06-15T00:00:03.000Z",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "done"}],
                },
            },
        ],
    )

    state = StreamState()
    _dispatch(_line("thread.started", thread_id="parent-thread"), state)

    result = build_run_result(state, wall_time_seconds=1.0, user_prompt="x")

    assert result.trajectory[1].skill_invocations[0].skill_name == "probe-skill"
    assert result.trajectory[1].skill_invocations[0].trigger_kind == "explicit_skill"
    assert result.trajectory[1].skill_invocations[0].triggered_by_tool_use_id is None


def test_build_run_result_keeps_explicit_skill_when_killed_before_assistant_content(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _write_codex_session(
        tmp_path,
        "parent-thread",
        [
            {
                "type": "event_msg",
                "timestamp": "2026-06-15T00:00:01.000Z",
                "payload": {
                    "type": "user_message",
                    "message": "Please use $probe-skill.",
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-06-15T00:00:02.000Z",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "<skill>\n"
                                "<name>probe-skill</name>\n"
                                "<path>/tmp/.agents/skills/probe-skill/SKILL.md</path>\n"
                                "# Probe skill\n"
                                "</skill>"
                            ),
                        }
                    ],
                },
            },
        ],
    )

    state = StreamState()
    _dispatch(_line("thread.started", thread_id="parent-thread"), state)

    result = build_run_result(state, wall_time_seconds=1.0, user_prompt="x")

    assert len(result.trajectory) == 2
    assert result.trajectory[1].content == []
    assert result.trajectory[1].skill_invocations[0].skill_name == "probe-skill"
    assert result.trajectory[1].skill_invocations[0].trigger_kind == "explicit_skill"


def test_find_session_explicit_skill_invocation_uses_persisted_injection(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _write_codex_session(
        tmp_path,
        "parent-thread",
        [
            {
                "type": "response_item",
                "timestamp": "2026-06-15T00:00:02.000Z",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "<skill>\n"
                                "<name>probe-skill</name>\n"
                                "<path>/tmp/.agents/skills/probe-skill/SKILL.md</path>\n"
                                "# Probe skill\n"
                                "</skill>"
                            ),
                        }
                    ],
                },
            },
        ],
    )

    inv = find_session_explicit_skill_invocation("parent-thread")

    assert inv is not None
    assert inv.skill_name == "probe-skill"
    assert inv.trigger_kind == "explicit_skill"


def test_build_run_result_uses_codex_home_for_sessions(monkeypatch, tmp_path):
    codex_home = tmp_path / "custom-codex-home"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "regular-home")
    session_path = _write_codex_session_under(
        codex_home,
        "parent-thread",
        [
            {
                "type": "event_msg",
                "timestamp": "2026-06-15T00:00:01.000Z",
                "payload": {"type": "user_message", "message": "hello"},
            },
            {
                "type": "response_item",
                "timestamp": "2026-06-15T00:00:02.000Z",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "ok"}],
                },
            },
            {
                "type": "event_msg",
                "timestamp": "2026-06-15T00:00:03.000Z",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": {
                            "input_tokens": 3,
                            "cached_input_tokens": 1,
                            "output_tokens": 2,
                        }
                    },
                },
            },
        ],
    )

    state = StreamState()
    _dispatch(_line("thread.started", thread_id="parent-thread"), state)

    result = build_run_result(state, wall_time_seconds=1.0, user_prompt="x")

    assert result.trajectory[1].content[0].text == "ok"
    assert result.metrics.raw["codex_session_paths"] == [str(session_path)]


def test_build_run_result_prefers_actual_codex_session_model(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _write_codex_session(
        tmp_path,
        "parent-thread",
        [
            {
                "type": "turn_context",
                "timestamp": "2026-06-15T00:00:00.000Z",
                "payload": {"model": "gpt-real"},
            },
            {
                "type": "event_msg",
                "timestamp": "2026-06-15T00:00:01.000Z",
                "payload": {"type": "user_message", "message": "hello"},
            },
            {
                "type": "response_item",
                "timestamp": "2026-06-15T00:00:02.000Z",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "ok"}],
                },
            },
        ],
    )

    state = StreamState()
    _dispatch(_line("thread.started", thread_id="parent-thread"), state)

    result = build_run_result(
        state,
        wall_time_seconds=1.0,
        user_prompt="x",
        model="requested-model",
    )

    assert result.model == "gpt-real"


def test_build_run_result_uses_session_when_stream_omits_denied_tool(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    parent_path = _write_codex_session(
        tmp_path,
        "parent-thread",
        [
            {
                "type": "turn_context",
                "timestamp": "2026-06-15T00:00:00.000Z",
                "payload": {"model": "gpt-real"},
            },
            {
                "type": "event_msg",
                "timestamp": "2026-06-15T00:00:01.000Z",
                "payload": {"type": "task_started", "started_at": 100.0},
            },
            {
                "type": "event_msg",
                "timestamp": "2026-06-15T00:00:02.000Z",
                "payload": {"type": "user_message", "message": "write a file"},
            },
            {
                "type": "response_item",
                "timestamp": "2026-06-15T00:00:03.000Z",
                "payload": {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "exec_command",
                    "arguments": json.dumps(
                        {
                            "cmd": "printf denied > denied.txt",
                            "workdir": "/private/tmp/probe",
                        }
                    ),
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-06-15T00:00:04.000Z",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": (
                        "Process exited with code 1\n"
                        "Output:\nzsh:1: operation not permitted: denied.txt\n"
                    ),
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-06-15T00:00:05.000Z",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "command failed"}],
                },
            },
            {
                "type": "event_msg",
                "timestamp": "2026-06-15T00:00:06.000Z",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": {
                            "input_tokens": 7,
                            "cached_input_tokens": 3,
                            "output_tokens": 2,
                            "reasoning_output_tokens": 1,
                        }
                    },
                },
            },
        ],
    )

    state = StreamState()
    _dispatch(_line("thread.started", thread_id="parent-thread"), state)

    result = build_run_result(state, wall_time_seconds=1.0, user_prompt="x")

    assert result.model == "gpt-real"
    assert len(result.trajectory) == 2
    assert result.trajectory[0].content[0].text == "write a file"
    tool = result.trajectory[1].content[0]
    assert isinstance(tool, ToolCallBlock)
    assert tool.name == "command_execution"
    assert tool.input["command"] == "printf denied > denied.txt"
    assert tool.status == "permission_denied"
    assert tool.duration_ms == 1000
    assert result.metrics.n_tool_calls == 1
    assert result.metrics.n_permission_denied == 1
    assert result.metrics.tokens.input == 7
    assert result.metrics.tokens.output == 2
    assert result.metrics.tokens.cache_read == 3
    assert result.metrics.raw["codex_session_paths"] == [str(parent_path)]


def test_build_run_result_counts_codex_escalation_rejection_as_permission_denied(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _write_codex_session(
        tmp_path,
        "parent-thread",
        [
            {
                "type": "event_msg",
                "timestamp": "2026-06-15T00:00:01.000Z",
                "payload": {"type": "user_message", "message": "write a file"},
            },
            {
                "type": "response_item",
                "timestamp": "2026-06-15T00:00:02.000Z",
                "payload": {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "exec_command",
                    "arguments": json.dumps(
                        {
                            "cmd": "touch denied.txt",
                            "sandbox_permissions": "require_escalated",
                        }
                    ),
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-06-15T00:00:03.000Z",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": (
                        "approval policy is Never; reject command - you cannot ask "
                        "for escalated permissions if the approval policy is Never"
                    ),
                },
            },
        ],
    )

    state = StreamState()
    _dispatch(_line("thread.started", thread_id="parent-thread"), state)

    result = build_run_result(state, wall_time_seconds=1.0, user_prompt="x")

    tool = result.trajectory[1].content[0]
    assert isinstance(tool, ToolCallBlock)
    assert tool.status == "permission_denied"
    assert result.metrics.n_permission_denied == 1


def test_build_run_result_counts_codex_network_denial_as_permission_denied(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _write_codex_session(
        tmp_path,
        "parent-thread",
        [
            {
                "type": "event_msg",
                "timestamp": "2026-06-15T00:00:01.000Z",
                "payload": {"type": "user_message", "message": "fetch a URL"},
            },
            {
                "type": "response_item",
                "timestamp": "2026-06-15T00:00:02.000Z",
                "payload": {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "exec_command",
                    "arguments": json.dumps({"cmd": "curl https://example.com"}),
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-06-15T00:00:03.000Z",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "Network access was denied by the Codex sandbox network proxy.",
                },
            },
        ],
    )

    state = StreamState()
    _dispatch(_line("thread.started", thread_id="parent-thread"), state)

    result = build_run_result(state, wall_time_seconds=1.0, user_prompt="x")

    tool = result.trajectory[1].content[0]
    assert isinstance(tool, ToolCallBlock)
    assert tool.status == "permission_denied"
    assert result.metrics.n_permission_denied == 1


def test_successful_command_mentioning_permission_denied_is_not_misclassified(
    monkeypatch, tmp_path
):
    """A command that exits 0 but whose own output happens to contain the
    words "permission denied" (e.g. grep matching log lines) must not be
    misclassified — the exit-code header is authoritative."""
    content, metrics = _single_tool_call(
        monkeypatch,
        tmp_path,
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "exec_command",
            "arguments": json.dumps({"cmd": "grep -r denied logs/"}),
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": (
                "Process exited with code 0\n"
                "Output:\nlogs/app.log:1: permission denied for user bob\n"
            ),
        },
    )
    tool = content[0]
    assert tool.status == "ok"
    assert metrics.n_permission_denied == 0


def test_failing_command_with_nested_success_text_is_not_masked(monkeypatch, tmp_path):
    """A failing command whose captured output happens to contain
    "exited with code 0" (e.g. from a supervised child process) must not
    be masked as ok by a naive whole-blob substring search."""
    content, metrics = _single_tool_call(
        monkeypatch,
        tmp_path,
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "exec_command",
            "arguments": json.dumps({"cmd": "./run_spider.sh"}),
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": (
                "Process exited with code 1\n"
                "Output:\nchild process exited with code 0\nfatal: crawl failed\n"
            ),
        },
    )
    tool = content[0]
    assert tool.status == "error"
    assert metrics.n_tool_errors == 1


def _single_tool_call(monkeypatch, tmp_path, *items):
    """Write a session whose assistant turn contains the given response_item
    payloads, run the parser, and return that turn's content blocks."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    events = [
        {
            "type": "event_msg",
            "timestamp": "2026-06-15T00:00:01.000Z",
            "payload": {"type": "user_message", "message": "go"},
        }
    ]
    for i, payload in enumerate(items):
        events.append(
            {
                "type": "response_item",
                "timestamp": f"2026-06-15T00:00:0{2 + i}.000Z",
                "payload": payload,
            }
        )
    _write_codex_session(tmp_path, "parent-thread", events)
    state = StreamState()
    _dispatch(_line("thread.started", thread_id="parent-thread"), state)
    result = build_run_result(state, wall_time_seconds=1.0, user_prompt="go")
    return result.trajectory[1].content, result.metrics


def test_web_search_call_becomes_tool_call(monkeypatch, tmp_path):
    content, metrics = _single_tool_call(
        monkeypatch,
        tmp_path,
        {
            "type": "web_search_call",
            "id": "ws_1",
            "status": "completed",
            "action": {"type": "search", "query": "weather in SF"},
        },
    )
    tool = content[0]
    assert isinstance(tool, ToolCallBlock)
    assert tool.name == "web_search"
    assert tool.tool_use_id == "ws_1"
    assert tool.input == {"action_type": "search", "query": "weather in SF"}
    assert tool.status == "ok"
    assert metrics.n_tool_calls == 1


def test_web_search_call_in_progress_is_aborted(monkeypatch, tmp_path):
    content, _ = _single_tool_call(
        monkeypatch,
        tmp_path,
        {
            "type": "web_search_call",
            "id": "ws_2",
            "status": "in_progress",
            "action": {"type": "open_page", "url": "https://example.com"},
        },
    )
    tool = content[0]
    assert tool.name == "web_search"
    assert tool.input == {"action_type": "open_page", "url": "https://example.com"}
    assert tool.status == "aborted"


def test_local_shell_call_normalizes_to_command_execution(monkeypatch, tmp_path):
    content, metrics = _single_tool_call(
        monkeypatch,
        tmp_path,
        {
            "type": "local_shell_call",
            "call_id": "ls_1",
            "status": "completed",
            "action": {"type": "exec", "command": ["bash", "-lc", "ls -la"]},
        },
        {
            "type": "function_call_output",
            "call_id": "ls_1",
            "output": "Process exited with code 0\nOutput:\nok\n",
        },
    )
    tool = content[0]
    assert isinstance(tool, ToolCallBlock)
    assert tool.name == "command_execution"
    assert tool.tool_use_id == "ls_1"
    assert tool.input == {"command": "bash -lc ls -la"}
    assert tool.status == "ok"
    assert metrics.n_tool_calls == 1


def test_custom_tool_call_keeps_name_and_completes(monkeypatch, tmp_path):
    content, _ = _single_tool_call(
        monkeypatch,
        tmp_path,
        {
            "type": "custom_tool_call",
            "call_id": "ct_1",
            "name": "my_custom_tool",
            "input": json.dumps({"x": 1}),
        },
        {
            "type": "custom_tool_call_output",
            "call_id": "ct_1",
            "output": "done",
        },
    )
    tool = content[0]
    assert isinstance(tool, ToolCallBlock)
    assert tool.name == "my_custom_tool"
    assert tool.input == {"x": 1}
    assert tool.result == "done"
    assert tool.status == "ok"


def test_image_generation_call_elides_base64(monkeypatch, tmp_path):
    content, _ = _single_tool_call(
        monkeypatch,
        tmp_path,
        {
            "type": "image_generation_call",
            "id": "ig_1",
            "status": "completed",
            "revised_prompt": "a gray tabby cat",
            "result": "iVBORw0KGgoAAAANSUhEUgAA" * 1000,
        },
    )
    tool = content[0]
    assert tool.name == "image_generation"
    assert tool.result == "a gray tabby cat"
    assert "iVBOR" not in tool.result
    assert tool.status == "ok"


def test_stream_error_messages_collects_dedupes_and_unwraps():
    blob = (
        '{"type":"error","status":400,"error":'
        '{"type":"invalid_request_error","message":"model not supported"}}'
    )
    events = [
        {"type": "thread.started", "thread_id": "t1"},
        {
            "type": "item.completed",
            "item": {"type": "error", "message": "metadata fallback warning"},
        },
        {"type": "error", "message": "You have hit your usage limit."},
        {"type": "error", "message": blob},
        {"type": "turn.failed", "error": {"message": blob}},  # dupe → dropped
        {"type": "error", "message": ""},  # empty → skipped
        {"type": "turn.failed", "error": "not-a-dict"},  # malformed → skipped
        {"type": "item.completed", "item": {"type": "agent_message"}},
    ]
    assert stream_error_messages(events) == [
        "metadata fallback warning",
        "You have hit your usage limit.",
        "model not supported",
    ]
