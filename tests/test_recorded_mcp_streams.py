"""Tool-trigger wiring against streams recorded from real harness runs.

Each file under `data/` is the stream of an attempt whose task targeted the
`search` tool of an MCP server named `files`, cut short the moment the
harness announced that call. They are the runs verbatim, minus Copilot's
ephemeral message deltas, the developer skills its session events list, and
the opaque reasoning blobs it round-trips; the Codex rollout is the two
response items its trajectory is built from.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent_exam.providers.copilot_cli.provider import CopilotCliProvider
from agent_exam.providers.dummy import DummyProvider
from agent_exam.providers.opencode.stream_parser import StreamState, _dispatch
from agent_exam.trajectory_walk import iter_tool_calls

_TARGET = "mcp__files__search"
_DATA = Path(__file__).parent / "data"


def _lines(name: str) -> list[str]:
    return (_DATA / name).read_text().splitlines()


def _names(trajectory) -> list[str]:
    return [call.name for call in iter_tool_calls(trajectory)]


def test_codex_cuts_on_the_announced_call():
    from agent_exam.providers.codex_cli.stream_parser import StreamState, _dispatch

    state = StreamState(skill_detection_enabled=True, target_tool=_TARGET)

    for line in _lines("codex_mcp_stream.jsonl"):
        _dispatch(line, state)

    assert state.detected_tool == _TARGET
    assert state.kill_signal.is_set()


def test_codex_names_an_mcp_call_after_its_server():
    """Codex keeps the server in a `namespace` field of its own, so without
    joining the two the call grades as a bare `search`."""
    from agent_exam.providers.codex_cli.transcripts import _session_trajectory

    events = [json.loads(line) for line in _lines("codex_mcp_rollout.jsonl")]

    assert _names(_session_trajectory(events)) == [_TARGET]


def test_copilot_cuts_on_the_requested_call_and_keeps_it():
    from agent_exam.providers.copilot_cli.stream_parser import StreamState, _dispatch
    from agent_exam.providers.copilot_cli.transcripts import build_run_result

    state = StreamState(provider=CopilotCliProvider())
    state.skill_detection_enabled = True
    state.target_tool = _TARGET

    for line in _lines("copilot_mcp_stream.jsonl"):
        _dispatch(line, state)

    assert state.kill_signal.is_set()
    result = build_run_result(state, wall_time_seconds=0.0)
    assert _names(result.trajectory) == [_TARGET]


def test_opencode_cuts_on_the_tool_part():
    state = StreamState(provider=DummyProvider())
    state.skill_detection_enabled = True
    state.target_tool = _TARGET
    state.mcp_server_names = ("files",)

    for line in _lines("opencode_mcp_stream.jsonl"):
        _dispatch(line, state)

    assert state.detected_tool == "files_search"
    assert state.kill_signal.is_set()
