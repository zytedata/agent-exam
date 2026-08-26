"""Connection status of the attached MCP servers, read off whatever each
harness says at session start. A server that dies on startup leaves the agent
without its tools and says nothing else about it.
"""

from __future__ import annotations

import io
import json

from agent_exam.mcp import connection_check
from agent_exam.providers.claude_code.stream_parser import StreamState, drain_stream

_INIT = {
    "type": "system",
    "subtype": "init",
    "session_id": "abc",
    "mcp_servers": [
        {"name": "files", "status": "connected"},
        {"name": "remote", "status": "failed"},
    ],
}


def _drain(events: list[dict]) -> StreamState:
    state = StreamState()
    payload = "".join(json.dumps(e) + "\n" for e in events).encode()
    drain_stream(io.BytesIO(payload), state)
    return state


def test_stream_records_server_statuses():
    state = _drain([_INIT, {"type": "result", "total_cost_usd": 0.1}])

    assert state.mcp_server_status == {"files": "connected", "remote": "failed"}
    # The init event still seeds the session id, and the result event is
    # still parsed after it.
    assert state.session_id == "abc"
    assert state.total_cost_usd == 0.1


def test_stream_records_nothing_without_servers():
    state = _drain([{"type": "system", "subtype": "init", "mcp_servers": []}])

    assert state.mcp_server_status == {}


_COPILOT_INIT = {
    "type": "session.mcp_servers_loaded",
    "data": {
        "servers": [
            {
                "name": "remote",
                "status": "failed",
                "error": "failed to spawn MCP server process: No such file or directory",
                "transport": "stdio",
            },
            {"name": "files", "status": "connected", "transport": "stdio"},
        ]
    },
    "ephemeral": True,
}


def test_copilot_stream_records_server_statuses():
    from agent_exam.providers.copilot_cli.provider import CopilotCliProvider
    from agent_exam.providers.copilot_cli.stream_parser import (
        StreamState as CopilotState,
    )
    from agent_exam.providers.copilot_cli.stream_parser import _dispatch
    from agent_exam.providers.copilot_cli.transcripts import build_run_result

    state = CopilotState(provider=CopilotCliProvider())
    _dispatch(json.dumps(_COPILOT_INIT), state)

    assert state.mcp_server_status == {"files": "connected", "remote": "failed"}
    result = build_run_result(state, wall_time_seconds=0.0)
    assert result.mcp_server_status == {"files": "connected", "remote": "failed"}


# Recorded from `opencode run --print-logs --log-level INFO` over four
# servers: one of each transport that came up, and one of each that did not.
_OPENCODE_LOGS = """\
INFO  2026-08-19T08:57:53 +5ms service=mcp key=files type=local found
INFO  2026-08-19T08:57:53 +5ms service=mcp key=dead type=local found
INFO  2026-08-19T09:05:57 +6ms service=mcp key=remote type=remote found
INFO  2026-08-19T09:05:57 +18ms service=mcp key=unreachable type=remote found
ERROR 2026-08-19T08:57:53 +5ms service=mcp key=dead command=["nope"] \
error=Executable not found in $PATH: "nope" local mcp startup failed
INFO  2026-08-19T08:57:55 +1652ms service=mcp key=files mcp stderr: Starting \
default (STDIO) server...
INFO  2026-08-19T09:05:58 +387ms service=mcp key=remote \
transport=StreamableHTTP connected
INFO  2026-08-19T08:57:55 +74ms service=mcp key=files toolCount=13 create() \
successfully created client
INFO  2026-08-19T09:05:58 +129ms service=mcp key=remote toolCount=13 create() \
successfully created client
INFO  2026-08-19T08:59:41 +316ms service=file init
"""


def _opencode_drain(logs: str):
    from agent_exam.providers.dummy import DummyProvider
    from agent_exam.providers.opencode.stream_parser import StreamState, drain_stderr

    state = StreamState(provider=DummyProvider())
    drain_stderr(io.BytesIO(logs.encode()), state)
    return state


def test_opencode_logs_record_server_statuses():
    """A local server that fails says so; a remote one that fails is never
    mentioned again, so `found` has to start every server off as failed."""
    state = _opencode_drain(_OPENCODE_LOGS)

    assert state.mcp_server_status == {
        "files": "connected",
        "remote": "connected",
        "dead": "failed",
        "unreachable": "failed",
    }


def test_opencode_keeps_routine_logging_out_of_the_stderr_tail():
    state = _opencode_drain(_OPENCODE_LOGS + "opencode: something broke\n")
    tail = bytes(state.stderr_tail).decode()

    assert "something broke" in tail
    assert "local mcp startup failed" in tail
    assert "service=file init" not in tail


def test_opencode_records_nothing_without_mcp_logs():
    """`--print-logs` is only passed when servers are attached, so a run
    without them says nothing either way."""
    assert _opencode_drain("INFO  service=file init\n").mcp_server_status is None


def test_opencode_connected_status_does_not_regress_on_a_stray_found_line():
    """A repeat `found` line for an already-connected server (e.g. a second
    config read racing the first) must not flip it back to failed."""
    logs = (
        "INFO  2026-08-19T08:57:53 +5ms service=mcp key=files type=local found\n"
        "INFO  2026-08-19T08:57:55 +74ms service=mcp key=files toolCount=13 "
        "create() successfully created client\n"
        "INFO  2026-08-19T08:57:56 +1ms service=mcp key=files type=local found\n"
    )
    state = _opencode_drain(logs)

    assert state.mcp_server_status == {"files": "connected"}


def test_codex_drops_the_path_alias_warning_from_the_stderr_tail():
    """A staged CODEX_HOME is always a temp dir, so codex warns on every run
    and would otherwise be the whole tail of an unrelated failure."""
    from agent_exam.providers.codex_cli.stream_parser import strip_path_alias_warning

    text = (
        "WARNING: proceeding, even though we could not create PATH "
        "aliases: Refusing to create helper binaries under temporary "
        'dir "/tmp" (codex_home: AbsolutePathBuf("/tmp/x"))\n'
        "codex: something broke\n"
    )

    assert strip_path_alias_warning(text) == "codex: something broke"


def test_opencode_stderr_tail_captures_bytes_before_a_trailing_newline_arrives():
    """A hung child that writes a diagnostic with no newline yet must not
    leave the tail waiting on one that may never come."""
    import os
    import threading
    import time

    from agent_exam.providers.dummy import DummyProvider
    from agent_exam.providers.opencode.stream_parser import StreamState, drain_stderr

    read_fd, write_fd = os.pipe()
    state = StreamState(provider=DummyProvider())
    thread = threading.Thread(
        target=drain_stderr, args=(os.fdopen(read_fd, "rb"), state)
    )
    thread.start()
    try:
        os.write(write_fd, b"opencode: stalled waiting on a dead mcp server")
        deadline = time.monotonic() + 2.0
        while not state.stderr_tail and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (
            bytes(state.stderr_tail)
            == b"opencode: stalled waiting on a dead mcp server"
        )
    finally:
        os.close(write_fd)
        thread.join(timeout=2.0)


def test_opencode_stderr_tail_withholds_a_partial_routine_prefix():
    """A partial line that could still turn out to be `INFO `/`DEBUG ` is not
    flushed until enough of it has arrived to rule that out."""
    import os
    import threading
    import time

    from agent_exam.providers.dummy import DummyProvider
    from agent_exam.providers.opencode.stream_parser import StreamState, drain_stderr

    read_fd, write_fd = os.pipe()
    state = StreamState(provider=DummyProvider())
    thread = threading.Thread(
        target=drain_stderr, args=(os.fdopen(read_fd, "rb"), state)
    )
    thread.start()
    try:
        os.write(write_fd, b"INFO")
        time.sleep(0.1)
        assert not state.stderr_tail
    finally:
        os.close(write_fd)
        thread.join(timeout=2.0)


def test_codex_stderr_tail_captures_bytes_before_a_trailing_newline_arrives():
    """A hung child that writes a diagnostic with no newline yet must not
    leave the tail waiting on one that may never come."""
    import os
    import threading
    import time

    from agent_exam.providers.codex_cli.stream_parser import StreamState, drain_stderr

    read_fd, write_fd = os.pipe()
    state = StreamState()
    thread = threading.Thread(
        target=drain_stderr, args=(os.fdopen(read_fd, "rb"), state)
    )
    thread.start()
    try:
        os.write(write_fd, b"codex: stalled waiting on a dead mcp server")
        deadline = time.monotonic() + 2.0
        while not state.stderr_tail and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (
            bytes(state.stderr_tail) == b"codex: stalled waiting on a dead mcp server"
        )
    finally:
        os.close(write_fd)
        thread.join(timeout=2.0)


def test_check_fails_on_a_server_that_did_not_connect():
    result = connection_check({"files": "connected", "remote": "failed"})

    assert result.status == "FAIL"
    assert "remote (failed)" in result.hint
    assert "files" not in result.hint


def test_check_passes_when_every_server_connected():
    assert connection_check({"files": "connected"}).status == "OK"


def test_check_fails_when_an_expected_server_is_missing():
    """The config never reached the CLI, so the agent has no tools at all."""
    result = connection_check({}, ["files"])

    assert result.status == "FAIL"
    assert "files (not attached)" in result.hint


def test_check_passes_with_nothing_attached():
    assert connection_check(None, ["files"]).status == "OK"
    assert connection_check({}).status == "OK"


_CONFIG = """\
default_harness: dummy
mcp_servers:
  files:
    command: sh
providers:
  dummy:
    judge_model: haiku
"""


def _run_with_statuses(root, monkeypatch, statuses):
    """Run one dummy attempt whose harness announced *statuses*."""
    from agent_exam.config import load_config
    from agent_exam.providers.dummy import DummyProvider
    from agent_exam.runner import RunRequest, run

    (root / "evals" / "suites" / "s" / "tasks").mkdir(parents=True)
    (root / "skills" / "skill-a").mkdir(parents=True)
    (root / "skills" / "skill-a" / "SKILL.md").write_text("# skill-a")
    (root / "pyproject.toml").write_text('[tool.agent-exam]\nevals_dir = "evals"\n')
    (root / "evals" / "config.yaml").write_text(_CONFIG)
    (root / "evals" / "suites" / "s" / "tasks" / "t.yaml").write_text(
        "kind: execute\nprompt: x\nassertions: []\n"
    )

    invoke = DummyProvider.invoke

    def with_statuses(self, *args, **kwargs):
        result = invoke(self, *args, **kwargs)
        result.mcp_server_status = statuses
        return result

    monkeypatch.setattr(DummyProvider, "invoke", with_statuses)

    exit_code = run(
        load_config(root),
        RunRequest(
            specs=[("s", None)],
            provider="dummy",
            model="",
            k=1,
            n_parallel=1,
            without_skill=False,
        ),
    )
    run_dir = next(iter((root / "evals" / "runs").iterdir()))
    report = json.loads(next(iter((run_dir / "reports").iterdir())).read_text())
    attempt = json.loads(
        (run_dir / "artifacts" / "s" / "t" / "attempt-1" / "attempt.json").read_text()
    )
    return exit_code, report, attempt


def test_attempt_errors_when_a_server_did_not_connect(tmp_path, monkeypatch):
    """Without this the agent just lacks its tools, and the task fails as if
    the skill had routed wrong."""
    exit_code, report, attempt = _run_with_statuses(
        tmp_path / "proj", monkeypatch, {"files": "failed"}
    )

    assert exit_code != 0
    assert [a["verdict"] for a in report["attempts"]] == ["error"]
    assert attempt["mcp_server_status"] == {"files": "failed"}


def test_attempt_is_graded_when_every_server_connected(tmp_path, monkeypatch):
    exit_code, report, attempt = _run_with_statuses(
        tmp_path / "proj", monkeypatch, {"files": "connected"}
    )

    assert exit_code == 0
    assert [a["verdict"] for a in report["attempts"]] == ["pass"]
    assert attempt["mcp_server_status"] == {"files": "connected"}
