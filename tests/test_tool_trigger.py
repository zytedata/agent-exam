"""`kind: trigger` aimed at a tool — typically an MCP one — instead of a skill."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest

from agent_exam.assertions.first_mcp_tool import FirstMcpToolConfig
from agent_exam.assertions.first_mcp_tool import check as first_mcp_tool
from agent_exam.config import DEFAULT_TASK_TIMEOUT, load_config
from agent_exam.errors import UsageError
from agent_exam.providers.claude_code.stream_parser import StreamState, _dispatch
from agent_exam.providers.dummy import DummyProvider
from agent_exam.runner import RunRequest, run
from agent_exam.schemas import Metrics, RunResult, TextBlock, Tokens, Turn
from agent_exam.tasks import load_task
from agent_exam.trajectory_walk import record_detected_tool

_TARGET = "mcp__files__search"
_CONFIG = FirstMcpToolConfig(server="files", tool="search")


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "trigger.yaml"
    path.write_text(dedent(body))
    return path


def test_cases_grade_on_the_tool_instead_of_the_skill(tmp_path):
    path = _write(
        tmp_path,
        """
        kind: trigger
        mcp_tool: {server: files, tool: search}
        positive: [Find the invoice for March.]
        negative: ["What is Python?"]
        """,
    )

    positive, negative = load_task(path, suite="s")
    assert [t.target_tool for t in (positive, negative)] == [_TARGET, _TARGET]
    assert positive.target_skill is None
    assert positive.assertions[0].type == "first_mcp_tool"
    assert positive.assertions[0].config == {"server": "files", "tool": "search"}
    assert negative.assertions[0].type == "mcp_tool_not_called"


def test_a_bare_tool_name_targets_it_on_any_server(tmp_path):
    path = _write(tmp_path, "kind: trigger\nmcp_tool: search\npositive: [hi]\n")

    (positive,) = load_task(path, suite="s")
    assert positive.target_tool == "mcp__*__search"
    assert positive.assertions[0].config == "search"
    result = RunResult(
        trajectory=[Turn(role="assistant", content=[TextBlock(text="on it")])],
        metrics=Metrics(0.0, Tokens(), 0.0, 0, 0),
    )
    record_detected_tool(result.trajectory, "mcp__notes__search")
    assert first_mcp_tool(positive.assertions[0].parsed_config, result, Path()).pass_


def test_a_trigger_targets_a_skill_or_a_tool_but_not_both(tmp_path):
    path = _write(
        tmp_path,
        """
        kind: trigger
        skill: files
        mcp_tool: search
        positive: [hi]
        """,
    )

    with pytest.raises(UsageError, match="exactly one of"):
        load_task(path, suite="s")


def _stream(state: StreamState, *tool_names: str, message_stop: bool = False) -> None:
    for name in tool_names:
        _dispatch(
            json.dumps(
                {
                    "type": "stream_event",
                    "event": {
                        "type": "content_block_start",
                        "content_block": {"type": "tool_use", "id": "x", "name": name},
                    },
                }
            ),
            state,
        )
    if message_stop:
        _dispatch(
            json.dumps({"type": "stream_event", "event": {"type": "message_stop"}}),
            state,
        )


def test_the_run_is_cut_on_the_target_tool():
    state = StreamState(skill_detection_enabled=True, target_tool=_TARGET)

    _stream(state, "Read", _TARGET)

    assert state.detected_tool == _TARGET
    assert state.kill_signal.is_set()


def test_a_positive_case_is_cut_on_another_servers_tool():
    """Which tool the agent picks is the whole question, so the first MCP call
    answers it whichever tool it is — the `first_skill` economics, for tools."""
    state = StreamState(skill_detection_enabled=True, target_tool=_TARGET)

    _stream(state, "Read", "mcp__notes__search")

    assert state.detected_tool == "mcp__notes__search"
    assert state.kill_signal.is_set()


def test_a_wrong_tool_before_the_target_is_a_routing_miss():
    """The kill can leave both calls in the transcript — Copilot announces
    every tool of a turn in one message — so the grading is what decides."""
    trajectory = [Turn(role="assistant", content=[TextBlock(text="on it")])]
    record_detected_tool(trajectory, "mcp__notes__search")
    record_detected_tool(trajectory, _TARGET)

    result = RunResult(trajectory=trajectory, metrics=Metrics(0.0, Tokens(), 0.0, 0, 0))
    assert not first_mcp_tool(_CONFIG, result, Path()).pass_


def test_native_calls_before_the_target_are_not_a_miss():
    trajectory = [Turn(role="assistant", content=[TextBlock(text="on it")])]
    record_detected_tool(trajectory, "Read")
    record_detected_tool(trajectory, _TARGET)

    result = RunResult(trajectory=trajectory, metrics=Metrics(0.0, Tokens(), 0.0, 0, 0))
    assert first_mcp_tool(_CONFIG, result, Path()).pass_


def test_a_positive_case_keeps_going_while_the_agent_looks_around():
    state = StreamState(skill_detection_enabled=True, target_tool=_TARGET)

    _stream(state, "Read", "Bash", message_stop=True)

    assert state.detected_tool is None
    assert not state.kill_signal.is_set()


def test_a_negative_case_also_keeps_going_while_the_agent_looks_around():
    """The eager negative kill is for skill targets, where the skill fires
    first or not at all. An agent reaches for a tool after grepping and
    reading, so cutting on the first tool call — or on the message_stop that
    ends the message announcing it — would settle every negative case before
    the routing decision is observable."""
    state = StreamState(
        skill_detection_enabled=True, target_tool=_TARGET, negative_trigger_mode=True
    )

    _stream(state, "Bash", message_stop=True)

    assert not state.kill_signal.is_set()


def test_the_call_the_kill_landed_on_is_recorded():
    """The kill precedes the tool_use reaching the transcript, so without
    this the generated assertion has nothing to grade."""
    trajectory = [Turn(role="assistant", content=[TextBlock(text="on it")])]

    record_detected_tool(trajectory, _TARGET)

    result = RunResult(trajectory=trajectory, metrics=Metrics(0.0, Tokens(), 0.0, 0, 0))
    assert first_mcp_tool(_CONFIG, result, Path()).pass_
    # A second pass over a transcript that already has the call adds nothing.
    record_detected_tool(trajectory, _TARGET)
    assert len(trajectory[0].content) == 2


def _run_a_tool_trigger(tmp_path, monkeypatch) -> list[dict]:
    """Run one positive tool-trigger case on the dummy provider, returning
    the options each `invoke` was called with."""
    root = tmp_path / "proj"
    (root / "evals" / "suites" / "s" / "tasks").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[tool.agent-exam]\nevals_dir = "evals"\n')
    (root / "evals" / "config.yaml").write_text(
        "default_harness: dummy\nskills_dirs: []\n"
        "mcp_servers:\n  files:\n    command: sh\n"
        "providers:\n  dummy:\n    judge_model: haiku\n"
    )
    (root / "evals" / "suites" / "s" / "tasks" / "t.yaml").write_text(
        "kind: trigger\nmcp_tool: {server: files, tool: search}\npositive: [hi]\n"
    )

    calls: list[dict] = []
    original = DummyProvider.invoke

    def spy(self, *args, **kwargs):
        calls.append(kwargs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr("agent_exam.providers.dummy.DummyProvider.invoke", spy)

    run(
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
    return calls


def test_a_tool_case_gets_the_whole_task_budget(tmp_path, monkeypatch):
    """The 60-second trigger default assumes a skill fires immediately. Here
    the agent looks around first, and a booting stdio server eats into it."""
    calls = _run_a_tool_trigger(tmp_path, monkeypatch)

    assert [c["timeout_seconds"] for c in calls] == [DEFAULT_TASK_TIMEOUT]


def test_a_tool_of_an_undeclared_server_fails_validation(tmp_path):
    """Every positive case of such a trigger would fail as a routing miss."""
    from agent_exam.validation import validate_suite

    root = tmp_path / "proj"
    (root / "evals" / "suites" / "s" / "tasks").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[tool.agent-exam]\nevals_dir = "evals"\n')
    (root / "evals" / "config.yaml").write_text(
        "default_harness: dummy\nskills_dirs: []\n"
        "mcp_servers:\n  files:\n    command: sh\n"
    )
    (root / "evals" / "suites" / "s" / "tasks" / "typo.yaml").write_text(
        "kind: trigger\nmcp_tool: {server: flies, tool: search}\npositive: [hi]\n"
    )
    (root / "evals" / "suites" / "s" / "tasks" / "anywhere.yaml").write_text(
        "kind: trigger\nmcp_tool: search\npositive: [hi]\n"
    )

    fails = [c for c in validate_suite(load_config(root), "s") if c.status == "FAIL"]

    assert [c.hint for c in fails] == [
        "no attached mcp_servers entry serves: mcp__flies__search"
    ]


def test_a_tool_of_a_server_the_task_leaves_out_fails_validation(tmp_path):
    """Declared is not enough — the task has to attach the server too."""
    from agent_exam.validation import validate_suite

    root = tmp_path / "proj"
    (root / "evals" / "suites" / "s" / "tasks").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[tool.agent-exam]\nevals_dir = "evals"\n')
    (root / "evals" / "config.yaml").write_text(
        "default_harness: dummy\nskills_dirs: []\n"
        "mcp_servers:\n  files:\n    command: sh\n  tickets:\n    command: sh\n"
    )
    (root / "evals" / "suites" / "s" / "tasks" / "elsewhere.yaml").write_text(
        "kind: trigger\nmcp_tool: {server: files, tool: search}\n"
        "mcp_servers: [tickets]\npositive: [hi]\n"
    )
    (root / "evals" / "suites" / "s" / "tasks" / "attached.yaml").write_text(
        "kind: trigger\nmcp_tool: {server: files, tool: search}\n"
        "mcp_servers: [files]\npositive: [hi]\n"
    )

    fails = [c for c in validate_suite(load_config(root), "s") if c.status == "FAIL"]

    assert [c.hint for c in fails] == [
        "no attached mcp_servers entry serves: mcp__files__search"
    ]


def _mcp_project(tmp_path: Path, *tasks: tuple[str, str], servers=("files",)) -> Path:
    root = tmp_path / "proj"
    (root / "evals" / "suites" / "s" / "tasks").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[tool.agent-exam]\nevals_dir = "evals"\n')
    (root / "evals" / "config.yaml").write_text(
        "default_harness: dummy\nskills_dirs: []\nmcp_servers:\n"
        + "".join(f"  {name}:\n    command: sh\n" for name in servers)
    )
    for name, body in tasks:
        (root / "evals" / "suites" / "s" / "tasks" / name).write_text(dedent(body))
    return root


def _dummy_request() -> RunRequest:
    return RunRequest(
        specs=[("s", None)],
        provider="dummy",
        model="",
        k=1,
        n_parallel=1,
        without_skill=False,
    )


def _trigger_checks(root: Path, inventory) -> list:
    from agent_exam.tasks import load_suite
    from agent_exam.validation import check_trigger_tools

    cfg = load_config(root)
    return check_trigger_tools("s", cfg, load_suite(cfg.evals_dir, "s"), inventory)


def test_a_tool_named_after_its_server_is_not_a_misspelling(tmp_path):
    """`files_search` is what a call to `search` on `files` looks like on some
    harnesses, and also a perfectly good name for a tool: only the server
    can tell, so static validation no longer guesses."""
    from agent_exam.mcp import ToolInventory
    from agent_exam.validation import validate_suite

    root = _mcp_project(
        tmp_path, ("t.yaml", "kind: trigger\nmcp_tool: files_search\npositive: [hi]\n")
    )

    assert not [c for c in validate_suite(load_config(root), "s") if c.status == "FAIL"]
    inventory = ToolInventory(tools={"files": frozenset({"files_search", "read"})})
    [check] = _trigger_checks(root, inventory)
    assert (check.status, check.hint) == (
        "OK",
        "1 target(s) found on the attached servers",
    )


def test_a_bare_target_no_attached_server_serves_fails_with_the_likely_fix(tmp_path):
    from agent_exam.mcp import ToolInventory

    root = _mcp_project(
        tmp_path,
        ("folded.yaml", "kind: trigger\nmcp_tool: files_search\npositive: [hi]\n"),
        ("typo.yaml", "kind: trigger\nmcp_tool: serach\npositive: [hi]\n"),
        ("fine.yaml", "kind: trigger\nmcp_tool: read\nnegative: [hi]\n"),
    )
    inventory = ToolInventory(tools={"files": frozenset({"search", "read"})})

    [check] = _trigger_checks(root, inventory)

    assert check.status == "FAIL"
    assert check.hint.split("\n  ") == [
        (
            "no attached server has a tool named 'files_search'; "
            "did you mean {server: files, tool: search}?"
        ),
        "no attached server has a tool named 'serach'; did you mean search?",
    ]


def test_a_pinned_target_its_server_does_not_serve_fails(tmp_path):
    from agent_exam.mcp import ToolInventory

    root = _mcp_project(
        tmp_path,
        (
            "t.yaml",
            "kind: trigger\nmcp_tool: {server: files, tool: serach}\npositive: [hi]\n",
        ),
    )
    inventory = ToolInventory(tools={"files": frozenset({"search"})})

    [check] = _trigger_checks(root, inventory)

    assert check.status == "FAIL"
    assert check.hint == (
        "mcp__files__serach: server files has no tool 'serach'; did you mean search?"
    )


def test_a_bare_target_is_looked_for_on_every_server_the_task_attaches(tmp_path):
    from agent_exam.mcp import ToolInventory

    root = _mcp_project(
        tmp_path,
        (
            "t.yaml",
            "kind: trigger\nmcp_tool: search\nmcp_servers: [tickets]\npositive: [hi]\n",
        ),
        servers=("files", "tickets"),
    )
    inventory = ToolInventory(
        tools={"files": frozenset({"search"}), "tickets": frozenset({"open"})}
    )

    [check] = _trigger_checks(root, inventory)

    assert check.status == "FAIL"
    assert check.hint == "no attached server has a tool named 'search'"


def test_the_targets_of_a_server_that_could_not_be_asked_go_unchecked(tmp_path):
    """No listing, no verdict: the server may well serve the tool."""
    from agent_exam.mcp import ToolInventory

    root = _mcp_project(
        tmp_path,
        ("a.yaml", "kind: trigger\nmcp_tool: search\npositive: [hi]\n"),
        (
            "b.yaml",
            "kind: trigger\nmcp_tool: {server: tickets, tool: open}\npositive: [hi]\n",
        ),
        servers=("files", "tickets"),
    )
    nothing = ToolInventory(errors={"files": "cannot reach", "tickets": "cannot reach"})
    assert _trigger_checks(root, nothing) == []

    partial = ToolInventory(
        tools={"files": frozenset({"search"})}, errors={"tickets": "cannot reach"}
    )
    [check] = _trigger_checks(root, partial)
    assert check.status == "OK"
    assert check.hint == (
        "1 target(s) found on the attached servers, 1 unchecked (server not listed)"
    )


def test_only_the_servers_the_trigger_tasks_attach_are_asked(tmp_path):
    from agent_exam.tasks import load_suite
    from agent_exam.validation import trigger_servers

    root = _mcp_project(
        tmp_path,
        (
            "t.yaml",
            "kind: trigger\nmcp_tool: search\nmcp_servers: [files]\npositive: [hi]\n",
        ),
        ("e.yaml", "kind: execute\nprompt: x\nassertions: []\n"),
        servers=("files", "tickets"),
    )
    cfg = load_config(root)

    assert trigger_servers(cfg, load_suite(cfg.evals_dir, "s")) == {"files"}


def test_a_run_refuses_a_target_the_servers_do_not_serve(tmp_path, monkeypatch):
    from agent_exam.mcp import ToolInventory

    root = _mcp_project(
        tmp_path, ("t.yaml", "kind: trigger\nmcp_tool: serach\npositive: [hi]\n")
    )
    asked: list[set[str]] = []

    def listing(cfg, names, **kwargs):
        asked.append(set(names))
        return ToolInventory(tools={"files": frozenset({"search"})})

    monkeypatch.setattr("agent_exam.runner.tool_inventory", listing)

    with pytest.raises(UsageError, match="did you mean search") as excinfo:
        run(load_config(root), _dummy_request())

    assert asked == [{"files"}]
    assert str(excinfo.value).startswith(
        "trigger tools the attached mcp_servers do not serve"
    )


def test_a_run_warns_and_goes_on_when_a_server_cannot_be_asked(
    tmp_path, monkeypatch, capsys
):
    from agent_exam.mcp import ToolInventory

    root = _mcp_project(
        tmp_path, ("t.yaml", "kind: trigger\nmcp_tool: search\npositive: [hi]\n")
    )
    monkeypatch.setattr(
        "agent_exam.runner.tool_inventory",
        lambda cfg, names, **kwargs: ToolInventory(errors={"files": "cannot reach it"}),
    )

    # The run goes ahead — the dummy harness then misses the target, which
    # is the case's verdict, not a refusal to start.
    run(load_config(root), _dummy_request())

    assert (
        "[WARN] mcp tool listing: files: cannot reach it; the trigger tools it may "
        "serve were not checked" in capsys.readouterr().err
    )


def _graded(result: RunResult) -> bool:
    """Whether the generated positive assertion passes on *result*."""
    return first_mcp_tool(_CONFIG, result, Path()).pass_


def test_copilot_negative_case_does_not_settle_on_the_first_message():
    from agent_exam.providers.copilot_cli.provider import CopilotCliProvider
    from agent_exam.providers.copilot_cli.stream_parser import (
        StreamState as CopilotState,
    )
    from agent_exam.providers.copilot_cli.stream_parser import _dispatch as copilot

    state = CopilotState(provider=CopilotCliProvider())
    state.skill_detection_enabled = True
    state.target_tool = _TARGET
    state.mcp_server_names = ("files",)
    state.negative_trigger_mode = True

    copilot(
        json.dumps(
            {
                "type": "assistant.message",
                "data": {"toolRequests": [{"name": "bash", "toolCallId": "c1"}]},
            }
        ),
        state,
    )

    assert not state.kill_signal.is_set()


def test_copilot_negative_case_does_not_settle_on_turn_end():
    from agent_exam.providers.copilot_cli.provider import CopilotCliProvider
    from agent_exam.providers.copilot_cli.stream_parser import (
        StreamState as CopilotState,
    )
    from agent_exam.providers.copilot_cli.stream_parser import _dispatch as copilot

    state = CopilotState(provider=CopilotCliProvider())
    state.skill_detection_enabled = True
    state.target_tool = _TARGET
    state.negative_trigger_mode = True

    copilot(json.dumps({"type": "assistant.turn_end"}), state)

    assert not state.kill_signal.is_set()


def test_opencode_negative_case_does_not_settle_on_a_finished_tool():
    from agent_exam.providers.opencode.stream_parser import StreamState as OpenCodeState
    from agent_exam.providers.opencode.stream_parser import _dispatch as opencode

    state = OpenCodeState(provider=DummyProvider())
    state.skill_detection_enabled = True
    state.target_tool = _TARGET
    state.negative_trigger_mode = True

    opencode(
        json.dumps(
            {
                "type": "tool_use",
                "part": {"tool": "grep", "state": {"status": "completed"}},
            }
        ),
        state,
    )

    assert not state.kill_signal.is_set()


def test_opencode_does_not_settle_on_a_running_target_tool():
    """A `running` status is not decisive — the same part repeats once the
    call actually finishes, and cutting early would kill it mid-execution."""
    from agent_exam.providers.opencode.stream_parser import StreamState as OpenCodeState
    from agent_exam.providers.opencode.stream_parser import _dispatch as opencode

    state = OpenCodeState(provider=DummyProvider())
    state.skill_detection_enabled = True
    state.target_tool = _TARGET
    state.mcp_server_names = ("files",)

    opencode(
        json.dumps(
            {
                "type": "tool_use",
                "part": {"tool": "files_search", "state": {"status": "running"}},
            }
        ),
        state,
    )

    assert not state.kill_signal.is_set()


def test_opencode_negative_case_does_not_settle_on_turn_finish():
    from agent_exam.providers.opencode.stream_parser import StreamState as OpenCodeState
    from agent_exam.providers.opencode.stream_parser import _dispatch as opencode

    state = OpenCodeState(provider=DummyProvider())
    state.skill_detection_enabled = True
    state.target_tool = _TARGET
    state.negative_trigger_mode = True

    opencode(json.dumps({"type": "step_finish", "part": {"reason": "stop"}}), state)

    assert not state.kill_signal.is_set()


def test_codex_negative_case_does_not_settle_on_a_command():
    from agent_exam.providers.codex_cli.stream_parser import StreamState as CodexState
    from agent_exam.providers.codex_cli.stream_parser import _dispatch as codex

    state = CodexState(skill_detection_enabled=True)
    state.target_tool = _TARGET
    state.negative_trigger_mode = True

    codex(
        json.dumps(
            {
                "type": "item.completed",
                "item": {"type": "command_execution", "command": "rg invoice"},
            }
        ),
        state,
    )

    assert not state.kill_signal.is_set()


def test_codex_negative_case_does_not_settle_on_turn_completed():
    from agent_exam.providers.codex_cli.stream_parser import StreamState as CodexState
    from agent_exam.providers.codex_cli.stream_parser import _dispatch as codex

    state = CodexState(skill_detection_enabled=True)
    state.target_tool = _TARGET
    state.negative_trigger_mode = True

    codex(json.dumps({"type": "turn.completed", "usage": {}}), state)

    assert not state.kill_signal.is_set()


def test_opencode_records_the_call_its_database_may_not_have():
    """OpenCode's trajectory comes from its database, which the kill can
    beat."""
    from agent_exam.providers.opencode.stream_parser import StreamState
    from agent_exam.providers.opencode.transcripts import build_run_result

    state = StreamState(provider=DummyProvider())
    state.mcp_server_names = ("files",)

    result = build_run_result(
        state,
        wall_time_seconds=0.0,
        stream_detected_tool="files_search",
        user_prompt="Find the invoice for March.",
    )

    assert _graded(result)


def test_codex_records_the_call_its_session_may_not_have():
    """A kill can leave Codex's session unflushed, leaving the run with the
    minimal trajectory built from the stream alone."""
    from agent_exam.providers.codex_cli.stream_parser import StreamState
    from agent_exam.providers.codex_cli.transcripts import build_run_result

    result = build_run_result(
        StreamState(),
        wall_time_seconds=0.0,
        stream_detected_tool=_TARGET,
        user_prompt="Find the invoice for March.",
        allow_minimal_trigger_result=True,
    )

    assert _graded(result)
    assert result.metrics.n_tool_calls == 1


def test_copilot_cuts_a_positive_case_on_another_servers_tool():
    from agent_exam.providers.copilot_cli.provider import CopilotCliProvider
    from agent_exam.providers.copilot_cli.stream_parser import (
        StreamState as CopilotState,
    )
    from agent_exam.providers.copilot_cli.stream_parser import _dispatch as copilot

    state = CopilotState(provider=CopilotCliProvider())
    state.skill_detection_enabled = True
    state.target_tool = _TARGET
    state.mcp_server_names = ("files", "notes")

    copilot(
        json.dumps(
            {
                "type": "assistant.message",
                "data": {
                    "toolRequests": [
                        {
                            "name": "notes-search",
                            "toolCallId": "c1",
                            "mcpServerName": "notes",
                            "mcpToolName": "search",
                        }
                    ]
                },
            }
        ),
        state,
    )

    assert state.kill_signal.is_set()


def test_opencode_cuts_a_positive_case_on_another_servers_tool():
    from agent_exam.providers.opencode.stream_parser import StreamState as OpenCodeState
    from agent_exam.providers.opencode.stream_parser import _dispatch as opencode

    state = OpenCodeState(provider=DummyProvider())
    state.skill_detection_enabled = True
    state.target_tool = _TARGET
    state.mcp_server_names = ("files", "notes")

    opencode(
        json.dumps(
            {
                "type": "tool_use",
                "part": {"tool": "notes_search", "state": {"status": "completed"}},
            }
        ),
        state,
    )

    assert state.detected_tool == "notes_search"
    assert state.kill_signal.is_set()


def test_codex_cuts_a_positive_case_on_another_servers_tool():
    from agent_exam.providers.codex_cli.stream_parser import StreamState as CodexState
    from agent_exam.providers.codex_cli.stream_parser import _dispatch as codex

    state = CodexState(skill_detection_enabled=True)
    state.target_tool = _TARGET

    codex(
        json.dumps(
            {
                "type": "item.completed",
                "item": {"type": "mcp_tool_call", "server": "notes", "tool": "search"},
            }
        ),
        state,
    )

    assert state.detected_tool == "mcp__notes__search"
    assert state.kill_signal.is_set()
