from __future__ import annotations

import io
import json
import posixpath
import re
import shlex
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import IO

from ...schemas import SkillInvocation

_SKILL_PATH_RE = re.compile(
    r"(?:^|/)(?:\.agents/)?skills/(?P<name>[^/\s\"']+)/SKILL\.md$"
)
_READERS = {
    "cat",
    "sed",
    "head",
    "tail",
    "less",
    "more",
    "bat",
    "batcat",
    "awk",
    "nl",
}
_SHELLS = {"sh", "bash", "zsh", "dash", "ksh"}


@dataclass
class StreamState:
    """Mutable shared state filled by the Codex JSONL reader thread."""

    thread_id: str | None = None
    stderr_tail: deque = field(default_factory=lambda: deque(maxlen=8192))
    events: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0

    skill_detection_enabled: bool = False
    detected_skill: SkillInvocation | None = None
    kill_signal: threading.Event = field(default_factory=threading.Event)
    negative_trigger_mode: bool = False


def drain_stream(
    stdout: IO[bytes],
    state: StreamState,
    raw_out: IO[bytes] | None = None,
) -> None:
    """Read Codex `exec --json` JSONL events from stdout."""
    for raw_line in io.TextIOWrapper(stdout, encoding="utf-8", errors="replace"):
        line = raw_line.rstrip("\n")
        if raw_out is not None:
            try:
                raw_out.write((line + "\n").encode())
            except ValueError:
                # Parent gave up waiting on this thread and closed the
                # raw-stream file (see provider._invoke_once); drop the
                # rest of the raw transcript rather than crashing the
                # thread.
                raw_out = None
        if line.strip():
            _dispatch(line, state)


def drain_stderr(stderr: IO[bytes], state: StreamState) -> None:
    while True:
        chunk = stderr.read1(4096)
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

    state.events.append(event)

    event_type = event.get("type")
    if event_type == "thread.started":
        thread_id = event.get("thread_id")
        if thread_id and state.thread_id is None:
            state.thread_id = thread_id
        return

    if event_type == "turn.completed":
        usage = event.get("usage") or {}
        state.input_tokens = int(usage.get("input_tokens") or 0)
        state.cached_input_tokens = int(usage.get("cached_input_tokens") or 0)
        state.output_tokens = int(usage.get("output_tokens") or 0)
        state.reasoning_output_tokens = int(usage.get("reasoning_output_tokens") or 0)
        if (
            state.skill_detection_enabled
            and state.negative_trigger_mode
            and state.detected_skill is None
        ):
            state.kill_signal.set()
        return

    if event_type not in ("item.started", "item.completed"):
        return

    item = event.get("item") or {}
    if not isinstance(item, dict):
        return

    if state.skill_detection_enabled:
        _dispatch_skill_detection(item, state)


def stream_error_messages(events: list[dict]) -> list[str]:
    """Error messages Codex reported on its JSON stream, deduped, in order.

    Codex surfaces fatal API failures (auth, usage limits, …) as `error`
    events — turn-level failures as `turn.failed`, plus non-fatal error
    *items* — on stdout, while stderr stays uninformative. This is the
    only place the actual reason for a non-zero exit is recorded.
    """
    messages: list[str] = []
    for event in events:
        event_type = event.get("type")
        message = None
        if event_type == "error":
            message = event.get("message")
        elif event_type == "turn.failed":
            error = event.get("error")
            message = error.get("message") if isinstance(error, dict) else None
        elif event_type in ("item.started", "item.completed"):
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "error":
                message = item.get("message")
        if isinstance(message, str) and message:
            message = _unwrap_error_json(message)
            if message not in messages:
                messages.append(message)
    return messages


def _unwrap_error_json(message: str) -> str:
    """Codex often stuffs the raw API error response into `message` as a
    JSON blob (`{"type":"error","status":400,"error":{"message":...}}`);
    pull out the human-readable part when that's the case."""
    if not message.startswith("{"):
        return message
    try:
        parsed = json.loads(message)
    except json.JSONDecodeError:
        return message
    if isinstance(parsed, dict):
        error = parsed.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
    return message


def _dispatch_skill_detection(item: dict, state: StreamState) -> None:
    detected = _skill_detection_from_item(item)
    if detected:
        skill, trigger_kind = detected
        state.detected_skill = SkillInvocation(
            skill_name=skill,
            trigger_kind=trigger_kind,
            triggered_by_tool_use_id=item.get("id"),
        )
        state.kill_signal.set()
        return

    # For negative triggers, non-skill tool work is enough evidence that
    # routing went elsewhere. Plain text alone is not decisive. Reader
    # commands are exempt: an agent legitimately inspects files (e.g. via
    # `cat`) before deciding whether to invoke a skill, mirroring Claude
    # Code's Read exemption for the same reason. Codex's dedicated tools
    # (web_search etc.) are decisive too — e.g. an agent that starts
    # fetching a live URL has routed away from a file-analysis skill;
    # without this it would burn to the trigger timeout instead of being
    # killed on the routing decision.
    if not state.negative_trigger_mode:
        return
    item_type = item.get("type")
    if (
        item_type == "command_execution" and not _item_is_file_read(item)
    ) or item_type in ("web_search", "file_change", "mcp_tool_call"):
        state.kill_signal.set()


def _item_is_file_read(item: dict) -> bool:
    command = item.get("command")
    return isinstance(command, str) and _command_reads_file(
        _effective_tokens(_tokenize_command(command))
    )


def _skill_detection_from_item(item: dict) -> tuple[str, str] | None:
    item_type = item.get("type")
    if item_type == "command_execution":
        command = item.get("command")
        if isinstance(command, str):
            return _skill_detection_from_command(
                command,
                _string_or_none(item.get("workdir"))
                or _string_or_none(item.get("cwd")),
            )
    return None


def _skill_detection_from_command(
    command: str,
    workdir: str | None = None,
) -> tuple[str, str] | None:
    tokens = _effective_tokens(_tokenize_command(command))
    if not _command_reads_file(tokens):
        return None
    for token in tokens[1:]:
        if token.startswith("-"):
            continue
        for path in _candidate_paths(token, workdir):
            match = _SKILL_PATH_RE.search(path)
            if match:
                return match.group("name"), "skill_md_read"
    return None


def _tokenize_command(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _effective_tokens(tokens: list[str], depth: int = 0) -> list[str]:
    """Unwrap `sh -c '<command>'`-style shell wrapping.

    Codex runs every command through a login shell (e.g.
    `/bin/zsh -lc 'cat SKILL.md'`), so argv[0] is always a shell rather
    than the actual program being invoked. Recurses (bounded) in case the
    inner command is itself shell-wrapped.
    """
    if depth >= 4 or not tokens:
        return tokens
    if _command_basename(tokens[0]).lower() not in _SHELLS:
        return tokens
    for index, token in enumerate(tokens[1:], start=1):
        if not token.startswith("-"):
            break
        # Short-option cluster (e.g. -c, -lc) containing `c`, not a
        # long-form flag like --rcfile that merely contains the letter.
        is_short_cluster = not token.startswith("--")
        if is_short_cluster and "c" in token[1:] and index + 1 < len(tokens):
            return _effective_tokens(_tokenize_command(tokens[index + 1]), depth + 1)
    return tokens


def _command_reads_file(tokens: list[str]) -> bool:
    program = tokens[0] if tokens else None
    if not program:
        return False
    return _command_basename(program).lower() in _READERS


def _candidate_paths(path: str, workdir: str | None) -> list[str]:
    path = _normalize_command_path(path)
    candidates = [path]
    if workdir and not _is_absolute_path(path):
        candidates.append(
            posixpath.normpath(f"{_normalize_command_path(workdir).rstrip('/')}/{path}")
        )
    return candidates


def _normalize_command_path(path: str) -> str:
    return posixpath.normpath(path.replace("\\", "/"))


def _is_absolute_path(path: str) -> bool:
    return path.startswith("/") or re.match(r"^[A-Za-z]:/", path) is not None


def _command_basename(command: str) -> str:
    return command.replace("\\", "/").rsplit("/", 1)[-1]


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
