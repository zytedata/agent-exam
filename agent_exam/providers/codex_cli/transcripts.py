from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ...errors import FrameworkError
from ...schemas import (
    Metrics,
    RunResult,
    SkillInvocation,
    TextBlock,
    ThinkingBlock,
    Tokens,
    ToolCallBlock,
    Turn,
)
from ...trajectory_walk import iter_tool_calls, walk_turns
from .paths import codex_home
from .stream_parser import StreamState, _skill_detection_from_item

_EXPLICIT_SKILL_RE = re.compile(
    r"<skill>\s*<name>(?P<name>[^<]+)</name>\s*<path>(?P<path>[^<]+)</path>",
    re.DOTALL,
)
_EXIT_CODE_RE = re.compile(r"process exited with code (\d+)")
_PERMISSION_DENIAL_PATTERNS = (
    "operation not permitted",
    "permission denied",
    "network access was denied by the codex sandbox",
    "network access denied",
    "command denied by sandbox",
    "denied by sandbox",
)


def build_run_result(
    state: StreamState,
    wall_time_seconds: float,
    stream_detected_skill: SkillInvocation | None = None,
    raw_transcript_path: Path | None = None,
    user_prompt: str | None = None,
    model: str | None = None,
    allow_minimal_trigger_result: bool = False,
    env: dict[str, str] | None = None,
) -> RunResult:
    try:
        session = _load_session_snapshot(state.thread_id, env=env)
    except FrameworkError:
        if not allow_minimal_trigger_result:
            raise
        # A killed-on-skill trigger run can leave a spawned subagent's
        # session unflushed; the minimal result below only needs the
        # stream-detected skill, not a fully attached subagent tree.
        session = None
    if session is None or not session.trajectory:
        if allow_minimal_trigger_result:
            return _build_minimal_trigger_result(
                state,
                wall_time_seconds=wall_time_seconds,
                stream_detected_skill=stream_detected_skill,
                raw_transcript_path=raw_transcript_path,
                user_prompt=user_prompt,
                model=model,
            )
        detail = f"thread_id={state.thread_id or 'unknown'}"
        if raw_transcript_path is not None:
            detail += f", raw_stream={raw_transcript_path}"
        raise FrameworkError(
            "Codex completed without a persisted session transcript; "
            f"cannot build a reliable trajectory ({detail})"
        )

    trajectory = session.trajectory
    _attach_session_skill_invocations(trajectory)
    stream_invocations = _extract_stream_skill_invocations(
        state.events, stream_detected_skill
    )
    _attach_skill_invocations(trajectory, stream_invocations)
    metrics = _build_metrics(
        wall_time_seconds,
        trajectory,
        state,
        session_metrics=session.metrics,
    )
    return RunResult(
        trajectory=trajectory,
        metrics=metrics,
        raw_transcript_path=raw_transcript_path,
        model=session.model or model,
    )


def _build_minimal_trigger_result(
    state: StreamState,
    wall_time_seconds: float,
    stream_detected_skill: SkillInvocation | None,
    raw_transcript_path: Path | None,
    user_prompt: str | None,
    model: str | None,
) -> RunResult:
    trajectory: list[Turn] = []
    if user_prompt:
        trajectory.append(Turn(role="user", content=[TextBlock(text=user_prompt)]))

    invocations = _extract_stream_skill_invocations(state.events, stream_detected_skill)
    trajectory.append(
        Turn(
            role="assistant",
            content=[],
            tokens=Tokens(
                input=state.input_tokens,
                output=state.output_tokens,
                cache_read=state.cached_input_tokens,
            ),
            skill_invocations=invocations,
        )
    )

    tokens = Tokens(
        input=state.input_tokens,
        output=state.output_tokens,
        cache_read=state.cached_input_tokens,
    )
    return RunResult(
        trajectory=trajectory,
        metrics=Metrics(
            wall_time_seconds=wall_time_seconds,
            tokens=tokens,
            cost_usd=None,
            peak_context=state.input_tokens + state.output_tokens,
            turn_count=1,
            raw={
                "reasoning_output_tokens": state.reasoning_output_tokens,
                "minimal_trigger_result": True,
            },
        ),
        raw_transcript_path=raw_transcript_path,
        model=model or None,
    )


def _tokens_from_usage(usage: dict) -> Tokens:
    return Tokens(
        input=int(usage.get("input_tokens") or 0),
        output=int(usage.get("output_tokens") or 0),
        cache_read=int(usage.get("cached_input_tokens") or 0),
    )


def _context_from_usage(usage: dict) -> int:
    return (
        int(usage.get("input_tokens") or 0)
        + int(usage.get("output_tokens") or 0)
        + int(usage.get("reasoning_output_tokens") or 0)
    )


@dataclass
class _SessionMetrics:
    tokens: Tokens
    reasoning_output_tokens: int = 0
    peak_context: int = 0
    subagent_count: int = 0
    session_paths: list[str] | None = None


@dataclass
class _SessionSnapshot:
    thread_id: str
    path: Path
    model: str | None
    trajectory: list[Turn]
    metrics: _SessionMetrics


def _add_session_metrics(
    total: _SessionMetrics, extra: _SessionMetrics, count_session: bool = False
) -> None:
    total.tokens.input += extra.tokens.input
    total.tokens.output += extra.tokens.output
    total.tokens.cache_read += extra.tokens.cache_read
    total.reasoning_output_tokens += extra.reasoning_output_tokens
    total.peak_context = max(total.peak_context, extra.peak_context)
    total.subagent_count += extra.subagent_count + (1 if count_session else 0)
    if extra.session_paths:
        if total.session_paths is None:
            total.session_paths = []
        total.session_paths.extend(extra.session_paths)


def _load_session_snapshot(
    thread_id: str | None,
    visited: set[str] | None = None,
    env: dict[str, str] | None = None,
) -> _SessionSnapshot | None:
    if not thread_id:
        return None
    visited = visited or set()
    if thread_id in visited:
        return None
    visited.add(thread_id)
    path = _find_session_path(thread_id, env=env)
    if path is None:
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    events: list[dict] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    model = _session_model(events)
    trajectory = _session_trajectory(events)
    metrics = _session_metrics(events, path)
    child_metrics = _attach_session_subagents(trajectory, visited, env=env)
    _add_session_metrics(metrics, child_metrics)
    return _SessionSnapshot(
        thread_id=thread_id,
        path=path,
        model=model,
        trajectory=trajectory,
        metrics=metrics,
    )


def _attach_session_subagents(
    trajectory: list[Turn], visited: set[str], env: dict[str, str] | None = None
) -> _SessionMetrics:
    metrics = _SessionMetrics(tokens=Tokens(), session_paths=[])
    for block in iter_tool_calls(trajectory, include_subagents=False):
        if block.name != "spawn_agent":
            continue
        receiver_ids = block.input.get("receiver_thread_ids")
        if not isinstance(receiver_ids, list):
            continue
        subagent_turns: list[Turn] = []
        for child_id in receiver_ids:
            if not isinstance(child_id, str):
                continue
            child = _load_session_snapshot(child_id, visited, env=env)
            if child is None or not child.trajectory:
                raise FrameworkError(
                    "Codex subagent session transcript missing or empty "
                    f"(thread_id={child_id})"
                )
            subagent_turns.extend(child.trajectory)
            _add_session_metrics(metrics, child.metrics, count_session=True)
        if subagent_turns:
            block.subagent = subagent_turns
    return metrics


def _find_session_path(
    thread_id: str, env: dict[str, str] | None = None
) -> Path | None:
    sessions_root = codex_home(env) / "sessions"
    if not sessions_root.exists():
        return None
    matches = list(sessions_root.rglob(f"*{thread_id}.jsonl"))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def find_session_explicit_skill_invocation(
    thread_id: str | None,
    env: dict[str, str] | None = None,
) -> SkillInvocation | None:
    """Return the first explicit Codex skill injection already persisted.

    Codex `exec --json` does not currently emit a native skill event for
    `$skill` text selections. It does persist the selected skill as a
    synthetic user `response_item` whose content starts with `<skill>`.
    This helper is intentionally based on that persisted injection rather
    than on the original prompt text.
    """
    if not thread_id:
        return None
    path = _find_session_path(thread_id, env=env)
    if path is None:
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        for inv in _explicit_skill_invocations_from_event(event):
            return inv
    return None


def _session_model(events: list[dict]) -> str | None:
    for event in events:
        if event.get("type") != "turn_context":
            continue
        payload = event.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        model = payload.get("model")
        if isinstance(model, str) and model:
            return model
    return None


def _session_trajectory(events: list[dict]) -> list[Turn]:
    turns: list[Turn] = []
    assistant = _SessionAssistantBuilder()
    for event in events:
        payload = event.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        event_type = event.get("type")
        payload_type = payload.get("type")
        ts = _session_ts(event)

        if event_type == "event_msg" and payload_type == "task_started":
            started_at = payload.get("started_at")
            if isinstance(started_at, (int, float)):
                assistant.started_at = float(started_at)
            elif ts is not None:
                assistant.started_at = ts
            continue

        if event_type == "event_msg" and payload_type == "user_message":
            message = payload.get("message")
            if isinstance(message, str) and message:
                turns.append(
                    Turn(role="user", content=[TextBlock(text=message)], started_at=ts)
                )
            continue

        if (
            event_type == "response_item"
            and payload_type == "message"
            and payload.get("role") == "user"
        ):
            assistant.add_explicit_skill_invocations(
                _explicit_skill_invocations_from_payload(payload)
            )
            continue

        if event_type == "event_msg" and payload_type == "token_count":
            usage = _session_total_usage(payload)
            if usage:
                assistant.set_tokens(_tokens_from_usage(usage))
                assistant.set_context(_context_from_usage(usage))
            continue

        if event_type != "response_item":
            continue

        if payload_type == "message" and payload.get("role") == "assistant":
            text = _session_message_text(payload)
            if text:
                assistant.add_text(text)
        elif payload_type == "reasoning":
            for text in _session_reasoning_summary_texts(payload):
                assistant.add_thinking(text)
        elif payload_type in (
            "function_call",
            "tool_search_call",
            "local_shell_call",
            "custom_tool_call",
        ):
            assistant.add_session_tool_call(
                _canonical_tool_call_payload(payload), started_at=ts
            )
        elif payload_type in (
            "function_call_output",
            "tool_search_output",
            "custom_tool_call_output",
        ):
            assistant.complete_session_tool_call(payload, completed_at=ts)
        elif payload_type == "web_search_call":
            assistant.add_completed_tool_call(_web_search_tool_call(payload, ts))
        elif payload_type == "image_generation_call":
            assistant.add_completed_tool_call(_image_generation_tool_call(payload, ts))

    turn = assistant.build()
    if turn is not None:
        turns.append(turn)
    return turns


class _SessionAssistantBuilder:
    def __init__(self) -> None:
        self.started_at: float | None = None
        self.content: list[Any] = []
        self.tokens: Tokens | None = None
        self.context: int | None = None
        self.skill_invocations: list[SkillInvocation] = []
        self._pending_tool_indexes: dict[str, int] = {}

    def add_text(self, text: str) -> None:
        self.content.append(TextBlock(text=text))

    def add_thinking(self, text: str) -> None:
        self.content.append(ThinkingBlock(text=text))

    def add_explicit_skill_invocations(
        self, invocations: list[SkillInvocation]
    ) -> None:
        for inv in invocations:
            if not _has_invocation(self.skill_invocations, inv):
                self.skill_invocations.append(inv)

    def add_session_tool_call(self, payload: dict, started_at: float | None) -> None:
        block = _session_tool_call_from_payload(
            payload,
            started_at=started_at,
            status="aborted",
        )
        index = len(self.content)
        self.content.append(block)
        if block.tool_use_id:
            self._pending_tool_indexes[block.tool_use_id] = index

    def add_completed_tool_call(self, block: ToolCallBlock) -> None:
        # Self-contained tool calls (web_search, image_generation) carry their
        # own terminal status inline and have no paired *_output event, so they
        # are appended fully-formed rather than tracked as pending.
        self.content.append(block)

    def complete_session_tool_call(
        self, payload: dict, completed_at: float | None
    ) -> None:
        call_id = payload.get("call_id")
        if not isinstance(call_id, str):
            call_id = ""
        pending_index = self._pending_tool_indexes.pop(call_id, None)
        pending: ToolCallBlock | None = None
        if pending_index is not None:
            maybe_pending = self.content[pending_index]
            if isinstance(maybe_pending, ToolCallBlock):
                pending = maybe_pending

        if pending is None:
            block = _session_tool_result_without_call(payload, completed_at)
            self.content.append(block)
            return

        block = _complete_session_tool_call(pending, payload, completed_at)
        self.content[pending_index] = block

    def set_tokens(self, tokens: Tokens) -> None:
        self.tokens = tokens

    def set_context(self, context: int) -> None:
        self.context = context

    def build(self) -> Turn | None:
        if not self.content and not self.skill_invocations:
            return None
        return Turn(
            role="assistant",
            content=self.content,
            tokens=self.tokens,
            context=self.context,
            started_at=self.started_at,
            skill_invocations=self.skill_invocations,
        )


def _session_tool_call_from_payload(
    payload: dict,
    started_at: float | None,
    status: str,
) -> ToolCallBlock:
    call_id = payload.get("call_id")
    if not isinstance(call_id, str):
        call_id = ""
    name = _normalize_session_tool_name(payload.get("name"))
    input_data = _session_tool_input(name, payload.get("arguments"))
    return ToolCallBlock(
        tool_use_id=call_id,
        name=name,
        input=input_data,
        status=status,
        result="",
        started_at=started_at,
    )


def _complete_session_tool_call(
    pending: ToolCallBlock, payload: dict, completed_at: float | None
) -> ToolCallBlock:
    output = payload.get("output")
    result = output if isinstance(output, str) else json.dumps(output, default=str)
    input_data = dict(pending.input)
    _fill_session_tool_result_metadata(pending.name, input_data, result)
    duration_ms = None
    if pending.started_at is not None and completed_at is not None:
        duration_ms = round((completed_at - pending.started_at) * 1000)
    return ToolCallBlock(
        tool_use_id=pending.tool_use_id,
        name=pending.name,
        input=input_data,
        status=_session_tool_status(pending.name, result),
        result=result,
        started_at=pending.started_at,
        duration_ms=duration_ms,
        subagent=pending.subagent,
    )


def _session_tool_result_without_call(
    payload: dict, completed_at: float | None
) -> ToolCallBlock:
    call_id = payload.get("call_id")
    if not isinstance(call_id, str):
        call_id = ""
    output = payload.get("output")
    result = output if isinstance(output, str) else json.dumps(output, default=str)
    return ToolCallBlock(
        tool_use_id=call_id,
        name="tool_result",
        input={},
        status="ok",
        result=result,
        started_at=completed_at,
    )


def _session_tool_status(name: str, result: str) -> str:
    if name != "command_execution":
        return "ok"
    low = result.lower()
    # The exit-code header always precedes the command's own captured
    # stdout/stderr, so the first match is authoritative — this avoids
    # both false positives (a successful command whose output happens to
    # mention "permission denied") and false negatives (a failing command
    # whose output happens to mention "exited with code 0").
    match = _EXIT_CODE_RE.search(low)
    if match:
        if match.group(1) == "0":
            return "ok"
        return "permission_denied" if _is_session_permission_denial(low) else "error"
    return "permission_denied" if _is_session_permission_denial(low) else "ok"


def _is_session_permission_denial(low_result: str) -> bool:
    if any(pattern in low_result for pattern in _PERMISSION_DENIAL_PATTERNS):
        return True
    return (
        "approval policy is" in low_result
        and "ask for escalated permissions" in low_result
    )


def _normalize_session_tool_name(name: Any) -> str:
    # All of Codex's shell-exec variants normalize to a single name so tool
    # assertions are stable regardless of which one a model/version emits.
    if name in ("exec_command", "unified_exec", "local_shell", "shell"):
        return "command_execution"
    if name == "wait_agent":
        return "wait"
    if isinstance(name, str) and name:
        return name
    return "tool"


def _session_tool_input(name: str, raw_arguments: Any) -> dict[str, Any]:
    args: Any
    if isinstance(raw_arguments, str):
        try:
            args = json.loads(raw_arguments)
        except json.JSONDecodeError:
            args = {"arguments": raw_arguments}
    elif isinstance(raw_arguments, dict):
        args = dict(raw_arguments)
    else:
        args = {}
    if not isinstance(args, dict):
        args = {"arguments": args}
    if name == "command_execution" and "cmd" in args and "command" not in args:
        args["command"] = args["cmd"]
    if name == "spawn_agent" and "message" in args and "prompt" not in args:
        args["prompt"] = args["message"]
    return args


def _canonical_tool_call_payload(payload: dict) -> dict:
    """Reshape Codex's dedicated tool-call ResponseItems into the function_call
    shape (name + arguments + call_id) so they flow through the normal tool-call
    builder. `function_call`/`tool_search_call` are passed through unchanged.

    `local_shell_call` is the Responses-API shell; it carries an `action`
    (`{type: exec, command: [...]}`) instead of `arguments`, and its result
    arrives as a `function_call_output` keyed by the same call_id. Codex emits
    no `local_shell_call_output` variant, so the existing completion handler
    finishes it.
    """
    ptype = payload.get("type")
    if ptype == "local_shell_call":
        action = payload.get("action") or {}
        command = action.get("command")
        if isinstance(command, list):
            command = " ".join(str(part) for part in command)
        return {
            "type": ptype,
            "name": "local_shell",
            "arguments": json.dumps({"command": command}),
            "call_id": payload.get("call_id") or payload.get("id") or "",
        }
    if ptype == "custom_tool_call":
        return {
            "type": ptype,
            "name": payload.get("name"),
            "arguments": payload.get("input"),
            "call_id": payload.get("call_id") or "",
        }
    return payload


def _web_search_tool_call(payload: dict, started_at: float | None) -> ToolCallBlock:
    """Build a tool call from a self-contained `web_search_call` ResponseItem."""
    action = payload.get("action") or {}
    action_type = action.get("type")
    input_data: dict[str, Any] = {}
    if isinstance(action_type, str):
        input_data["action_type"] = action_type
    result = ""
    if action_type == "search":
        queries = action.get("queries")
        query = action.get("query")
        if isinstance(queries, list) and queries:
            input_data["queries"] = queries
            result = "; ".join(str(q) for q in queries)
        elif isinstance(query, str):
            input_data["query"] = query
            result = query
    elif action_type == "open_page":
        url = action.get("url")
        if isinstance(url, str):
            input_data["url"] = url
            result = url
    elif action_type == "find_in_page":
        for key in ("url", "pattern"):
            value = action.get(key)
            if isinstance(value, str):
                input_data[key] = value
        result = input_data.get("pattern", "")
    return ToolCallBlock(
        tool_use_id=payload.get("id") or "",
        name="web_search",
        input=input_data,
        status="ok" if payload.get("status") in (None, "completed") else "aborted",
        result=result,
        started_at=started_at,
    )


def _image_generation_tool_call(
    payload: dict, started_at: float | None
) -> ToolCallBlock:
    """Build a tool call from an `image_generation_call` ResponseItem.

    `result` is base64 image data; it is elided to keep the transcript small —
    the revised prompt is kept as the human-readable trace.
    """
    revised = payload.get("revised_prompt")
    return ToolCallBlock(
        tool_use_id=payload.get("id") or "",
        name="image_generation",
        input={},
        status="ok" if payload.get("status") in (None, "completed") else "aborted",
        result=revised if isinstance(revised, str) else "<image>",
        started_at=started_at,
    )


def _fill_session_tool_result_metadata(
    name: str, input_data: dict[str, Any], result: str
) -> None:
    try:
        decoded = json.loads(result)
    except json.JSONDecodeError:
        return
    if not isinstance(decoded, dict):
        return
    if name == "spawn_agent":
        agent_id = decoded.get("agent_id")
        if isinstance(agent_id, str):
            input_data["receiver_thread_ids"] = [agent_id]
    elif name == "wait":
        status = decoded.get("status")
        if isinstance(status, dict):
            input_data["receiver_thread_ids"] = list(status.keys())


def _session_message_text(payload: dict) -> str:
    content = payload.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") in ("output_text", "text"):
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _session_reasoning_summary_texts(payload: dict) -> list[str]:
    summary = payload.get("summary")
    if not isinstance(summary, list):
        return []
    texts: list[str] = []
    for part in summary:
        if isinstance(part, str):
            if part:
                texts.append(part)
            continue
        if not isinstance(part, dict):
            continue
        text = part.get("text") or part.get("summary")
        if isinstance(text, str) and text:
            texts.append(text)
    return texts


def _explicit_skill_invocations_from_event(event: dict) -> list[SkillInvocation]:
    if event.get("type") != "response_item":
        return []
    payload = event.get("payload") or {}
    if not isinstance(payload, dict):
        return []
    if payload.get("type") != "message" or payload.get("role") != "user":
        return []
    return _explicit_skill_invocations_from_payload(payload)


def _explicit_skill_invocations_from_payload(payload: dict) -> list[SkillInvocation]:
    invocations: list[SkillInvocation] = []
    for text in _session_text_parts(payload):
        if not text.lstrip().startswith("<skill>"):
            continue
        for match in _EXPLICIT_SKILL_RE.finditer(text):
            name = match.group("name").strip()
            path = match.group("path").strip()
            if not name or not path.replace("\\", "/").endswith("/SKILL.md"):
                continue
            inv = SkillInvocation(
                skill_name=name,
                trigger_kind="explicit_skill",
            )
            if not _has_invocation(invocations, inv):
                invocations.append(inv)
    return invocations


def _session_text_parts(payload: dict) -> list[str]:
    content = payload.get("content")
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") in ("input_text", "output_text", "text"):
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
    return parts


def _session_total_usage(payload: dict) -> dict | None:
    info = payload.get("info") or {}
    if not isinstance(info, dict):
        return None
    usage = info.get("total_token_usage")
    return usage if isinstance(usage, dict) else None


def _session_metrics(events: list[dict], path: Path) -> _SessionMetrics:
    tokens = Tokens()
    reasoning = 0
    peak_context = 0
    for event in events:
        payload = event.get("payload") or {}
        if not isinstance(payload, dict) or payload.get("type") != "token_count":
            continue
        usage = _session_total_usage(payload)
        if not usage:
            continue
        tokens = _tokens_from_usage(usage)
        reasoning = int(usage.get("reasoning_output_tokens") or 0)
        peak_context = _context_from_usage(usage)
    return _SessionMetrics(
        tokens=tokens,
        reasoning_output_tokens=reasoning,
        peak_context=peak_context,
        session_paths=[str(path)],
    )


def _session_ts(event: dict) -> float | None:
    ts = event.get("timestamp")
    if not isinstance(ts, str):
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _extract_stream_skill_invocations(
    events: list[dict],
    stream_detected_skill: SkillInvocation | None,
) -> list[SkillInvocation]:
    invocations: list[SkillInvocation] = []
    for event in events:
        if event.get("type") not in ("item.started", "item.completed"):
            continue
        item = event.get("item") or {}
        if not isinstance(item, dict):
            continue
        detected = _skill_detection_from_item(item)
        if not detected:
            continue
        skill_name, trigger_kind = detected
        inv = SkillInvocation(
            skill_name=skill_name,
            trigger_kind=trigger_kind,
            triggered_by_tool_use_id=item.get("id"),
        )
        if not _has_invocation(invocations, inv):
            invocations.append(inv)

    if stream_detected_skill is not None and not _has_invocation(
        invocations, stream_detected_skill
    ):
        invocations.append(stream_detected_skill)
    return invocations


def _attach_session_skill_invocations(trajectory: list[Turn]) -> None:
    for turn in trajectory:
        if turn.role != "assistant":
            continue
        for block in turn.content:
            if not isinstance(block, ToolCallBlock):
                continue
            inv = _skill_invocation_from_tool_call(block)
            if inv is not None and not _has_invocation(turn.skill_invocations, inv):
                turn.skill_invocations.append(inv)
            if block.subagent:
                _attach_session_skill_invocations(block.subagent)


def _skill_invocation_from_tool_call(block: ToolCallBlock) -> SkillInvocation | None:
    if block.name != "command_execution":
        return None
    command = block.input.get("command") or block.input.get("cmd")
    item = {
        "id": block.tool_use_id,
        "type": "command_execution",
        "command": command,
        "workdir": block.input.get("workdir"),
        "cwd": block.input.get("cwd"),
    }
    detected = _skill_detection_from_item(item)
    if not detected:
        return None
    skill_name, trigger_kind = detected
    return SkillInvocation(
        skill_name=skill_name,
        trigger_kind=trigger_kind,
        triggered_by_tool_use_id=block.tool_use_id,
    )


def _has_invocation(
    invocations: list[SkillInvocation], candidate: SkillInvocation
) -> bool:
    return any(
        i.skill_name == candidate.skill_name
        and i.triggered_by_tool_use_id == candidate.triggered_by_tool_use_id
        for i in invocations
    )


def _attach_skill_invocations(
    trajectory: list[Turn], invocations: list[SkillInvocation]
) -> None:
    if not invocations:
        return
    invocations = [
        inv for inv in invocations if not _trajectory_has_invocation(trajectory, inv)
    ]
    if not invocations:
        return
    for turn in trajectory:
        if turn.role == "assistant":
            turn.skill_invocations.extend(invocations)
            return


def _trajectory_has_invocation(
    trajectory: list[Turn], candidate: SkillInvocation
) -> bool:
    # Matches on (skill_name, trigger_kind) rather than
    # triggered_by_tool_use_id: this checks a stream-derived invocation
    # against ones already attached from the session, but the stream's
    # item ids ("item_3") and the session's function-call ids ("call_abc")
    # are different namespaces for the same underlying tool call, so an
    # id-based match would never dedup a genuine repeat.
    return any(
        inv.skill_name == candidate.skill_name
        and inv.trigger_kind == candidate.trigger_kind
        for turn in walk_turns(trajectory)
        for inv in turn.skill_invocations
    )


def _build_metrics(
    wall_time_seconds: float,
    trajectory: list[Turn],
    state: StreamState,
    session_metrics: _SessionMetrics | None = None,
) -> Metrics:
    n_tool_calls = n_tool_errors = n_permission_denied = n_tool_rejected = 0
    for block in iter_tool_calls(trajectory):
        n_tool_calls += 1
        if block.status == "error":
            n_tool_errors += 1
        elif block.status == "permission_denied":
            n_permission_denied += 1
        elif block.status == "rejected":
            n_tool_rejected += 1

    turn_count = sum(1 for t in trajectory if t.role == "assistant")
    if session_metrics is not None and _has_tokens(session_metrics.tokens):
        tokens = Tokens(
            input=session_metrics.tokens.input,
            output=session_metrics.tokens.output,
            cache_read=session_metrics.tokens.cache_read,
        )
        reasoning_output_tokens = session_metrics.reasoning_output_tokens
        peak_context = session_metrics.peak_context
        raw: dict[str, Any] = {
            "reasoning_output_tokens": reasoning_output_tokens,
            "codex_session_paths": session_metrics.session_paths or [],
        }
        if session_metrics.subagent_count:
            raw["subagent_count"] = session_metrics.subagent_count
    else:
        tokens = Tokens(
            input=state.input_tokens,
            output=state.output_tokens,
            cache_read=state.cached_input_tokens,
        )
        reasoning_output_tokens = state.reasoning_output_tokens
        peak_context = state.input_tokens + state.output_tokens
        raw = {"reasoning_output_tokens": reasoning_output_tokens}

    return Metrics(
        wall_time_seconds=wall_time_seconds,
        tokens=tokens,
        cost_usd=None,
        peak_context=peak_context,
        turn_count=turn_count,
        n_tool_calls=n_tool_calls,
        n_tool_errors=n_tool_errors,
        n_permission_denied=n_permission_denied,
        n_tool_rejected=n_tool_rejected,
        raw=raw,
    )


def _has_tokens(tokens: Tokens) -> bool:
    return bool(tokens.input or tokens.output or tokens.cache_read)
