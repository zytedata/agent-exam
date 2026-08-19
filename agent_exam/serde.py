from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from .schemas import (
    Metrics,
    RunResult,
    SkillInvocation,
    TextBlock,
    ThinkingBlock,
    Tokens,
    ToolCallBlock,
    Turn,
)


def _prepare(value: Any) -> Any:
    """Recursively convert dataclasses, Paths, etc. into JSON-safe values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        out = {}
        for f in fields(value):
            name = "pass" if f.name == "pass_" else f.name
            out[name] = _prepare(getattr(value, f.name))
        return out
    if isinstance(value, dict):
        return {str(k): _prepare(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_prepare(v) for v in value]
    return str(value)


def to_json_dict(value: Any) -> Any:
    return _prepare(value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(_prepare(value), fh, indent=2, sort_keys=False)
        fh.write("\n")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Deserialization — reverse of _prepare for the schema types used by rescore.
# ---------------------------------------------------------------------------


def _block_from_dict(d: dict):
    t = d.get("type")
    if t == "text":
        return TextBlock(text=d.get("text", ""))
    if t == "thinking":
        return ThinkingBlock(text=d.get("text", ""))
    if t == "tool_call":
        subagent = None
        if d.get("subagent") is not None:
            subagent = [_turn_from_dict(s) for s in d["subagent"]]
        return ToolCallBlock(
            tool_use_id=d.get("tool_use_id", ""),
            name=d.get("name", ""),
            input=dict(d.get("input") or {}),
            status=d.get("status", "ok"),
            result=d.get("result", ""),
            started_at=d.get("started_at"),
            duration_ms=d.get("duration_ms"),
            subagent=subagent,
        )
    return None  # unknown block type — skip for forward-compat


def _turn_from_dict(d: dict) -> Turn:
    content = []
    for raw_block in d.get("content") or []:
        block = _block_from_dict(raw_block)
        if block is not None:
            content.append(block)
    tokens = None
    tok = d.get("tokens")
    if tok is not None:
        tokens = Tokens(
            input=tok.get("input", 0),
            output=tok.get("output", 0),
            cache_read=tok.get("cache_read", 0),
        )
    skill_invs = [
        SkillInvocation(
            skill_name=si.get("skill_name", ""),
            trigger_kind=si.get("trigger_kind", ""),
            triggered_by_tool_use_id=si.get("triggered_by_tool_use_id"),
        )
        for si in (d.get("skill_invocations") or [])
    ]
    return Turn(
        role=d.get("role", "user"),
        content=content,
        model=d.get("model"),
        tokens=tokens,
        context=d.get("context"),
        started_at=d.get("started_at"),
        skill_invocations=skill_invs,
    )


def trajectory_from_dict(trajectory_json: dict) -> list[Turn]:
    return [_turn_from_dict(t) for t in trajectory_json.get("turns", [])]


def metrics_from_dict(metrics_json: dict) -> Metrics:
    tok = metrics_json.get("tokens") or {}
    return Metrics(
        wall_time_seconds=metrics_json.get("wall_time_seconds", 0.0),
        tokens=Tokens(
            input=tok.get("input", 0),
            output=tok.get("output", 0),
            cache_read=tok.get("cache_read", 0),
        ),
        cost_usd=metrics_json.get("cost_usd"),  # None preserved as-is
        peak_context=metrics_json.get("peak_context", 0),
        turn_count=metrics_json.get("turn_count", 0),
        n_tool_calls=metrics_json.get("n_tool_calls", 0),
        n_tool_errors=metrics_json.get("n_tool_errors", 0),
        n_permission_denied=metrics_json.get("n_permission_denied", 0),
        n_tool_rejected=metrics_json.get("n_tool_rejected", 0),
        raw=metrics_json.get("raw") or {},
    )


def run_result_from_artifacts(attempt_json: dict, trajectory_json: dict) -> RunResult:
    """Reconstruct a RunResult from archived `attempt.json` + `trajectory.json`."""
    trajectory = trajectory_from_dict(trajectory_json)
    metrics = metrics_from_dict(attempt_json.get("metrics") or {})
    raw_path_str = attempt_json.get("raw_transcript_path")
    raw_path = Path(raw_path_str) if raw_path_str else None
    return RunResult(
        trajectory=trajectory,
        metrics=metrics,
        raw_transcript_path=raw_path,
        mcp_server_status=attempt_json.get("mcp_server_status"),
    )
