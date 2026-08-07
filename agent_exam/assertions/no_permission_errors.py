"""Assert that no tool call in the trajectory was blocked by the harness's
permission system.

Catches skill-side bugs where the skill tries to do something that would
require user approval at runtime — even if the eval framework has the
command allowlisted. A permission prompt is bad UX for a real user; the
user granting it is worse (may leak secrets). Either way, the skill
shouldn't be trying.

Works on the normalized trajectory — each provider translates its own
permission-signal strings into `ToolCallBlock.status = "permission_denied"`
in its transcript parser.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic import model_validator

from .._models import _StrictModel
from ..schemas import AssertionResult, RunResult, ToolCallBlock

if TYPE_CHECKING:
    from pathlib import Path


class NoPermissionErrorsConfig(_StrictModel):
    """`no_permission_errors:` — no tunables.

    Accepts the bare form (config is `None` in YAML) and the empty
    mapping; anything else (a stray option the author thinks does
    something) is rejected by `extra="forbid"`.
    """

    @model_validator(mode="before")
    @classmethod
    def _accept_none(cls, v: Any) -> Any:
        # `- no_permission_errors:` in YAML loads as None; normalise so
        # pydantic sees an empty mapping.
        return {} if v is None else v


def check(
    config: NoPermissionErrorsConfig, result: RunResult, cwd: Path
) -> AssertionResult:
    offending: list[dict] = []

    def _walk(turns, parent_turn: str | None = None):
        for i, turn in enumerate(turns):
            turn_label = parent_turn if parent_turn is not None else f"turn {i}"
            for block in turn.content:
                if not isinstance(block, ToolCallBlock):
                    continue
                if block.status == "permission_denied":
                    offending.append(
                        {
                            "turn": turn_label,
                            "name": block.name,
                            "input": block.input,
                            "result_head": (block.result or "")[:200],
                        }
                    )
                if block.subagent:
                    _walk(block.subagent, parent_turn=f"{turn_label} (subagent)")

    _walk(result.trajectory)

    if not offending:
        return AssertionResult(pass_=True, reason="no permission-denied tool calls")

    count = len(offending)
    lines = [f"{count} tool call{'s' if count != 1 else ''} hit permission-denied:"]
    for o in offending:
        inp = o.get("input") or {}
        cmd = None
        if isinstance(inp, dict):
            cmd = inp.get("command") or inp.get("file_path") or inp.get("pattern")
        if not cmd:
            cmd = json.dumps(inp, default=str)
        cmd = str(cmd)
        if len(cmd) > 120:
            cmd = cmd[:120] + "…"
        lines.append(f"  {o.get('turn', '?')}  [{o['name']}] {cmd}")
        result_head = (o.get("result_head") or "").strip()
        if result_head:
            lines.append(f"    {result_head}")
    return AssertionResult(
        pass_=False,
        reason="\n".join(lines),
        details={"offending": offending},
    )
