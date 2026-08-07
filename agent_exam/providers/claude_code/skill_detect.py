"""Skill-invocation detection shared by the transcript walker and stream reader.

Two regexes cover both paths Claude Code surfaces a skill invocation:

- `Skill` tool call — input `{skill: "<name>"}` (some variants use `name`).
- `Read` tool call — input `{file_path: ".../skills/<name>/SKILL.md"}`.

The transcript walker (providers/transcripts.py) inspects finalized
ToolCallBlock instances; the stream reader (providers/stream_parser.py)
runs the same regexes against partial `input_json_delta` fragments so
`stop_on_first_skill` can kill the subprocess before the tool executes.
"""

from __future__ import annotations

import re

_SKILL_TOOL_RE = re.compile(r'"(?:skill|name)"\s*:\s*"([^"]+)"')
_SKILL_MD_RE = re.compile(r'skills/([^/"]+)/SKILL\.md')


def detect_from_partial(tool_name: str, accumulated: str) -> str | None:
    """Return the skill slug if `accumulated` (partial JSON) reveals one.

    Returns None on no match; caller keeps accumulating and re-checks.
    """
    if tool_name == "Skill":
        m = _SKILL_TOOL_RE.search(accumulated)
        return m.group(1) if m else None
    if tool_name == "Read":
        m = _SKILL_MD_RE.search(accumulated)
        return m.group(1) if m else None
    return None


def detect_from_input(
    tool_name: str, input_dict: dict
) -> tuple[str | None, str | None]:
    """Return (skill_name, trigger_kind) for a finalized tool input dict, or (None, None)."""
    if not isinstance(input_dict, dict):
        return None, None
    if tool_name == "Skill":
        name = input_dict.get("skill") or input_dict.get("name")
        if name:
            return str(name), "skill_tool"
    elif tool_name == "Read":
        path = str(input_dict.get("file_path") or "")
        m = _SKILL_MD_RE.search(path)
        if m:
            return m.group(1), "skill_md_read"
    return None, None
