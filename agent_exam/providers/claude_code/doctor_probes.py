"""Claude-Code-specific doctor probes evaluated against the round-trip transcript.

Moved out of `commands/doctor.py` so the generic doctor orchestrator
doesn't import Claude-specific modules. `ClaudeCodeProvider.probe_checks()`
calls these; other providers implement their own equivalents if needed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ...schemas import CheckResult
from .blocked_plugins import blocked_skills_in_listing


def hermetic_check(transcript_path: Path | None) -> CheckResult:
    """Scan the probe transcript for signs that user-level context leaked in.

    Claude Code in `-p` mode does NOT inject user-level auto-memory or
    CLAUDE.md walk-up content into subprocess sessions (verified
    empirically for 2.1.119). This check guards against future
    regressions:

    - Attachment types that would carry memory (`auto_memory`,
      `user_memory`, `claude_md`, `memory`).
    - A distinctive sentinel phrase pulled from the caller's
      most-recent `~/.claude/projects/*/memory/*.md` file.
    """
    if transcript_path is None or not transcript_path.exists():
        return CheckResult(
            name="hermetic (no memory leak)",
            status="WARN",
            hint="transcript path missing — skipped",
        )

    offenders: list[str] = []
    leaky_attachment_types = {"auto_memory", "user_memory", "claude_md", "memory"}
    try:
        with transcript_path.open() as fh:
            for line in fh:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("type") != "attachment":
                    continue
                at = entry.get("attachment") or {}
                at_type = at.get("type") if isinstance(at, dict) else None
                if at_type in leaky_attachment_types:
                    offenders.append(f"attachment type: {at_type}")
    except OSError:
        return CheckResult(
            name="hermetic (no memory leak)",
            status="WARN",
            hint=f"could not read transcript {transcript_path}",
        )

    sentinel = _pick_memory_sentinel()
    if sentinel:
        phrase, source = sentinel
        try:
            blob = transcript_path.read_text(errors="replace")
            if phrase in blob:
                offenders.append(f"auto-memory phrase from {source}")
        except OSError:
            pass

    if offenders:
        return CheckResult(
            name="hermetic (no memory leak)",
            status="WARN",
            hint=(
                f"signs of user-level context leaking into claude -p: "
                f"{'; '.join(offenders)}. Evals may be over-optimistic."
            ),
        )
    return CheckResult(
        name="hermetic (no memory leak)",
        status="OK",
        hint="no auto-memory or CLAUDE.md content in probe transcript",
    )


def blocked_plugins_in_probe(
    transcript_path: Path | None, blocked: list[str]
) -> CheckResult:
    """Scan the probe's skill_listing for any blocklisted `<plugin>:<skill>`.

    Catches cases the settings.json check misses — plugins loaded via
    marketplace cache, or paths not reflected in `enabledPlugins`.
    """
    if not blocked:
        return CheckResult(
            name="blocked plugins in probe",
            status="OK",
            hint="no blocklist configured",
        )
    if transcript_path is None or not transcript_path.exists():
        return CheckResult(
            name="blocked plugins in probe",
            status="WARN",
            hint="transcript path missing — skipped",
        )
    listing = ""
    try:
        for line in transcript_path.open():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("type") != "attachment":
                continue
            att = entry.get("attachment") or {}
            if isinstance(att, dict) and att.get("type") == "skill_listing":
                listing = att.get("content") or ""
                break
    except OSError:
        return CheckResult(
            name="blocked plugins in probe",
            status="WARN",
            hint=f"could not read transcript {transcript_path}",
        )

    hits = blocked_skills_in_listing(listing, blocked)
    if hits:
        return CheckResult(
            name="blocked plugins in probe",
            status="WARN",
            hint=(
                f"{', '.join(hits)} loaded into probe session — "
                "disable the plugin or remove from blocked_plugins if intentional"
            ),
        )
    return CheckResult(
        name="blocked plugins in probe",
        status="OK",
        hint=f"none of {blocked} appeared in skill listing",
    )


def _pick_memory_sentinel() -> tuple[str, str] | None:
    """Pull a distinctive phrase from the caller's auto-memory, or None.

    Walks `~/.claude/projects/*/memory/*.md` for the most-recently-touched
    non-MEMORY.md file and returns a representative line from it.
    """
    memory_root = Path.home() / ".claude" / "projects"
    if not memory_root.is_dir():
        return None
    candidates: list[Path] = []
    for project_dir in memory_root.iterdir():
        mem_dir = project_dir / "memory"
        if not mem_dir.is_dir():
            continue
        for md in mem_dir.glob("*.md"):
            if md.name == "MEMORY.md":
                continue
            candidates.append(md)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for md in candidates:
        try:
            for line in md.read_text().splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith(("---", "#", "- ", "*")):
                    continue
                if len(stripped) < 40:
                    continue
                plain = re.sub(r"[*_`]", "", stripped)
                return plain[:80], md.name
        except OSError:
            continue
    return None
