"""Detect blocked Claude Code plugins that would contaminate eval runs.

Claude Code loads user-enabled plugins from `~/.claude/settings.json` into
every `claude -p` subprocess, *additively* — on top of any skills the
framework stages under `.claude/skills/`. If one of those user-global
plugins ships skills we're trying to evaluate under controlled
conditions, the eval isn't controlled anymore: both `<plugin>:<skill>`
and the staged `<skill>` load, and `--without-skill` can't touch the
user-global copy.

This module has two detectors:

- `enabled_blocked_in_settings(settings_path, blocked)` — static check:
  parse the user's settings.json, return which blocked plugins are
  enabled. Cheap, runs at run start.
- `blocked_skills_in_listing(skill_listing, blocked)` — dynamic check:
  parse a probe's `skill_listing` attachment content, return which
  individual `<plugin>:<skill>` entries belong to a blocked plugin. Run
  by doctor against the round-trip probe transcript.
"""

from __future__ import annotations

import json
from pathlib import Path


def enabled_blocked_in_settings(settings_path: Path, blocked: list[str]) -> list[str]:
    """Return the subset of `blocked` plugin names that are enabled.

    `settings.json`'s `enabledPlugins` keys are `<plugin>@<marketplace>`;
    we split on `@` to get the plugin name. Missing file / malformed
    JSON / missing key → empty list (nothing to warn about).
    """
    if not blocked:
        return []
    try:
        data = json.loads(Path(settings_path).read_text())
    except (OSError, json.JSONDecodeError):
        return []
    enabled = data.get("enabledPlugins") or {}
    if not isinstance(enabled, dict):
        return []
    blocked_set = set(blocked)
    found: list[str] = []
    for key, value in enabled.items():
        if not value:
            continue
        name = str(key).split("@", 1)[0]
        if name in blocked_set:
            found.append(name)
    return sorted(set(found))


def blocked_skills_in_listing(skill_listing_text: str, blocked: list[str]) -> list[str]:
    """Return `<plugin>:<skill>` entries from a skill_listing attachment
    whose plugin name is in `blocked`.

    The `skill_listing` attachment lists one skill per line as
    `- <name>: <description>`. Plugin-loaded skills are prefixed
    `<plugin>:<skill>`; `.claude/skills/` walk-up-loaded skills are
    unprefixed.
    """
    if not blocked or not skill_listing_text:
        return []
    blocked_set = set(blocked)
    hits: list[str] = []
    for raw_line in skill_listing_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("- "):
            continue
        body = line[2:]
        # "<name>:<description>" — the name is up to the first ": ".
        name_part = body.split(":", 2)
        if len(name_part) < 2:
            continue
        # A plugin-loaded skill has a second colon: "plugin:skill: desc".
        # A bare skill has: "skill: desc" — name_part[0] is the whole name.
        if len(name_part) >= 3:
            plugin, _skill, _rest = name_part[0], name_part[1], name_part[2]
            if plugin in blocked_set:
                hits.append(f"{plugin}:{_skill}")
    return sorted(set(hits))
