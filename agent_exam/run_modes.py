"""The `run_mode` values carried in `run.json`, and their shared rendering.

- `run` — normal: every skill under `skills_dirs` is staged.
- `without-skill` — `--without-skill`: the suite's `evaluated_skills` are
  dropped from the staged bundle (the rest stay).
- `no-skills` — `--no-skills`: nothing is staged at all, so the harness
  runs as a plain agent.

The latter two are *reality checks*: trigger tasks are skipped (no skill
to fire), verdicts are informational so the run exits 0 regardless, and
the inspection commands keep them out of lift-style comparisons against
normal runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

NORMAL = "run"
WITHOUT_SKILL = "without-skill"
NO_SKILLS = "no-skills"

_REALITY_CHECK = frozenset({WITHOUT_SKILL, NO_SKILLS})

_COMPACT = {WITHOUT_SKILL: "w/o-skill", NO_SKILLS: "no-skills"}


def is_reality_check(run_mode: str) -> bool:
    """True for the modes whose verdicts are informational."""
    return run_mode in _REALITY_CHECK


def compact_mode(run_mode: str) -> str:
    """Short mode tag for the history tables."""
    return _COMPACT.get(run_mode, run_mode)


def banner_lines(run_mode: str, skills_excluded: Sequence[str]) -> list[str]:
    """Header lines announcing a reality-check run; `[]` for normal runs.

    Shared by the runner (at run start) and `show` (when inspecting the
    archived run) so both describe the run the same way.
    """
    if not is_reality_check(run_mode):
        return []
    if run_mode == NO_SKILLS:
        lines = ["REALITY CHECK (no skills loaded) — verdicts informational"]
        if skills_excluded:
            lines.append(f"All {len(skills_excluded)} skills excluded")
        return lines
    lines = ["REALITY CHECK — verdicts informational"]
    excluded = list(skills_excluded)
    if len(excluded) == 1:
        lines.append(f"Skill excluded: {excluded[0]}")
    elif excluded:
        lines.append(f"Skills excluded: {', '.join(excluded)}")
    return lines
