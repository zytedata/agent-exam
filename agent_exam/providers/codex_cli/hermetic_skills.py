from __future__ import annotations

from typing import TYPE_CHECKING

from ..skill_staging import stage_skills_into as _stage_skills_into

if TYPE_CHECKING:
    from pathlib import Path


def stage_skills_into(
    run_dir: Path,
    skills_dirs: list[Path],
    exclude: frozenset[str] = frozenset(),
) -> list[str]:
    """Stage skills under `<run_dir>/.agents/skills/<name>` for Codex.

    Codex discovers project skills by walking up from the current working
    directory and reading `.agents/skills/<name>/SKILL.md`.
    """
    return _stage_skills_into(run_dir, skills_dirs, ".agents/skills", exclude=exclude)
