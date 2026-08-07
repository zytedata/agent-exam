from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from ..schemas import CheckResult

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def discover_skills(
    skills_dirs: list[Path] | None,
    exclude: frozenset[str] = frozenset(),
) -> list[tuple[str, Path]]:
    """Return the skills discovered under *skills_dirs*.

    Each entry is a ``(skill_name, skill_dir)`` tuple for directories
    that contain a ``SKILL.md`` file.  Entries across all directories are
    de-duplicated by skill name (earlier directories win).  The result is
    sorted by skill name.

    This is provider-agnostic and can be used both for staging and for
    preflight checks that need to know which skills are under test.
    """
    found: dict[str, Path] = {}
    if not skills_dirs:
        return []
    for src in skills_dirs:
        if not src.is_dir():
            continue
        for entry in src.iterdir():
            if not entry.is_dir():
                continue
            if entry.name in exclude:
                continue
            if not (entry / "SKILL.md").is_file():
                continue
            if entry.name not in found:
                found[entry.name] = entry
    return sorted(found.items())


def check_skill_clashes(
    global_skills: list[str],
    staged_names: set[str],
    *,
    check_name: str = "global skills",
    normalize: Callable[[str], str] | None = None,
) -> CheckResult:
    """Compare *global_skills* against *staged_names* and report clashes.

    ``normalize`` is applied to each global skill name before comparing
    with *staged_names*.  The original (un-normalized) names are shown
    in warning messages so the user knows exactly which global skill
    to remove or rename.
    """
    if not global_skills:
        return CheckResult(
            name=check_name,
            status="OK",
            hint="no global skills loaded",
        )

    if normalize is None:

        def normalize(s: str) -> str:
            return s

    clashes: set[str] = set()
    for full in global_skills:
        bare = normalize(full)
        if bare in staged_names:
            clashes.add(full)

    if clashes:
        return CheckResult(
            name=check_name,
            status="WARN",
            hint=(
                f"{', '.join(sorted(clashes))} loaded globally and clash with staged skills"
            ),
        )

    return CheckResult(
        name=check_name,
        status="OK",
        hint=(
            f"{len(global_skills)} global skill(s) present but none clash "
            "with staged skills"
        ),
    )


def check_global_skills_against_staged(
    global_skills: list[str],
    cfg,
    provider_name: str,
    *,
    check_name: str = "global skills",
    normalize: Callable[[str], str] | None = None,
) -> CheckResult:
    """Return a CheckResult comparing *global_skills* against staged skills.

    When *cfg* is ``None`` and skills are present, returns a generic WARN.
    When *cfg* is available, uses `check_skill_clashes` for precise
    comparison against the staged harness skills.
    """
    if not global_skills:
        return CheckResult(
            name=check_name,
            status="OK",
            hint="no global skills loaded",
        )

    if cfg is None or getattr(cfg, "skills_dirs", None) is None:
        return CheckResult(
            name=check_name,
            status="WARN",
            hint=(
                f"{', '.join(sorted(set(global_skills)))} loaded globally — "
                "skipping staged-skill clash check because skills_dirs is not "
                "configured"
            ),
        )

    skills_dirs = cfg.skills_dirs or []
    if not skills_dirs:
        return CheckResult(
            name=check_name,
            status="WARN",
            hint=(
                f"{', '.join(sorted(set(global_skills)))} loaded globally; "
                "staged skills are not configured yet, so clash check was skipped"
            ),
        )

    staged_names = {name for name, _ in discover_skills(skills_dirs)}
    return check_skill_clashes(
        global_skills,
        staged_names,
        check_name=check_name,
        normalize=normalize,
    )


def stage_skills_into(
    run_dir: Path,
    skills_dirs: list[Path] | None,
    target_rel_path: str,
    exclude: frozenset[str] = frozenset(),
) -> list[str]:
    """Populate `<run_dir>/<target_rel_path>/<name>` with a real directory copy
    of every skill subdir under each entry in `skills_dirs`, minus anything in
    `exclude`.

    Directory copies rather than symlinks: some host agents' bash sandboxes
    (notably Copilot CLI's) cannot traverse directory symlinks to paths outside
    the working tree, so bash/view/glob fail silently on symlinked skill dirs.
    Real copies are fully accessible to those tools, and staging the same way
    across every provider keeps behaviour uniform and easy to reason about.

    `target_rel_path` is a relative path like `.claude/skills`,
    `.opencode/skills`, or `.github/skills` — each provider picks its own
    walk-up discovery path, which subprocesses under `run_dir` (at any depth)
    pick up.

    Returns the list of skill names actually staged, sorted.

    `dirs_exist_ok=True` keeps concurrent calls safe for parallel trigger-eval
    workers that all stage into the same run_tmp_root: every worker copies from
    the same source, so racing writes produce identical files.

    If `skills_dirs` is empty, no directory is created and `[]` is returned.
    """
    skills = discover_skills(skills_dirs, exclude)
    if not skills:
        return []
    target = run_dir / target_rel_path
    target.mkdir(parents=True, exist_ok=True)
    for name, src_path in skills:
        shutil.copytree(src_path.resolve(), target / name, dirs_exist_ok=True)
    return [name for name, _ in skills]
