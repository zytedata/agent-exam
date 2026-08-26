"""Static validation of an eval suite.

Catches the structural problems that today only surface late — after an
agent run, during scoring (a typo'd assertion type or malformed assertion
config) or at attempt-staging time (a missing fixture). Shared by two
callers so they can't drift:

- the **runner** calls it before building the plan and fails fast on any
  FAIL, so a malformed suite never burns tokens;
- **doctor** calls it for every suite and renders the results.

Checks are environment-independent with one exception: fixture hygiene
shells out to `git` to find fixture content that isn't under version
control. It stays silent where git can't be asked, so a project that
doesn't use git is unaffected.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from .errors import UsageError
from .mcp import canonical_tool_name, canonical_tool_server, is_mcp_tool
from .schemas import CheckResult
from .tasks import load_suite, load_suite_config

if TYPE_CHECKING:
    from .config import Config


def _git(fixtures_dir: Path, *args: str) -> subprocess.CompletedProcess | None:
    """Run a git command in *fixtures_dir*; None if git can't be asked."""
    try:
        return subprocess.run(
            ["git", "-C", str(fixtures_dir), *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _git_ignored_files(fixtures_dir: Path, fixtures: list[str]) -> list[str] | None:
    """Every ignored file under *fixtures*, relative to *fixtures_dir*.

    Exact per-file output — deliberately no ``--directory``. That flag
    collapses wholly-untracked directories, and since the collapsed
    directory itself usually isn't ignored it then drops out of an
    ``--ignored`` listing altogether, so a ``.venv`` under a brand-new
    subdirectory would go unreported. :func:`_collapse_ignored` does the
    collapsing instead, and does it safely.

    None when git can't answer — no binary, or not a work tree — which
    the caller treats as "this project doesn't keep its fixtures in git",
    not as a problem.
    """
    proc = _git(
        fixtures_dir,
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "-z",
        "--",
        *fixtures,
    )
    if proc is None or proc.returncode != 0:
        return None
    return [p for p in proc.stdout.split("\0") if p]


def _git_ignored_of(fixtures_dir: Path, paths: list[str]) -> set[str] | None:
    """Subset of *paths* that git ignores in its own right."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(fixtures_dir), "check-ignore", "-z", "--stdin"],
            input="\0".join(paths),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    # 0 = some ignored, 1 = none ignored; anything else is a real error.
    if proc.returncode not in (0, 1):
        return None
    return {p for p in proc.stdout.split("\0") if p}


def _collapse_ignored(fixtures_dir: Path, ignored_files: list[str]) -> list[str] | None:
    """Smallest set of paths whose removal clears every ignored file.

    Walks down from each fixture root and stops at the first ancestor git
    ignores in its own right — everything beneath an ignored directory is
    ignored too, so deleting it is both complete and safe. A file whose
    ancestors are all unignored reports as itself.

    Collapsing any directory all of whose *known* ignored files were listed
    would be unsafe: that directory may also hold committed files, or
    untracked fixture files someone is still writing, and ``rm -rf`` on it
    would take them out.
    """
    minimal: set[str] = set()
    remaining = set(ignored_files)
    depth = 1
    while remaining:
        prefixes = sorted({"/".join(p.split("/")[:depth]) for p in remaining})
        ignored_prefixes = _git_ignored_of(fixtures_dir, prefixes)
        if ignored_prefixes is None:
            return None
        minimal |= ignored_prefixes
        remaining = {
            p
            for p in remaining
            if not any(
                p == pref or p.startswith(f"{pref}/") for pref in ignored_prefixes
            )
        }
        depth += 1
    return sorted(minimal)


def _listing(root: Path, fixtures_dir: Path, paths: list[str]) -> str:
    """Offending *paths*, one indented line each, relative to *root*.

    Project-root-relative like ``git status``, and short enough to read —
    but still pasteable into ``rm -rf``, because the project root is where
    `agent-exam` runs from. Absolute for anything outside the root, which
    shouldn't happen but shouldn't print a pile of ``../`` if it does.
    """
    lines = []
    for path in paths:
        absolute = fixtures_dir / path
        try:
            lines.append(f"  {absolute.relative_to(root)}")
        except ValueError:
            lines.append(f"  {absolute}")
    return "\n".join(lines)


def _fileless_dirs(fixtures_dir: Path, fixtures: set[str]) -> list[str]:
    """Topmost directories in *fixtures* whose subtree holds no files.

    Git has no way to record a directory, only files in one, so an empty
    directory cannot be committed and won't exist on a fresh checkout —
    yet it *is* staged into the attempt cwd, where the agent sees it and
    may write into it. Same reproducibility failure as an ignored file,
    and invisible to ``git status`` in the same way.

    Reports the shallowest offender: a chain like ``out/logs/`` (both
    fileless) is one problem, fixed by one :file:`.gitkeep` or one delete.

    Needs no git itself, but the caller only runs it for projects whose
    fixtures are in git — git is what can't represent an empty directory,
    so without it there is nothing being lost.
    """
    tops: list[str] = []
    for fixture in sorted(fixtures):
        fixture_root = fixtures_dir / fixture
        has_file: dict[str, bool] = {}
        for dirpath, dirnames, filenames in os.walk(fixture_root, topdown=False):
            has_file[dirpath] = bool(filenames) or any(
                has_file.get(str(Path(dirpath) / sub), False) for sub in dirnames
            )
        for dirpath, dirnames, _ in os.walk(fixture_root):
            if not has_file.get(dirpath, True):
                tops.append(str(Path(dirpath).relative_to(fixtures_dir)))
                dirnames.clear()  # its children are the same one problem
    return sorted(tops)


def _fixture_hygiene(
    suite: str, root: Path, fixtures_dir: Path, fixtures: set[str]
) -> list[CheckResult]:
    """Refuse to run when a fixture holds anything a fresh checkout won't.

    Fixtures must be fully version-controlled — an eval whose input only
    exists on one machine is not the same eval anywhere else — so these
    are FAILs and the runner stops before spending tokens. Both kinds are
    invisible to ``git status``, which is how an 85 MB ``.venv`` once sat
    in a fixture unnoticed.

    Two separate findings, because they have different fixes:

    - **Files git ignores** — remove, or stop ignoring.
    - **Empty directories** — remove, or :file:`.gitkeep` (see
      :func:`_fileless_dirs`). An ignored directory reports here rather
      than above, since it holds no files to list; adding the
      :file:`.gitkeep` then trips the ignored-files check, which is the
      nudge to narrow the ignore rule.

    Deliberately scoped to *ignored* paths, not merely untracked ones: a
    fixture being authored is untracked until it's committed, and runs
    routinely happen before that commit.

    **Silent when git can't be asked** — no binary, or the evals dir isn't
    inside a work tree. Both findings only mean anything for a project
    whose fixtures live in git: "won't exist on a fresh checkout" needs a
    checkout, and an empty directory is only unrepresentable because git
    is what can't represent it. A project that doesn't use git (or any
    VCS) is not misconfigured, so there's nothing to report — and no
    name-matching stand-in either, since guessing at the usual suspects
    would pass a fixture whose problem has any other shape.
    """
    ignored_files = (
        _git_ignored_files(fixtures_dir, sorted(fixtures)) if fixtures else []
    )
    if ignored_files is None:
        return []

    offenders = _collapse_ignored(fixtures_dir, ignored_files) if ignored_files else []
    if offenders is None:
        return []

    # Named for the problem, not the expectation: these only ever render
    # when they fail (clean fixtures return no CheckResult at all), so
    # "fixtures have …" reads as the finding rather than inverting it.
    results: list[CheckResult] = []

    empty = _fileless_dirs(fixtures_dir, fixtures)
    if empty:
        results.append(
            CheckResult(
                name=f"{suite}: fixtures have empty directories",
                status="FAIL",
                hint=(
                    "git can't commit an empty directory, so it won't exist on "
                    "a fresh checkout — remove it, or add a .gitkeep inside:\n"
                    f"{_listing(root, fixtures_dir, empty)}"
                ),
            )
        )

    if offenders:
        results.append(
            CheckResult(
                name=f"{suite}: fixtures have git-ignored files",
                status="FAIL",
                hint=(
                    "remove them, or stop ignoring them:\n"
                    f"{_listing(root, fixtures_dir, offenders)}"
                ),
            )
        )

    return results


def validate_suite(
    cfg: Config, suite: str, task_filter: str | None = None
) -> list[CheckResult]:
    """Structural checks for a suite: task files parse, every referenced
    fixture exists on disk and holds nothing git ignores, every
    `concurrency_group` and every tag is declared, and `suite.yml` (if
    present) is well-formed.

    Returns a list of CheckResult. A FAIL is fatal for a run (the runner
    raises); OK is informational and only `doctor` renders it. No check
    here returns WARN today — a run reads FAILs only, so an advisory
    finding would need the runner taught to print it. `task_filter`
    narrows the check to a single task, matching `load_suite` — so a
    single-task run isn't failed by an unrelated sibling task being
    malformed.
    """
    results: list[CheckResult] = []

    # suite.yml is optional, but a malformed one (or a typo'd key) should
    # surface here rather than only when the runner loads it.
    try:
        load_suite_config(cfg.evals_dir, suite)
    except UsageError as exc:
        results.append(
            CheckResult(name=f"{suite}: suite.yml", status="FAIL", hint=str(exc))
        )

    try:
        tasks = load_suite(cfg.evals_dir, suite, task_filter=task_filter)
    except UsageError as exc:
        # Parse / schema errors — unknown assertion types, malformed env
        # blocks, bad trigger shape, etc.
        results.append(
            CheckResult(name=f"{suite}: task files parse", status="FAIL", hint=str(exc))
        )
        return results

    # Count source files, not tasks: a `kind: trigger` file fans out into
    # one task per case, but parsing/validation happens per file — there's
    # no interesting per-case work, so per-task counts would just mislead.
    file_count = len({t.source_path for t in tasks})
    results.append(
        CheckResult(
            name=f"{suite}: task files parse",
            status="OK",
            hint=f"{file_count} file(s)",
        )
    )

    fixtures_dir = cfg.evals_dir / "fixtures"
    seen_fixtures: set[str] = set()
    missing: set[str] = set()
    for t in tasks:
        if t.fixture and t.fixture not in seen_fixtures:
            seen_fixtures.add(t.fixture)
            if not (fixtures_dir / t.fixture).is_dir():
                missing.add(t.fixture)
    if missing:
        results.append(
            CheckResult(
                name=f"{suite}: fixtures exist",
                status="FAIL",
                hint=f"missing under {fixtures_dir}: {', '.join(sorted(missing))}",
            )
        )
    elif seen_fixtures:
        results.append(
            CheckResult(
                name=f"{suite}: fixtures exist",
                status="OK",
                hint=f"{len(seen_fixtures)} fixture(s)",
            )
        )

    results.extend(
        _fixture_hygiene(suite, cfg.project_root, fixtures_dir, seen_fixtures - missing)
    )

    # Every `concurrency_group` a task tags must be declared in
    # config.yaml. The pool enforces this too, but only mid-run — surface
    # it here so doctor and the runner catch it before any subprocess.
    declared = set(cfg.concurrency_groups)
    undeclared = sorted(
        {
            t.concurrency_group
            for t in tasks
            if t.concurrency_group and t.concurrency_group not in declared
        }
    )
    if undeclared:
        results.append(
            CheckResult(
                name=f"{suite}: concurrency groups declared",
                status="FAIL",
                hint=(
                    f"not in config.yaml concurrency_groups: {', '.join(undeclared)}"
                ),
            )
        )

    # Same for tags. A misspelled tag excludes nothing, and one misspelled
    # into another declared name takes the task out of every wide run —
    # both silent, so refuse the run instead.
    undeclared_tags = sorted(
        {tag for t in tasks for tag in t.tags if tag not in cfg.tags}
    )
    if undeclared_tags:
        results.append(
            CheckResult(
                name=f"{suite}: tags declared",
                status="FAIL",
                hint=f"not in config.yaml tags: {', '.join(undeclared_tags)}",
            )
        )

    # A task can only attach servers config.yaml declares. A typo would
    # otherwise leave the agent quietly short of the tools the task is
    # about, which reads as the skill failing.
    undeclared_servers = sorted(
        {
            name
            for t in tasks
            for name in (t.mcp_servers or ())
            if name not in cfg.mcp_servers
        }
    )
    if undeclared_servers:
        results.append(
            CheckResult(
                name=f"{suite}: mcp servers declared",
                status="FAIL",
                hint=(
                    f"not in config.yaml mcp_servers: {', '.join(undeclared_servers)}"
                ),
            )
        )

    # A trigger aimed at a tool of a server the task does not attach can never
    # fire, so every one of its positive cases would fail as a routing miss. A
    # task that names no subset attaches everything config.yaml declares.
    unreachable = sorted(
        {
            t.target_tool
            for t in tasks
            if t.target_tool
            and is_mcp_tool(t.target_tool)
            and canonical_tool_server(t.target_tool)
            not in (cfg.mcp_servers if t.mcp_servers is None else t.mcp_servers)
        }
    )
    if unreachable:
        results.append(
            CheckResult(
                name=f"{suite}: trigger tools reachable",
                status="FAIL",
                hint=(
                    "no attached mcp_servers entry serves: " + ", ".join(unreachable)
                ),
            )
        )

    # A tool: value that isn't already canonical but starts with a declared
    # server's name reads as a typo of the mcp__<server>__<tool> spelling
    # the docs ask for, not a native tool target — left as written it can
    # never match a canonicalized trajectory, so every positive case would
    # silently fail and every negative case would silently pass.
    miscanonical = sorted(
        {
            t.target_tool
            for t in tasks
            if t.target_tool
            and not is_mcp_tool(t.target_tool)
            and canonical_tool_name(
                t.target_tool,
                cfg.mcp_servers if t.mcp_servers is None else t.mcp_servers,
            )
            != t.target_tool
        }
    )
    if miscanonical:
        results.append(
            CheckResult(
                name=f"{suite}: trigger tool canonical",
                status="FAIL",
                hint=(
                    "looks like a non-canonical mcp__<server>__<tool> spelling: "
                    + ", ".join(miscanonical)
                ),
            )
        )

    return results
