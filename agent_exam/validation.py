"""Static, environment-independent validation of an eval suite.

Catches the structural problems that today only surface late — after an
agent run, during scoring (a typo'd assertion type or malformed assertion
config) or at attempt-staging time (a missing fixture). Shared by two
callers so they can't drift:

- the **runner** calls it before building the plan and fails fast on any
  FAIL, so a malformed suite never burns tokens;
- **doctor** calls it for every suite and renders the results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .errors import UsageError
from .schemas import CheckResult
from .tasks import load_suite, load_suite_config

if TYPE_CHECKING:
    from .config import Config


def validate_suite(
    cfg: Config, suite: str, task_filter: str | None = None
) -> list[CheckResult]:
    """Structural checks for a suite: task files parse, every referenced
    fixture exists on disk, every `concurrency_group` is declared, and
    `suite.yml` (if present) is well-formed.

    Returns a list of CheckResult. A FAIL is fatal for a run (the runner
    raises); WARN/OK are informational. `task_filter` narrows the check
    to a single task, matching `load_suite` — so a single-task run isn't
    failed by an unrelated sibling task being malformed.
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

    return results
