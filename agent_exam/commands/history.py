"""`agent-exam history <scope>` — trends across runs.

Scope shapes (mirroring `show`, with the run-id stripped):
- `<suite>::<task>` — per-run row showing this task's latest attempt in each run.
- `<suite>`         — per-run row aggregating all tasks in this suite.

(`runs` covers the empty scope — every run, no suite filter.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from ..run_modes import compact_mode, is_reality_check
from ._format import (
    fmt_cost,
    fmt_ctx,
    fmt_iso_age,
    fmt_pass_ratio,
    fmt_wall,
    render_table,
)
from ._loader import list_runs, load_run, parse_task_spec

if TYPE_CHECKING:
    from pathlib import Path


def run(
    evals_dir: Path, scope_spec: str, limit: int = 10, all_runs: bool = False
) -> int:
    suite, task = parse_task_spec(scope_spec)
    run_ids = list_runs(evals_dir)
    if not run_ids:
        click.echo(f"no runs in {evals_dir / 'runs'}.")
        return 0

    if task is not None:
        return _history_task(evals_dir, suite, task, run_ids, limit, all_runs)
    return _history_suite(evals_dir, suite, run_ids, limit, all_runs)


# --- per-task history -------------------------------------------------------


def _history_task(
    evals_dir: Path,
    suite: str,
    task: str,
    run_ids: list[str],
    limit: int,
    all_runs: bool,
) -> int:
    rows: list[list[str]] = []
    reality_check_count = 0
    for rid in reversed(run_ids):
        try:
            data = load_run(evals_dir, rid)
        except Exception:  # noqa: S112 -- a half-written or corrupt run dir is skipped, not fatal
            continue
        latest = data.latest_for_task(suite, task)
        if latest is None:
            continue
        meta = data.load_attempt_metadata(suite, task, latest["attempt"])
        metrics = (meta or {}).get("metrics") or {}
        mode = data.run_json.get("run_mode", "run")
        if is_reality_check(mode):
            reality_check_count += 1
        rows.append(
            [
                rid,
                compact_mode(mode),
                f"attempt-{latest['attempt']}",
                fmt_pass_ratio(latest),
                fmt_cost(metrics.get("cost_usd")),
                fmt_wall(metrics.get("wall_time_seconds")),
                fmt_ctx(metrics.get("peak_context")),
                fmt_iso_age(data.run_json.get("started_at", "")),
            ]
        )
        if not all_runs and len(rows) >= limit:
            break

    if not rows:
        click.echo(f"no runs covered {suite}::{task} yet.")
        return 0

    header = f"History of {suite}::{task} (last {len(rows)} run{'s' if len(rows) != 1 else ''})"
    header = _append_reality_check_note(header, reality_check_count)
    click.echo(header)
    click.echo("")
    click.echo(
        render_table(
            rows, ["RUN", "MODE", "ATTEMPT", "PASSED", "COST", "WALL", "CTX", "AGE"]
        )
    )
    return 0


# --- per-suite history ------------------------------------------------------


def _history_suite(
    evals_dir: Path, suite: str, run_ids: list[str], limit: int, all_runs: bool
) -> int:
    """One row per run: aggregate across every task in the suite that the
    run's latest report covered. Columns surface pass/fail task counts,
    total cost, max wall + peak context (across all of that run's attempts
    in this suite).
    """
    rows: list[list[str]] = []
    reality_check_count = 0
    for rid in reversed(run_ids):
        try:
            data = load_run(evals_dir, rid)
        except Exception:  # noqa: S112 -- a half-written or corrupt run dir is skipped, not fatal
            continue
        report = data.latest_report
        if report is None:
            continue
        suite_attempts = [a for a in report.attempts if a["suite"] == suite]
        if not suite_attempts:
            continue

        mode = data.run_json.get("run_mode", "run")
        if is_reality_check(mode):
            reality_check_count += 1

        passed = sum(1 for a in suite_attempts if a.get("verdict") == "pass")
        total = len(suite_attempts)
        tasks_counter = _color_task_counter(passed, total)

        # Aggregate metrics across all attempts in the suite.
        costs, walls, ctxs = [], [], []
        for a in suite_attempts:
            meta = data.load_attempt_metadata(a["suite"], a["task"], a["attempt"])
            metrics = (meta or {}).get("metrics") or {}
            if metrics.get("cost_usd") is not None:
                costs.append(metrics["cost_usd"])
            if metrics.get("wall_time_seconds") is not None:
                walls.append(metrics["wall_time_seconds"])
            if metrics.get("peak_context") is not None:
                ctxs.append(metrics["peak_context"])

        rows.append(
            [
                rid,
                compact_mode(mode),
                tasks_counter,
                fmt_cost(sum(costs) if costs else None),
                fmt_wall(max(walls) if walls else None),
                fmt_ctx(max(ctxs) if ctxs else None),
                fmt_iso_age(data.run_json.get("started_at", "")),
            ]
        )
        if not all_runs and len(rows) >= limit:
            break

    if not rows:
        click.echo(f"no runs covered suite {suite!r} yet.")
        return 0

    header = f"History of {suite} (last {len(rows)} run{'s' if len(rows) != 1 else ''})"
    header = _append_reality_check_note(header, reality_check_count)
    click.echo(header)
    click.echo("")
    click.echo(
        render_table(
            rows,
            [
                "RUN",
                "MODE",
                "TASKS (pass/total)",
                "ΣCOST",
                "MAX WALL",
                "MAX CTX",
                "AGE",
            ],
        )
    )
    return 0


# --- shared helpers ---------------------------------------------------------


def _color_task_counter(passed: int, total: int) -> str:
    """Colored `<passed>/<total>` — same scheme as `fmt_pass_ratio` but at
    the attempt-count level, not assertion-count.
    """
    text = f"{passed}/{total}"
    if total == 0 or passed == 0:
        return click.style(text, fg="red")
    if passed == total:
        return click.style(text, fg="green")
    return click.style(text, fg="yellow")


def _append_reality_check_note(header: str, count: int) -> str:
    if not count:
        return header
    return (
        header
        + f" — {count} reality-check run{'s' if count != 1 else ''} "
        + "(verdicts informational, metrics not comparable)"
    )
