"""`agent-exam diff <a> <b>` — compare two reports."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from ..errors import UsageError
from ..report_compare import AttemptKey, compare_reports
from ._format import THRESHOLDS, fmt_cost, fmt_ctx, fmt_wall
from ._loader import RunData, load_run

if TYPE_CHECKING:
    from pathlib import Path


def run(
    evals_dir: Path,
    a_spec: str,
    b_spec: str,
    report_a: str | None = None,
    report_b: str | None = None,
    scope: str | None = None,
) -> int:
    a_run = load_run(evals_dir, a_spec)
    b_run = load_run(evals_dir, b_spec)
    ra = a_run.report_for(report_a)
    rb = b_run.report_for(report_b)
    if ra is None:
        raise UsageError(f"no report in run {a_run.run_id}")
    if rb is None:
        raise UsageError(f"no report in run {b_run.run_id}")

    a_attempts, a_meta = _collect_attempts(a_run, ra, scope)
    b_attempts, b_meta = _collect_attempts(b_run, rb, scope)

    result = compare_reports(
        a_attempts,
        b_attempts,
        before_attempt_meta=a_meta,
        after_attempt_meta=b_meta,
    )

    a_mode = a_run.run_json.get("run_mode", "run")
    b_mode = b_run.run_json.get("run_mode", "run")
    click.echo(
        f"Diff: {a_run.run_id} ({ra.timestamp}) → {b_run.run_id} ({rb.timestamp})"
    )
    click.echo(f"Mode: {a_mode} → {b_mode}")
    if a_mode != b_mode:
        click.echo(
            click.style(
                "WARNING: comparing across modes — different skill sets were "
                "loaded. Metric deltas and verdict flips may reflect the mode "
                "change rather than the skill's behavior.",
                fg="yellow",
            )
        )
    if scope:
        click.echo(f"Scope: {scope}")
    click.echo("")

    if result.verdict_changes:
        click.echo(f"Verdict changes ({len(result.verdict_changes)}):")
        for c in result.verdict_changes:
            click.echo(f"  {c.attempt}   {c.before.upper()} → {c.after.upper()}")
        click.echo("")
    else:
        click.echo("Verdict changes: none")
        click.echo("")

    if result.metric_deltas:
        click.echo(f"Metric deltas ({len(result.metric_deltas)}):")
        for d in result.metric_deltas:
            click.echo(
                f"  {d.attempt}   {_metric_label(d.metric)} "
                f"{_fmt_metric(d.metric, d.before)} → {_fmt_metric(d.metric, d.after)}  "
                f"({_pct(d.delta_pct)}, threshold ±{int(THRESHOLDS[d.metric] * 100)}%)"
            )
        click.echo("")
    else:
        click.echo("Metric deltas: none above threshold")
        click.echo("")

    if result.grader_changes:
        click.echo(f"Grader changes ({len(result.grader_changes)}):")
        for g in result.grader_changes:
            click.echo(f"  {g.label}  [{g.kind}] {g.assertion_key}")
        click.echo("")
    else:
        click.echo("Grader changes: none")
    return 0


def _collect_attempts(data: RunData, report, scope: str | None):
    attempts = list(report.attempts)
    if scope:
        suite, _, task = scope.partition("::")
        attempts = [
            a
            for a in attempts
            if a["suite"] == suite and (not task or a["task"] == task)
        ]
    meta: dict[AttemptKey, dict] = {}
    for a in attempts:
        m = data.load_attempt_metadata(a["suite"], a["task"], a["attempt"])
        if m is not None:
            meta[AttemptKey(a["suite"], a["task"], a["attempt"])] = m
    return attempts, meta


def _metric_label(metric: str) -> str:
    return {"cost_usd": "cost", "peak_context": "ctx", "wall_time_seconds": "wall"}.get(
        metric, metric
    )


def _fmt_metric(metric: str, val: float | None) -> str:
    if val is None:
        return "—"
    if metric == "cost_usd":
        return fmt_cost(val)
    if metric == "wall_time_seconds":
        return fmt_wall(val)
    if metric == "peak_context":
        return fmt_ctx(int(val))
    return str(val)


def _pct(delta: float) -> str:
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta * 100:.0f}%"
