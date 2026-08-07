"""`agent-exam runs` — list recent runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from ._format import fmt_cost, fmt_iso_age, iso_duration, render_table
from ._loader import list_runs, load_run

if TYPE_CHECKING:
    from pathlib import Path


def run(evals_dir: Path, limit: int = 20) -> int:
    run_ids = list_runs(evals_dir)
    if not run_ids:
        click.echo(
            f"no runs found in {evals_dir / 'runs'}. Try 'uv run agent-exam <suite>' first."
        )
        return 0
    run_ids = list(reversed(run_ids))[:limit]

    header = ["RUN", "STARTED", "MODE", "P/F/K/T", "COST", "DURATION"]
    rows = []
    for rid in run_ids:
        try:
            data = load_run(evals_dir, rid)
        except Exception as exc:  # incomplete run dir — surface but keep going
            rows.append([rid, "", "?", "?", "?", f"(unreadable: {exc})"])
            continue
        pfe = _verdict_counts(data.latest_report.attempts if data.latest_report else [])
        cost = _total_cost(data)
        dur = iso_duration(
            data.run_json.get("started_at", ""), data.run_json.get("finished_at", "")
        )
        rows.append(
            [
                rid,
                fmt_iso_age(data.run_json.get("started_at", "")),
                data.run_json.get("run_mode", "?"),
                f"{pfe['pass']}/{pfe['fail']}/{pfe['known']}/{pfe['other']}",
                fmt_cost(cost),
                _fmt_dur(dur),
            ]
        )
    click.echo(render_table(rows, header))
    return 0


def _verdict_counts(attempts: list[dict]) -> dict:
    """Tally into four buckets.

    P: clean passes. F: real failures (gating). K: known_issue +
    unexpected_pass (informational, doesn't gate the suite). T: timeout /
    error / unknown.
    """
    counts = {"pass": 0, "fail": 0, "known": 0, "other": 0}
    for a in attempts:
        v = a.get("verdict")
        if v == "pass":
            counts["pass"] += 1
        elif v == "fail":
            counts["fail"] += 1
        elif v in ("known_issue", "unexpected_pass"):
            counts["known"] += 1
        else:
            counts["other"] += 1
    return counts


def _total_cost(data) -> float | None:
    total = 0.0
    found = False
    if data.latest_report is None:
        return None
    for entry in data.latest_report.attempts:
        meta = data.load_attempt_metadata(
            entry["suite"], entry["task"], entry["attempt"]
        )
        if not meta:
            continue
        c = (meta.get("metrics") or {}).get("cost_usd")
        if c is not None:
            total += float(c)
            found = True
    return total if found else None


def _fmt_dur(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"
