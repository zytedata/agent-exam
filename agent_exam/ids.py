from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .errors import UsageError

if TYPE_CHECKING:
    from pathlib import Path


def new_run_id(runs_dir: Path, now: datetime | None = None) -> str:
    """Generate a run id like `run-YYYY-MM-DD-HHMM`, adding `-2`, `-3`, ... on collision.

    Uses UTC so that daylight-saving transitions can't cause a later run to
    sort before an earlier one.
    """
    now = now or datetime.now(timezone.utc)
    base = now.strftime("run-%Y-%m-%d-%H%M")
    candidate = base
    suffix = 2
    while (runs_dir / candidate).exists():
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def list_run_ids(runs_dir: Path) -> list[str]:
    if not runs_dir.is_dir():
        return []
    ids = [
        p.name for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("run-")
    ]
    ids.sort()
    return ids


def resolve_run_id(runs_dir: Path, spec: str) -> str:
    """Resolve `latest`, `prev`, or an explicit run id against runs_dir."""
    ids = list_run_ids(runs_dir)
    if spec == "latest":
        if not ids:
            raise UsageError(
                f"no runs found in {runs_dir}. Try 'uv run agent-exam <suite>' first."
            )
        return ids[-1]
    if spec == "prev":
        if len(ids) < 2:
            raise UsageError("no previous run (need at least 2 runs in evals/runs/).")
        return ids[-2]
    if spec in ids:
        return spec
    if not ids:
        raise UsageError(f"run {spec!r} not found; no runs exist yet.")
    recent = ", ".join(ids[-5:])
    raise UsageError(f"run {spec!r} not found in {runs_dir}. Recent: {recent}")
