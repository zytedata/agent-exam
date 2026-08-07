"""Shared formatters for the inspection CLI.

Plain text, fixed-width columns. Stdlib only — no `rich`/`tabulate`. Metric
delta thresholds are defined once here and reused by `show`, `history`, and
`diff`.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from collections.abc import Iterable

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _visual_width(s: str) -> int:
    """Length as rendered to a terminal — excludes ANSI escape sequences."""
    return len(_ANSI_RE.sub("", s))


THRESHOLDS = {
    "cost_usd": 0.15,  # ±15%
    "peak_context": 0.20,  # ±20%
    "wall_time_seconds": 0.25,  # ±25%
}


def fmt_cost(usd: float | None) -> str:
    if usd is None:
        return "   ?   "
    return f"${usd:.4f}"


def fmt_wall(seconds: float | None) -> str:
    if seconds is None:
        return "  —  "
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def fmt_ctx(tokens: int | None) -> str:
    if tokens is None:
        return "  —  "
    if tokens < 1000:
        return f"{tokens}"
    if tokens < 1_000_000:
        return f"{tokens / 1000:.1f}k"
    return f"{tokens / 1_000_000:.1f}M"


def fmt_iso_age(iso: str, now: datetime | None = None) -> str:
    if not iso:
        return ""
    try:
        # Accept both "Z" and offset forms.
        ts = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return iso
    now = now or datetime.now(timezone.utc)
    delta = now - ts
    s = int(delta.total_seconds())
    if s < 60:
        return f"{s}s ago"
    if s < 3600:
        return f"{s // 60}m ago"
    if s < 86400:
        return f"{s // 3600}h ago"
    return f"{s // 86400}d ago"


def iso_duration(started: str, finished: str) -> float | None:
    if not started or not finished:
        return None
    try:
        a = datetime.fromisoformat(started.replace("Z", "+00:00"))
        b = datetime.fromisoformat(finished.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (b - a).total_seconds()


def delta_marker(new: float | None, old: float | None, metric: str) -> str:
    """Return a short delta description, or '' if within threshold / missing.

    Example outputs:
        "" (within threshold)
        "+42%"
        "-25%"
        "new"
        "gone"
    """
    threshold = THRESHOLDS.get(metric)
    if threshold is None:
        return ""
    if new is None and old is None:
        return ""
    if old in (None, 0) and new not in (None, 0):
        return "new"
    if new in (None, 0) and old not in (None, 0):
        return "gone"
    if old is None or old == 0:
        return ""
    delta = (new - old) / old  # type: ignore[operator]
    if abs(delta) < threshold:
        return ""
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta * 100:.0f}%"


def render_table(rows: Iterable[list[str]], header: list[str]) -> str:
    """ANSI-aware table renderer. Column widths are computed from the
    visual width (no escape chars), but cells are emitted verbatim so
    colored strings (`click.style(...)`) display correctly.
    """
    rows = list(rows)
    cols = len(header)
    widths = [_visual_width(h) for h in header]
    for row in rows:
        for i in range(min(cols, len(row))):
            widths[i] = max(widths[i], _visual_width(row[i]))

    def _fmt_row(cells: list[str]) -> str:
        padded_cells = cells + [""] * (cols - len(cells))
        out = []
        for i, cell in enumerate(padded_cells[:cols]):
            out.append(cell + " " * (widths[i] - _visual_width(cell)))
        return "  ".join(out)

    lines = [_fmt_row(header)]
    lines.extend(_fmt_row(row) for row in rows)
    return "\n".join(lines)


def fmt_pass_ratio(attempt_entry: dict) -> str:
    """Colored `<passed>/<total>` for a report attempts[] entry.

    The ratio is computed over **ungated** assertions — ones that
    actually contribute to the task's aggregate verdict. Assertions
    carrying `known_issue` (and assertions skipped via `providers:`)
    are excluded, matching the verdict logic: their individual outcomes
    don't gate pass/fail.

    When ungated assertions all pass, a `+N` suffix hints at gated
    assertions so the user knows there's more to look at in the detail
    view (e.g. `4/4 +1` → check attempt-detail for the known-issue /
    skipped entry).

    Green when every ungated assertion passed, yellow when some did,
    red when none did. Timeout / error verdicts bypass the ratio.
    Task-level `known_issue` / `unexpected_pass` verdicts tint the
    whole ratio (yellow / cyan).
    """
    verdict = attempt_entry.get("verdict", "")
    if verdict in ("timeout", "error"):
        return click.style(verdict.upper(), fg="red")
    assertions = attempt_entry.get("assertions") or []
    ungated = [
        a
        for a in assertions
        if not a.get("known_issue") and not a.get("skipped_reason")
    ]
    xpass_n = sum(1 for a in assertions if a.get("known_issue"))
    skipped_n = sum(1 for a in assertions if a.get("skipped_reason"))
    total = len(ungated)
    passed = sum(1 for a in ungated if a.get("pass"))
    text = f"{passed}/{total}"
    extras = []
    if xpass_n:
        extras.append(f"+{xpass_n} xpass")
    if skipped_n:
        extras.append(f"{skipped_n} skipped")
    if extras:
        text = f"{text} {', '.join(extras)}"
    if total == 0:
        return verdict.upper() if verdict else text
    if verdict == "unexpected_pass":
        return click.style(text, fg="bright_cyan")
    if verdict == "known_issue":
        return click.style(text, fg="yellow")
    if passed == total:
        return click.style(text, fg="green")
    if passed == 0:
        return click.style(text, fg="red")
    return click.style(text, fg="yellow")
