"""OpenCode-specific doctor probes.

`OpenCodeProvider.preflight()` calls the static checks (DB existence and
schema); `probe_checks()` calls the post-probe check (model populated).
"""

from __future__ import annotations

import sqlite3

from ...schemas import CheckResult
from .transcripts import _db_path

_REQUIRED_TABLES = {"session", "message", "part"}


def check_db_exists() -> CheckResult:
    """Verify the opencode SQLite DB exists and is readable."""
    db = _db_path()
    if not db.exists():
        return CheckResult(
            name="opencode DB exists",
            status="FAIL",
            hint=f"DB not found at {db} — run opencode at least once, or set OPENCODE_DATA_DIR",
        )
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        conn.close()
    except sqlite3.OperationalError as exc:
        return CheckResult(
            name="opencode DB exists",
            status="FAIL",
            hint=f"DB at {db} is not readable: {exc}",
        )
    return CheckResult(name="opencode DB exists", status="OK", hint=str(db))


def check_db_schema() -> CheckResult:
    """Verify the expected tables are present in the opencode DB."""
    db = _db_path()
    if not db.exists():
        return CheckResult(
            name="opencode DB schema",
            status="WARN",
            hint="DB missing — skipped",
        )
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            present = {r[0] for r in rows}
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return CheckResult(
            name="opencode DB schema",
            status="FAIL",
            hint=f"could not query DB schema: {exc}",
        )
    missing = _REQUIRED_TABLES - present
    if missing:
        return CheckResult(
            name="opencode DB schema",
            status="FAIL",
            hint=f"missing tables: {', '.join(sorted(missing))} — opencode version mismatch?",
        )
    return CheckResult(
        name="opencode DB schema",
        status="OK",
        hint=f"tables present: {', '.join(sorted(_REQUIRED_TABLES))}",
    )


def check_probe_model(probe_result) -> CheckResult:
    """Verify the probe attempt recorded a non-empty model name.

    An empty model means opencode ran but the DB didn't capture providerID/
    modelID — typically an auth failure or an older opencode version.
    """
    model = (probe_result.model or "").strip() if probe_result is not None else ""
    if not model:
        return CheckResult(
            name="opencode probe model",
            status="WARN",
            hint=(
                "probe completed but no model name was recorded — "
                "check API key and opencode version"
            ),
        )
    return CheckResult(
        name="opencode probe model",
        status="OK",
        hint=model,
    )
