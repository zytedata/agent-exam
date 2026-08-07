"""Read on-disk run artifacts into typed structures.

All four inspection commands (`runs`, `show`, `history`, `diff`) read the
same files, so the parsing + scope resolution lives here. Commands stay
thin wrappers over these helpers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ..artifacts import RunPaths
from ..errors import UsageError
from ..ids import list_run_ids, resolve_run_id


@dataclass
class Report:
    timestamp: str  # filename stem e.g. "2026-04-23-1423"
    path: Path
    started_at: str
    finished_at: str
    scope: dict | None
    attempts: list[dict]  # raw entries from the report file

    def covers(self, suite: str, task: str, attempt_n: int | None = None) -> bool:
        for a in self.attempts:
            if a["suite"] != suite or a["task"] != task:
                continue
            if attempt_n is None or a["attempt"] == attempt_n:
                return True
        return False


@dataclass
class RunData:
    run_id: str
    paths: RunPaths
    run_json: dict
    reports: list[Report]  # sorted oldest first

    @property
    def latest_report(self) -> Report | None:
        return self.reports[-1] if self.reports else None

    def report_for(self, timestamp: str | None) -> Report | None:
        if timestamp is None:
            return self.latest_report
        for r in self.reports:
            if r.timestamp == timestamp:
                return r
        return None

    def latest_for_task(
        self, suite: str, task: str, attempt_n: int | None = None
    ) -> dict | None:
        """Walk newest→oldest, return the first attempt entry that covers scope."""
        for r in reversed(self.reports):
            for a in r.attempts:
                if a["suite"] != suite or a["task"] != task:
                    continue
                if attempt_n is not None and a["attempt"] != attempt_n:
                    continue
                return a
        return None

    def load_attempt_metadata(
        self, suite: str, task: str, attempt_n: int
    ) -> dict | None:
        p = self.paths.attempt_json(suite, task, attempt_n)
        if not p.exists():
            return None
        return json.loads(p.read_text())


@dataclass
class Scope:
    run_id: str
    suite: str | None = None
    task: str | None = None
    attempt: int | None = None


def _read_run_json(paths: RunPaths) -> dict:
    if not paths.run_json.exists():
        raise UsageError(f"run.json missing in {paths.run_dir}")
    return json.loads(paths.run_json.read_text())


def _read_reports(paths: RunPaths) -> list[Report]:
    if not paths.reports_dir.is_dir():
        return []
    out: list[Report] = []
    for p in sorted(paths.reports_dir.glob("*.json")):
        data = json.loads(p.read_text())
        out.append(
            Report(
                timestamp=p.stem,
                path=p,
                started_at=data.get("started_at", ""),
                finished_at=data.get("finished_at", ""),
                scope=data.get("scope"),
                attempts=list(data.get("attempts", [])),
            )
        )
    return out


def load_run(evals_dir: Path, run_id_spec: str) -> RunData:
    runs_dir = evals_dir / "runs"
    run_id = resolve_run_id(runs_dir, run_id_spec)
    paths = RunPaths(evals_dir, run_id)
    return RunData(
        run_id=run_id,
        paths=paths,
        run_json=_read_run_json(paths),
        reports=_read_reports(paths),
    )


def list_runs(evals_dir: Path) -> list[str]:
    return list_run_ids(evals_dir / "runs")


def parse_run_spec(spec: str) -> Scope:
    """Parse `<run-id>[::<suite>[::<task>[::attempt-N]]]`."""
    parts = spec.split("::")
    run_id = parts[0]
    suite = parts[1] if len(parts) > 1 else None
    task = parts[2] if len(parts) > 2 else None
    attempt: int | None = None
    if len(parts) > 4:
        raise UsageError(f"too many '::' segments in {spec!r}")
    if len(parts) == 4:
        m = re.match(r"^attempt-(\d+)$", parts[3])
        if not m:
            raise UsageError(f"expected 'attempt-N', got {parts[3]!r}")
        attempt = int(m.group(1))
    if suite == "" or task == "":
        raise UsageError(f"empty segment in scope {spec!r}")
    return Scope(run_id=run_id, suite=suite, task=task, attempt=attempt)


def parse_task_spec(spec: str) -> tuple[str, str | None]:
    """Parse `<suite>` or `<suite>::<task>`. Used by `history`.

    Returns `(suite, task)` where `task` is None for suite-only scopes.
    """
    parts = spec.split("::")
    if len(parts) > 2:
        raise UsageError(f"too many '::' segments in {spec!r}")
    if not parts[0]:
        raise UsageError(f"empty suite name in {spec!r}")
    if len(parts) == 1:
        return parts[0], None
    if not parts[1]:
        raise UsageError(f"empty task name in {spec!r}")
    return parts[0], parts[1]
