"""Compare two reports, producing verdict / metric-delta / grader-change sections.

Used by `agent-exam diff` and reused by `show` for per-attempt delta highlighting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .commands._format import THRESHOLDS, delta_marker


@dataclass
class AttemptKey:
    suite: str
    task: str
    attempt: int

    def __hash__(self):
        return hash((self.suite, self.task, self.attempt))

    def __str__(self):
        return f"{self.suite}::{self.task} attempt-{self.attempt}"


@dataclass
class VerdictChange:
    attempt: AttemptKey
    before: str
    after: str


@dataclass
class MetricDelta:
    attempt: AttemptKey
    metric: str
    before: float | None
    after: float | None
    delta_pct: float


@dataclass
class GraderChange:
    suite: str
    task: str
    assertion_key: str
    kind: str  # "added" | "removed" | "definition-changed"
    before_config: object = None
    after_config: object = None

    @property
    def label(self) -> str:
        return f"{self.suite}::{self.task}"


@dataclass
class DiffResult:
    verdict_changes: list[VerdictChange] = field(default_factory=list)
    metric_deltas: list[MetricDelta] = field(default_factory=list)
    grader_changes: list[GraderChange] = field(default_factory=list)


def assertion_key(type_name: str, config: object) -> str:
    """Stable, comparable identity for an assertion within a task.

    Lets `diff` decide whether two assertion entries are "the same assertion"
    across runs. When both runs have the same key, we compare config to
    decide if the *definition* changed; keys present in only one run are
    added/removed.

    Keys chosen so the disambiguating field (path, tool name, etc.) is part
    of the key:
      - file_exists / file_contains → by path
      - tool_called / tool_not_called / tool_count → by tool name
      - judge → by first 40 chars of criterion (semantic-hash if collision hurts)
      - everything else → by type alone (one instance per task)
    """
    cfg = config
    if type_name == "file_exists":
        path = cfg if isinstance(cfg, str) else (cfg or {}).get("path", "")
        return f"{type_name}:{path}"
    if type_name == "file_contains":
        path = (cfg or {}).get("path", "") if isinstance(cfg, dict) else ""
        return f"{type_name}:{path}"
    if type_name in ("tool_called", "tool_not_called", "tool_count"):
        if isinstance(cfg, str):
            name = cfg
        elif isinstance(cfg, dict):
            name = cfg.get("name", "")
        else:
            name = ""
        return f"{type_name}:{name}"
    if type_name == "judge":
        criterion = cfg if isinstance(cfg, str) else (cfg or {}).get("criterion", "")
        head = (criterion or "")[:40]
        return f"judge:{head}"
    if type_name == "first_skill":
        return "first_skill"
    return f"{type_name}:{json.dumps(cfg, sort_keys=True, default=str)}"


def _index_attempts(attempts: list[dict]) -> dict[AttemptKey, dict]:
    return {AttemptKey(a["suite"], a["task"], a["attempt"]): a for a in attempts}


def _index_assertions(attempt: dict) -> dict[str, dict]:
    return {
        assertion_key(a["type"], a["config"]): a for a in attempt.get("assertions", [])
    }


def compare_reports(
    before_attempts: list[dict],
    after_attempts: list[dict],
    *,
    before_attempt_meta: dict[AttemptKey, dict] | None = None,
    after_attempt_meta: dict[AttemptKey, dict] | None = None,
) -> DiffResult:
    """Compare two reports' attempts[].

    `*_attempt_meta` map AttemptKey→attempt.json contents for per-attempt metric
    comparison. Missing meta means the metric section for that attempt is
    skipped (no false alarms when artifacts aren't on disk).
    """
    before = _index_attempts(before_attempts)
    after = _index_attempts(after_attempts)
    result = DiffResult()

    # Verdict changes — limited to attempts present in both.
    for key in sorted(
        set(before) & set(after), key=lambda k: (k.suite, k.task, k.attempt)
    ):
        if before[key]["verdict"] != after[key]["verdict"]:
            result.verdict_changes.append(
                VerdictChange(
                    attempt=key,
                    before=before[key]["verdict"],
                    after=after[key]["verdict"],
                )
            )

    # Metric deltas — need attempt.json for both sides.
    if before_attempt_meta and after_attempt_meta:
        for key in sorted(
            set(before_attempt_meta) & set(after_attempt_meta),
            key=lambda k: (k.suite, k.task, k.attempt),
        ):
            b = before_attempt_meta[key].get("metrics", {})
            a = after_attempt_meta[key].get("metrics", {})
            for metric in THRESHOLDS:
                b_val = _metric_val(b, metric)
                a_val = _metric_val(a, metric)
                if b_val is None or a_val is None or b_val == 0:
                    continue
                pct = (a_val - b_val) / b_val
                if abs(pct) < THRESHOLDS[metric]:
                    continue
                result.metric_deltas.append(
                    MetricDelta(
                        attempt=key,
                        metric=metric,
                        before=b_val,
                        after=a_val,
                        delta_pct=pct,
                    )
                )

    # Grader changes — comparison is at the (suite, task) level, not per attempt.
    # A task's assertion set is the same across its attempts within a run; the
    # per-attempt perspective would flag spurious adds/removes whenever two
    # runs used different -k values.
    result.grader_changes = _compare_grader_defs(before, after)
    return result


def _compare_grader_defs(before: dict, after: dict) -> list[GraderChange]:
    def _per_task_assertions(attempts: dict) -> dict[tuple[str, str], dict]:
        out: dict[tuple[str, str], dict] = {}
        for key, attempt in attempts.items():
            st = (key.suite, key.task)
            if st in out:
                continue  # first attempt wins — all attempts share assertions
            out[st] = _index_assertions(attempt)
        return out

    b_by_task = _per_task_assertions(before)
    a_by_task = _per_task_assertions(after)

    changes: list[GraderChange] = []
    for st in sorted(set(b_by_task) & set(a_by_task)):
        suite, task = st
        b = b_by_task[st]
        a = a_by_task[st]
        for ak, entry in a.items():
            if ak not in b:
                changes.append(
                    GraderChange(
                        suite=suite,
                        task=task,
                        assertion_key=ak,
                        kind="added",
                        after_config=entry.get("config"),
                    )
                )
                continue
            if b[ak].get("config") != entry.get("config"):
                changes.append(
                    GraderChange(
                        suite=suite,
                        task=task,
                        assertion_key=ak,
                        kind="definition-changed",
                        before_config=b[ak].get("config"),
                        after_config=entry.get("config"),
                    )
                )
        for bk, entry in b.items():
            if bk not in a:
                changes.append(
                    GraderChange(
                        suite=suite,
                        task=task,
                        assertion_key=bk,
                        kind="removed",
                        before_config=entry.get("config"),
                    )
                )
    return changes


def _metric_val(metrics: dict, metric: str) -> float | None:
    if metric == "cost_usd":
        return metrics.get("cost_usd")
    if metric == "peak_context":
        return metrics.get("peak_context")
    if metric == "wall_time_seconds":
        return metrics.get("wall_time_seconds")
    return None


__all__ = [
    "AttemptKey",
    "DiffResult",
    "GraderChange",
    "MetricDelta",
    "VerdictChange",
    "assertion_key",
    "compare_reports",
    "delta_marker",
]
