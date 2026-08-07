"""`agent-exam rescore <run-id>[::<suite>[::<task>]]` — re-grade archived attempts.

Does NOT re-run the agent. Reads archived `attempt.json` + `trajectory.json`
per attempt, reconstructs a RunResult, loads the CURRENT task YAML, and runs
the scoring pass. Writes a new `reports/<ts>.json` alongside prior reports.

Judge assertions consult the per-run `judge-cache.json` (populated during
the original run in step 4); same criterion + same archived output + same
judge model → cache hit, no LLM call.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import click

from ..config import Config, load_config
from ..errors import UsageError
from ..judge import JudgeCache, JudgeCall
from ..providers import get_provider
from ..report import AttemptReport, report_to_dict, score_attempt
from ..scoring_context import ScoringContext
from ..serde import run_result_from_artifacts, write_json
from ..tasks import Task, load_task
from ._loader import RunData, load_run, parse_run_spec

if TYPE_CHECKING:
    from pathlib import Path


def run(evals_dir: Path, scope_spec: str) -> int:
    scope = parse_run_spec(scope_spec)
    if scope.attempt is not None:
        raise UsageError(
            "rescore does not accept ::attempt-N — re-scores every attempt in scope"
        )

    cfg = load_config()
    data = load_run(evals_dir, scope.run_id)

    targets = _find_targets(data, scope.suite, scope.task)
    if not targets:
        if scope.suite and scope.task:
            raise UsageError(
                f"no archived attempts for {scope.suite}::{scope.task} in {data.run_id}"
            )
        if scope.suite:
            raise UsageError(
                f"no archived attempts for suite {scope.suite!r} in {data.run_id}"
            )
        raise UsageError(f"no archived attempts in {data.run_id}")

    context = _build_rescore_context(cfg, data)

    started = _utc_now_iso()
    reports: list[AttemptReport] = []
    for suite, task_name, attempt_n in targets:
        task, _task_yaml = _load_current_task(evals_dir, suite, task_name)

        attempt_json_path = data.paths.attempt_json(suite, task_name, attempt_n)
        trajectory_path = data.paths.trajectory_json(suite, task_name, attempt_n)
        if not attempt_json_path.exists() or not trajectory_path.exists():
            raise UsageError(
                f"attempt artifacts missing for {suite}::{task_name}::attempt-{attempt_n}: "
                f"need both attempt.json and trajectory.json under {data.paths.run_dir}"
            )
        attempt_json = json.loads(attempt_json_path.read_text())
        trajectory_json = json.loads(trajectory_path.read_text())
        run_result = run_result_from_artifacts(attempt_json, trajectory_json)
        attempt_cwd = data.paths.attempt_cwd(suite, task_name, attempt_n)

        report = score_attempt(
            task,
            attempt_n,
            run_result,
            attempt_cwd,
            context=context,
            provider_name=data.run_json.get("config", {}).get("provider", ""),
        )
        reports.append(report)
        click.echo(
            f"{report.verdict.upper():<4} {report.suite}::{report.task} attempt-{report.attempt}"
        )

    finished = _utc_now_iso()
    scope_dict = _scope_dict(scope.suite, scope.task)
    out_path = _unique_report_path(data, _report_ts())
    write_json(
        out_path,
        report_to_dict(started, finished, scope=scope_dict, attempts=reports),
    )

    click.echo("")
    click.echo(f"New report: {out_path}")
    click.echo(f"Inspect:    uv run agent-exam show {data.run_id}")
    if all(r.verdict == "pass" for r in reports):
        return 0
    return 1


def _load_current_task(
    evals_dir: Path, suite: str, task_name: str
) -> tuple[Task, Path]:
    """Find the task YAML that produced `task_name` and return the matching Task.

    For `kind: execute` the YAML stem equals the task name. For `kind: trigger`
    each YAML expands into N cases named `<stem>-<index>`; strip the trailing
    `-<digits>` to find the source YAML.
    """
    tasks_dir = evals_dir / "suites" / suite / "tasks"
    # 1) Exact match — covers kind: execute.
    direct = tasks_dir / f"{task_name}.yaml"
    if direct.exists():
        expanded = load_task(direct, suite)
        for t in expanded:
            if t.name == task_name:
                return t, direct
    # 2) Trigger case: strip trailing `-<digits>` and try the base YAML.
    m = re.match(r"^(.*)-(\d+)$", task_name)
    if m:
        base = tasks_dir / f"{m.group(1)}.yaml"
        if base.exists():
            expanded = load_task(base, suite)
            for t in expanded:
                if t.name == task_name:
                    return t, base
    raise UsageError(
        f"task YAML not found for {suite}::{task_name} under {tasks_dir}. "
        "Rescore needs the current task YAML — was it renamed or removed?"
    )


def _find_targets(
    data: RunData, suite: str | None, task: str | None
) -> list[tuple[str, str, int]]:
    """Walk `artifacts/` for every (suite, task, attempt-N) actually on disk.

    Uses the filesystem as source of truth rather than the initial report,
    so rescore works even on runs that partially failed before a report
    file was written.
    """
    artifacts = data.paths.artifacts_dir
    if not artifacts.is_dir():
        return []

    targets: list[tuple[str, str, int]] = []
    for suite_dir in sorted(artifacts.iterdir()):
        if not suite_dir.is_dir():
            continue
        if suite is not None and suite_dir.name != suite:
            continue
        for task_dir in sorted(suite_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            if task is not None and task_dir.name != task:
                continue
            for attempt_dir in sorted(task_dir.iterdir()):
                if not attempt_dir.is_dir():
                    continue
                if not attempt_dir.name.startswith("attempt-"):
                    continue
                try:
                    n = int(attempt_dir.name.split("-", 1)[1])
                except (IndexError, ValueError):
                    continue
                targets.append((suite_dir.name, task_dir.name, n))
    return targets


def _build_rescore_context(cfg: Config, data: RunData) -> ScoringContext:
    """Build a ScoringContext using CURRENT config for the run's provider.

    Current judge_model wins over whatever was configured when the run
    originally ran — the point of rescore is to apply current graders /
    judging to frozen outputs.
    """
    provider_name = (
        data.run_json.get("config", {}).get("provider") or cfg.default_harness
    )
    provider_cfg = cfg.provider(provider_name)
    judge_model = provider_cfg.judge_model or provider_cfg.default_model or ""
    provider = get_provider(provider_name)

    judge_call = JudgeCall(
        provider=provider,
        judge_model=provider_cfg.resolve_model(judge_model),
        provider_options={"extra_args": list(provider_cfg.extra_args)},
        timeout_seconds=cfg.judge.timeout_seconds,
        agent_timeout_seconds=cfg.judge.agent_timeout_seconds,
    )

    skills_excluded = frozenset(
        data.run_json.get("config", {}).get("skills_excluded") or []
    )
    return ScoringContext(
        provider=provider,
        judge_call=judge_call,
        judge_cache=JudgeCache(data.paths.judge_cache),
        judge_pass_on=list(cfg.judge.pass_on),
        skills_excluded=skills_excluded,
    )


def _scope_dict(suite: str | None, task: str | None) -> dict | None:
    if suite is None:
        return None
    if task is None:
        return {"suite": suite}
    return {"suite": suite, "task": task}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _report_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")


def _unique_report_path(data: RunData, base_ts: str) -> Path:
    """Avoid overwriting an existing report file when a rescore runs in the
    same second as the initial report. Appends `-2`, `-3`, ... on collision.
    """
    candidate = data.paths.report_file(base_ts)
    if not candidate.exists():
        return candidate
    suffix = 2
    while True:
        candidate = data.paths.report_file(f"{base_ts}-{suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1
