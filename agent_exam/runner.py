from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import click

from . import run_modes
from ._validate import reject_unknown_keys
from .artifacts import RunPaths
from .errors import ProviderTimeout, RateLimitExhausted, UsageError
from .hooks import call_pre_run_hook
from .ids import new_run_id
from .judge import JudgeCache, JudgeCall
from .mcp import preflight as mcp_preflight
from .pool import AttemptOutcome, PoolPlan, forget_mcp_staging, run_plan
from .providers import get_provider
from .providers.skill_staging import discover_skills
from .report import AttemptReport, report_to_dict, score_attempt
from .scoring_context import ScoringContext
from .serde import write_json
from .tasks import Task, expand_specs, load_specs, load_suite_config, select_by_tags
from .validation import validate_suite

if TYPE_CHECKING:
    from .config import Config
    from .providers.base import Provider


@dataclass
class RunRequest:
    specs: list[tuple[str, str | None]]
    provider: str
    model: str
    k: int
    n_parallel: int
    without_skill: bool
    # Reality-check with *every* skill dropped, not just the suite's
    # evaluated ones — the harness runs as a plain agent.
    no_skills: bool = False
    # Reality-check with the skills in place but no MCP server attached —
    # the counterfactual that shows how much of the work the servers do.
    no_mcp: bool = False
    # Drop `kind: trigger` tasks from the plan. Implied by the skill-
    # withholding modes (no skill loaded → nothing to fire); settable on its
    # own so the with-skill half of the comparison covers the same tasks.
    no_triggers: bool = False
    # Tag selection. `tags` (--tag) lifts a tag's default exclusion,
    # `exclude_tags` (--exclude-tag) drops the tasks wearing one, and
    # `all_tags` lifts every default exclusion at once.
    tags: list[str] = field(default_factory=list)
    exclude_tags: list[str] = field(default_factory=list)
    all_tags: bool = False
    # When False the ephemeral run-tmp root is left on disk so tests can
    # inspect the runtime filesystem state (e.g. where skills were staged).
    cleanup_tmp_root: bool = True

    def __post_init__(self) -> None:
        if self.without_skill and self.no_skills:
            raise UsageError(
                "--without-skill and --no-skills are mutually exclusive; "
                "--no-skills already drops every skill"
            )
        if self.no_mcp and self.skills_withheld:
            raise UsageError(
                "--no-mcp is mutually exclusive with --without-skill and "
                "--no-skills; a run that withholds both cannot say which of "
                "the two the difference is down to"
            )

    @property
    def skills_withheld(self) -> bool:
        """True when skills are (partly or wholly) withheld from the harness."""
        return self.without_skill or self.no_skills

    @property
    def reality_check(self) -> bool:
        """True when the run withholds something the suite is meant to have,
        so its verdicts describe a counterfactual rather than a regression."""
        return self.skills_withheld or self.no_mcp


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _report_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")


def _format_run_json(
    run_id: str,
    run_mode: str,
    started: str,
    finished: str,
    config: dict,
) -> dict:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "run_mode": run_mode,
        "started_at": started,
        "finished_at": finished,
        "config": config,
    }


_HEARTBEAT_INTERVAL = 15.0


def _heartbeat_interval() -> float | None:
    """Seconds between heartbeat lines; None disables (env override, <=0 = off)."""
    env_value = os.environ.get("AGENT_EXAM_HEARTBEAT_INTERVAL")
    if env_value is None:
        return _HEARTBEAT_INTERVAL
    try:
        val = float(env_value)
    except ValueError:
        return _HEARTBEAT_INTERVAL
    return val if val > 0 else None


def _heartbeat_enabled() -> bool:
    """On for interactive terminals; AGENT_EXAM_HEARTBEAT forces on/off.

    Off by default when stdout isn't a TTY so piped/CI output and test
    snapshots don't get periodic progress lines interleaved in.
    """
    force = os.environ.get("AGENT_EXAM_HEARTBEAT")
    if force is not None:
        return force.strip().lower() in ("1", "true", "yes", "on")
    return sys.stdout.isatty()


class _Heartbeat:
    """Periodically reprints in-flight attempts with elapsed time.

    Fills the dead air between an attempt's `started` line and its `scoring`/verdict lines so a
    long attempt reads as "still working (142s)" rather than looking stuck. Runs in a daemon
    thread; the main thread reports lifecycle via `attempt_started`/`attempt_scoring`/
    `attempt_finished`, all cheap and lock-guarded.

    An attempt stays in flight from dispatch through the end of scoring, so slow judge calls are
    covered too. Running vs queued is derived, not measured. `attempt_started` fires at *submit*
    time, so len(in_flight) counts submitted attempts, not executing ones. Since
    ProcessPoolExecutor dispatches futures FIFO and keeps ~n_parallel workers saturated,
    `running = min(n_parallel, active)` and the remainder is queued. The oldest active attempt
    (started first, not yet done) is necessarily one of the running ones.
    """

    def __init__(
        self, total: int, n_parallel: int, interval: float | None, enabled: bool
    ) -> None:
        self._total = total
        self._n_parallel = max(1, n_parallel)
        self._interval = interval
        self._enabled = enabled and interval is not None
        self._lock = threading.Lock()
        # key -> [start_monotonic, phase]  (phase: "running" | "scoring")
        self._in_flight: dict[tuple[str, str, int], list] = {}
        self._done = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self._enabled:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def attempt_started(self, key: tuple[str, str, int]) -> None:
        with self._lock:
            self._in_flight[key] = [time.monotonic(), "running"]

    def attempt_scoring(self, key: tuple[str, str, int]) -> None:
        with self._lock:
            entry = self._in_flight.get(key)
            if entry is not None:
                entry[1] = "scoring"

    def attempt_finished(self, key: tuple[str, str, int]) -> None:
        with self._lock:
            self._in_flight.pop(key, None)
            self._done += 1

    def _run(self) -> None:
        assert self._interval is not None
        while not self._stop.wait(self._interval):
            self._emit()

    def _emit(self) -> None:
        now = time.monotonic()
        with self._lock:
            if not self._in_flight:
                return
            # "active" = executing or queued (not yet handed to scoring).
            active = [
                (key, start)
                for key, (start, phase) in self._in_flight.items()
                if phase == "running"
            ]
            scoring = sum(
                1 for _, phase in self._in_flight.values() if phase == "scoring"
            )
            done = self._done
        n_active = len(active)
        running = min(self._n_parallel, n_active)
        queued = n_active - running

        segments: list[str] = []
        oldest_desc = ""
        if active:
            oldest_key, oldest_start = min(active, key=lambda ks: ks[1])
            suite, task, attempt = oldest_key
            oldest_desc = (
                f"oldest {suite}::{task} attempt-{attempt} ({now - oldest_start:.0f}s)"
            )

        if n_active == 1 and queued == 0 and scoring == 0:
            # Single attempt (the common one-long-task case): name it inline.
            suite, task, attempt = active[0][0]
            body = (
                f"{suite}::{task} attempt-{attempt} running ({now - active[0][1]:.0f}s)"
            )
        else:
            if running:
                segments.append(f"{running} running")
            if queued:
                segments.append(f"{queued} queued")
            if scoring:
                segments.append(f"{scoring} scoring")
            lead = " · ".join(segments)
            body = f"{lead} · {oldest_desc}" if oldest_desc else lead

        # Single write (nl in the message) so the line is atomic against
        # main-thread output as much as possible without a shared lock.
        click.echo(
            click.style("running", fg="cyan")
            + f"  {body} · {done}/{self._total} done\n",
            nl=False,
        )


def _provider_for(req: RunRequest) -> Provider:
    return get_provider(req.provider)


def _build_scoring_context(
    cfg: Config,
    req: RunRequest,
    provider: Provider,
    paths: RunPaths,
    skills_excluded: frozenset[str] = frozenset(),
) -> ScoringContext:
    """Assemble the judge-call + judge-cache once per run.

    Judge runs through the same provider as the agent, with that provider's
    `judge_model` when configured, otherwise its `default_model`. Providers
    that accept an omitted model may receive an empty model string and use
    their own default.
    """
    provider_cfg = cfg.provider(req.provider)
    judge_model = provider_cfg.judge_model or provider_cfg.default_model or ""
    judge_call = JudgeCall(
        provider=provider,
        judge_model=provider_cfg.resolve_model(judge_model),
        provider_options={"extra_args": list(provider_cfg.extra_args)},
        timeout_seconds=cfg.judge.timeout_seconds,
        agent_timeout_seconds=cfg.judge.agent_timeout_seconds,
    )

    judge_cache = JudgeCache(paths.judge_cache)
    return ScoringContext(
        provider=provider,
        judge_call=judge_call,
        judge_cache=judge_cache,
        judge_pass_on=list(cfg.judge.pass_on),
        skills_excluded=skills_excluded,
    )


def _resolve_model(cfg: Config, req: RunRequest) -> str:
    provider_cfg = (
        cfg.provider(req.provider) if req.provider != "dummy" else cfg.provider("dummy")
    )
    if req.model:
        return provider_cfg.resolve_model(req.model)
    if provider_cfg.default_model:
        return provider_cfg.default_model
    return ""


def run(cfg: Config, req: RunRequest) -> int:
    concrete = expand_specs(cfg.evals_dir, req.specs)
    tasks = load_specs(cfg.evals_dir, concrete)
    if not tasks:
        specs_str = ", ".join(f"{s}::{t}" if t else s for s, t in req.specs)
        raise UsageError(f"no tasks matched: {specs_str}")

    suite_config_map = {s: load_suite_config(cfg.evals_dir, s) for s, _ in concrete}

    for names, label in ((req.tags, "--tag"), (req.exclude_tags, "--exclude-tag")):
        reject_unknown_keys(names, cfg.tags, label=label, noun="tag")
    tasks, tags_excluded = select_by_tags(
        tasks,
        concrete,
        default_excluded=[n for n, t in cfg.tags.items() if t.exclude_by_default],
        suite_tags={s: sc.tags for s, sc in suite_config_map.items()},
        include=req.tags,
        exclude=req.exclude_tags,
        all_tags=req.all_tags,
    )
    if not tasks:
        specs_str = ", ".join(f"{s}::{t}" if t else s for s, t in req.specs)
        raise UsageError(
            f"every task in {specs_str} is excluded by tag "
            f"({_tags_excluded_str(tags_excluded)}); pass --tag <tag> or "
            f"--all-tags to include them"
        )

    # Drop trigger tasks, either because the user asked (--no-triggers) or
    # because a reality-check mode implies it: with the skills withheld there
    # is nothing for a trigger case to fire.
    effective_k = req.k
    drop_triggers = req.no_triggers or req.skills_withheld
    if drop_triggers:
        tasks = [t for t in tasks if t.kind != "trigger"]
        if not tasks:
            specs_str = ", ".join(f"{s}::{t}" if t else s for s, t in req.specs)
            reason = (
                "to reality-check"
                if req.skills_withheld
                else "left after --no-triggers"
            )
            raise UsageError(f"no kind: execute tasks found in {specs_str!r} {reason}")

    # The reality-check modes also withhold skills from the bundle we hand to
    # the harness — the suite's configured skills for `--without-skill`, every
    # discovered skill for `--no-skills` (resolved below, once skills_dirs is
    # final). Exits 0 regardless of assertion verdicts — the verdicts are
    # informational, not regression.
    if req.without_skill:
        exclude: set[str] = set()
        for suite, _ in concrete:
            sc = suite_config_map[suite]
            exclude.update(sc.evaluated_skills or [suite])
        skills_to_exclude = frozenset(exclude)
    else:
        skills_to_exclude = frozenset()

    # Fail fast on structural problems (unknown assertion types, missing
    # fixtures) before spending tokens. Parse errors already surfaced
    # from load_specs above; this adds the fixture check and is the same
    # call doctor makes.
    validation_fails = [
        c
        for suite, task_filter in concrete
        for c in validate_suite(cfg, suite, task_filter=task_filter)
        if c.status == "FAIL"
    ]
    if validation_fails:
        specs_str = ", ".join(f"{s}::{t}" if t else s for s, t in req.specs)
        raise UsageError(
            f"specs {specs_str!r} failed validation:\n  "
            + "\n  ".join(f"{c.name}: {c.hint}" for c in validation_fails)
        )

    # Captured before --no-mcp wipes the definitions below, so the run
    # record says which servers it withheld.
    mcp_servers_declared = sorted(cfg.mcp_servers)

    if req.no_mcp:
        if not cfg.mcp_servers:
            raise UsageError(
                "--no-mcp: no mcp_servers are declared in evals/config.yaml, "
                "so the run would be identical to a normal one"
            )
        # Detaching every server is the same thing as declaring none, so
        # staging, the preflight checks and the reports all follow without
        # a second code path. Task selections are already validated above.
        cfg = cfg.model_copy(update={"mcp_servers": {}})
        # A tool trigger grades on MCP calls, which cannot happen with no
        # server attached: the positives fail and the negatives pass on an
        # empty trajectory. Skill triggers stay — routing does not need the
        # tools it routes to. What the survivors select is cleared along with
        # the definitions, since a selection naming a server no longer in
        # config.yaml is a load-time error everywhere else.
        tasks = [replace(t, mcp_servers=[]) for t in tasks if not t.target_tool]
        if not tasks:
            specs_str = ", ".join(f"{s}::{t}" if t else s for s, t in req.specs)
            raise UsageError(
                f"every task in {specs_str} targets an MCP tool, so there is "
                f"nothing left to run under --no-mcp"
            )

    paths = RunPaths(cfg.evals_dir, new_run_id(cfg.evals_dir / "runs"))
    paths.run_dir.mkdir(parents=True)
    paths.reports_dir.mkdir()

    provider = _provider_for(req)
    model = _resolve_model(cfg, req)

    hook_result = call_pre_run_hook(cfg, req.provider)
    if (
        hook_result is not None
        and not cfg._skills_dirs_locked
        and hook_result.skills_dirs is not None
    ):
        cfg = cfg.model_copy(update={"skills_dirs": hook_result.skills_dirs})

    if cfg.skills_dirs is None:
        raise UsageError(
            "skills_dirs is not configured; put your skills in a `skills/` "
            "directory at the project root, or set skills_dirs in "
            "evals/config.yaml or evals/config.local.yaml, or return it from "
            "your pre_run_hook"
        )

    if req.no_skills:
        # "Load nothing" is expressed as "exclude everything": staging then
        # copies no skill dir at all, and `skill_invoked` assertions treat
        # every skill as absent-by-design rather than as a failure.
        skills_to_exclude = frozenset(
            name for name, _ in discover_skills(cfg.skills_dirs)
        )

    context = _build_scoring_context(
        cfg, req, provider, paths, skills_excluded=skills_to_exclude
    )

    run_started = _utc_now_iso()
    if req.no_skills:
        run_mode = run_modes.NO_SKILLS
    elif req.without_skill:
        run_mode = run_modes.WITHOUT_SKILL
    elif req.no_mcp:
        run_mode = run_modes.NO_MCP
    else:
        run_mode = run_modes.NORMAL

    banner = run_modes.banner_lines(run_mode, sorted(skills_to_exclude))
    if banner:
        for line in banner:
            click.echo(line)
        click.echo("")
    elif req.no_triggers:
        click.echo("Trigger tasks skipped (--no-triggers)")
        click.echo("")

    plan = PoolPlan(
        tasks=tasks,
        attempts_per_task=effective_k,
        n_parallel=max(1, req.n_parallel),
        # Dispatch attempts in attempt-number batches when every
        # task in the plan is a fixtureless trigger: attempt-1 of every
        # task in parallel, then attempt-2, etc. Each repeat hits the
        # prompt cache populated by the prior attempt of the same task.
        # Other plan shapes don't benefit — fixtures force per-task
        # cwds, the cwd path is embedded in the system prompt, so
        # different tasks get different cache prefixes regardless.
        serial_within_task=bool(tasks)
        and all(t.kind == "trigger" and not t.fixture for t in tasks),
    )

    attempt_reports: list[AttemptReport] = []
    rate_limit_hit = False

    # Materializes the default section when the run config omits this
    # provider, so later lookups find one to read.
    cfg.provider(req.provider)

    # Provider-specific pre-run warnings (e.g. Claude Code warns if a
    # blocked plugin is enabled in settings.json). Generic runner stays
    # provider-agnostic — each provider owns its own check set.
    for warning in provider.pre_run_warnings(cfg):
        _emit_warning(warning)

    # MCP servers whose command or credentials can't be resolved would only
    # surface as the agent silently missing its tools, so refuse the run.
    # Only the servers the planned tasks attach are checked, so a run that
    # leaves a credentialed server out doesn't need its credential.
    mcp_checks = mcp_preflight(cfg, provider, tasks)
    mcp_fails = [c for c in mcp_checks if c.status == "FAIL"]
    if mcp_fails:
        raise UsageError(
            "mcp_servers are not usable:\n  "
            + "\n  ".join(f"{c.name}: {c.hint}" for c in mcp_fails)
        )
    for warning in mcp_checks:
        if warning.status == "WARN":
            _emit_warning(warning)

    _emit_run_header(
        paths.run_id, req, plan, model, effective_k, concrete, tags_excluded
    )

    attempt_starts: dict[tuple[str, str, int], float] = {}
    heartbeat = _Heartbeat(
        total=plan.attempts_per_task * len(plan.tasks),
        n_parallel=plan.n_parallel,
        interval=_heartbeat_interval(),
        enabled=_heartbeat_enabled(),
    )

    def _on_attempt_start(task: Task, attempt_n: int) -> None:
        key = (task.suite, task.name, attempt_n)
        attempt_starts[key] = time.monotonic()
        heartbeat.attempt_started(key)
        click.echo(
            click.style("started", fg="blue")
            + f"  {task.suite}::{task.name} attempt-{attempt_n}"
        )

    # Attempt cwds live under an ephemeral /tmp root — the path Claude sees
    # in its system prompt contains no "evals/runs/..." markers.
    #
    # Prefix is intentionally the stdlib default (`tmp`) — any
    # eval-identifying string here (e.g. `agent-exam-run-`) would leak
    # into Claude's system prompt via the cwd path.
    run_tmp_root = Path(tempfile.mkdtemp())
    session_checked = False
    heartbeat.start()
    try:
        try:
            for outcome in run_plan(
                cfg,
                plan,
                run_tmp_root,
                req.provider,
                model,
                paths,
                on_attempt_start=_on_attempt_start,
                skills_to_exclude=skills_to_exclude,
            ):
                if not session_checked and outcome.run_result is not None:
                    # Provider-specific checks on what the first finished
                    # attempt reveals about the skills under test, so a run
                    # that can only score misses says so while there is still
                    # time to abandon it. Generic runner stays
                    # provider-agnostic and just prints what comes back.
                    session_checked = True
                    for warning in provider.session_checks(outcome.run_result, cfg):
                        if warning.status != "OK":
                            _emit_warning(warning)
                key = (outcome.suite, outcome.task_name, outcome.attempt_n)
                heartbeat.attempt_scoring(key)
                # Scoring can be slow when judge assertions dispatch LLM
                # calls. Emit a line so the user sees the phase change
                # rather than wondering if the attempt hung.
                _emit_scoring_start(outcome)
                score_t0 = time.monotonic()
                try:
                    report = _score_outcome(outcome, tasks, context, req.provider)
                except ProviderTimeout as exc:
                    task = _lookup_task(tasks, outcome.suite, outcome.task_name)
                    report = score_attempt(
                        task,
                        outcome.attempt_n,
                        run_result=None,
                        attempt_cwd=outcome.attempt_cwd,
                        error_verdict="error",
                    )
                    click.echo(f"scoring error (judge timed out): {exc}", err=True)
                attempt_reports.append(report)
                heartbeat.attempt_finished(key)
                elapsed = _elapsed(attempt_starts, report)
                scoring_elapsed = time.monotonic() - score_t0
                _emit_progress(report, elapsed, scoring_elapsed)
        except RateLimitExhausted as exc:
            rate_limit_hit = True
            click.echo(f"rate-limit exhausted: {exc}", err=True)
    finally:
        heartbeat.stop()
        forget_mcp_staging(run_tmp_root)
        if req.cleanup_tmp_root:
            shutil.rmtree(run_tmp_root, ignore_errors=True)

    run_finished = _utc_now_iso()
    attempt_reports.sort(key=lambda a: (a.suite, a.task, a.attempt))
    report_ts = _report_timestamp()
    write_json(
        paths.report_file(report_ts),
        report_to_dict(run_started, run_finished, scope=None, attempts=attempt_reports),
    )
    write_json(
        paths.run_json,
        _format_run_json(
            run_id=paths.run_id,
            run_mode=run_mode,
            started=run_started,
            finished=run_finished,
            config={
                "k": effective_k,
                "models": [model],
                "n_parallel": plan.n_parallel,
                "without_skill": req.without_skill,
                "no_skills": req.no_skills,
                "no_mcp": req.no_mcp,
                "mcp_servers": mcp_servers_declared,
                "no_triggers": drop_triggers,
                "skills_excluded": sorted(skills_to_exclude),
                "tags": sorted(req.tags),
                "exclude_tags": sorted(req.exclude_tags),
                "all_tags": req.all_tags,
                "tasks_excluded_by_tag": dict(sorted(tags_excluded.items())),
                "specs_requested": [f"{s}::{t}" if t else s for s, t in req.specs],
                "provider": req.provider,
                "tmp_root": str(run_tmp_root),
            },
        ),
    )

    _print_summary(paths)
    if rate_limit_hit:
        return 3
    # Reality-check runs are informational — don't translate assertion
    # failures into a non-zero exit code. Framework errors would have
    # bubbled up before this point.
    if req.reality_check:
        return 0
    # Known-issue outcomes don't gate the suite: the whole point of the
    # annotation is to land a failing check without failing the run.
    # Only a plain `fail` / `timeout` / `error` is suite-fatal.
    non_gating = {"pass", "known_issue", "unexpected_pass"}
    return 0 if all(a.verdict in non_gating for a in attempt_reports) else 1


def _score_outcome(
    outcome: AttemptOutcome,
    tasks: list[Task],
    context: ScoringContext,
    provider_name: str,
) -> AttemptReport:
    task = _lookup_task(tasks, outcome.suite, outcome.task_name)
    return score_attempt(
        task,
        outcome.attempt_n,
        outcome.run_result,
        outcome.attempt_cwd,
        error_verdict=outcome.error_verdict,
        context=context,
        provider_name=provider_name,
    )


def _lookup_task(tasks: list[Task], suite: str, name: str) -> Task:
    for t in tasks:
        if t.suite == suite and t.name == name:
            return t
    raise KeyError(f"task {suite}::{name} not found in plan")


def _emit_warning(check) -> None:
    """Print a provider-returned CheckResult as a yellow stderr line."""
    line = f"[{check.status}] {check.name}"
    if check.hint:
        line += f": {check.hint}"
    click.echo(click.style(line, fg="yellow"), err=True)


def _tags_excluded_str(counts: dict[str, int]) -> str:
    return ", ".join(f"{tag} ({n})" for tag, n in sorted(counts.items()))


def _emit_run_header(
    run_id: str,
    req: RunRequest,
    plan: PoolPlan,
    model: str,
    effective_k: int,
    concrete: list[tuple[str, str | None]],
    tags_excluded: dict[str, int],
) -> None:
    total_attempts = plan.attempts_per_task * len(plan.tasks)
    task_count = len(plan.tasks)
    task_word = "task" if task_count == 1 else "tasks"
    parallelism = min(plan.n_parallel, total_attempts)
    click.echo(f"Run:      {run_id}")
    specs_str = ", ".join(f"{s}::{t}" if t else s for s, t in req.specs)
    if req.specs != concrete:
        expanded = ", ".join(f"{s}::{t}" if t else s for s, t in concrete)
        specs_str = f"{specs_str} → {expanded}"
    label = "Suite" if len(concrete) == 1 else "Specs"
    click.echo(
        f"{label}:    {specs_str} "
        f"({task_count} {task_word} × k={effective_k} = {total_attempts} attempts)"
    )
    if tags_excluded:
        total = sum(tags_excluded.values())
        click.echo(
            f"Skipped:  {total} task(s) by tag: {_tags_excluded_str(tags_excluded)}"
        )
    click.echo(f"Provider: {req.provider} ({model or 'default'})")
    click.echo(f"Parallel: up to {parallelism}")
    click.echo("")


def _elapsed(
    starts: dict[tuple[str, str, int], float], report: AttemptReport
) -> float | None:
    key = (report.suite, report.task, report.attempt)
    start = starts.get(key)
    return time.monotonic() - start if start is not None else None


def _emit_scoring_start(outcome: AttemptOutcome) -> None:
    click.echo(
        click.style("scoring", fg="blue")
        + f"  {outcome.suite}::{outcome.task_name} attempt-{outcome.attempt_n}"
    )


def _emit_progress(
    report: AttemptReport, elapsed: float | None, scoring_elapsed: float | None
) -> None:
    verdict = report.verdict.upper()
    # KNOWN_ISSUE     = yellow       (still failing, acknowledged).
    # UNEXPECTED_PASS = bright_cyan  (hint: annotation can be removed).
    color = {
        "PASS": "green",
        "FAIL": "red",
        "KNOWN_ISSUE": "yellow",
        "UNEXPECTED_PASS": "bright_cyan",
    }.get(verdict)
    # Widen the column so the longer verdict labels don't misalign.
    tag = click.style(f"{verdict:<16}", fg=color) if color else f"{verdict:<16}"
    parts = []
    if elapsed is not None:
        parts.append(f"{elapsed:.0f}s total")
    if scoring_elapsed is not None:
        parts.append(f"{scoring_elapsed:.0f}s scoring")
    suffix = f"  ({', '.join(parts)})" if parts else ""
    click.echo(f"{tag} {report.suite}::{report.task} attempt-{report.attempt}{suffix}")


def _print_summary(paths: RunPaths) -> None:
    click.echo("")
    click.echo(f"Run dir:     {paths.run_dir}")
    reports = sorted(paths.reports_dir.glob("*.json"))
    if reports:
        click.echo(f"Report:      {reports[-1]}")
    click.echo("Inspect:     uv run agent-exam show latest")
