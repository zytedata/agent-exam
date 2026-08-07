"""Parallel attempt execution.

The parent submits (task, attempt_n) pairs to a ProcessPoolExecutor; each
worker constructs a fresh Provider, runs one attempt, writes attempt.json +
trajectory.json + cwd/, and returns an `AttemptOutcome` back to the parent.
Scoring (including judge dispatch) runs on the parent so judge-cache access
is single-threaded and picklability of the judge context is a non-issue.

Concurrency groups are implemented via `multiprocessing.Manager().Semaphore`
proxies passed to workers through the pool initializer.
"""

from __future__ import annotations

import multiprocessing as mp
import shutil
import uuid
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .errors import ProviderTimeout, UsageError
from .providers import get_provider
from .schemas import RunResult
from .serde import to_json_dict, write_json
from .tasks import Task

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from .artifacts import RunPaths
    from .config import Config

_WORKER_SEMAPHORES: dict[str, object] | None = None


def _init_worker(semaphores: dict[str, object]) -> None:
    global _WORKER_SEMAPHORES  # noqa: PLW0603 -- process-pool initializer
    _WORKER_SEMAPHORES = semaphores


@dataclass
class AttemptOutcome:
    """What a worker returns for one attempt."""

    suite: str
    task_name: str
    attempt_n: int
    attempt_cwd: Path
    run_result: RunResult | None
    error_verdict: str | None  # "timeout" | None


def _copy_fixture(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, symlinks=False, dirs_exist_ok=True)


def _stage_attempt_cwd(task: Task, evals_dir: Path, attempt_cwd: Path) -> None:
    attempt_cwd.mkdir(parents=True, exist_ok=True)
    if task.fixture:
        fixture_src = evals_dir / "fixtures" / task.fixture
        if not fixture_src.is_dir():
            raise UsageError(
                f"{task.source_path}: fixture {task.fixture!r} not found under "
                f"{evals_dir / 'fixtures'}"
            )
        _copy_fixture(fixture_src, attempt_cwd)


_SKILL_DISCOVERY_DIRS = frozenset({".opencode", ".claude", ".github", ".agents"})


def _mirror_cwd(src: Path, dst: Path) -> None:
    """Archive the end-of-attempt cwd into the run's artifacts tree.

    Called after `provider.invoke()` returns. Uses copy (not move/rename)
    so that `src` under the run-tmp root can be cleaned up wholesale when
    the run finishes.

    Skill-discovery directories (``.opencode/skills``, ``.claude/skills``,
    ``.github/skills``, ``.agents/skills``) are skipped so archives stay
    small — the symlinks would otherwise be resolved and all skill files
    copied into every attempt's archive.
    """

    def _ignore_skills(directory: str, contents: list[str]) -> set[str]:
        # Only prune the `skills` subdir of a top-level discovery dir —
        # not the whole dir at any depth — so fixture/agent content that
        # happens to live under e.g. `.agents/` (for anything other than
        # `.agents/skills`) still gets archived.
        if (
            Path(directory).parent != src
            or Path(directory).name not in _SKILL_DISCOVERY_DIRS
        ):
            return set()
        return {"skills"} & set(contents)

    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, symlinks=False, ignore=_ignore_skills)


def _execute_attempt(
    task: Task,
    attempt_n: int,
    cfg: Config,
    provider_name: str,
    model: str,
    paths: RunPaths,
    attempt_started: str,
    attempt_finished_writer,
    run_tmp_root: Path,
    skills_to_exclude: frozenset[str] = frozenset(),
) -> AttemptOutcome:
    """Body of one attempt. Runs in a pool worker (or inline for serial mode).

    The attempt's cwd lives under `run_tmp_root` (an ephemeral `/tmp/...`
    dir) so the path Claude sees doesn't leak "evals/runs/..." into the
    agent's context. After the provider returns, we mirror the
    end-of-attempt cwd into the archive under `paths.attempt_cwd(...)` for
    scoring + inspection.
    """
    provider = get_provider(provider_name)
    provider_cfg = cfg.provider(provider_name)
    provider_options: dict = {
        "extra_args": list(provider_cfg.extra_args),
        "env_overrides": dict(task.env),
    }
    # Provider-specific options (permission_mode, allowed_tools,
    # opencode.permission/pure, …) are the provider's concern — it owns
    # its own typed config model and knows how to map it onto invoke
    # options. Pool stays provider-agnostic for the dict assembly.
    provider_options.update(
        provider.task_options(
            task.provider_configs.get(provider_name), provider_cfg, task.kind
        )
    )
    if task.target_skill:
        provider_options["target_skill"] = task.target_skill
    if task.should_trigger is False:
        # Negative trigger case: signal the provider to cut early once
        # the routing decision is evident (first non-Skill tool use or
        # first message_stop). See stream_parser.negative_trigger_mode.
        provider_options["negative_trigger"] = True

    # Runtime cwd is ephemeral and opaquely-named — no "evals/runs",
    # no "attempt", no suite name — so nothing in the cwd path tips the
    # agent off that it's under evaluation.
    #
    # Fixtureless triggers share a single cwd across attempts: nothing
    # is staged and the agent gets killed within seconds (first skill
    # fire / non-Skill tool / message_stop) before it has time to
    # write meaningful state. Shared cwd path means an identical
    # system prompt prefix across attempts, so the API's prompt cache
    # hits and input cost drops sharply for the suite.
    #
    # Fixtured triggers can't share cwd: attempt 2+ would see leftover
    # state from attempt 1. They take the execute path (per-attempt
    # uuid cwd + fresh fixture copy) and forfeit the cache win.
    trigger_shares_cwd = task.kind == "trigger" and not task.fixture
    if trigger_shares_cwd:
        runtime_cwd = run_tmp_root / "triggers"
        runtime_cwd.mkdir(parents=True, exist_ok=True)
    else:
        runtime_cwd = run_tmp_root / uuid.uuid4().hex[:12]
        _stage_attempt_cwd(task, cfg.evals_dir, runtime_cwd)

    # Stage skills into the attempt cwd so the agent discovers them at
    # the project root (cwd) rather than in a parent directory. This
    # matches how real users set up project-specific skills and avoids
    # permission issues (e.g. OpenCode's external_directory checks) that
    # block access to parent-dir skills.
    provider.stage_run_env(runtime_cwd, cfg, skills_to_exclude=skills_to_exclude)

    # Trigger tasks default to 60s: positives get killed on first skill
    # fire; negatives get killed on first non-Skill tool use or first
    # message_stop (see negative_trigger_mode). Wall-clock is a
    # fallback for cold-start latency — stream signals handle the
    # fast path. 60s accommodates slower providers (e.g. opencode
    # with z-ai/glm-5.1 takes ~8s to first byte).
    if task.timeout_seconds is not None:
        timeout = task.timeout_seconds
    elif task.kind == "trigger":
        timeout = min(60, cfg.default_task_timeout_seconds)
    else:
        timeout = cfg.default_task_timeout_seconds

    sem = None
    if task.concurrency_group and _WORKER_SEMAPHORES is not None:
        sem = _WORKER_SEMAPHORES.get(task.concurrency_group)

    run_result: RunResult | None = None
    error_verdict: str | None = None
    try:
        if sem is not None:
            sem.acquire()
        try:
            run_result = provider.invoke(
                prompt=task.prompt,
                model=model,
                cwd=runtime_cwd,
                provider_options=provider_options,
                stop_on_first_skill=task.stop_on_first_skill,
                timeout_seconds=timeout,
            )
        finally:
            if sem is not None:
                sem.release()
    except ProviderTimeout as exc:
        error_verdict = "timeout"
        # The provider recovers a trajectory from the killed session
        # when possible; archive it so `show` can surface what the
        # agent was doing when the timeout fired. Verdict stays
        # "timeout" — we just don't throw away the evidence.
        run_result = exc.partial_run_result

    attempt_finished = attempt_finished_writer()

    # Mirror the end-of-attempt cwd into the archive (for scoring assertions
    # and `agent-exam show` inspection). Always — even on timeout, the cwd
    # may hold partial output worth keeping.
    #
    # Fixtureless triggers share one cwd across attempts and the agent gets
    # killed before doing meaningful filesystem work — every per-attempt
    # mirror would either be empty or identical to its siblings, so we skip
    # the copy and just create an empty archive dir for any code that
    # expects the path to exist. Fixtured triggers use per-attempt cwds
    # and get the normal mirror.
    archive_cwd = paths.attempt_cwd(task.suite, task.name, attempt_n)
    if trigger_shares_cwd:
        archive_cwd.mkdir(parents=True, exist_ok=True)
    elif runtime_cwd.is_dir():
        _mirror_cwd(runtime_cwd, archive_cwd)
    else:
        archive_cwd.mkdir(parents=True, exist_ok=True)

    # Copy raw NDJSON stream from tmp root to archive tree.
    raw_path = run_result.raw_transcript_path if run_result is not None else None
    if raw_path is not None and raw_path.exists():
        archive_raw = archive_cwd.parent / "raw_stream.jsonl"
        shutil.copy2(raw_path, archive_raw)
        raw_path = archive_raw

    if run_result is not None:
        write_json(
            paths.attempt_json(task.suite, task.name, attempt_n),
            {
                "provider": provider_name,
                "model": run_result.model or model,
                "started_at": attempt_started,
                "finished_at": attempt_finished,
                "raw_transcript_path": str(raw_path) if raw_path else None,
                "metrics": to_json_dict(run_result.metrics),
            },
        )
        write_json(
            paths.trajectory_json(task.suite, task.name, attempt_n),
            {"turns": to_json_dict(run_result.trajectory)},
        )

    return AttemptOutcome(
        suite=task.suite,
        task_name=task.name,
        attempt_n=attempt_n,
        attempt_cwd=archive_cwd,
        run_result=run_result,
        error_verdict=error_verdict,
    )


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _worker_entry(
    task: Task,
    attempt_n: int,
    cfg: Config,
    provider_name: str,
    model: str,
    paths: RunPaths,
    attempt_started: str,
    run_tmp_root: Path,
    skills_to_exclude: frozenset[str] = frozenset(),
) -> AttemptOutcome:
    return _execute_attempt(
        task,
        attempt_n,
        cfg,
        provider_name,
        model,
        paths,
        attempt_started,
        _utc_now_iso,
        run_tmp_root,
        skills_to_exclude,
    )


@dataclass
class PoolPlan:
    tasks: list[Task]
    attempts_per_task: int
    n_parallel: int
    # When True, dispatch in two batches: attempt-1 of every task in
    # parallel (warms each task's prompt cache), then every remaining
    # attempt (2..K) of every task in parallel (all hit the warm
    # cache). The second batch waits for the first to finish. Wins both
    # ways: ~serial cost (one full-price attempt per task, the rest
    # cache-hit) and near-parallel wall-clock (2 batches regardless of
    # K). Useful for trigger sweeps where every attempt shares the
    # system-prompt prefix; pointless for plans where each task has
    # unique fixture state.
    serial_within_task: bool = False


def _build_semaphores(cfg: Config, plan: PoolPlan) -> dict[str, object]:
    """Create a managed Semaphore for each concurrency group actually used.

    Returns an empty dict (and skips Manager startup) when no task in the
    plan tags a group — keeps the simple serial path free of manager
    overhead.
    """
    groups_used = {t.concurrency_group for t in plan.tasks if t.concurrency_group}
    if not groups_used:
        return {}
    manager = mp.Manager()
    semaphores: dict[str, object] = {}
    for name in groups_used:
        limit = cfg.concurrency_groups.get(name)
        if limit is None:
            raise UsageError(
                f"task tags concurrency_group: {name!r} but it's not declared "
                f"in config.yaml's concurrency_groups"
            )
        semaphores[name] = manager.Semaphore(int(limit))
    # Keep the manager alive by attaching it to the dict under a reserved key.
    semaphores["__manager__"] = manager
    return semaphores


def run_plan(
    cfg: Config,
    plan: PoolPlan,
    run_tmp_root: Path,
    provider_name: str,
    model: str,
    paths: RunPaths,
    on_attempt_start: Callable[[Task, int], None] | None = None,
    skills_to_exclude: frozenset[str] = frozenset(),
) -> Iterator[AttemptOutcome]:
    """Execute every (task, attempt) in `plan`, yielding outcomes as they finish.

    Serial when `n_parallel <= 1`; otherwise uses a ProcessPoolExecutor.

    `on_attempt_start` fires from the parent just before each attempt is
    dispatched — used to stream progress so long-running attempts don't look
    hung.
    """
    semaphores = _build_semaphores(cfg, plan)

    def _announce(task: Task, attempt_n: int) -> None:
        if on_attempt_start is not None:
            on_attempt_start(task, attempt_n)

    if plan.n_parallel <= 1:
        for task in plan.tasks:
            for attempt_n in range(1, plan.attempts_per_task + 1):
                _announce(task, attempt_n)
                yield _execute_attempt(
                    task,
                    attempt_n,
                    cfg,
                    provider_name,
                    model,
                    paths,
                    _utc_now_iso(),
                    _utc_now_iso,
                    run_tmp_root,
                    skills_to_exclude,
                )
        return

    ctx = mp.get_context("spawn")
    # Filter out the __manager__ sentinel before passing into workers.
    worker_semaphores = {k: v for k, v in semaphores.items() if k != "__manager__"}

    # Warm-then-fan-out dispatch when serial_within_task is on. Only the
    # first attempt of each task needs to run alone: it pays full input
    # cost and populates the prompt cache for that task's system-prompt
    # prefix. Every later attempt hits that warm cache regardless of the
    # order they run in, so there's no reason to keep them serial with
    # each other. Batch 1 = attempt-1 of every task; batch 2 = every
    # remaining attempt (2..K) of every task, all at once. That keeps
    # the ~serial cost win (one full-price attempt per task, rest
    # cache-hit) while collapsing wall-clock from K serial batches to 2.
    if plan.serial_within_task:
        batches = [
            [(task, 1) for task in plan.tasks],
            [
                (task, n)
                for task in plan.tasks
                for n in range(2, plan.attempts_per_task + 1)
            ],
        ]
    else:
        # All attempts go in one batch — every (task, attempt_n) at once.
        batches = [
            [
                (task, n)
                for task in plan.tasks
                for n in range(1, plan.attempts_per_task + 1)
            ]
        ]

    with ProcessPoolExecutor(
        max_workers=plan.n_parallel,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(worker_semaphores,),
    ) as pool:
        for batch in batches:
            if not batch:
                continue
            futures: list[Future[AttemptOutcome]] = []
            for task, attempt_n in batch:
                _announce(task, attempt_n)
                started = _utc_now_iso()
                futures.append(
                    pool.submit(
                        _worker_entry,
                        task,
                        attempt_n,
                        cfg,
                        provider_name,
                        model,
                        paths,
                        started,
                        run_tmp_root,
                        skills_to_exclude,
                    )
                )
            for fut in as_completed(futures):
                yield fut.result()
