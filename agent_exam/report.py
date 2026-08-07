from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .assertions.registry import call_check, get_check
from .schemas import AssertionResult, RunResult

if TYPE_CHECKING:
    from pathlib import Path

    from .scoring_context import ScoringContext
    from .tasks import Task

# Cap concurrent judge dispatches per attempt. Judges are I/O-bound
# (LLM calls), so a small pool is enough — the ceiling exists mostly
# to avoid stampeding a provider's rate limit when a task has many
# judges. Non-judge assertions are fast and don't gate on this pool.
_JUDGE_PARALLELISM = 8


# pass / fail        — normal outcomes
# known_issue        — task is annotated known_issue and would otherwise fail
# unexpected_pass    — task is annotated known_issue but all ungated assertions
#                      pass (hint to remove the annotation — pytest-style xpass)
# timeout / error    — framework didn't get a trajectory to grade
Verdict = Literal["pass", "fail", "known_issue", "unexpected_pass", "timeout", "error"]


def report_to_dict(
    started: str,
    finished: str,
    scope: dict | None,
    attempts: list[AttemptReport],
) -> dict:
    """Produce the on-disk shape for a single `reports/<timestamp>.json` file.

    Shared between the runner's initial report and `rescore`'s re-grading so
    both produce the same schema.
    """
    return {
        "schema_version": 1,
        "started_at": started,
        "finished_at": finished,
        "scope": scope,
        "attempts": [
            {
                "suite": a.suite,
                "task": a.task,
                "attempt": a.attempt,
                "verdict": a.verdict,
                "known_issue": a.known_issue,
                "assertions": [
                    {
                        "type": x.type,
                        "config": x.config,
                        "pass": x.result.pass_,
                        "reason": x.result.reason,
                        "details": x.result.details,
                        "known_issue": x.known_issue,
                        "skipped_reason": x.skipped_reason,
                    }
                    for x in a.assertions
                ],
            }
            for a in attempts
        ],
    }


@dataclass
class AssertionOutcome:
    type: str
    config: object
    result: AssertionResult
    known_issue: str | None = None
    # When set, the check was not evaluated (e.g. provider filter
    # excluded it). `result` is a placeholder in that case — ignore it.
    # Excluded from the task's aggregate verdict.
    skipped_reason: str | None = None


@dataclass
class AttemptReport:
    suite: str
    task: str
    attempt: int
    verdict: Verdict
    assertions: list[AssertionOutcome]
    known_issue: str | None = None


def score_attempt(
    task: Task,
    attempt_n: int,
    run_result: RunResult | None,
    attempt_cwd: Path,
    error_verdict: Verdict | None = None,
    context: ScoringContext | None = None,
    provider_name: str = "",
) -> AttemptReport:
    """Score a single attempt against its task's assertions.

    `run_result=None` + `error_verdict` handles the timeout / framework-error
    paths where we never got a trajectory to check.

    `provider_name` is the harness that ran the attempt; used to filter
    assertions tagged with `providers: [...]`. Pass empty string to disable
    the filter (tests that don't care about provider-specificity).

    Known-issue + provider-filter semantics:
    - Assertion `known_issue` or provider-mismatch skip: still reported,
      excluded from the task's aggregate pass/fail.
    - Task-level `known_issue`: wraps the final verdict — `unexpected_pass`
      when the aggregate passes (hint to drop the marker), else
      `known_issue`. The aggregate still honors per-assertion markers.
    """
    if error_verdict is not None or run_result is None:
        return AttemptReport(
            suite=task.suite,
            task=task.name,
            attempt=attempt_n,
            verdict=error_verdict or "error",
            assertions=[],
            known_issue=task.known_issue,
        )

    # Preserve per-assertion slots so final order matches task.yaml
    # regardless of which judge finishes first.
    outcomes: list[AssertionOutcome | None] = [None] * len(task.assertions)

    def _eval(i: int, a) -> tuple[int, AssertionOutcome]:
        skipped_reason = _provider_skip_reason(a, provider_name)
        if skipped_reason is not None:
            return i, AssertionOutcome(
                type=a.type,
                config=a.config,
                result=AssertionResult(pass_=True, reason=skipped_reason, details={}),
                known_issue=a.known_issue,
                skipped_reason=skipped_reason,
            )
        check = get_check(a.type)
        # `parsed_config` is the typed pydantic model populated by
        # `_parse_assertion`. None only when an Assertion is built
        # directly (in tests) without going through parsing — fall back
        # to the raw `config` there so monkey-patched check stubs keep
        # working.
        config = a.parsed_config if a.parsed_config is not None else a.config
        res = call_check(check, config, run_result, attempt_cwd, context)
        return i, AssertionOutcome(
            type=a.type,
            config=a.config,
            result=res,
            known_issue=a.known_issue,
        )

    # Judges dispatch LLM calls. With 2+ judges, running the first
    # one alone before spawning the parallel cohort populates
    # Anthropic's prompt cache with the shared (trajectory + final
    # message) prefix; the rest then hit a warm cache instead of
    # stampeding a cold one. Same pattern proved out on trigger
    # sweeps (see research doc, "Dispatch shape" section).
    #
    # Non-judge assertions are deterministic and cheap; we dispatch
    # them in the same pool for code uniformity, but they don't
    # benefit from caching.
    judge_indices = [i for i, a in enumerate(task.assertions) if a.type == "judge"]
    if len(judge_indices) <= 1:
        # Zero or one judge → no cache opportunity, no parallelism.
        for i, a in enumerate(task.assertions):
            idx, oc = _eval(i, a)
            outcomes[idx] = oc
    else:
        first_judge_i = judge_indices[0]
        # Warmup: run the first judge alone to seed the cache.
        idx, oc = _eval(first_judge_i, task.assertions[first_judge_i])
        outcomes[idx] = oc
        # Everyone else runs in parallel against the warm cache.
        remaining = [
            (i, a) for i, a in enumerate(task.assertions) if i != first_judge_i
        ]
        if remaining:
            workers = min(_JUDGE_PARALLELISM, len(remaining))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for idx, oc in pool.map(lambda ia: _eval(*ia), remaining):
                    outcomes[idx] = oc

    all_ungated_pass = True
    for a, oc in zip(task.assertions, outcomes, strict=True):
        if a.known_issue is None and oc.skipped_reason is None and not oc.result.pass_:
            all_ungated_pass = False

    if task.known_issue is not None:
        verdict: Verdict = "unexpected_pass" if all_ungated_pass else "known_issue"
    else:
        verdict = "pass" if all_ungated_pass else "fail"

    return AttemptReport(
        suite=task.suite,
        task=task.name,
        attempt=attempt_n,
        verdict=verdict,
        assertions=outcomes,
        known_issue=task.known_issue,
    )


def _provider_skip_reason(a, provider_name: str) -> str | None:
    """Return a skip message if this assertion doesn't apply to the current
    provider, else None. Empty `provider_name` disables the filter.
    """
    if a.providers is None or not provider_name:
        return None
    if provider_name in a.providers:
        return None
    return f"provider-specific: {' or '.join(a.providers)} only"
