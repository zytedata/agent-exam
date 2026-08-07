"""known_issue: non-restrictive marker that reports verdict but doesn't gate.

At the assertion level: the assertion still runs and its pass/fail shows
in the report, but it's excluded from the task's aggregate verdict.

At the task level: the task's verdict is wrapped — a would-be `pass`
becomes `known_pass` (hint: drop the annotation), a would-be `fail`
becomes `known_issue` (don't gate the suite).
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from agent_exam.errors import UsageError
from agent_exam.report import score_attempt
from agent_exam.schemas import AssertionResult, RunResult
from agent_exam.tasks import Assertion, Task, load_task


def _task(assertions: list[Assertion], *, task_known_issue: str | None = None) -> Task:
    return Task(
        suite="s",
        name="t",
        kind="execute",
        prompt="x",
        description=None,
        assertions=assertions,
        fixture=None,
        env={},
        timeout_seconds=None,
        concurrency_group=None,
        raw={},
        source_path=Path("/tmp/t.yaml"),
        known_issue=task_known_issue,
    )


class _StubCheck:
    """Mimic the assertion registry for tests — pass/fail map by type."""

    def __init__(self, mapping: dict[str, bool]):
        self.mapping = mapping

    def __call__(self, cfg, result, cwd, ctx):  # pragma: no cover - not used
        raise RuntimeError("use monkeypatch instead")


def _fake_run_result() -> RunResult:
    from agent_exam.schemas import Metrics, Tokens

    return RunResult(
        trajectory=[],
        metrics=Metrics(
            wall_time_seconds=0.0,
            tokens=Tokens(input=0, output=0, cache_read=0),
            cost_usd=0.0,
            peak_context=0,
            turn_count=0,
            raw={},
        ),
        raw_transcript_path=None,
    )


@pytest.fixture
def stub_registry(monkeypatch):
    """Stub assertion registry so a type name encodes pass/fail."""
    verdicts: dict[str, bool] = {}

    def fake_get_check(type_name: str):
        return type_name  # any truthy identifier; we ignore it

    def fake_call_check(check, cfg, run_result, cwd, ctx):
        # Config is the type name in our tests.
        return AssertionResult(
            pass_=verdicts[cfg],
            reason="stub",
            details={},
        )

    monkeypatch.setattr("agent_exam.report.get_check", fake_get_check)
    monkeypatch.setattr("agent_exam.report.call_check", fake_call_check)
    return verdicts


def test_assertion_known_issue_excluded_from_aggregate(stub_registry):
    """A failing known_issue assertion does not fail the task."""
    stub_registry["ok"] = True
    stub_registry["broken"] = False
    task = _task(
        [
            Assertion(type="judge", config="ok"),
            Assertion(
                type="judge", config="broken", known_issue="flaky since 2026-04-01"
            ),
        ]
    )
    rep = score_attempt(task, 1, _fake_run_result(), Path("/tmp"))
    assert rep.verdict == "pass"
    # Both assertions reported; the known-issue one carries its marker.
    assert rep.assertions[1].known_issue == "flaky since 2026-04-01"
    assert rep.assertions[1].result.pass_ is False
    # ungated one passes
    assert rep.assertions[0].result.pass_ is True


def test_assertion_known_issue_does_not_force_fail_when_passing(stub_registry):
    """known_issue is non-restrictive: a passing known_issue doesn't flip
    the verdict to error (unlike pytest-xfail). The report surfaces the
    pass; user decides whether to remove the annotation.
    """
    stub_registry["ok1"] = True
    stub_registry["was_flaky_now_ok"] = True
    task = _task(
        [
            Assertion(type="judge", config="ok1"),
            Assertion(type="judge", config="was_flaky_now_ok", known_issue="old bug"),
        ]
    )
    rep = score_attempt(task, 1, _fake_run_result(), Path("/tmp"))
    assert rep.verdict == "pass"
    assert rep.assertions[1].known_issue == "old bug"
    assert rep.assertions[1].result.pass_ is True  # surface; don't hide


def test_ungated_failure_still_fails_with_known_issue_siblings(stub_registry):
    stub_registry["ok"] = True
    stub_registry["regression"] = False
    stub_registry["acknowledged"] = False
    task = _task(
        [
            Assertion(type="judge", config="ok"),
            Assertion(type="judge", config="regression"),  # real fail
            Assertion(type="judge", config="acknowledged", known_issue="skill bug #42"),
        ]
    )
    rep = score_attempt(task, 1, _fake_run_result(), Path("/tmp"))
    assert rep.verdict == "fail"  # the ungated fail gates


def test_task_level_known_issue_failing(stub_registry):
    """Task-level known_issue: failing → known_issue verdict."""
    stub_registry["broken"] = False
    task = _task(
        [Assertion(type="judge", config="broken")],
        task_known_issue="whole task is a known regression",
    )
    rep = score_attempt(task, 1, _fake_run_result(), Path("/tmp"))
    assert rep.verdict == "known_issue"
    assert rep.known_issue == "whole task is a known regression"


def test_task_level_known_issue_passing(stub_registry):
    """Task-level known_issue: passing → unexpected_pass (hint to remove marker)."""
    stub_registry["ok"] = True
    task = _task(
        [Assertion(type="judge", config="ok")],
        task_known_issue="used to fail, check if still needed",
    )
    rep = score_attempt(task, 1, _fake_run_result(), Path("/tmp"))
    assert rep.verdict == "unexpected_pass"
    assert rep.known_issue == "used to fail, check if still needed"


def test_yaml_parses_assertion_known_issue(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(
        dedent(
            """\
            kind: execute
            prompt: do the thing
            assertions:
              - judge: "simple criterion"
              - judge: "criterion with known issue"
                known_issue: "skill bug #42"
            """
        )
    )
    tasks = load_task(p, "my-suite")
    assert len(tasks) == 1
    assertions = tasks[0].assertions
    assert len(assertions) == 2
    assert assertions[0].known_issue is None
    assert assertions[1].known_issue == "skill bug #42"


def test_yaml_parses_task_known_issue(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(
        dedent(
            """\
            kind: execute
            prompt: do the thing
            known_issue: "whole task flaky; track in Linear"
            assertions:
              - judge: "simple"
            """
        )
    )
    tasks = load_task(p, "my-suite")
    assert tasks[0].known_issue == "whole task flaky; track in Linear"


def test_yaml_rejects_non_string_known_issue(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(
        dedent(
            """\
            kind: execute
            prompt: x
            assertions:
              - judge: "x"
                known_issue: 42
            """
        )
    )
    with pytest.raises(UsageError, match="known_issue"):
        load_task(p, "my-suite")


def test_yaml_requires_exactly_one_type_key(tmp_path):
    p = tmp_path / "t.yaml"
    # Two type keys and no meta — should fail.
    p.write_text(
        dedent(
            """\
            kind: execute
            prompt: x
            assertions:
              - judge: "a"
                file_exists: "b"
            """
        )
    )
    with pytest.raises(UsageError, match="exactly one"):
        load_task(p, "my-suite")
