"""Per-assertion providers: filter.

An assertion tagged `providers: [claude_code]` is skipped when the current
run's provider doesn't match. Skipped assertions don't evaluate (no LLM
calls), appear in the report with a `skipped_reason`, and are excluded
from the task's aggregate verdict.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from agent_exam.errors import UsageError
from agent_exam.report import score_attempt
from agent_exam.schemas import AssertionResult, Metrics, RunResult, Tokens
from agent_exam.tasks import Assertion, Task, load_task


def _task(assertions: list[Assertion]) -> Task:
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
    )


def _fake_run_result() -> RunResult:
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
    verdicts: dict[str, bool] = {}
    called: list[str] = []

    def fake_get_check(type_name: str):
        return type_name

    def fake_call_check(check, cfg, run_result, cwd, ctx):
        called.append(cfg)
        return AssertionResult(pass_=verdicts[cfg], reason="stub", details={})

    monkeypatch.setattr("agent_exam.report.get_check", fake_get_check)
    monkeypatch.setattr("agent_exam.report.call_check", fake_call_check)
    return verdicts, called


def test_assertion_skipped_when_provider_not_in_list(stub_registry):
    verdicts, called = stub_registry
    verdicts["portable"] = True  # other assertion still runs
    task = _task(
        [
            Assertion(type="judge", config="portable"),
            Assertion(
                type="tool_called",
                config="Bash",
                providers=["claude_code"],
            ),
        ]
    )
    rep = score_attempt(
        task, 1, _fake_run_result(), Path("/tmp"), provider_name="dummy"
    )
    # The provider-filtered assertion was not evaluated.
    assert called == ["portable"]
    # It still appears in the report with a skipped_reason.
    assert rep.assertions[1].skipped_reason is not None
    assert "claude_code" in rep.assertions[1].skipped_reason
    # Aggregate ignores the skipped one — task passes on the portable one.
    assert rep.verdict == "pass"


def test_assertion_runs_when_provider_matches(stub_registry):
    verdicts, called = stub_registry
    verdicts["matches"] = True
    task = _task(
        [
            Assertion(
                type="judge",
                config="matches",
                providers=["claude_code"],
            )
        ]
    )
    rep = score_attempt(
        task, 1, _fake_run_result(), Path("/tmp"), provider_name="claude_code"
    )
    assert called == ["matches"]
    assert rep.assertions[0].skipped_reason is None
    assert rep.verdict == "pass"


def test_skipped_assertion_does_not_gate_failure(stub_registry):
    """A passing known-good assertion combined with a would-be-failing
    provider-filtered assertion: task still passes, because the filtered
    one is skipped, not evaluated.
    """
    verdicts, called = stub_registry
    verdicts["ok"] = True
    # If this ran, it would fail — but the providers filter skips it.
    verdicts["would_fail"] = False
    task = _task(
        [
            Assertion(type="judge", config="ok"),
            Assertion(
                type="judge",
                config="would_fail",
                providers=["claude_code"],
            ),
        ]
    )
    rep = score_attempt(
        task, 1, _fake_run_result(), Path("/tmp"), provider_name="dummy"
    )
    assert called == ["ok"]
    assert rep.verdict == "pass"


def test_empty_provider_name_disables_filter(stub_registry):
    """When called without a provider (e.g. from tests), the filter is
    disabled and every assertion runs.
    """
    verdicts, called = stub_registry
    verdicts["ok"] = True
    task = _task([Assertion(type="judge", config="ok", providers=["claude_code"])])
    rep = score_attempt(task, 1, _fake_run_result(), Path("/tmp"))
    assert called == ["ok"]
    assert rep.assertions[0].skipped_reason is None


def test_yaml_parses_assertion_providers(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(
        dedent(
            """\
            kind: execute
            prompt: x
            assertions:
              - judge: "portable"
              - tool_called: Bash
                providers: [claude_code]
            """
        )
    )
    tasks = load_task(p, "s")
    assert tasks[0].assertions[0].providers is None
    assert tasks[0].assertions[1].providers == ["claude_code"]


def test_yaml_rejects_empty_providers_list(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(
        dedent(
            """\
            kind: execute
            prompt: x
            assertions:
              - tool_called: Bash
                providers: []
            """
        )
    )
    with pytest.raises(UsageError, match="non-empty list"):
        load_task(p, "s")


def test_yaml_rejects_non_list_providers(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(
        dedent(
            """\
            kind: execute
            prompt: x
            assertions:
              - tool_called: Bash
                providers: claude_code
            """
        )
    )
    with pytest.raises(UsageError, match="non-empty list"):
        load_task(p, "s")
