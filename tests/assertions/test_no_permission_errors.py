"""no_permission_errors: flag any tool call blocked by the harness's
permission system.
"""

from __future__ import annotations

from pathlib import Path

from agent_exam.assertions.no_permission_errors import (
    NoPermissionErrorsConfig,
    check,
)
from agent_exam.schemas import (
    Metrics,
    RunResult,
    Tokens,
    ToolCallBlock,
    Turn,
)


def _cfg() -> NoPermissionErrorsConfig:
    return NoPermissionErrorsConfig()


def _run_result(tool_calls: list[ToolCallBlock]) -> RunResult:
    return RunResult(
        trajectory=[
            Turn(
                role="assistant",
                content=tool_calls,
                model=None,
                tokens=None,
                context=None,
                started_at=None,
                skill_invocations=[],
            )
        ],
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


def _tool_call(name: str, status: str, result: str = "") -> ToolCallBlock:
    return ToolCallBlock(
        tool_use_id="t1",
        name=name,
        input={},
        status=status,
        result=result,
        started_at=None,
        duration_ms=None,
    )


def test_passes_when_no_permission_denials():
    rr = _run_result(
        [_tool_call("Bash", "ok", "hello"), _tool_call("Read", "error", "ENOENT")]
    )
    res = check(_cfg(), rr, Path("/tmp"))
    assert res.pass_ is True
    assert "no permission-denied" in res.reason


def test_fails_on_single_permission_denial():
    rr = _run_result(
        [
            _tool_call("Bash", "ok", ""),
            _tool_call(
                "Read",
                "permission_denied",
                "Claude requested permissions to read from /Users/x/.foo",
            ),
        ]
    )
    res = check(_cfg(), rr, Path("/tmp"))
    assert res.pass_ is False
    assert "Read" in res.reason
    assert len(res.details["offending"]) == 1


def test_fails_on_multiple_permission_denials_reports_all():
    rr = _run_result(
        [
            _tool_call("Read", "permission_denied", "..."),
            _tool_call("Bash", "permission_denied", "requires approval"),
        ]
    )
    res = check(_cfg(), rr, Path("/tmp"))
    assert res.pass_ is False
    assert len(res.details["offending"]) == 2
    assert {o["name"] for o in res.details["offending"]} == {"Read", "Bash"}


def test_tool_status_counter():
    """_count_tool_statuses sums tool calls by status across nested
    subagents, excludes the synthetic `Skill` redirect, and tracks
    ok/error/permission_denied/rejected separately.
    """
    from agent_exam.providers.claude_code.transcripts import (
        _count_tool_statuses,
    )

    subagent = [
        Turn(
            role="assistant",
            content=[_tool_call("Grep", "ok"), _tool_call("Bash", "error")],
            model=None,
            tokens=None,
            context=None,
            started_at=None,
            skill_invocations=[],
        )
    ]
    outer = ToolCallBlock(
        tool_use_id="t-outer",
        name="Agent",
        input={},
        status="ok",
        result="",
        started_at=None,
        duration_ms=None,
        subagent=subagent,
    )
    trajectory = [
        Turn(
            role="assistant",
            content=[
                outer,
                _tool_call("Bash", "ok"),
                _tool_call("Read", "permission_denied"),
                _tool_call("Edit", "rejected"),
                # Skill redirects are excluded from all counters.
                _tool_call("Skill", "error", "Execute skill: foo"),
            ],
            model=None,
            tokens=None,
            context=None,
            started_at=None,
            skill_invocations=[],
        )
    ]
    total, errs, denied, rejected = _count_tool_statuses(trajectory)
    # outer + 2 subagent + 3 outer non-Skill = 6
    assert total == 6
    assert errs == 1  # one from subagent
    assert denied == 1
    assert rejected == 1


def test_permission_detector_patterns():
    """The Claude Code provider's permission-denial detector should
    match the real phrasings surfaced in tool results."""
    from agent_exam.providers.claude_code.transcripts import (
        _is_permission_denial,
    )

    assert _is_permission_denial(
        "Claude requested permissions to read from /Users/x/.scrapinghub.yml, "
        "but you haven't granted it yet."
    )
    assert _is_permission_denial(
        "This Bash command contains multiple operations. The following parts "
        "require approval: env, grep ..."
    )
    # Single-command denial: different phrasing ("requires approval" with
    # the `s`) — used to slip past the detector when we only had
    # "require approval" (base form).
    assert _is_permission_denial("This command requires approval")
    assert _is_permission_denial("The permission prompt keeps getting declined.")
    # Generic errors should not be flagged as permission denials.
    assert not _is_permission_denial("ENOENT: no such file")
    assert not _is_permission_denial("")
    assert not _is_permission_denial(None)


def test_rejection_detector_patterns():
    """Harness-level input-validation rejections come wrapped in a
    distinctive envelope. Patterns captured from a real probe.
    """
    from agent_exam.providers.claude_code.transcripts import (
        _is_harness_rejection,
        _is_permission_denial,
    )

    # Exact shape emitted when a required parameter is missing.
    msg = (
        "<tool_use_error>InputValidationError: Edit failed due to the "
        "following issue:\nThe required parameter `new_string` is missing"
        "</tool_use_error>"
    )
    assert _is_harness_rejection(msg)
    # Should NOT be classified as a permission denial — different concept.
    assert not _is_permission_denial(msg)

    # Generic tool-side errors don't look like rejections.
    assert not _is_harness_rejection("ENOENT: no such file")
    assert not _is_harness_rejection("")
    assert not _is_harness_rejection(None)
