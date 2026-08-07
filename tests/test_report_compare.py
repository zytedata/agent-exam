from __future__ import annotations

from agent_exam.report_compare import (
    AttemptKey,
    assertion_key,
    compare_reports,
)


def _attempt(suite, task, attempt, verdict, assertions=None):
    return {
        "suite": suite,
        "task": task,
        "attempt": attempt,
        "verdict": verdict,
        "assertions": assertions or [],
    }


def _assertion(type_, config, pass_=True):
    return {"type": type_, "config": config, "pass": pass_, "reason": "", "details": {}}


def test_assertion_keys_by_disambiguator():
    assert assertion_key("file_exists", "a/b.txt") == "file_exists:a/b.txt"
    assert assertion_key("file_exists", {"path": "a/b.txt"}) == "file_exists:a/b.txt"
    assert (
        assertion_key("file_contains", {"path": "x.py", "pattern": "foo"})
        == "file_contains:x.py"
    )
    assert assertion_key("tool_called", "Read") == "tool_called:Read"
    assert (
        assertion_key("tool_count", {"name": "Read", "exactly": 2}) == "tool_count:Read"
    )
    assert assertion_key("judge", "long criterion here" + "x" * 100).startswith(
        "judge:long criterion here"
    )
    assert assertion_key("first_skill", "scrape") == "first_skill"


def test_verdict_changes_pairs_matching_attempts():
    before = [_attempt("s", "t", 1, "pass"), _attempt("s", "t", 2, "fail")]
    after = [_attempt("s", "t", 1, "fail"), _attempt("s", "t", 2, "fail")]
    r = compare_reports(before, after)
    assert len(r.verdict_changes) == 1
    assert r.verdict_changes[0].attempt == AttemptKey("s", "t", 1)
    assert (r.verdict_changes[0].before, r.verdict_changes[0].after) == ("pass", "fail")


def test_metric_deltas_require_meta_and_threshold():
    ak = AttemptKey("s", "t", 1)
    before = [_attempt("s", "t", 1, "pass")]
    after = [_attempt("s", "t", 1, "pass")]
    # Cost +40% with 15% threshold → flagged.
    b_meta = {
        ak: {
            "metrics": {"cost_usd": 0.01, "peak_context": 1000, "wall_time_seconds": 10}
        }
    }
    a_meta = {
        ak: {
            "metrics": {
                "cost_usd": 0.014,
                "peak_context": 1050,
                "wall_time_seconds": 10.5,
            }
        }
    }
    r = compare_reports(
        before, after, before_attempt_meta=b_meta, after_attempt_meta=a_meta
    )
    metrics_flagged = {d.metric for d in r.metric_deltas}
    assert "cost_usd" in metrics_flagged
    assert "peak_context" not in metrics_flagged
    assert "wall_time_seconds" not in metrics_flagged


def test_grader_changes_added_removed_definition_changed():
    before = [
        _attempt(
            "s",
            "t",
            1,
            "pass",
            [
                _assertion("file_exists", "a.txt"),
                _assertion("tool_called", "Read"),
                _assertion("judge", "be accurate"),
            ],
        )
    ]
    after = [
        _attempt(
            "s",
            "t",
            1,
            "pass",
            [
                _assertion("file_exists", "a.txt"),  # unchanged
                # tool_called removed
                _assertion("judge", "be accurate"),  # same key
                _assertion("file_contains", {"path": "a.txt", "pattern": "x"}),  # added
            ],
        )
    ]
    r = compare_reports(before, after)
    kinds = {(g.kind, g.assertion_key) for g in r.grader_changes}
    assert ("added", "file_contains:a.txt") in kinds
    assert ("removed", "tool_called:Read") in kinds


def test_grader_changes_are_per_task_not_per_attempt():
    # Same task assertions in both runs, but run B has extra attempts.
    before = [_attempt("s", "t", 1, "pass", [_assertion("file_exists", "a.txt")])]
    after = [
        _attempt("s", "t", 1, "pass", [_assertion("file_exists", "a.txt")]),
        _attempt("s", "t", 2, "pass", [_assertion("file_exists", "a.txt")]),
    ]
    r = compare_reports(before, after)
    assert r.grader_changes == [], "no spurious grader changes from extra attempts"


def test_definition_changed_flagged_when_config_differs_same_key():
    before = [
        _attempt(
            "s",
            "t",
            1,
            "pass",
            [_assertion("file_contains", {"path": "x.py", "pattern": "foo"})],
        )
    ]
    after = [
        _attempt(
            "s",
            "t",
            1,
            "pass",
            [_assertion("file_contains", {"path": "x.py", "pattern": "bar"})],
        )
    ]
    r = compare_reports(before, after)
    assert len(r.grader_changes) == 1
    assert r.grader_changes[0].kind == "definition-changed"
