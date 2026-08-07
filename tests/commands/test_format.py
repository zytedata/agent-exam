from __future__ import annotations

from datetime import datetime, timezone

from agent_exam.commands._format import (
    THRESHOLDS,
    _visual_width,
    delta_marker,
    fmt_cost,
    fmt_ctx,
    fmt_iso_age,
    fmt_pass_ratio,
    fmt_wall,
    iso_duration,
    render_table,
)
from agent_exam.commands.show import _render_assertion


def test_fmt_cost():
    assert fmt_cost(0.025) == "$0.0250"
    assert fmt_cost(None) == "   ?   "


def test_fmt_wall():
    assert fmt_wall(12.1) == "12.1s"
    assert fmt_wall(62) == "1m02s"
    assert fmt_wall(3725) == "1h02m"
    assert fmt_wall(None) == "  —  "


def test_fmt_ctx():
    assert fmt_ctx(500) == "500"
    assert fmt_ctx(1500) == "1.5k"
    assert fmt_ctx(1_500_000) == "1.5M"


def test_delta_marker_within_threshold():
    # cost 10% change, threshold 15% → no marker.
    assert delta_marker(0.011, 0.010, "cost_usd") == ""


def test_delta_marker_above_threshold():
    assert delta_marker(0.015, 0.010, "cost_usd") == "+50%"
    assert delta_marker(0.007, 0.010, "cost_usd") == "-30%"


def test_delta_marker_new_and_gone():
    assert delta_marker(10, None, "cost_usd") == "new"
    assert delta_marker(None, 10, "cost_usd") == "gone"


def test_delta_marker_zero_old():
    assert delta_marker(0.01, 0, "cost_usd") == "new"


def test_iso_duration():
    assert iso_duration("2026-04-23T14:23:00Z", "2026-04-23T14:23:42Z") == 42
    assert iso_duration("", "x") is None
    assert iso_duration("not-iso", "also-not-iso") is None


def test_fmt_iso_age_buckets():
    now = datetime(2026, 4, 23, 15, 0, 0, tzinfo=timezone.utc)
    assert fmt_iso_age("2026-04-23T14:59:30+00:00", now=now) == "30s ago"
    assert fmt_iso_age("2026-04-23T14:45:00+00:00", now=now) == "15m ago"
    assert fmt_iso_age("2026-04-23T12:00:00+00:00", now=now) == "3h ago"
    assert fmt_iso_age("2026-04-20T12:00:00+00:00", now=now) == "3d ago"


def test_render_table_pads_columns():
    out = render_table([["a", "bb", "ccc"], ["aaaa", "b", "c"]], ["X", "Y", "Z"])
    lines = out.splitlines()
    # All lines should have the same width.
    assert len({len(line) for line in lines}) == 1


def test_thresholds_shape():
    assert set(THRESHOLDS) == {"cost_usd", "peak_context", "wall_time_seconds"}


# --- fmt_pass_ratio + render_table ANSI handling ---------------------------


def _assertion(passed: bool) -> dict:
    return {"type": "x", "config": None, "pass": passed, "reason": "", "details": {}}


def _trial(verdict: str, n_pass: int, n_total: int) -> dict:
    asserts = [_assertion(True) for _ in range(n_pass)] + [
        _assertion(False) for _ in range(n_total - n_pass)
    ]
    return {
        "suite": "s",
        "task": "t",
        "trial": 1,
        "verdict": verdict,
        "assertions": asserts,
    }


def test_pass_ratio_all_pass_is_green():
    out = fmt_pass_ratio(_trial("pass", 6, 6))
    assert "6/6" in out
    assert "\x1b[32m" in out  # green


def test_pass_ratio_none_pass_is_red():
    out = fmt_pass_ratio(_trial("fail", 0, 6))
    assert "0/6" in out
    assert "\x1b[31m" in out  # red


def test_pass_ratio_partial_is_yellow():
    out = fmt_pass_ratio(_trial("fail", 4, 6))
    assert "4/6" in out
    assert "\x1b[33m" in out  # yellow


def test_pass_ratio_timeout_shows_label_in_red():
    out = fmt_pass_ratio({"verdict": "timeout", "assertions": []})
    assert "TIMEOUT" in out
    assert "\x1b[31m" in out


def test_pass_ratio_empty_assertions_falls_back_to_verdict():
    # Edge: a task with zero assertions (valid — runner still records pass).
    out = fmt_pass_ratio({"verdict": "pass", "assertions": []})
    assert out == "PASS"


def test_pass_ratio_excludes_known_issue_from_ratio():
    """A known_issue assertion that passed is NOT counted in the ratio —
    otherwise 4 regular passes + 1 unexpected-pass would render as
    a misleading 5/5.
    """
    assertions = [
        _assertion(True),
        _assertion(True),
        _assertion(True),
        _assertion(True),
        {
            "type": "x",
            "config": None,
            "pass": True,
            "reason": "",
            "details": {},
            "known_issue": "tracked as bug #42",
        },
    ]
    entry = {"verdict": "pass", "assertions": assertions}
    out = fmt_pass_ratio(entry)
    assert "4/4" in out
    assert "+1 xpass" in out
    assert "\x1b[32m" in out  # green — ungated all passed


def test_pass_ratio_excludes_skipped_from_ratio():
    """provider-filtered skipped assertions also don't count."""
    assertions = [
        _assertion(True),
        _assertion(True),
        {
            "type": "x",
            "config": None,
            "pass": True,
            "reason": "",
            "details": {},
            "skipped_reason": "provider mismatch",
        },
    ]
    out = fmt_pass_ratio({"verdict": "pass", "assertions": assertions})
    assert "2/2" in out
    assert "1 skipped" in out


def test_render_table_measures_visual_width_excluding_ansi():
    import click as _click

    rows = [["plain", _click.style("colored", fg="red")]]
    out = render_table(rows, ["A", "B"])
    # Header row + data row, both should end up with identical visible width.
    lines = out.splitlines()
    assert _visual_width(lines[0]) == _visual_width(lines[1])


# --- _render_assertion -------------------------------------------------------


def _capture_render(a: dict) -> str:
    """Return the text _render_assertion would print, ANSI stripped."""
    import re

    import click

    lines = []
    original = click.echo

    def capturing_echo(msg="", **kw):
        lines.append(re.sub(r"\x1b\[[0-9;]*m", "", str(msg)))

    click.echo = capturing_echo
    try:
        _render_assertion(a)
    finally:
        click.echo = original
    return "\n".join(lines)


def _judge_assertion(type_: str, passed: bool, verdict: str, reasoning: str) -> dict:
    return {
        "type": type_,
        "config": "did the thing",
        "pass": passed,
        "reason": f"{type_} said {verdict}",
        "details": {"verdict": verdict, "reasoning": reasoning},
    }


def test_render_assertion_judge_shows_reasoning():
    out = _capture_render(
        _judge_assertion("judge", True, "YES", "The output looks correct.")
    )
    assert "The output looks correct." in out
    assert "YES" in out
    # The generic `reason` field ("judge said YES") should NOT appear separately.
    assert "judge said YES" not in out


def test_render_assertion_judge_agent_shows_reasoning():
    out = _capture_render(
        _judge_assertion("judge_agent", True, "YES", "File was created as expected.")
    )
    assert "File was created as expected." in out
    assert "YES" in out
    assert "judge_agent said YES" not in out


def test_render_assertion_judge_agent_fail_shows_reasoning():
    out = _capture_render(
        _judge_assertion("judge_agent", False, "NO", "Output was empty.")
    )
    assert "Output was empty." in out
    assert "NO" in out


def test_render_assertion_non_judge_shows_reason_not_details():
    a = {
        "type": "file_exists",
        "config": "out.txt",
        "pass": False,
        "reason": "out.txt not found",
        "details": {},
    }
    out = _capture_render(a)
    assert "out.txt not found" in out
