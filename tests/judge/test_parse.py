from __future__ import annotations

import pytest

from agent_exam.judge.parse import parse_verdict


def test_canonical_yes():
    v, r = parse_verdict("Reasoning here.\nVERDICT: YES")
    assert v == "YES"
    assert r == "Reasoning here."


def test_canonical_no():
    v, _r = parse_verdict("Did not work.\nVERDICT: NO")
    assert v == "NO"


def test_canonical_unclear():
    v, _r = parse_verdict("Not sure.\nVERDICT: UNCLEAR")
    assert v == "UNCLEAR"


def test_extra_whitespace():
    v, _ = parse_verdict("ok\n  VERDICT  :   YES  \n\n")
    assert v == "YES"


def test_case_insensitive():
    v, _ = parse_verdict("ok\nverdict: yes")
    assert v == "YES"


def test_missing_verdict_returns_unclear():
    v, r = parse_verdict("Just some text, no verdict marker here.")
    assert v == "UNCLEAR"
    assert "Just some text" in r


def test_empty_response():
    v, _r = parse_verdict("")
    assert v == "UNCLEAR"


def test_multiline_reasoning_preserved():
    v, r = parse_verdict("Line 1.\nLine 2.\nLine 3.\nVERDICT: YES")
    assert v == "YES"
    assert r == "Line 1.\nLine 2.\nLine 3."


def test_verdict_not_on_last_nonempty_line_fails():
    # Spec: "Look for VERDICT: on the last non-empty line."
    v, _r = parse_verdict("VERDICT: YES\nsomething after")
    assert v == "UNCLEAR", "verdict must be on last non-empty line"


@pytest.mark.parametrize("suffix", ["", ".", " ", "!"])
def test_trailing_noise_tolerated(suffix: str):
    v, _ = parse_verdict(f"reason\nVERDICT: YES{suffix}")
    assert v == "YES"
