from __future__ import annotations

from agent_exam.judge.prompt import build_prompt


def test_includes_all_sections():
    p = build_prompt("Be correct.", "final message", "traj text")
    assert "CRITERION:" in p
    assert "Be correct." in p
    assert "AGENT'S FINAL MESSAGE:" in p
    assert "final message" in p
    assert "AGENT'S TRAJECTORY:" in p
    assert "traj text" in p
    assert "VERDICT: YES | NO | UNCLEAR" in p


def test_omits_trajectory_when_disabled():
    p = build_prompt("c", "out", "traj", include_trajectory=False)
    assert "AGENT'S TRAJECTORY:" not in p
    assert "traj" not in p


def test_empty_output_marker():
    p = build_prompt("c", "", "traj")
    assert "(empty)" in p


def test_empty_trajectory_marker():
    p = build_prompt("c", "out", "")
    assert "(empty trajectory)" in p
