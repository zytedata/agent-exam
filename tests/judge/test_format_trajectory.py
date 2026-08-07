from __future__ import annotations

from fixtures.canned_run_result import (
    assistant_turn,
    text,
    thinking,
    tool_call,
    user_turn,
)

from agent_exam.judge.format_trajectory import (
    final_output_text,
    format_trajectory,
)


def test_basic_render_has_turns_and_tool_calls():
    trajectory = [
        user_turn("do the thing"),
        assistant_turn(
            thinking("thinking through it"),
            text("will do"),
            tool_call("Read", input_={"file_path": "/a"}, result="file contents"),
        ),
    ]
    rendered = format_trajectory(trajectory)
    assert "[turn 0, user]" in rendered
    assert "[turn 1, assistant]" in rendered
    assert "thinking: thinking through it" in rendered
    assert "text: will do" in rendered
    assert "tool_use Read(" in rendered
    assert "file contents" in rendered


def test_tool_result_truncation_keeps_head_and_tail():
    long_result = "A" * 400 + "B" * 400 + "C" * 400
    trajectory = [assistant_turn(tool_call("Read", result=long_result))]
    rendered = format_trajectory(trajectory)
    assert "chars removed" in rendered
    assert "AAAA" in rendered  # head preserved
    assert "CCCC" in rendered  # tail preserved


def test_subagent_indented():
    sub = assistant_turn(text("sub output"))
    parent = assistant_turn(tool_call("Skill", input_={"skill": "x"}, subagent=[sub]))
    rendered = format_trajectory([parent])
    sub_line = next(line for line in rendered.splitlines() if "sub output" in line)
    assert sub_line.startswith("    "), "subagent turns should render indented"


def test_truncates_when_over_limit():
    # Build a trajectory whose formatted text clearly exceeds max_chars.
    big = assistant_turn(text("x" * 2000))
    rendered = format_trajectory([big] * 50, max_chars=5000)
    assert "[trajectory truncated]" in rendered
    assert len(rendered) < 6000


def test_final_output_text_picks_last_assistant_text():
    trajectory = [
        user_turn("u1"),
        assistant_turn(text("first")),
        user_turn("u2"),
        assistant_turn(text("second"), text("third")),
    ]
    assert final_output_text(trajectory) == "second\nthird"


def test_final_output_text_ignores_tool_only_assistant_turn():
    # Last assistant turn is just a tool call → returns its (empty) text.
    trajectory = [
        user_turn("u"),
        assistant_turn(text("hello")),
        assistant_turn(tool_call("Read")),
    ]
    assert final_output_text(trajectory) == ""
