from __future__ import annotations

from pathlib import Path

from fixtures.canned_run_result import (
    assistant_turn,
    run_result,
    skill_inv,
    text,
    thinking,
    tool_call,
    user_turn,
)

from agent_exam.serde import (
    metrics_from_dict,
    run_result_from_artifacts,
    to_json_dict,
    trajectory_from_dict,
    write_json,
)


def _sample_trajectory():
    sub = assistant_turn(text("subagent reply"))
    parent = assistant_turn(
        thinking("let me think"),
        text("I'll call a skill"),
        tool_call(
            "Skill",
            tool_use_id="tu_parent",
            input_={"skill": "x"},
            result="subagent returned",
            subagent=[sub],
        ),
        skill_invocations=[skill_inv("x", trigger_kind="slash_command")],
    )
    return [user_turn("go"), parent]


def test_trajectory_roundtrip():
    trajectory = _sample_trajectory()
    rr = run_result(
        trajectory, wall_time_seconds=3.14, cost_usd=0.025, peak_context=1234
    )
    as_dict = to_json_dict(rr)
    # Roundtrip trajectory.
    reborn_traj = trajectory_from_dict({"turns": as_dict["trajectory"]})
    assert len(reborn_traj) == 2
    parent = reborn_traj[1]
    assert parent.role == "assistant"
    assert parent.model == rr.trajectory[1].model
    # Blocks preserved with types intact.
    assert parent.content[0].__class__.__name__ == "ThinkingBlock"
    assert parent.content[1].__class__.__name__ == "TextBlock"
    call = parent.content[2]
    assert call.__class__.__name__ == "ToolCallBlock"
    assert call.tool_use_id == "tu_parent"
    # Subagent restored as a list of Turns.
    assert call.subagent is not None
    assert len(call.subagent) == 1
    assert call.subagent[0].content[0].text == "subagent reply"
    # Skill invocations survive.
    assert parent.skill_invocations[0].skill_name == "x"


def test_metrics_roundtrip():
    rr = run_result([], cost_usd=0.042, peak_context=9999, wall_time_seconds=2.5)
    d = to_json_dict(rr.metrics)
    reborn = metrics_from_dict(d)
    assert reborn.cost_usd == 0.042
    assert reborn.peak_context == 9999
    assert reborn.wall_time_seconds == 2.5
    assert reborn.tokens.input == 0


def test_run_result_from_artifacts(tmp_path):
    trajectory = _sample_trajectory()
    rr = run_result(trajectory, cost_usd=0.01)
    rr.raw_transcript_path = Path("/tmp/session.jsonl")
    trial_path = tmp_path / "trial.json"
    traj_path = tmp_path / "trajectory.json"
    write_json(
        trial_path,
        {
            "provider": "dummy",
            "model": "x",
            "started_at": "",
            "finished_at": "",
            "raw_transcript_path": "/tmp/session.jsonl",
            "metrics": to_json_dict(rr.metrics),
        },
    )
    write_json(traj_path, {"turns": to_json_dict(rr.trajectory)})

    import json

    trial = json.loads(trial_path.read_text())
    traj = json.loads(traj_path.read_text())
    reborn = run_result_from_artifacts(trial, traj)
    assert reborn.metrics.cost_usd == 0.01
    assert str(reborn.raw_transcript_path) == "/tmp/session.jsonl"
    assert len(reborn.trajectory) == 2


def test_unknown_block_type_skipped():
    # Forward-compat: readers skip unknown `type` values.
    traj = {
        "turns": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "known"},
                    {"type": "some-future-block", "foo": "bar"},
                ],
            }
        ]
    }
    reborn = trajectory_from_dict(traj)
    assert len(reborn) == 1
    assert [b.__class__.__name__ for b in reborn[0].content] == ["TextBlock"]
