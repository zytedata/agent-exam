"""End-to-end test for `agent-exam rescore`.

Builds a minimal project root (pyproject.toml + evals/ tree + one run dir
with frozen attempt artifacts) inside tmp_path, swaps in a stub provider
so no real LLM calls happen, then drives `commands.rescore.run()`
directly and inspects the new report file.

Exercises the bug that originally slipped through at step 8:
`load_task` returns `list[Task]` (for trigger expansion), not a single
Task — rescore has to match the archived attempt by name.
"""

from __future__ import annotations

import json
from textwrap import dedent
from typing import TYPE_CHECKING

import pytest

from agent_exam.commands import rescore
from agent_exam.errors import UsageError
from agent_exam.providers.base import Provider
from agent_exam.schemas import Metrics, RunResult, TextBlock, Tokens, Turn

if TYPE_CHECKING:
    from pathlib import Path


class _StubJudgeProvider(Provider):
    """Provider stub: routes `invoke()` through a verdict map, records calls.

    Used so rescore tests can drive `commands.rescore.run()` end-to-end
    without spawning a real harness — `get_provider` is monkey-patched
    to return an instance of this class.
    """

    name = "claude_code"

    def __init__(self, verdict_map: dict[str, str], calls: list[tuple[str, str]]):
        self.verdict_map = verdict_map
        self.calls = calls

    def invoke(
        self,
        prompt: str,
        model: str,
        cwd: Path,
        provider_options: dict,
        stop_on_first_skill: bool,
        timeout_seconds: int,
    ) -> RunResult:
        response = self._response_for(prompt)
        return RunResult(
            trajectory=[
                Turn(role="user", content=[TextBlock(text=prompt)]),
                Turn(
                    role="assistant",
                    content=[TextBlock(text=response)],
                    model=model,
                    tokens=Tokens(input=0, output=0, cache_read=0),
                    context=0,
                ),
            ],
            metrics=Metrics(
                wall_time_seconds=0.0,
                tokens=Tokens(input=0, output=0, cache_read=0),
                cost_usd=0.0,
                peak_context=0,
                turn_count=1,
                raw={"provider": "stub"},
            ),
        )

    def _response_for(self, prompt: str) -> str:
        for key, response in self.verdict_map.items():
            if key in prompt:
                self.calls.append((key, prompt))
                if response in ("YES", "NO", "UNCLEAR"):
                    return f"reasoning cites {key}\nVERDICT: {response}"
                return response
        self.calls.append(("(default)", prompt))
        return "default reasoning\nVERDICT: YES"


_PYPROJECT = """\
[tool.agent-exam]
evals_dir = "evals"
"""

_CONFIG = """\
default_harness: claude_code
providers:
  claude_code:
    judge_model: haiku-test
judge:
  timeout_seconds: 10
"""


def _write_project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "evals" / "suites" / "my-skill" / "tasks").mkdir(parents=True)
    (root / "evals" / "fixtures").mkdir(parents=True)
    (root / "evals" / "runs").mkdir(parents=True)
    (root / "pyproject.toml").write_text(_PYPROJECT)
    (root / "evals" / "config.yaml").write_text(_CONFIG)
    return root


def _write_run(
    root: Path,
    run_id: str,
    suite: str,
    task: str,
    attempt: int,
    *,
    final_output: str,
    initial_report_ts: str,
    initial_attempts: list[dict],
) -> Path:
    run_dir = root / "evals" / "runs" / run_id
    (run_dir / "reports").mkdir(parents=True)
    attempt_dir = run_dir / "artifacts" / suite / task / f"attempt-{attempt}"
    (attempt_dir / "cwd").mkdir(parents=True)

    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "run_mode": "run",
                "started_at": "2026-04-23T14:00:00Z",
                "finished_at": "2026-04-23T14:00:30Z",
                "config": {
                    "k": 1,
                    "models": ["claude-sonnet-4-6"],
                    "provider": "claude_code",
                },
            }
        )
    )
    (attempt_dir / "attempt.json").write_text(
        json.dumps(
            {
                "provider": "claude_code",
                "model": "claude-sonnet-4-6",
                "started_at": "2026-04-23T14:00:01Z",
                "finished_at": "2026-04-23T14:00:25Z",
                "raw_transcript_path": None,
                "metrics": {
                    "wall_time_seconds": 24.0,
                    "tokens": {"input": 1000, "output": 100, "cache_read": 0},
                    "cost_usd": 0.01,
                    "peak_context": 1100,
                    "turn_count": 2,
                    "raw": {},
                },
            }
        )
    )
    (attempt_dir / "trajectory.json").write_text(
        json.dumps(
            {
                "turns": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "do the thing"}],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": final_output}],
                        "model": "claude-sonnet-4-6",
                        "tokens": {"input": 1000, "output": 100, "cache_read": 0},
                        "context": 1100,
                    },
                ]
            }
        )
    )
    (run_dir / "reports" / f"{initial_report_ts}.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "started_at": "2026-04-23T14:00:25Z",
                "finished_at": "2026-04-23T14:00:30Z",
                "scope": None,
                "attempts": initial_attempts,
            }
        )
    )
    return run_dir


def _stub_judge(monkeypatch, verdict_map: dict[str, str]):
    r"""Replace `commands.rescore.get_provider` with one that hands back
    a stub provider — every judge LLM call routes through it.

    `verdict_map` is `{criterion_substring_match: "YES"|"NO"|raw_response}`.
    Default when no key matches: "reasoning\\nVERDICT: YES".
    """
    calls: list[tuple[str, str]] = []  # (criterion_key, prompt)
    stub_provider = _StubJudgeProvider(verdict_map, calls)
    monkeypatch.setattr(
        "agent_exam.commands.rescore.get_provider", lambda _name: stub_provider
    )
    return calls


def test_rescore_unchanged_yaml_hits_cache(tmp_path, monkeypatch):
    """First run populates cache; rescore with same YAML hits cache and skips LLM calls."""
    root = _write_project(tmp_path)
    (root / "evals" / "suites" / "my-skill" / "tasks" / "basic.yaml").write_text(
        dedent(
            """\
            kind: execute
            prompt: do the thing
            assertions:
              - judge: The response reports success.
            """
        )
    )
    run_dir = _write_run(
        root,
        "run-2026-04-23-1500",
        "my-skill",
        "basic",
        1,
        final_output="I did the thing successfully.",
        initial_report_ts="2026-04-23-150000",
        initial_attempts=[
            {
                "suite": "my-skill",
                "task": "basic",
                "attempt": 1,
                "verdict": "pass",
                "assertions": [],
            }
        ],
    )
    from agent_exam.judge.cache import JudgeCache, agent_output_hash, key_for
    from agent_exam.serde import trajectory_from_dict

    traj = trajectory_from_dict(
        json.loads(
            (run_dir / "artifacts/my-skill/basic/attempt-1/trajectory.json").read_text()
        )
    )
    oh = agent_output_hash(traj)
    criterion = "The response reports success."
    k = key_for(criterion, oh, "haiku-test")
    cache = JudgeCache(run_dir / "judge-cache.json")
    cache.put(k, criterion, "YES", "cached reasoning")

    calls = _stub_judge(monkeypatch, {})

    monkeypatch.chdir(root)
    exit_code = rescore.run(root / "evals", "run-2026-04-23-1500")
    assert exit_code == 0

    assert calls == []

    reports = sorted((run_dir / "reports").glob("*.json"))
    assert len(reports) == 2
    new = json.loads(reports[-1].read_text())
    assert new["scope"] is None
    assert new["attempts"][0]["verdict"] == "pass"
    assert new["attempts"][0]["assertions"][0]["details"]["cache"] == "hit"
    assert new["attempts"][0]["assertions"][0]["details"]["verdict"] == "YES"


def test_rescore_changed_criterion_is_cache_miss(tmp_path, monkeypatch):
    """Tightening the criterion between run and rescore forces a fresh judge call."""
    root = _write_project(tmp_path)
    (root / "evals" / "suites" / "my-skill" / "tasks" / "basic.yaml").write_text(
        dedent(
            """\
            kind: execute
            prompt: do the thing
            assertions:
              - judge: NEW CRITERION that was never cached.
            """
        )
    )
    run_dir = _write_run(
        root,
        "run-2026-04-23-1500",
        "my-skill",
        "basic",
        1,
        final_output="I did the thing successfully.",
        initial_report_ts="2026-04-23-150000",
        initial_attempts=[
            {
                "suite": "my-skill",
                "task": "basic",
                "attempt": 1,
                "verdict": "pass",
                "assertions": [],
            }
        ],
    )
    from agent_exam.judge.cache import JudgeCache, agent_output_hash, key_for
    from agent_exam.serde import trajectory_from_dict

    traj = trajectory_from_dict(
        json.loads(
            (run_dir / "artifacts/my-skill/basic/attempt-1/trajectory.json").read_text()
        )
    )
    oh = agent_output_hash(traj)
    cache = JudgeCache(run_dir / "judge-cache.json")
    cache.put(
        key_for("The response reports success.", oh, "haiku-test"),
        "The response reports success.",
        "YES",
        "stale reasoning",
    )

    calls = _stub_judge(monkeypatch, {"NEW CRITERION": "NO"})

    monkeypatch.chdir(root)
    exit_code = rescore.run(root / "evals", "run-2026-04-23-1500")

    assert len(calls) == 1
    assert exit_code == 1

    reports = sorted((run_dir / "reports").glob("*.json"))
    new = json.loads(reports[-1].read_text())
    assertion = new["attempts"][0]["assertions"][0]
    assert assertion["details"]["verdict"] == "NO"
    assert assertion["details"]["cache"] == "miss"


def test_rescore_matches_trigger_case_by_name(tmp_path, monkeypatch):
    """Trigger YAMLs expand into N Tasks. Rescore must pick the one whose
    `name` matches the archived attempt (regression test for the list/Task bug).
    """
    root = _write_project(tmp_path)
    (root / "evals" / "suites" / "my-skill" / "tasks" / "trigger.yaml").write_text(
        dedent(
            """\
            kind: trigger
            skill: my-skill
            positive:
              - Should fire the skill.
            negative:
              - Should not fire the skill.
            """
        )
    )
    run_dir = _write_run(
        root,
        "run-2026-04-23-1600",
        "my-skill",
        "trigger-1",
        1,
        final_output="(skill wasn't supposed to fire — agent said nothing)",
        initial_report_ts="2026-04-23-160000",
        initial_attempts=[
            {
                "suite": "my-skill",
                "task": "trigger-1",
                "attempt": 1,
                "verdict": "pass",
                "assertions": [],
            }
        ],
    )

    _stub_judge(monkeypatch, {})

    monkeypatch.chdir(root)
    exit_code = rescore.run(root / "evals", "run-2026-04-23-1600::my-skill::trigger-1")
    assert exit_code == 0

    reports = sorted((run_dir / "reports").glob("*.json"))
    new = json.loads(reports[-1].read_text())
    assert new["scope"] == {"suite": "my-skill", "task": "trigger-1"}
    assert new["attempts"][0]["assertions"][0]["type"] == "skill_not_invoked"


def test_rescore_errors_cleanly_when_yaml_missing(tmp_path, monkeypatch):
    root = _write_project(tmp_path)
    _write_run(
        root,
        "run-2026-04-23-1700",
        "my-skill",
        "ghost",
        1,
        final_output="x",
        initial_report_ts="2026-04-23-170000",
        initial_attempts=[],
    )
    monkeypatch.chdir(root)
    with pytest.raises(UsageError, match="task YAML not found"):
        rescore.run(root / "evals", "run-2026-04-23-1700")


def test_rescore_scope_dict_written(tmp_path, monkeypatch):
    """Scope field is null for whole-run, {suite} for suite, {suite, task} for task."""
    root = _write_project(tmp_path)
    (root / "evals" / "suites" / "my-skill" / "tasks" / "basic.yaml").write_text(
        dedent(
            """\
            kind: execute
            prompt: x
            assertions:
              - judge: x.
            """
        )
    )
    _write_run(
        root,
        "run-2026-04-23-1800",
        "my-skill",
        "basic",
        1,
        final_output="x",
        initial_report_ts="2026-04-23-180000",
        initial_attempts=[],
    )
    _stub_judge(monkeypatch, {})
    monkeypatch.chdir(root)

    rescore.run(root / "evals", "run-2026-04-23-1800::my-skill::basic")
    reports = sorted((root / "evals/runs/run-2026-04-23-1800/reports").glob("*.json"))
    assert json.loads(reports[-1].read_text())["scope"] == {
        "suite": "my-skill",
        "task": "basic",
    }
