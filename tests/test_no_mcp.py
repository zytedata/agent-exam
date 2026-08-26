"""`--no-mcp`: the counterfactual run with the servers detached."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest

from agent_exam.config import load_config
from agent_exam.errors import UsageError
from agent_exam.providers.claude_code.provider import ClaudeCodeProvider
from agent_exam.runner import RunRequest, run

_CONFIG = """\
default_harness: dummy
skills_dirs: []
mcp_servers:
  files:
    command: sh
providers:
  dummy:
    judge_model: haiku
"""


def _project(tmp_path: Path, config: str = _CONFIG) -> Path:
    root = tmp_path / "proj"
    (root / "evals" / "suites" / "s" / "tasks").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[tool.agent-exam]\nevals_dir = "evals"\n')
    (root / "evals" / "config.yaml").write_text(dedent(config))
    (root / "evals" / "suites" / "s" / "tasks" / "t.yaml").write_text(
        "kind: execute\nprompt: x\nassertions: []\n"
    )
    return root


def _req(**kwargs) -> RunRequest:
    return RunRequest(
        specs=[("s", None)],
        provider="dummy",
        model="",
        k=1,
        n_parallel=1,
        without_skill=False,
        cleanup_tmp_root=False,
        **kwargs,
    )


def test_run_attaches_nothing_and_stays_informational(tmp_path, monkeypatch):
    root = _project(tmp_path)
    monkeypatch.setattr(
        "agent_exam.providers.dummy.DummyProvider.stage_mcp_config",
        ClaudeCodeProvider.stage_mcp_config,
        raising=False,
    )

    assert run(load_config(root), _req(no_mcp=True)) == 0

    run_dir = next(iter((root / "evals" / "runs").iterdir()))
    run_json = json.loads((run_dir / "run.json").read_text())
    assert run_json["run_mode"] == "no-mcp"
    assert run_json["config"]["no_mcp"] is True
    # The record says what the run withheld.
    assert run_json["config"]["mcp_servers"] == ["files"]
    # Nothing was rendered, so nothing was attached.
    tmp_root = Path(run_json["config"]["tmp_root"])
    assert not list(tmp_root.glob("*.mcp.json"))


def test_a_normal_run_still_attaches_them(tmp_path, monkeypatch):
    root = _project(tmp_path)
    monkeypatch.setattr(
        "agent_exam.providers.dummy.DummyProvider.stage_mcp_config",
        ClaudeCodeProvider.stage_mcp_config,
        raising=False,
    )

    assert run(load_config(root), _req()) == 0

    run_dir = next(iter((root / "evals" / "runs").iterdir()))
    run_json = json.loads((run_dir / "run.json").read_text())
    assert run_json["run_mode"] == "run"
    assert list(Path(run_json["config"]["tmp_root"]).glob("*.mcp.json"))
    # What ran is readable back from the artifacts alone.
    assert run_json["config"]["mcp_servers"] == ["files"]
    attempt = json.loads(
        (run_dir / "artifacts" / "s" / "t" / "attempt-1" / "attempt.json").read_text()
    )
    assert attempt["mcp_servers_attached"] == ["files"]


def test_a_task_selecting_a_server_by_name_still_runs(tmp_path, monkeypatch):
    """`--no-mcp` drops the definitions; a selection naming one must go too."""
    root = _project(tmp_path)
    (root / "evals" / "suites" / "s" / "tasks" / "t.yaml").write_text(
        "kind: execute\nprompt: x\nmcp_servers: [files]\nassertions: []\n"
    )
    monkeypatch.setattr(
        "agent_exam.providers.dummy.DummyProvider.stage_mcp_config",
        ClaudeCodeProvider.stage_mcp_config,
        raising=False,
    )

    assert run(load_config(root), _req(no_mcp=True)) == 0

    run_dir = next(iter((root / "evals" / "runs").iterdir()))
    run_json = json.loads((run_dir / "run.json").read_text())
    assert not list(Path(run_json["config"]["tmp_root"]).glob("*.mcp.json"))


def test_detaching_nothing_is_refused(tmp_path):
    root = _project(tmp_path, "default_harness: dummy\nskills_dirs: []\n")

    with pytest.raises(UsageError, match="no mcp_servers are declared"):
        run(load_config(root), _req(no_mcp=True))


def _trigger(root: Path, name: str, target: str) -> None:
    (root / "evals" / "suites" / "s" / "tasks" / f"{name}.yaml").write_text(
        f"kind: trigger\n{target}\npositive: [ping]\nnegative: [pong]\n"
    )


def test_tool_triggers_are_dropped_and_skill_triggers_stay(tmp_path):
    root = _project(tmp_path)
    _trigger(root, "tool", "tool: mcp__files__search")
    _trigger(root, "skill", "skill: s")

    assert run(load_config(root), _req(no_mcp=True)) == 0

    run_dir = next(iter((root / "evals" / "runs").iterdir()))
    suite_dir = run_dir / "artifacts" / "s"
    assert sorted(p.name for p in suite_dir.iterdir()) == ["skill-0", "skill-1", "t"]


def test_a_tool_trigger_only_suite_is_refused(tmp_path):
    root = _project(tmp_path)
    (root / "evals" / "suites" / "s" / "tasks" / "t.yaml").unlink()
    _trigger(root, "tool", "tool: mcp__files__search")

    with pytest.raises(UsageError, match="targets an MCP tool"):
        run(load_config(root), _req(no_mcp=True))


@pytest.mark.parametrize("mode", ["without_skill", "no_skills"])
def test_withholding_skills_too_is_refused(mode):
    """A run that withholds both cannot attribute the difference to either."""
    with pytest.raises(UsageError, match="mutually exclusive"):
        RunRequest(
            specs=[("s", None)],
            provider="dummy",
            model="",
            k=1,
            n_parallel=1,
            no_mcp=True,
            **{"without_skill": False, mode: True},
        )
