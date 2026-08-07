"""Tests for the shared pre-run hook helper."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import pytest

from agent_exam.config import Config, PreRunResult
from agent_exam.errors import UsageError
from agent_exam.hooks import call_pre_run_hook

if TYPE_CHECKING:
    from pathlib import Path


def _cfg(project_root: Path, hook: str | None = None) -> Config:
    return Config(
        project_root=project_root,
        evals_dir=project_root / "evals",
        pre_run_hook=hook,
    )


def test_no_hook_returns_none(tmp_path):
    assert call_pre_run_hook(_cfg(tmp_path), "claude_code") is None


def test_hook_without_colon_raises(tmp_path):
    with pytest.raises(UsageError, match="module:function"):
        call_pre_run_hook(_cfg(tmp_path, "no_colon_here"), "claude_code")


def test_hook_receives_harness_name_and_returns_result(tmp_path):
    skills_dir = tmp_path / "built-skills"
    skills_dir.mkdir()

    # Write a hook module under project_root so call_pre_run_hook adds it
    # to sys.path and can import it.
    (tmp_path / "my_hook_mod.py").write_text(
        textwrap.dedent(f"""\
            from pathlib import Path
            from agent_exam.config import PreRunRequest, PreRunResult

            SKILLS_DIR = Path({str(skills_dir)!r})

            def hook(req: PreRunRequest) -> PreRunResult:
                return PreRunResult(skills_dirs=[SKILLS_DIR])
        """)
    )

    result = call_pre_run_hook(_cfg(tmp_path, "my_hook_mod:hook"), "copilot_cli")

    assert isinstance(result, PreRunResult)
    assert result.skills_dirs == [skills_dir]


def test_hook_returning_none_propagates_as_none(tmp_path):
    (tmp_path / "noop_hook.py").write_text(
        textwrap.dedent("""\
            from agent_exam.config import PreRunRequest

            def hook(req: PreRunRequest):
                return None
        """)
    )

    result = call_pre_run_hook(_cfg(tmp_path, "noop_hook:hook"), "opencode")

    assert result is None
