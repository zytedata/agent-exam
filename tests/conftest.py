"""Make tests/fixtures importable and inject shared cwd / ctx fixtures."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))


@pytest.fixture(autouse=True)
def _fresh_oauth_tokens(monkeypatch):
    """Give each test its own MCP OAuth token cache.

    The cache is module state a whole run shares, so a token one test mints
    would otherwise spare the next one the grant it is there to exercise.
    """
    monkeypatch.setattr("agent_exam.mcp._TOKENS", {})


@pytest.fixture(autouse=True)
def _no_mcp_tool_listing(monkeypatch):
    """Keep runs and doctor from asking the MCP servers tests declare for
    their tools — a ``command: sh`` stands in for a server in most of them.

    They see an empty inventory instead, which leaves every trigger target
    unchecked. Tests of the check itself patch these names again.
    """
    from agent_exam.mcp import ToolInventory

    def nothing_listed(cfg, names, **kwargs):
        return ToolInventory()

    monkeypatch.setattr("agent_exam.runner.tool_inventory", nothing_listed)
    monkeypatch.setattr("agent_exam.commands.doctor.tool_inventory", nothing_listed)


def git_init(root: Path, gitignore: str = "", commit: bool = False) -> None:
    """Make *root* a git work tree.

    Needed by tests that exercise fixture hygiene: `validate_suite` asks git
    what it ignores under a fixture, and stays silent when git can't be asked —
    so in a bare tmp_path those checks never fire.

    *commit* matters when a test cares about ``--others`` semantics, which read
    differently in a repo where nothing is tracked yet.
    """
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_text(gitignore)
    if commit:
        subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "-c",
                "user.email=t@example.com",
                "-c",
                "user.name=t",
                "commit",
                "-qm",
                "fixtures",
            ],
            check=True,
        )


@pytest.fixture
def cwd(tmp_path):
    return tmp_path


@pytest.fixture
def ctx():
    """Minimal ScoringContext for assertion tests that don't care about
    judges, exclusions, etc. — just need a context with a provider."""
    from agent_exam.providers.base import Provider
    from agent_exam.scoring_context import ScoringContext

    return ScoringContext(provider=Provider())
