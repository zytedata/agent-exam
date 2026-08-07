"""Make tests/fixtures importable and inject shared cwd / ctx fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))


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
