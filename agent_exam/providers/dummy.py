from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from .._models import _StrictModel
from ..schemas import (
    CheckResult,
    Metrics,
    RunResult,
    TextBlock,
    Tokens,
    Turn,
)
from .base import Provider

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import BaseModel


class DummyTaskConfig(_StrictModel):
    """The `dummy:` block on a task YAML — no tunables."""


class DummyProvider(Provider):
    """Canned provider for pipeline testing.

    Returns a fixed two-turn trajectory (one user, one assistant) and zero
    metrics. Never touches the cwd. Useful forever for testing the runner in
    isolation from `claude -p`.
    """

    name = "dummy"
    # Stages like Claude Code so integration tests can verify skill placement.
    skills_rel_path: ClassVar[str] = ".claude/skills"
    task_config_model: ClassVar[type[BaseModel]] = DummyTaskConfig

    def invoke(
        self,
        prompt: str,
        model: str,
        cwd: Path,
        provider_options: dict,
        stop_on_first_skill: bool,
        timeout_seconds: int,
    ) -> RunResult:
        trajectory = [
            Turn(
                role="user",
                content=[TextBlock(text=prompt)],
                started_at=0.0,
            ),
            Turn(
                role="assistant",
                content=[TextBlock(text="[dummy provider output]")],
                model=model,
                tokens=Tokens(input=0, output=0, cache_read=0),
                context=0,
                started_at=0.01,
            ),
        ]
        metrics = Metrics(
            wall_time_seconds=0.0,
            tokens=Tokens(input=0, output=0, cache_read=0),
            cost_usd=0.0,
            peak_context=0,
            turn_count=len(trajectory),
            raw={"provider": "dummy"},
        )
        return RunResult(
            trajectory=trajectory, metrics=metrics, raw_transcript_path=None
        )

    def preflight(self, cfg=None) -> list[CheckResult]:
        return [CheckResult(name="dummy provider", status="OK")]

    def probe_checks(self, probe_result, cfg=None) -> list[CheckResult]:
        return []

    def pre_run_warnings(self, cfg=None) -> list[CheckResult]:
        return []
