from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    """All on-disk paths for a single run, derived from evals_dir + run_id."""

    evals_dir: Path
    run_id: str

    @property
    def run_dir(self) -> Path:
        return self.evals_dir / "runs" / self.run_id

    @property
    def run_json(self) -> Path:
        return self.run_dir / "run.json"

    @property
    def artifacts_dir(self) -> Path:
        return self.run_dir / "artifacts"

    @property
    def reports_dir(self) -> Path:
        return self.run_dir / "reports"

    @property
    def judge_cache(self) -> Path:
        return self.run_dir / "judge-cache.json"

    def attempt_dir(self, suite: str, task: str, attempt_n: int) -> Path:
        return self.artifacts_dir / suite / task / f"attempt-{attempt_n}"

    def attempt_json(self, suite: str, task: str, attempt_n: int) -> Path:
        return self.attempt_dir(suite, task, attempt_n) / "attempt.json"

    def trajectory_json(self, suite: str, task: str, attempt_n: int) -> Path:
        return self.attempt_dir(suite, task, attempt_n) / "trajectory.json"

    def attempt_cwd(self, suite: str, task: str, attempt_n: int) -> Path:
        return self.attempt_dir(suite, task, attempt_n) / "cwd"

    def report_file(self, report_ts: str) -> Path:
        return self.reports_dir / f"{report_ts}.json"
