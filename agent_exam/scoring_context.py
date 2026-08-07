from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .judge import JudgeCache, JudgeCall, cwd_hash
from .providers.base import Provider


@dataclass
class ScoringContext:
    """Everything an assertion checker might need beyond config/result/cwd.

    Most assertions ignore most fields. The `judge` and `judge_agent`
    assertions read the judge slots for dispatch, caching, and pass-on
    overrides. `skill_invoked` reads `provider.is_same_skill` and
    `skills_excluded` to handle the reality-check inversion.

    `provider` has no default — building a ScoringContext without one
    raises at construction so the runner / rescore command can't
    silently drop the per-provider hooks.
    """

    provider: Provider
    judge_call: JudgeCall | None = None
    judge_cache: JudgeCache | None = None
    judge_pass_on: list[str] = field(default_factory=lambda: ["YES"])
    # Skills the current run excluded from the bundle (populated by
    # `--without-skill` / `--no-skills`; empty otherwise). Assertions that check
    # "the skill fired" use this to invert when the asserted skill is the
    # one that was intentionally removed.
    skills_excluded: frozenset[str] = field(default_factory=frozenset)
    # Memoization for the `judge_agent` cwd-content hash. ScoringContext
    # is shared across attempts within a run, so the cache is keyed by
    # the attempt cwd path. First judge_agent assertion on an attempt
    # pays the walk cost; subsequent ones hit the memo.
    _cwd_hash_cache: dict[Path, str] = field(default_factory=dict)

    def cwd_hash_for(self, cwd: Path) -> str:
        """Return (memoized) content hash of an attempt's archived cwd."""
        cached = self._cwd_hash_cache.get(cwd)
        if cached is not None:
            return cached
        h = cwd_hash(cwd)
        self._cwd_hash_cache[cwd] = h
        return h
