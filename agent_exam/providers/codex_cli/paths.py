from __future__ import annotations

import os
from pathlib import Path


def codex_home(env: dict[str, str] | None = None) -> Path:
    """Resolve $CODEX_HOME.

    `env` lets callers resolve the home of the *subprocess* codex actually
    ran under (e.g. when a task's `env:` overrides CODEX_HOME) rather than
    the parent eval process's own environment, which may disagree.
    """
    value = (env if env is not None else os.environ).get("CODEX_HOME")
    if value:
        return Path(value).expanduser()
    return Path.home() / ".codex"
