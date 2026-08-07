"""Pre-run hook invocation.

The hook lets a project run arbitrary Python before each eval run — most
commonly to build/stage skills into a folder and return that folder via
`PreRunResult.skills_dirs`. Both the runner (before staging) and the
doctor command (as a preflight check) invoke the hook through here.
"""

from __future__ import annotations

import importlib
import sys

from .config import Config, PreRunRequest, PreRunResult
from .errors import UsageError


def call_pre_run_hook(cfg: Config, provider_name: str) -> PreRunResult | None:
    """Invoke `cfg.pre_run_hook` if configured.

    Returns the hook's result, or None when no hook is configured. Raises
    UsageError when the hook string is malformed; propagates any exception
    raised by the hook itself so callers can decide how to surface it
    (the runner aborts; doctor reports it as a FAIL check).
    """
    if not cfg.pre_run_hook:
        return None

    module_path, sep, func_name = cfg.pre_run_hook.partition(":")
    if not sep:
        raise UsageError(
            f"pre_run_hook {cfg.pre_run_hook!r} must be in 'module:function' format"
        )

    project_root_str = str(cfg.project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    module = importlib.import_module(module_path)
    func = getattr(module, func_name)
    return func(PreRunRequest(harness=provider_name))
