from __future__ import annotations

import contextlib
import os
import signal
import subprocess

_DEAD_PG = (ProcessLookupError, PermissionError)


def terminate_tree(process: subprocess.Popen, sigterm_timeout: float = 5.0) -> None:
    """Terminate a process and its entire process group.

    Sends SIGTERM first, waits up to `sigterm_timeout` seconds, then
    escalates to SIGKILL. Handles races where the process group has
    already exited (ProcessLookupError, PermissionError on macOS).

    Pass `sigterm_timeout=0` to skip SIGTERM and kill immediately.
    """
    if sigterm_timeout > 0:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except _DEAD_PG:
            return
        try:
            process.wait(timeout=sigterm_timeout)
            return
        except subprocess.TimeoutExpired:
            pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except _DEAD_PG:
        return
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=5)
