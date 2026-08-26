from __future__ import annotations

import json
import subprocess
from functools import cache
from pathlib import Path

from ...schemas import CheckResult


def check_binary() -> CheckResult:
    """Verify the copilot binary is on PATH and returns a version string."""
    try:
        out = subprocess.run(
            ["copilot", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except FileNotFoundError:
        return CheckResult(
            name="copilot binary",
            status="FAIL",
            hint="copilot not on PATH",
        )
    if out.returncode != 0:
        return CheckResult(
            name="copilot binary",
            status="FAIL",
            hint=f"`copilot --version` exited {out.returncode}",
        )
    return CheckResult(
        name="copilot binary",
        status="OK",
        hint=out.stdout.strip(),
    )


def check_probe_model(probe_result) -> CheckResult:
    """Verify the probe attempt recorded a non-empty model name."""
    model = (probe_result.model or "").strip() if probe_result is not None else ""
    if not model:
        return CheckResult(
            name="copilot probe model",
            status="WARN",
            hint=(
                "probe completed but no model name was recorded — "
                "check authentication and copilot version"
            ),
        )
    return CheckResult(
        name="copilot probe model",
        status="OK",
        hint=model,
    )


@cache
def _personal_mcp_servers() -> tuple[str, ...] | None:
    """The developer's personal MCP server names, or ``None`` if
    ``copilot mcp list --json`` could not be run or parsed."""
    try:
        out = subprocess.run(
            ["copilot", "mcp", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            # From the home directory, so nothing of the project under
            # evaluation counts as a workspace source.
            cwd=Path.home(),
        )
        data = json.loads(out.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None
    servers = data.get("mcpServers") if isinstance(data, dict) else None
    return tuple(sorted(servers)) if isinstance(servers, dict) else ()


def personal_mcp_servers() -> list[str]:
    """Names of the MCP servers the developer's own Copilot CLI setup loads,
    across every source it merges — user config, installed plugins, built-ins.

    ``--additional-mcp-config`` augments those rather than replacing them, so
    each of these is disabled by name to keep a trial hermetic. Whether one
    would have loaded is not worth establishing: disabling a server that was
    never going to load costs nothing, while missing one that does breaks the
    trial.

    Asked of ``copilot`` itself, which knows where each source lives, and
    asked once per process, since every attempt of a run has the same answer
    and the call takes a few seconds. Empty when the probe itself could not
    be run — see :func:`check_personal_mcp_servers` for that case surfaced as
    a check.
    """
    return list(_personal_mcp_servers() or ())


def check_personal_mcp_servers(cfg=None) -> CheckResult:
    """Report the developer's own MCP servers, and any name they share with a
    configured one.

    Copilot CLI merges ``--additional-mcp-config`` last, so a shared name
    resolves to the configured definition, and the developer's server of that
    name cannot be disabled without taking the configured one with it.
    """
    probed = _personal_mcp_servers()
    if probed is None:
        return CheckResult(
            name="personal mcp servers",
            status="WARN",
            hint=(
                "`copilot mcp list --json` failed; cannot confirm your own "
                "MCP servers are disabled for this trial"
            ),
        )
    personal = list(probed)
    if not personal:
        return CheckResult(
            name="personal mcp servers",
            status="OK",
            hint="none set up",
        )
    shared = sorted(set(personal) & set(cfg.mcp_servers if cfg else ()))
    if shared:
        return CheckResult(
            name="personal mcp servers",
            status="WARN",
            hint=(
                f"{', '.join(shared)} named both under mcp_servers: and in your own "
                "Copilot CLI setup, which stays enabled behind the configured one"
            ),
        )
    return CheckResult(
        name="personal mcp servers",
        status="OK",
        hint=f"{', '.join(personal)} disabled per trial",
    )
