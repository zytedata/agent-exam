from __future__ import annotations

import json
import os
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import TYPE_CHECKING

from .config import McpOAuthClientCredentials, McpStdioServer
from .errors import UsageError
from .schemas import CheckResult
from .trajectory_walk import iter_tool_calls

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from .config import Config, McpServerConfig
    from .providers.base import Provider
    from .schemas import RunResult, Turn
    from .tasks import Task

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

_PROJECT_ROOT_VAR = "PROJECT_ROOT"


def _expand_env_refs(value: str, where: str, project_root: Path) -> str:
    """Substitute ``${VAR}`` references from the parent environment.

    ``${PROJECT_ROOT}`` is a builtin, resolved from *project_root* instead —
    it takes priority over an OS environment variable of the same name, so a
    stdio server's ``command``/``args`` can reference this repo's own root
    without the user having to export anything.
    """

    def replace(match: re.Match) -> str:
        name = match.group(1)
        if name == _PROJECT_ROOT_VAR:
            return str(project_root)
        try:
            return os.environ[name]
        except KeyError:
            raise UsageError(
                f"{where}: ${{{name}}} is not set in the environment"
            ) from None

    return _ENV_REF.sub(replace, value)


def _env_refs(value: str) -> list[str]:
    return _ENV_REF.findall(value)


def _oauth_ref_values(oauth: McpOAuthClientCredentials) -> tuple[str, ...]:
    values = (oauth.token_url, oauth.client_id, oauth.client_secret)
    return (*values, oauth.scope) if oauth.scope else values


def _fetch_oauth_token(
    oauth: McpOAuthClientCredentials, where: str, project_root: Path
) -> str:
    """Run *oauth*'s client credentials grant and return the access token."""
    token_url = _expand_env_refs(oauth.token_url, f"{where}.token_url", project_root)
    if not token_url.startswith(("http://", "https://")):
        raise UsageError(f"{where}.token_url: must be an http:// or https:// URL")
    body = {
        "grant_type": "client_credentials",
        "client_id": _expand_env_refs(
            oauth.client_id, f"{where}.client_id", project_root
        ),
        "client_secret": _expand_env_refs(
            oauth.client_secret, f"{where}.client_secret", project_root
        ),
    }
    if oauth.scope:
        body["scope"] = _expand_env_refs(oauth.scope, f"{where}.scope", project_root)
    # Scheme checked above; ruff's S310 still flags Request() itself.
    request = urllib.request.Request(  # noqa: S310
        token_url, data=urllib.parse.urlencode(body).encode(), method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310
            payload = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError) as exc:
        raise UsageError(
            f"{where}: token request to {token_url} failed: {exc}"
        ) from exc
    token = payload.get("access_token")
    if not token:
        raise UsageError(
            f"{where}: token response from {token_url} has no access_token"
        )
    return token


def _selected(cfg: Config, names: list[str] | None) -> dict:
    if names is None:
        return dict(cfg.mcp_servers)
    return {name: cfg.mcp_servers[name] for name in names}


def resolve_servers(cfg: Config, names: list[str] | None = None) -> dict[str, dict]:
    """Return the selected servers as MCP JSON, with ``${VAR}`` expanded.

    *names* selects a subset of ``cfg.mcp_servers``; ``None`` selects all of
    them. Raises :py:class:`UsageError` when a referenced environment
    variable is unset, so a missing credential surfaces before the agent
    runs rather than as a tool failure mid-task. A stdio server's
    ``command``/``args`` also get ``${PROJECT_ROOT}`` expanded, the builtin
    ``${VAR}`` case handled by :py:func:`_expand_env_refs`.

    A server carrying ``oauth`` runs its client credentials grant here and
    exports the access token into ``os.environ[oauth.env_var]``, ahead of
    the server's own ``env``/``headers`` expansion below — so a ``${VAR}``
    reference to it resolves like any other credential.
    """
    out: dict[str, dict] = {}
    for name, server in _selected(cfg, names).items():
        data = server.model_dump()
        data.pop("oauth", None)
        if server.oauth is not None:
            os.environ[server.oauth.env_var] = _fetch_oauth_token(
                server.oauth, f"mcp_servers.{name}.oauth", cfg.project_root
            )
        if isinstance(server, McpStdioServer):
            # Optional in the MCP JSON everyone copy-pastes, and rejected
            # by some harnesses' own config schemas.
            data.pop("type")
            data["command"] = _expand_env_refs(
                data["command"], f"mcp_servers.{name}.command", cfg.project_root
            )
            data["args"] = [
                _expand_env_refs(v, f"mcp_servers.{name}.args[{i}]", cfg.project_root)
                for i, v in enumerate(data["args"])
            ]
        for key in ("env", "headers"):
            if key in data:
                data[key] = {
                    k: _expand_env_refs(
                        v, f"mcp_servers.{name}.{key}.{k}", cfg.project_root
                    )
                    for k, v in data[key].items()
                }
        out[name] = data
    return out


def render_mcp_json(run_tmp_root: Path, servers: dict[str, dict]) -> Path:
    """Write *servers* as an MCP config file and return its path.

    The file lands directly under *run_tmp_root*, next to the attempt cwd
    rather than in it, so a rendered credential is not archived with the
    run's artifacts. The name is random so that the configs of the several
    server sets a run attaches cannot collide.
    """
    path = run_tmp_root / f"{uuid.uuid4().hex[:12]}.mcp.json"
    path.write_text(json.dumps({"mcpServers": servers}))
    return path


def stage_mcp_json(run_tmp_root: Path, cfg: Config, servers: list[str] | None) -> dict:
    """Resolve and render *servers* as an MCP JSON file, for providers whose
    CLI takes one as a flag argument (Claude Code's ``--mcp-config``,
    Copilot CLI's ``--additional-mcp-config``).
    """
    resolved = resolve_servers(cfg, servers)
    if not resolved:
        return {}
    return {
        "mcp_config_path": render_mcp_json(run_tmp_root, resolved),
        "mcp_server_names": sorted(resolved),
    }


_CANONICAL_PREFIX = "mcp__"
_SEPARATORS = ("__", "_", "-")


def join_canonical_tool_name(server: str, tool: str) -> str:
    """Build the canonical ``mcp__<server>__<tool>`` spelling of a call.

    The one place that joins *server* and *tool* this way, so a harness that
    reports them as separate fields (rather than one joined string) doesn't
    hand-rolled its own copy of the format.
    """
    return f"{_CANONICAL_PREFIX}{server}__{tool}"


def canonical_server_prefix(server: str) -> str:
    """Build the ``mcp__<server>`` prefix that names every tool of *server*.

    What Claude Code's ``--allowed-tools`` expects to pre-approve a whole
    MCP server rather than one tool of it.
    """
    return f"{_CANONICAL_PREFIX}{server}"


def canonical_tool_server(name: str) -> str | None:
    """The server *name* belongs to, if it is a canonical MCP tool name."""
    if not is_mcp_tool(name):
        return None
    server, _, _ = name[len(_CANONICAL_PREFIX) :].partition("__")
    return server or None


def _match_server(bare: str, servers: list[str]) -> tuple[str, str] | None:
    # Longest first, so a server named `github` doesn't claim a tool of
    # `github-actions`.
    for server in servers:
        for separator in _SEPARATORS:
            prefix = f"{server}{separator}"
            if bare.startswith(prefix):
                return server, bare[len(prefix) :]
    return None


def canonical_tool_name(name: str, servers: Iterable[str]) -> str:
    """Rewrite an MCP tool name into Claude Code's ``mcp__<server>__<tool>``.

    For harnesses that report an MCP call as one joined string and nothing
    else, so the only way back to the server is to match the configured
    names against the spellings in use. A name that belongs to no
    configured server is returned unchanged.

    Harnesses that name the server in a field of their own resolve the call
    from that field instead; this guesses, and a native tool whose name
    happens to start with a server name would be guessed wrong.
    """
    if name.startswith(_CANONICAL_PREFIX):
        return name
    servers = sorted(servers, key=len, reverse=True)
    match = _match_server(name, servers)
    if match is None and name.startswith(("mcp_", "mcp-")):
        # A generic `mcp_`/`mcp-` marker some harnesses add on top of the
        # server name, tried only once the name itself matches no
        # configured server outright, so a server actually named `mcp`
        # still resolves from the name as given.
        match = _match_server(name[4:], servers)
    if match is None:
        return name
    server, tool = match
    return join_canonical_tool_name(server, tool)


def is_mcp_tool(name: str) -> bool:
    """Whether *name* is an MCP tool in its canonical spelling."""
    return name.startswith(_CANONICAL_PREFIX)


def settles_tool_trigger(name: str, target: str, negative: bool) -> bool:
    """Whether a call to *name* settles a trigger aimed at tool *target*.

    The target itself always does. A positive case is settled by a call to
    any MCP tool: the case grades on the first one, so reaching for another
    server's tool answers it just as decisively. A negative case has to run
    the turn out, since the agent can call one MCP tool and still reach for
    the target afterwards.
    """
    return name == target or (not negative and is_mcp_tool(name))


def canonicalize_tool_names(trajectory: list[Turn], servers: Iterable[str]) -> None:
    """Rename every MCP tool call in *trajectory* to its canonical spelling,
    in place, so one ``tool_called:`` line grades on any harness.
    """
    servers = list(servers)
    for call in iter_tool_calls(trajectory):
        call.name = canonical_tool_name(call.name, servers)


def _planned(cfg: Config, tasks: Iterable[Task]) -> dict[str, McpServerConfig]:
    names: set[str] = set()
    for task in tasks:
        if task.mcp_servers is None:
            return _selected(cfg, None)
        names.update(task.mcp_servers)
    return _selected(cfg, sorted(names))


def preflight(
    cfg: Config, provider: Provider, tasks: Iterable[Task] | None = None
) -> list[CheckResult]:
    """Static checks for the MCP servers in play: stdio commands resolve on
    ``PATH``, referenced environment variables are set, and the selected
    harness can actually attach servers.

    *tasks* narrows the checks to the servers those tasks attach between
    them, so a credentialed server that no selected task asks for does not
    stand in the way of the run. ``None`` checks every configured server.
    """
    servers = cfg.mcp_servers if tasks is None else _planned(cfg, tasks)
    if not servers:
        return []

    if not provider.supports_mcp:
        return [
            CheckResult(
                name="mcp servers supported",
                status="WARN",
                hint=(
                    f"{provider.name} attaches no MCP servers, so the "
                    f"{len(servers)} configured under mcp_servers: "
                    "do nothing in this run"
                ),
            )
        ]

    results: list[CheckResult] = []

    if not provider.reports_mcp_connections:
        results.append(
            CheckResult(
                name="mcp connection status",
                status="WARN",
                hint=(
                    f"{provider.name} reports no MCP connection status at "
                    "session start, so a server that fails to attach reads "
                    "as a plain task failure instead of an MCP error"
                ),
            )
        )

    problems = provider.validate_mcp_servers(servers)
    if problems:
        results.append(
            CheckResult(
                name="mcp server configuration",
                status="FAIL",
                hint="; ".join(problems),
            )
        )

    missing_cmd = sorted(
        name
        for name, server in servers.items()
        if isinstance(server, McpStdioServer) and not shutil.which(server.command)
    )
    if missing_cmd:
        results.append(
            CheckResult(
                name="mcp server commands",
                status="FAIL",
                hint=f"not on PATH: {', '.join(missing_cmd)}",
            )
        )

    missing_vars = sorted(
        {
            var
            for server in servers.values()
            for value in (
                *(
                    (server.command, *server.args)
                    if isinstance(server, McpStdioServer)
                    else ()
                ),
                *getattr(server, "env", {}).values(),
                *getattr(server, "headers", {}).values(),
                *(_oauth_ref_values(server.oauth) if server.oauth is not None else ()),
            )
            for var in _env_refs(value)
            if var != _PROJECT_ROOT_VAR and var not in os.environ
        }
    )
    if missing_vars:
        results.append(
            CheckResult(
                name="mcp server environment",
                status="FAIL",
                hint=f"referenced but unset: {', '.join(missing_vars)}",
            )
        )

    if not problems and not missing_cmd and not missing_vars:
        results.append(
            CheckResult(
                name="mcp servers",
                status="OK",
                hint=f"{len(servers)} configured",
            )
        )
    return results


def server_status_map(servers: object) -> dict[str, str] | None:
    """Read a harness's session-start server list as name -> status.

    Claude Code and Copilot CLI both announce the servers they attached as a
    list of ``{"name": ..., "status": ...}`` objects. Anything else reads as
    ``None``, the "the harness said nothing" :py:func:`connection_check`
    passes on.
    """
    if not isinstance(servers, list):
        return None
    return {
        str(s.get("name")): str(s.get("status"))
        for s in servers
        if isinstance(s, dict) and s.get("name")
    }


def connection_check(
    statuses: dict[str, str] | None, expected: Iterable[str] = ()
) -> CheckResult:
    """Report the MCP connection statuses a harness announced at session
    start, against the *expected* server names. A server that failed to
    connect — or that the harness never mentions, because the config never
    reached it — leaves the agent silently without its tools, which reads
    as a skill failure.

    *statuses* is ``None`` when the harness announces nothing, which says
    nothing either way.
    """
    if statuses is None:
        return CheckResult(
            name="mcp servers connected",
            status="OK",
            hint="harness reports no connection status",
        )
    problems = sorted(
        f"{name} ({statuses.get(name) or 'not attached'})"
        for name in {*statuses, *expected}
        if statuses.get(name) != "connected"
    )
    if problems:
        return CheckResult(
            name="mcp servers connected",
            status="FAIL",
            hint=f"did not connect: {', '.join(problems)}",
        )
    if not statuses:
        return CheckResult(
            name="mcp servers connected",
            status="OK",
            hint="no MCP servers attached",
        )
    return CheckResult(
        name="mcp servers connected",
        status="OK",
        hint=f"{len(statuses)} connected",
    )


def probe_connection_check(
    probe_result: RunResult, cfg: Config | None
) -> list[CheckResult]:
    """The `probe_checks` MCP connection check, when *cfg* attaches any
    servers — shared by every provider whose round-trip probe reports
    connection status.
    """
    if cfg is None or not cfg.mcp_servers:
        return []
    return [connection_check(probe_result.mcp_server_status, cfg.mcp_servers)]
