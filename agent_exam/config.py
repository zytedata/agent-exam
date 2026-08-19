from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import yaml
from pydantic import Field, PrivateAttr, StringConstraints, ValidationError

from ._models import _StrictModel, render_validation_error
from .errors import UsageError

DEFAULT_EVALS_DIR = "evals"
DEFAULT_SKILLS_DIR = "skills"
DEFAULT_HARNESS = "claude_code"
DEFAULT_TASK_TIMEOUT = 300


@dataclass
class PreRunRequest:
    harness: str


@dataclass
class PreRunResult:
    skills_dirs: list[Path] | None = None


class ProviderConfig(_StrictModel):
    """One model for every section under `providers:`, harness-specific
    fields included — a deliberate tradeoff. The provider name is the
    mapping *key*, not a field, so pydantic can't discriminate a union
    here; and the per-provider config models would have to come from the
    provider registry, which sits above this module in the import graph
    (task-level configs get their strict per-provider models that way —
    see `Provider.task_config_model` — because task loading happens above
    the registry). The cost: a harness-specific field under the wrong
    provider (e.g. `pure` under `codex_cli:`) validates and is silently
    ignored, unlike the unknown-key typos `_StrictModel` rejects.

    Some Codex CLI behavior is fixed in its provider rather than
    configurable here: `--ask-for-approval never`, `--ignore-user-config`,
    `--ignore-rules`, and the `workspace-write` sandbox (tasks can override
    the sandbox). User skill roots can still be discovered by Codex, so the
    provider keeps separate staged-vs-global skill checks.
    """

    default_model: str | None = None
    judge_model: str | None = None
    model_aliases: dict[str, str] = Field(default_factory=dict)
    extra_args: list[str] = Field(default_factory=list)
    # Default permission mode for agent calls. One of Claude Code's values
    # (`auto`, `bypassPermissions`, `acceptEdits`, `dontAsk`, `default`,
    # `plan`) or `None` to leave the flag off entirely. Per-task override
    # lives on the Task itself. Passed through as-is; typos surface as
    # `claude -p` errors.
    permission_mode: str | None = None
    # Names of Claude Code plugins whose presence in the agent session
    # would invalidate eval results — typically a plugin that ships the
    # same skills this repo stages via `skills_dirs`. Claude Code loads
    # user-enabled plugins from `~/.claude/settings.json` additively, so
    # `<plugin>:<skill>` would load alongside our staged `<skill>` and
    # `--without-skill` exclusion couldn't touch it. The runner warns at
    # run start if any blocked plugin is enabled; doctor's probe check
    # warns if one actually appears in the loaded skill listing.
    blocked_plugins: list[str] = Field(default_factory=list)
    # OpenCode-specific: run with --pure by default (disables external plugins).
    pure: bool = True
    # Codex CLI-specific: extra paths the workspace-write sandbox may write
    # to (`sandbox_workspace_write.writable_roots`). Codex expands `~`.
    # Needed for tools with home-directory caches — e.g. uv fails on
    # `~/.cache/uv` inside the default sandbox. A task's `codex_cli:` block
    # can override this list (replace, not extend). Ignored by tasks that
    # use permission profiles, which define their own write rules.
    writable_roots: list[str] = Field(default_factory=list)
    # Codex CLI-specific: default for `sandbox_workspace_write.network_access`
    # across all tasks. Codex's own default is off, but that makes codex runs
    # stricter than claude_code/opencode (which run with full network) — and
    # skills whose scripts declare PEP 723 deps need PyPI at run time. A
    # task's `codex_cli:` block overrides this. Judge invocations are not
    # affected (they keep the read-only, network-off judge profile).
    network_access: bool | None = None

    def resolve_model(self, name: str) -> str:
        return self.model_aliases.get(name, name)


class JudgeConfig(_StrictModel):
    timeout_seconds: int = 60
    # Separate budget for ``judge_agent`` — its multi-turn tool loop
    # rarely fits in the plain-judge default. Higher default reflects
    # the typical 3-tool-call round trip on real tasks.
    agent_timeout_seconds: int = 300
    include_trajectory: bool = True
    pass_on: list[str] = Field(default_factory=lambda: ["YES"])


class TagConfig(_StrictModel):
    """One entry under `tags:` in `evals/config.yaml`.

    A tag is a plain label — agent-exam attaches no meaning to the name, only
    to *exclude_by_default*, which keeps the tasks wearing it out of a run
    that did not ask for them narrowly enough (see `select_by_tags`).
    """

    exclude_by_default: bool = False


class McpStdioServer(_StrictModel):
    """A stdio MCP server entry under `mcp_servers:`, in the standard MCP
    JSON shape — so a server block can be copy-pasted from its README.
    """

    type: Literal["stdio"] = "stdio"
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class McpHttpServer(_StrictModel):
    """An HTTP or SSE MCP server entry under `mcp_servers:`. `type` defaults
    to `http`, the transport a bare `{url: ...}` block means.
    """

    type: Literal["http", "sse"] = "http"
    url: str
    headers: dict[str, str] = Field(default_factory=dict)


McpServerConfig = McpStdioServer | McpHttpServer
"""One entry under `mcp_servers:`. A plain union rather than a discriminated
one: `type` is optional in the MCP JSON everyone copy-pastes, so the two
branches are told apart by `command` vs `url`.
"""

McpServerName = Annotated[
    str, StringConstraints(pattern=r"^[A-Za-z0-9-]+(_[A-Za-z0-9-]+)*$")
]
"""A server name under `mcp_servers:`. Narrow because the name is both half
of the `mcp__<server>__<tool>` tool names assertions match on — where a
doubled underscore would split in the wrong place — and a bare TOML key
path in the config codex_cli renders, where a dot would nest.
"""


class Config(_StrictModel):
    """The eval framework's runtime config — `evals/config.yaml`
    overlaid with `evals/config.local.yaml`, plus the project + evals
    directories computed by `load_config`.
    """

    project_root: Path
    evals_dir: Path
    default_harness: str = DEFAULT_HARNESS
    skills_dirs: list[Path] | None = None
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    default_task_timeout_seconds: int = DEFAULT_TASK_TIMEOUT
    concurrency_groups: dict[str, int] = Field(default_factory=dict)
    # Every tag a suite or task may wear. Undeclared tags are a validation
    # error, so a typo can't silently exclude nothing — or everything.
    tags: dict[str, TagConfig] = Field(default_factory=dict)
    # MCP servers available to the agent under evaluation. Tasks attach a
    # subset with their own `mcp_servers:`; definitions live here so
    # credentials stay out of task files, which reports serialize verbatim.
    mcp_servers: dict[McpServerName, McpServerConfig] = Field(default_factory=dict)
    # Dotted module:callable path for the pre-run hook, e.g.
    # ``"evals.hooks:pre_run_hook"``. Loaded from ``pyproject.toml
    # [tool.agent-exam] pre_run_hook``.
    pre_run_hook: str | None = None

    # True when `skills_dirs` was set in `evals/config.local.yaml` — the
    # pre-run hook must not override an explicit local override. Set by
    # `load_config` after validation; preserved across `model_copy`.
    _skills_dirs_locked: bool = PrivateAttr(default=False)

    def provider(self, name: str) -> ProviderConfig:
        if name not in self.providers:
            self.providers[name] = ProviderConfig()
        return self.providers[name]


def find_project_root(start: Path | None = None) -> Path:
    path = (start or Path.cwd()).resolve()
    while True:
        if (path / "pyproject.toml").exists():
            return path
        if path == path.parent:
            raise UsageError(
                "no pyproject.toml found from cwd up to /; not in a project?"
            )
        path = path.parent


def _read_pyproject_section(project_root: Path) -> dict:
    pyproject = project_root / "pyproject.toml"
    if not pyproject.exists():
        return {}
    with pyproject.open("rb") as fh:
        data = tomllib.load(fh)
    return data.get("tool", {}).get("agent-exam", {})


def _read_evals_yaml(evals_dir: Path) -> dict:
    cfg_path = evals_dir / "config.yaml"
    if not cfg_path.exists():
        return {}
    with cfg_path.open() as fh:
        return yaml.safe_load(fh) or {}


def _read_evals_local_yaml(evals_dir: Path) -> dict:
    cfg_path = evals_dir / "config.local.yaml"
    if not cfg_path.exists():
        return {}
    with cfg_path.open() as fh:
        return yaml.safe_load(fh) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict with override merged on top of base.

    Dicts are merged recursively so a local config can set a single key
    inside a provider block without repeating the whole block.  All other
    types (scalars, lists) are replaced outright by the override value.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_dirs(entries: Any, project_root: Path) -> Any:
    """Resolve relative skill-dir string entries to absolute paths.

    Type-tolerant on purpose: anything that isn't a list of strings is
    passed through unchanged so pydantic's `list[Path]` validator
    surfaces a clean error (instead of `Path(1)` crashing here).
    """
    if not isinstance(entries, list):
        return entries
    out: list[Any] = []
    for entry in entries:
        if isinstance(entry, str):
            p = Path(entry)
            if not p.is_absolute():
                p = (project_root / p).resolve()
            out.append(p)
        else:
            out.append(entry)  # pydantic catches the bad element type
    return out


def load_config(project_root: Path | None = None) -> Config:
    project_root = project_root or find_project_root()
    pyproject_section = _read_pyproject_section(project_root)
    evals_dir = project_root / pyproject_section.get("evals_dir", DEFAULT_EVALS_DIR)
    local_raw = _read_evals_local_yaml(evals_dir)
    raw = _deep_merge(_read_evals_yaml(evals_dir), local_raw)

    # Resolve relative skills_dirs paths before validation — pydantic's
    # `Path` field has no concept of project_root.
    if "skills_dirs" in raw:
        raw["skills_dirs"] = _resolve_dirs(raw["skills_dirs"], project_root)
    elif (project_root / DEFAULT_SKILLS_DIR).is_dir():
        # A `skills/` directory at the project root is the common layout, so
        # it needs no configuration. A pre-run hook still overrides this, the
        # same way it overrides an explicit `skills_dirs` in `config.yaml`.
        raw["skills_dirs"] = [project_root / DEFAULT_SKILLS_DIR]

    # Inject the computed paths so pydantic validates them as fields.
    # YAML shouldn't define these (project_root is per-machine, evals_dir
    # is set in pyproject.toml, pre_run_hook lives in pyproject too); we
    # don't bother rejecting if it does — our injection wins.
    raw["project_root"] = project_root
    raw["evals_dir"] = evals_dir
    raw["pre_run_hook"] = pyproject_section.get("pre_run_hook")

    config_path = evals_dir / "config.yaml"
    try:
        cfg = Config.model_validate(raw)
    except ValidationError as exc:
        raise UsageError(render_validation_error(str(config_path), exc)) from exc

    cfg._skills_dirs_locked = "skills_dirs" in local_raw
    return cfg
