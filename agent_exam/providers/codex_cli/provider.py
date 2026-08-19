from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, field_validator

from ..._models import _StrictModel
from ...config import McpHttpServer
from ...errors import FrameworkError, ProviderTimeout, UsageError
from ...mcp import resolve_servers
from ...ratelimit import with_retries
from ...schemas import CheckResult, RunResult
from ..base import Provider
from ..child_env import build_child_env
from ..process_utils import terminate_tree
from .hermetic_skills import stage_skills_into
from .paths import codex_home
from .stream_parser import (
    StreamState,
    drain_stderr,
    drain_stream,
    stream_error_messages,
    strip_path_alias_warning,
)
from .transcripts import build_run_result, find_session_explicit_skill_invocation

if TYPE_CHECKING:
    from collections.abc import Callable

SandboxMode = Literal["read-only", "workspace-write", "danger-full-access"]
RuleDecision = Literal["allow", "prompt", "forbidden"]


class CodexCliPrefixRule(_StrictModel):
    pattern: list[str | list[str]]
    decision: RuleDecision = "allow"
    justification: str | None = None

    @field_validator("pattern")
    @classmethod
    def _validate_pattern(cls, value: list[str | list[str]]) -> list[str | list[str]]:
        if not value:
            raise ValueError("must not be empty")
        for item in value:
            if isinstance(item, str):
                if not item:
                    raise ValueError("pattern tokens must not be empty")
                continue
            if not item:
                raise ValueError("pattern alternatives must not be empty")
            if not all(isinstance(alt, str) and alt for alt in item):
                raise ValueError("pattern alternatives must be non-empty strings")
        return value


class CodexCliTaskConfig(_StrictModel):
    """The `codex_cli:` block on a task YAML."""

    sandbox: SandboxMode | None = None
    network_access: bool | None = None
    writable_roots: list[str] | None = None
    prefix_rules: list[CodexCliPrefixRule] | None = None


class CodexCliProvider(Provider):
    name = "codex_cli"
    safe_judge_tools = ("command_execution",)
    omitted_model_label = "Codex CLI default model"
    task_config_model: ClassVar[type[BaseModel]] = CodexCliTaskConfig
    supports_mcp: ClassVar[bool] = True
    # `codex exec --json`'s event stream has no session-level event for MCP
    # server startup/connection status (only per-call `mcp_tool_call` items
    # once the agent actually invokes one) — see `Provider.reports_mcp_connections`.
    reports_mcp_connections: ClassVar[bool] = False

    def task_options(
        self, task_cfg: CodexCliTaskConfig | None, framework_cfg, task_kind: str
    ) -> dict:
        opts: dict = {
            "sandbox": "workspace-write",
            "ask_for_approval": "never",
            "ignore_user_config": True,
            "ignore_rules": True,
        }
        if framework_cfg.extra_args:
            opts["extra_args"] = list(framework_cfg.extra_args)
        if framework_cfg.writable_roots:
            opts["writable_roots"] = list(framework_cfg.writable_roots)
        if framework_cfg.network_access is not None:
            opts["network_access"] = framework_cfg.network_access
        if task_cfg is not None:
            for key in ("sandbox", "network_access", "writable_roots"):
                value = getattr(task_cfg, key)
                if value is not None:
                    opts[key] = value
            if task_cfg.prefix_rules is not None:
                opts["prefix_rules"] = [
                    rule.model_dump(exclude_none=True) for rule in task_cfg.prefix_rules
                ]
        return opts

    def invoke(
        self,
        prompt: str,
        model: str,
        cwd: Path,
        provider_options: dict,
        stop_on_first_skill: bool,
        timeout_seconds: int,
    ) -> RunResult:
        return with_retries(
            lambda: self._invoke_once(
                prompt,
                model,
                cwd,
                provider_options,
                stop_on_first_skill,
                timeout_seconds,
            )
        )

    def _invoke_once(
        self,
        prompt: str,
        model: str,
        cwd: Path,
        provider_options: dict,
        stop_on_first_skill: bool,
        timeout_seconds: int,
    ) -> RunResult:
        provider_options = self._prepare_prefix_rules(cwd, provider_options)
        cmd = self._build_cmd(prompt, model, cwd, provider_options)
        env_overrides = provider_options.get("env_overrides")
        staged_home = provider_options.get("codex_home")
        if staged_home:
            # Wins over a task's own `env: {CODEX_HOME: ...}`: that staged
            # home is where `stage_mcp_config` just wrote the MCP config,
            # and it's what makes `--ignore-user-config` skippable below.
            env_overrides = {**(env_overrides or {}), "CODEX_HOME": staged_home}
        env = build_child_env(env_overrides)

        cwd_abs = cwd.resolve()
        raw_dir = cwd_abs.parent / ".raw_streams" / cwd_abs.name
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"stream_{uuid.uuid4().hex[:8]}.jsonl"
        raw_fh = raw_path.open("wb")

        started = time.time()
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            cwd=str(cwd_abs),
            env=env,
            start_new_session=True,
        )
        watchdog_stop = _start_orphan_watchdog(process)

        state = StreamState()
        if stop_on_first_skill:
            state.skill_detection_enabled = True
            state.target_tool = provider_options.get("target_tool")
            state.negative_trigger_mode = bool(provider_options.get("negative_trigger"))

        t_out = threading.Thread(
            target=drain_stream, args=(process.stdout, state, raw_fh), daemon=True
        )
        t_err = threading.Thread(
            target=drain_stderr, args=(process.stderr, state), daemon=True
        )
        t_out.start()
        t_err.start()

        timed_out = False
        killed_on_skill = False
        if stop_on_first_skill:
            killed_on_skill, timed_out = self._wait_with_skill_kill(
                process, state, timeout_seconds, env=env
            )
        else:
            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                terminate_tree(process)

        wall_time = time.time() - started
        watchdog_stop.set()
        t_out.join(timeout=5)
        t_err.join(timeout=5)
        if t_out.is_alive():
            # A background grandchild process can inherit codex's stdout
            # and keep the pipe open past codex's own exit, leaving the
            # drain thread blocked in a read that will never see EOF.
            # Close the read end so it unblocks with an error instead of
            # leaking a thread that writes to raw_fh after we close it.
            with contextlib.suppress(OSError):
                process.stdout.close()
            t_out.join(timeout=2)
        raw_fh.close()

        if timed_out:
            partial = None
            try:
                partial = build_run_result(
                    state,
                    wall_time_seconds=wall_time,
                    user_prompt=prompt,
                    raw_transcript_path=raw_path,
                    model=model,
                    env=env,
                )
            except Exception:
                partial = None
            raise ProviderTimeout(
                f"codex exec timed out after {timeout_seconds}s "
                f"(thread_id={state.thread_id})",
                partial_run_result=partial,
            )

        if process.returncode != 0 and not killed_on_skill:
            stderr_tail = strip_path_alias_warning(
                bytes(state.stderr_tail).decode("utf-8", errors="replace")
            ).strip()
            # The reason for a fatal exit (auth failure, usage limit, …)
            # arrives as error events on the JSON stream; stderr is
            # typically just noise. Quote the stream errors, and rescue
            # the raw stream out of the run tmp root — the runner deletes
            # that tree on abort, which would destroy the only evidence.
            stream_errors = "; ".join(stream_error_messages(state.events)[-3:])
            preserved = _preserve_failed_stream(raw_path)
            raise FrameworkError(
                f"codex exec exited with {process.returncode}. "
                f"stream errors: {stream_errors or '(none)'}\n"
                f"raw stream preserved at: {preserved}\n"
                f"stderr tail:\n{stderr_tail[-1024:]}"
            )

        return build_run_result(
            state,
            wall_time_seconds=wall_time,
            stream_detected_skill=state.detected_skill,
            stream_detected_tool=state.detected_tool,
            raw_transcript_path=raw_path,
            user_prompt=prompt,
            model=model,
            allow_minimal_trigger_result=killed_on_skill,
            env=env,
        )

    def _build_cmd(
        self, prompt: str, model: str, cwd: Path, provider_options: dict
    ) -> list[str]:
        allowed = provider_options.get("allowed_tools")
        if allowed is not None:
            unsupported = [tool for tool in allowed if tool != "command_execution"]
            if unsupported:
                raise UsageError(
                    "codex_cli.allowed_tools only supports 'command_execution' "
                    f"(got {unsupported!r})"
                )
        restricted_tools = allowed is not None
        config_overrides = dict(provider_options.get("config_overrides") or {})
        use_permission_profiles = (
            "default_permissions" in config_overrides
            or "permissions" in config_overrides
        )
        sandbox = provider_options.get("sandbox") or (
            "read-only" if restricted_tools else "workspace-write"
        )
        ask_for_approval = provider_options.get("ask_for_approval") or "never"

        cmd = ["codex", "--ask-for-approval", str(ask_for_approval)]
        if model:
            cmd.extend(["--model", model])
        network_access = provider_options.get("network_access")
        if use_permission_profiles:
            if network_access is not None:
                raise UsageError(
                    "codex_cli.network_access cannot be combined with "
                    "permission profiles; configure "
                    "permissions.<name>.network instead"
                )
            # writable_roots is deliberately not an error here: it is
            # usually a run-wide default from config.yaml, and raising
            # would break every profile task in the run. Profiles define
            # their own write rules, so the sandbox_workspace_write table
            # would be inert anyway.
        else:
            cmd.extend(["--sandbox", str(sandbox)])
            if restricted_tools and network_access is None:
                network_access = False
            if network_access is not None:
                value = "true" if network_access else "false"
                cmd.extend(["-c", f"sandbox_workspace_write.network_access={value}"])
            writable_roots = provider_options.get("writable_roots")
            if writable_roots:
                cmd.extend(
                    [
                        "-c",
                        "sandbox_workspace_write.writable_roots="
                        + _toml_value(list(writable_roots)),
                    ]
                )
        if restricted_tools:
            config_overrides.setdefault("web_search", "disabled")
        if provider_options.get("trust_project_config"):
            config_overrides = _deep_merge(
                config_overrides,
                {"projects": {str(cwd.resolve()): {"trust_level": "trusted"}}},
            )
        for key, value in _flatten_config(config_overrides):
            cmd.extend(["-c", f"{key}={_toml_value(value)}"])

        cmd.append("exec")
        cmd.extend(["--json", "--color", "never", "--skip-git-repo-check"])
        # A staged CODEX_HOME (see `stage_mcp_config`) holds only the
        # config.toml this run wrote, which codex has to load to find its
        # MCP servers.
        staged_home = provider_options.get("codex_home")
        if provider_options.get("ignore_user_config", True) and not staged_home:
            cmd.append("--ignore-user-config")
        if provider_options.get("ignore_rules", True):
            cmd.append("--ignore-rules")
        cmd.extend(["-C", str(cwd.resolve())])
        cmd.extend(provider_options.get("extra_args") or [])
        # `--` stops codex from parsing a prompt that itself starts with
        # `-` (e.g. "--verbose mode: ...") as a CLI option.
        cmd.append("--")
        cmd.append(prompt)
        return cmd

    def stage_mcp_config(self, run_tmp_root: Path, cfg, servers=None) -> dict:
        """Render the servers into the `mcp_servers` table of a
        :file:`config.toml` under a staged ``CODEX_HOME``, and point the
        attempt at it. Codex takes configuration either from that file or from
        `-c` overrides on its command line, and a server's ``env`` holds
        credentials, which argv would expose to any local process through
        ``ps``.
        """
        table: dict[str, dict] = {}
        for name, server in resolve_servers(cfg, servers).items():
            if "url" not in server:
                table[name] = {
                    k: v for k, v in server.items() if k in ("command", "args", "env")
                }
                continue
            problem = _codex_http_problem(name, cfg.mcp_servers[name])
            if problem is not None:
                raise UsageError(problem)
            table[name] = {"url": server["url"]}
            headers = cfg.mcp_servers[name].headers
            if headers:
                table[name]["bearer_token_env_var"] = _bearer_header_var(headers)
        if not table:
            return {}
        return {
            "codex_home": str(_stage_codex_home(run_tmp_root, table)),
            "mcp_server_names": sorted(table),
        }

    def validate_mcp_servers(self, servers: dict) -> list[str]:
        """Reject a ``type: sse`` or non-bearer HTTP server before the run
        starts, rather than only when the first codex_cli attempt stages it.
        """
        problems = []
        for name, server in servers.items():
            if isinstance(server, McpHttpServer):
                problem = _codex_http_problem(name, server)
                if problem is not None:
                    problems.append(problem)
        return problems

    def _prepare_prefix_rules(self, cwd: Path, provider_options: dict) -> dict:
        prefix_rules = provider_options.get("prefix_rules")
        if not prefix_rules:
            return provider_options

        rules_dir = cwd.resolve() / ".codex" / "rules"
        rules_dir.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: fixtureless trigger tasks share one cwd across
        # concurrent attempts, so a plain write_text() here could race with
        # another attempt's read of a half-written task.rules.
        tmp_path = rules_dir / f".task.rules.{uuid.uuid4().hex[:8]}.tmp"
        tmp_path.write_text(_render_prefix_rules(prefix_rules), encoding="utf-8")
        tmp_path.replace(rules_dir / "task.rules")

        prepared = dict(provider_options)
        prepared["ignore_rules"] = False
        prepared["trust_project_config"] = True
        return prepared

    def _wait_with_skill_kill(
        self,
        process: subprocess.Popen,
        state: StreamState,
        timeout_seconds: int,
        env: dict[str, str] | None = None,
    ) -> tuple[bool, bool]:
        deadline = time.time() + timeout_seconds
        next_session_poll = 0.0
        while True:
            if process.poll() is not None:
                return False, False
            if state.kill_signal.wait(timeout=0.05):
                terminate_tree(process, sigterm_timeout=1.0)
                return True, False
            now = time.time()
            if (
                state.thread_id
                and state.detected_skill is None
                and now >= next_session_poll
            ):
                next_session_poll = now + 0.25
                inv = find_session_explicit_skill_invocation(state.thread_id, env=env)
                if inv is not None:
                    state.detected_skill = inv
                    state.kill_signal.set()
                    terminate_tree(process, sigterm_timeout=1.0)
                    return True, False
            if now >= deadline:
                terminate_tree(process)
                return False, True

    def get_global_skills(self) -> list[str]:
        from ..skill_staging import discover_skills

        paths = [
            codex_home() / "skills",
            Path.home() / ".agents" / "skills",
        ]
        skills: list[str] = []
        for path in paths:
            if path.is_dir():
                skills.extend(name for name, _ in discover_skills([path]))
        return skills

    def preflight(self, cfg=None) -> list[CheckResult]:
        from ..skill_staging import check_global_skills_against_staged

        results: list[CheckResult] = []
        try:
            out = subprocess.run(
                ["codex", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except FileNotFoundError:
            return [
                CheckResult(
                    name="codex binary",
                    status="FAIL",
                    hint="codex not on PATH",
                )
            ]
        if out.returncode != 0:
            return [
                CheckResult(
                    name="codex binary",
                    status="FAIL",
                    hint=f"`codex --version` exited {out.returncode}",
                )
            ]
        results.append(
            CheckResult(name="codex binary", status="OK", hint=out.stdout.strip())
        )
        results.append(
            check_global_skills_against_staged(
                self.get_global_skills(),
                cfg,
                self.name,
                check_name="global codex skills",
            )
        )
        return results

    def probe_checks(self, probe_result, cfg=None) -> list[CheckResult]:
        if not probe_result.raw_transcript_path:
            return [
                CheckResult(
                    name="codex probe stream",
                    status="WARN",
                    hint="raw stream path missing",
                )
            ]
        return [
            CheckResult(
                name="codex probe stream",
                status="OK",
                hint=str(probe_result.raw_transcript_path),
            )
        ]

    def pre_run_warnings(self, cfg=None) -> list[CheckResult]:
        if cfg is None:
            return []
        from ..skill_staging import check_global_skills_against_staged

        result = check_global_skills_against_staged(
            self.get_global_skills(),
            cfg,
            self.name,
            check_name="global codex skills",
        )
        if result.status == "WARN":
            return [result]
        return []

    def stage_run_env(
        self,
        run_tmp_root: Path,
        cfg=None,
        skills_to_exclude: frozenset[str] = frozenset(),
    ) -> None:
        if cfg is None:
            return
        stage_skills_into(run_tmp_root, cfg.skills_dirs, exclude=skills_to_exclude)


def _stage_codex_home(run_tmp_root: Path, mcp_servers: dict[str, dict]) -> Path:
    """Build a ``CODEX_HOME`` holding this run's MCP servers, plus a copy of
    the developer's :file:`auth.json` — codex resolves credentials from
    ``CODEX_HOME`` too, and a copy leaves their own file untouched by a token
    refresh mid-attempt.

    A home of its own is also what leaves nothing of the developer's codex
    config in reach of the trial, the job ``--ignore-user-config`` does for
    every other run.
    """
    home = run_tmp_root / f"codex-home-{uuid.uuid4().hex[:12]}"
    home.mkdir(parents=True, exist_ok=True)
    lines = ["[mcp_servers]"]
    lines += [
        f"{_toml_key(name)} = {_toml_value(server)}"
        for name, server in mcp_servers.items()
    ]
    (home / "config.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    auth = codex_home() / "auth.json"
    if auth.is_file():
        shutil.copy2(auth, home / "auth.json")
    return home


def _render_prefix_rules(rules: list[dict]) -> str:
    chunks: list[str] = []
    for rule in rules:
        lines = [
            "prefix_rule(",
            f"    pattern = {_starlark_value(rule['pattern'])},",
            f"    decision = {_starlark_value(rule.get('decision', 'allow'))},",
        ]
        justification = rule.get("justification")
        if justification:
            lines.append(f"    justification = {_starlark_value(justification)},")
        lines.append(")")
        chunks.append("\n".join(lines))
    return "\n\n".join(chunks) + "\n"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            result[key] = _deep_merge(existing, value)
        else:
            result[key] = value
    return result


def _flatten_config(config: dict[str, Any]) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []

    def walk(parts: list[str], value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                walk([*parts, str(key)], child)
            return
        out.append((_config_key(parts), value))

    for key, value in config.items():
        if key == "permissions":
            # Codex 0.133 rejects permission profiles when the table is
            # decomposed into multiple dotted `-c permissions.<...>`
            # overrides. The official TOML shape is accepted when passed as
            # one inline table.
            out.append(("permissions", value))
            continue
        walk([str(key)], value)
    return out


def _config_key(parts: list[str]) -> str:
    return ".".join(part if part.isidentifier() else _toml_key(part) for part in parts)


def _starlark_value(value) -> str:
    return json.dumps(value)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return (
            "{"
            + ", ".join(
                f"{_toml_key(str(key))} = {_toml_value(child)}"
                for key, child in value.items()
            )
            + "}"
        )
    raise UsageError(
        f"codex_cli config override value {value!r} is not representable as TOML"
    )


def _start_orphan_watchdog(
    process: subprocess.Popen,
    poll_interval: float = 1.0,
    getppid: Callable[[], int] = os.getppid,
) -> threading.Event:
    """Kill the codex process tree if our own parent process dies.

    Pool workers survive a SIGKILL of the main agent-exam process, so
    without this an in-flight codex agent keeps running (and e.g. keeps
    deploying to Scrapy Cloud) with nobody left to score or stop it. A
    reparent (getppid() changes — to launchd/init or a subreaper) is the
    portable death signal on macOS, which has no PR_SET_PDEATHSIG.

    Returns an Event; set it to stop the watchdog once the process has
    been waited on (avoids a stray poll racing a reused pid).
    """
    initial_ppid = getppid()
    stop = threading.Event()

    def watch() -> None:
        while not stop.is_set() and process.poll() is None:
            if getppid() != initial_ppid:
                terminate_tree(process, sigterm_timeout=2.0)
                return
            stop.wait(poll_interval)

    threading.Thread(target=watch, daemon=True).start()
    return stop


def _preserve_failed_stream(raw_path: Path) -> Path:
    """Copy a failed attempt's raw stream out of the ephemeral run tree.

    The runner rmtree's the run tmp root when a run aborts, and a failed
    attempt never reaches the archive step — without this copy the JSON
    stream (the only record of why codex died) is destroyed.
    """
    try:
        stable_dir = Path(tempfile.gettempdir()) / "agent-exam-failed-streams"
        stable_dir.mkdir(parents=True, exist_ok=True)
        target = stable_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{raw_path.name}"
        shutil.copy2(raw_path, target)
        return target
    except OSError:
        return raw_path


_BEARER_ENV_REF = re.compile(r"^Bearer\s+\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def _bearer_header_var(headers: dict[str, str]) -> str | None:
    """The variable an ``Authorization: Bearer ${VAR}`` header reads, or
    ``None`` when *headers* isn't shaped exactly that way.
    """
    auth = next((v for k, v in headers.items() if k.lower() == "authorization"), None)
    match = _BEARER_ENV_REF.match(auth or "")
    if match is None or len(headers) > 1:
        return None
    return match.group(1)


def _codex_http_problem(name: str, server: McpHttpServer) -> str | None:
    """What keeps codex_cli from attaching HTTP *server* config as declared.

    Codex speaks streamable HTTP only, not classic SSE. It also sends no
    header of its own; it authenticates by reading a bearer token out of a
    named environment variable at launch, so it wants *headers* as written
    rather than resolved, and any header shape other than a single
    ``Authorization: Bearer ${VAR}`` has nowhere to go.
    """
    if server.type == "sse":
        return (
            f"mcp_servers.{name}: codex_cli has no sse transport, only streamable http"
        )
    if server.headers and _bearer_header_var(server.headers) is None:
        return (
            f"mcp_servers.{name}: codex_cli passes no HTTP header other than "
            'an `Authorization: "Bearer ${VAR}"` it reads from the '
            "environment; scope the task with `providers:` to leave codex_cli "
            "out of it"
        )
    return None


def _toml_key(value: str) -> str:
    return json.dumps(value)
