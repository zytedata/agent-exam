from __future__ import annotations

import json
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, field_validator

from ..._models import _StrictModel
from ...errors import FrameworkError, ProviderTimeout
from ...mcp import probe_connection_check, resolve_servers
from ...ratelimit import with_retries
from ...schemas import CheckResult, RunResult
from ..base import Provider
from ..child_env import apply_env_overrides, build_child_env
from ..process_utils import terminate_tree, wait_or_terminate
from .stream_parser import StreamState, drain_stderr, drain_stream
from .transcripts import build_run_result

_RATE_LIMIT_STATUSES = frozenset({429, 529})


class OpenCodeTaskConfig(_StrictModel):
    """The `opencode:` block on a task YAML.

    `permission` is `{tool: action}` or `{tool: {pattern: action}}` —
    actions are "allow" | "ask" | "deny". A bare `"deny"` on certain
    tools (bash, edit, write, read) can hang opencode in headless mode;
    use the object form (`{"bash": {"*": "deny"}}`) instead.
    """

    permission: dict[str, str | dict[str, str]] | None = None

    @field_validator("permission")
    @classmethod
    def _validate_permission(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        if v is None:
            return v
        for tool_name, rules in v.items():
            if isinstance(rules, str):
                if rules not in _VALID_PERMISSION_ACTIONS:
                    raise ValueError(
                        f"{tool_name}: invalid action {rules!r} "
                        f"(expected one of {sorted(_VALID_PERMISSION_ACTIONS)})"
                    )
                if rules == "deny" and tool_name in _BARE_DENY_HANG_TOOLS:
                    raise ValueError(
                        f'{tool_name}: bare string "deny" on '
                        f"{tool_name} may cause opencode to hang in "
                        f"headless mode; use object syntax instead, e.g. "
                        f'{{"{tool_name}": {{"*": "deny"}}}}'
                    )
            elif isinstance(rules, dict):
                for pattern, action in rules.items():
                    if action not in _VALID_PERMISSION_ACTIONS:
                        raise ValueError(
                            f"{tool_name}.{pattern}: invalid action "
                            f"{action!r} (expected one of "
                            f"{sorted(_VALID_PERMISSION_ACTIONS)})"
                        )
            else:
                raise ValueError(
                    f"{tool_name}: must be a string or mapping (got {rules!r})"
                )
        return v


_VALID_PERMISSION_ACTIONS = frozenset({"allow", "ask", "deny"})


class OpenCodeProvider(Provider):
    name = "opencode"
    safe_judge_tools = ("read", "glob", "grep")
    skills_rel_path: ClassVar[str] = ".opencode/skills"
    task_config_model: ClassVar[type[BaseModel]] = OpenCodeTaskConfig

    def task_options(
        self, task_cfg: OpenCodeTaskConfig | None, framework_cfg, task_kind: str
    ) -> dict:
        opts: dict = {"pure": framework_cfg.pure}
        explicit_perm = task_cfg.permission if task_cfg else None
        if explicit_perm is not None:
            opts["permission"] = dict(explicit_perm)
        elif task_kind == "trigger":
            opts["permission"] = {"*": "allow"}
        return opts

    def invoke(
        self,
        prompt: str,
        model: str,
        cwd: Path,
        provider_options: dict,
        stop_on_first_trigger: bool,
        timeout_seconds: int,
    ) -> RunResult:
        return with_retries(
            lambda: self._invoke_once(
                prompt,
                model,
                cwd,
                provider_options,
                stop_on_first_trigger,
                timeout_seconds,
            )
        )

    def _invoke_once(
        self,
        prompt: str,
        model: str,
        cwd: Path,
        provider_options: dict,
        stop_on_first_trigger: bool,
        timeout_seconds: int,
    ) -> RunResult:
        cmd = ["opencode", "run", "--format", "json"]
        pure = provider_options.get("pure", True)
        if pure:
            cmd.append("--pure")
        if model and model != "opencode":
            cmd.extend(["--model", model])
        agent = provider_options.get("agent")
        if agent:
            cmd.extend(["--agent", agent])
        cmd.extend(["--dir", str(cwd.resolve())])
        if provider_options.get("mcp_config"):
            # Whether each server connected is logged and nothing else
            # reports it — not the JSON stream, not the exit code.
            cmd.extend(["--print-logs", "--log-level", "INFO"])
        cmd.append(prompt)

        env = build_child_env()
        permission_config = build_permission_config(
            permission=provider_options.get("permission"),
            allowed_tools=provider_options.get("allowed_tools"),
            mcp_servers=provider_options.get("mcp_server_names"),
        )
        config: dict = {}
        if permission_config:
            config["permission"] = permission_config
        if provider_options.get("mcp_config"):
            config["mcp"] = provider_options["mcp_config"]
        if config:
            env["OPENCODE_CONFIG_CONTENT"] = json.dumps(config)
        if provider_options.get("xdg_config_home"):
            # OpenCode merges the developer's own ~/.config/opencode/opencode.json
            # into OPENCODE_CONFIG_CONTENT rather than being replaced by it, so
            # keeping the run hermetic means pointing it at an empty config dir.
            env["XDG_CONFIG_HOME"] = str(provider_options["xdg_config_home"])
        # Overrides last, so a task can still replace anything set above.
        apply_env_overrides(env, provider_options.get("env_overrides"))

        started = time.time()
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(cwd.resolve()),
            env=env,
            start_new_session=True,
        )

        state = StreamState(provider=self)
        # Needed whether or not the run is cut short: the trajectory is
        # named after these too, not just the kill decision.
        state.mcp_server_names = tuple(provider_options.get("mcp_server_names") or ())
        if stop_on_first_trigger:
            state.skill_detection_enabled = True
            state.target_skill = provider_options.get("target_skill")
            state.target_tool = provider_options.get("target_tool")
            state.negative_trigger_mode = bool(provider_options.get("negative_trigger"))

        # Open raw stream file before the reader thread starts so every
        # line is captured even if we kill early. The handle is closed after
        # the thread drains.
        raw_dir = cwd.parent / ".raw_streams" / cwd.name
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / "stream.jsonl"
        raw_fh = raw_path.open("wb")
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
        if stop_on_first_trigger:
            killed_on_skill, timed_out = self._wait_with_skill_kill(
                process, state, timeout_seconds
            )
        else:
            timed_out = wait_or_terminate(process, timeout_seconds)

        wall_time = time.time() - started
        t_out.join(timeout=5)
        t_err.join(timeout=5)
        raw_fh.close()

        if timed_out:
            partial = None
            if state.session_id:
                try:
                    partial = build_run_result(
                        state,
                        wall_time_seconds=wall_time,
                        user_prompt=prompt,
                        raw_transcript_path=raw_path,
                    )
                except Exception:
                    partial = None
            raise ProviderTimeout(
                f"opencode run timed out after {timeout_seconds}s "
                f"(session_id={state.session_id})",
                partial_run_result=partial,
            )

        if process.returncode != 0 and not killed_on_skill:
            stderr_tail = (
                bytes(state.stderr_tail).decode("utf-8", errors="replace").strip()
            )
            raise FrameworkError(
                f"opencode run exited with {process.returncode}. "
                f"stderr tail:\n{stderr_tail[-1024:]}"
            )

        if not state.session_id:
            stderr_tail = (
                bytes(state.stderr_tail).decode("utf-8", errors="replace").strip()
            )
            raise FrameworkError(
                "opencode run finished without emitting a sessionID; "
                f"stderr tail:\n{stderr_tail[-1024:]}"
            )

        return build_run_result(
            state,
            wall_time_seconds=wall_time,
            stream_detected_skill=state.detected_skill,
            stream_detected_tool=state.detected_tool,
            raw_transcript_path=raw_path,
            user_prompt=prompt,
        )

    def stage_mcp_config(self, run_tmp_root: Path, cfg, servers=None) -> dict:
        """Translate the servers into OpenCode's own `mcp` config block,
        which ships to the child through `OPENCODE_CONFIG_CONTENT`.
        """
        mcp: dict[str, dict] = {}
        for name, server in resolve_servers(cfg, servers).items():
            if "url" in server:
                entry = {"type": "remote", "url": server["url"], "enabled": True}
                if server.get("headers"):
                    entry["headers"] = server["headers"]
            else:
                entry = {
                    "type": "local",
                    "command": [server["command"], *server.get("args", [])],
                    "enabled": True,
                }
                if server.get("env"):
                    entry["environment"] = server["env"]
            mcp[name] = entry
        if not mcp:
            return {}
        # OpenCode always merges ~/.config/opencode/opencode.json into the
        # env-supplied config rather than being replaced by it, so a
        # developer's own MCP servers would otherwise leak into every run.
        # An isolated, empty XDG_CONFIG_HOME (see `_invoke_once`) keeps it
        # out — rendered next to the attempt cwd, not inside it, like the
        # MCP config itself.
        xdg_config_home = run_tmp_root / f"{uuid.uuid4().hex[:12]}.opencode-config"
        xdg_config_home.mkdir(parents=True, exist_ok=True)
        return {
            "mcp_config": mcp,
            "mcp_server_names": sorted(mcp),
            "xdg_config_home": xdg_config_home,
        }

    def _wait_with_skill_kill(
        self, process: subprocess.Popen, state: StreamState, timeout_seconds: int
    ) -> tuple[bool, bool]:
        deadline = time.time() + timeout_seconds
        try:
            while True:
                if process.poll() is not None:
                    return False, False
                if state.kill_signal.wait(timeout=0.05):
                    # Use a short SIGTERM window: OpenCode defers shutdown until
                    # the current inference step finishes (~2s), then runs tool
                    # calls in the extra step. 1s lets the DB flush before SIGKILL
                    # but stops before the next tool executes.
                    terminate_tree(process, sigterm_timeout=1.0)
                    return True, False
                if time.time() >= deadline:
                    terminate_tree(process)
                    return False, True
        except BaseException:
            terminate_tree(process, sigterm_timeout=0)
            raise

    def get_global_skills(self) -> list[str]:
        """Discover global skills from ``~/.config/opencode/skills/``
        (native) and ``~/.claude/skills/`` (Claude compatibility)."""
        from ..skill_staging import discover_skills

        global_paths = [
            Path.home() / ".config" / "opencode" / "skills",
            Path.home() / ".claude" / "skills",
        ]
        skills: list[str] = []
        for p in global_paths:
            if p.is_dir():
                skills.extend(name for name, _ in discover_skills([p]))
        return skills

    def preflight(self, cfg=None) -> list[CheckResult]:
        results: list[CheckResult] = []
        try:
            out = subprocess.run(
                ["opencode", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except FileNotFoundError:
            return [
                CheckResult(
                    name="opencode binary", status="FAIL", hint="opencode not on PATH"
                )
            ]
        if out.returncode != 0:
            return [
                CheckResult(
                    name="opencode binary",
                    status="FAIL",
                    hint=f"`opencode --version` exited {out.returncode}",
                )
            ]
        results.append(
            CheckResult(name="opencode binary", status="OK", hint=out.stdout.strip())
        )
        from ..skill_staging import check_global_skills_against_staged
        from .doctor_probes import check_db_exists, check_db_schema

        results.append(check_db_exists())
        results.append(check_db_schema())
        results.append(
            check_global_skills_against_staged(
                self.get_global_skills(),
                cfg,
                self.name,
                check_name="global opencode skills",
            )
        )
        return results

    def probe_checks(self, probe_result, cfg=None) -> list[CheckResult]:
        from .doctor_probes import check_probe_model

        results = [check_probe_model(probe_result)]
        results.extend(probe_connection_check(probe_result, cfg))
        return results

    def pre_run_warnings(self, cfg=None) -> list[CheckResult]:
        return []


# Bare-string "deny" on these tools may cause opencode to hang in headless
# mode (a known opencode bug). Object syntax {"*": "deny"} is safe.
_BARE_DENY_HANG_TOOLS = frozenset({"bash", "edit", "write", "read"})


def build_permission_config(
    *,
    permission: dict | None,
    allowed_tools: list[str] | tuple[str, ...] | None,
    mcp_servers: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Compose the OpenCode `permission` mapping for one invocation.

    - If ``permission`` is provided, it's used as-is.
    - Else if ``allowed_tools`` is provided, translate the allowlist
      into OpenCode's permission shape using the global ``"*"``
      wildcard: deny everything, then explicitly allow each listed
      tool. OpenCode honors this by hiding non-allowed tools from the
      model entirely (matching Claude Code's ``--allowed-tools``
      semantics).
    - ``external_directory`` is always force-denied if absent — its
      OpenCode default of ``ask`` would cause headless runs to hang
      waiting for user approval that never arrives.

    An allowlist also allows every tool of each server in *mcp_servers*:
    one that doesn't name them denies them, so the tools the task is about
    never run. A permission key is matched as a wildcard against the tool
    name, so ``<server>*`` covers a server's tool set whichever way
    OpenCode joins server and tool. Added even when ``permission`` is an
    explicit override, so an attached server isn't left to whatever that
    override's own default (e.g. a bare ``"*": "ask"``) would do to it.
    """
    permission_config = dict(permission or {})
    if not permission_config and allowed_tools is not None:
        permission_config["*"] = "deny"
        for tool in allowed_tools:
            permission_config[tool] = "allow"
    for name in mcp_servers or ():
        permission_config.setdefault(f"{name}*", "allow")
    if "external_directory" not in permission_config:
        permission_config["external_directory"] = "deny"
    return permission_config
