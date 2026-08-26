from __future__ import annotations

import json
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from claude_measure_usage.parse import find_transcript_path

from ..._models import _StrictModel
from ...errors import FrameworkError, ProviderTimeout, RateLimitError
from ...mcp import canonical_server_prefix, probe_connection_check, stage_mcp_json
from ...ratelimit import with_retries
from ...schemas import CheckResult, RunResult
from ..base import Provider
from ..child_env import build_child_env
from ..process_utils import terminate_tree, wait_or_terminate
from ..skill_staging import check_global_skills_against_staged, discover_skills
from .blocked_plugins import enabled_blocked_in_settings
from .doctor_probes import blocked_plugins_in_probe, hermetic_check
from .session_checks import skill_descriptions_in_session
from .stream_parser import StreamState, drain_stderr, drain_stream
from .transcripts import load_run_result

if TYPE_CHECKING:
    from pydantic import BaseModel

_RATE_LIMIT_STATUSES = frozenset({429, 529})


class ClaudeCodeTaskConfig(_StrictModel):
    """The `claude_code:` block on a task YAML."""

    permission_mode: str | None = None
    allowed_tools: list[str] | None = None


class ClaudeCodeProvider(Provider):
    name = "claude_code"
    safe_judge_tools = ("Read", "Glob", "Grep")
    skills_rel_path: ClassVar[str] = ".claude/skills"
    task_config_model: ClassVar[type[BaseModel]] = ClaudeCodeTaskConfig

    def task_options(
        self, task_cfg: ClaudeCodeTaskConfig | None, framework_cfg, task_kind: str
    ) -> dict:
        opts: dict = {}
        explicit_perm = task_cfg.permission_mode if task_cfg else None
        perm_mode = explicit_perm or (
            "bypassPermissions"
            if task_kind == "trigger"
            else framework_cfg.permission_mode
        )
        if perm_mode:
            opts["permission_mode"] = perm_mode
        if task_cfg and task_cfg.allowed_tools:
            opts["allowed_tools"] = list(task_cfg.allowed_tools)
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
        """Invoke with transparent rate-limit retry. See ratelimit.with_retries
        for the schedule. Other errors (timeout, framework) bubble up directly.
        """
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
        cmd = [
            "claude",
            "-p",
            prompt,
            "--output-format",
            "stream-json",
            "--verbose",
            # Leave the developer's `~/.claude/settings.json` out of the run, so
            # the plugins they happen to have enabled don't load into it. Their
            # skills compete with the staged ones for the fixed-size skill
            # listing Claude Code shows the model, and once that budget is
            # spent the remaining skills are listed by name alone, with no
            # description — so a staged skill can silently stop triggering
            # depending on what the developer installed.
            "--setting-sources",
            "project,local",
        ]
        if stop_on_first_trigger:
            cmd.append("--include-partial-messages")
        if model:
            cmd.extend(["--model", model])
        mode = provider_options.get("permission_mode")
        if mode:
            cmd.extend(["--permission-mode", str(mode)])
        mcp_config = provider_options.get("mcp_config_path")
        if mcp_config:
            # `--mcp-config` takes a variadic <configs...>: whatever argv
            # entry follows it must itself be a flag, or it gets swallowed
            # as another config path.
            cmd.extend(["--mcp-config", str(mcp_config)])
        # Always strict, even with no servers of our own: without it the
        # developer's `~/.claude.json` servers load into the trial, the
        # same hermeticity rule that keeps their plugins out.
        cmd.append("--strict-mcp-config")
        allowed = list(provider_options.get("allowed_tools") or [])
        if allowed:
            if "Skill" not in allowed:
                # Always pre-approve the Skill tool when an allowlist is set.
                # `--allowed-tools` is a pre-approval list; in headless `-p`
                # mode anything outside it falls through to the permission
                # prompt, which auto-rejects with no human to approve. Without
                # Skill pre-approved, the agent's natural-language route to a
                # skill returns is_error=true ("Execute skill: <name>") — the
                # agent reads that as a real error and gives up. An eval
                # suite exists to evaluate a skill, so blocking the very
                # mechanism it tests has no legitimate use case.
                allowed.append("Skill")
            # Same reasoning for every attached MCP server: an allowlist that
            # doesn't name it auto-rejects its calls, so the tools the task is
            # about never run. `mcp__<server>` covers that server's tool set.
            allowed.extend(
                canonical_server_prefix(name)
                for name in provider_options.get("mcp_server_names") or ()
                if canonical_server_prefix(name) not in allowed
            )
            # Claude Code's `--allowed-tools` takes a variadic <tools...>;
            # joining with commas keeps it a single argv entry so any
            # subsequent flags (including `extra_args`) aren't swallowed.
            cmd.extend(["--allowed-tools", ",".join(str(p) for p in allowed)])
        cmd.extend(provider_options.get("extra_args") or [])

        env = build_child_env(
            provider_options.get("env_overrides"), drop=("CLAUDECODE",)
        )

        cwd_abs = str(cwd.resolve())
        started = time.time()
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd_abs,
            env=env,
            start_new_session=True,
        )

        state = StreamState()
        if stop_on_first_trigger:
            state.skill_detection_enabled = True
            state.target_skill = provider_options.get("target_skill")
            state.target_tool = provider_options.get("target_tool")
            state.negative_trigger_mode = bool(provider_options.get("negative_trigger"))
        t_out = threading.Thread(
            target=drain_stream, args=(process.stdout, state), daemon=True
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

        if timed_out:
            # Best-effort: if claude -p had emitted a session_id before
            # the kill, the raw transcript on disk is usable — build a
            # partial RunResult so the caller can still archive
            # attempt.json + trajectory.json. Any step can fail quietly;
            # the timeout itself is still the outcome we report.
            partial = None
            if state.session_id:
                try:
                    transcript = find_transcript_path(state.session_id, cwd_abs)
                    if transcript:
                        partial = load_run_result(
                            Path(transcript),
                            wall_time_seconds=wall_time,
                            explicit_cost_usd=state.total_cost_usd,
                            stream_detected_skill=state.detected_skill,
                            stream_detected_tool=state.detected_tool,
                            mcp_server_status=state.mcp_server_status,
                        )
                except Exception:
                    partial = None
            raise ProviderTimeout(
                f"claude -p timed out after {timeout_seconds}s "
                f"(session_id={state.session_id})",
                partial_run_result=partial,
            )

        if state.api_error_status in _RATE_LIMIT_STATUSES:
            raise RateLimitError(
                f"claude -p reported api_error_status={state.api_error_status} "
                f"({state.result_error or 'no detail'})"
            )

        # A skill-kill terminates the process via SIGTERM/SIGKILL, so a
        # non-zero exit is expected; don't surface it as a framework error.
        if process.returncode != 0 and not killed_on_skill:
            stderr_tail = (
                bytes(state.stderr_tail).decode("utf-8", errors="replace").strip()
            )
            raise FrameworkError(
                f"claude -p exited with {process.returncode}. stderr tail:\n{stderr_tail[-1024:]}"
            )

        if not state.session_id:
            stderr_tail = (
                bytes(state.stderr_tail).decode("utf-8", errors="replace").strip()
            )
            raise FrameworkError(
                "claude -p finished without emitting a session_id; "
                f"stderr tail:\n{stderr_tail[-1024:]}"
            )

        transcript = find_transcript_path(state.session_id, cwd_abs)
        if not transcript:
            raise FrameworkError(
                f"transcript not found for session {state.session_id} under ~/.claude/projects/"
            )

        return load_run_result(
            Path(transcript),
            wall_time_seconds=wall_time,
            explicit_cost_usd=state.total_cost_usd,
            stream_detected_skill=state.detected_skill,
            stream_detected_tool=state.detected_tool,
            mcp_server_status=state.mcp_server_status,
        )

    def _wait_with_skill_kill(
        self, process: subprocess.Popen, state: StreamState, timeout_seconds: int
    ) -> tuple[bool, bool]:
        """Poll for either early skill fire, natural exit, or wall-clock timeout.

        Returns `(killed_on_skill, timed_out)`.
        """
        deadline = time.time() + timeout_seconds
        try:
            while True:
                if process.poll() is not None:
                    return False, False
                if state.kill_signal.wait(timeout=0.05):
                    _terminate_tree(process)
                    return True, False
                if time.time() >= deadline:
                    _terminate_tree(process)
                    return False, True
        except BaseException:
            terminate_tree(process, sigterm_timeout=0)
            raise

    def get_global_skills(self) -> list[str] | None:
        """Run a token-free clean ``claude -p`` probe and return the
        ``skills`` list from its ``system/init`` event.

        Returns ``None`` when the probe could not be run.  The process is
        terminated immediately after reading the init event (before any
        API call), so this costs zero tokens.
        """
        cmd = [
            "claude",
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "hi",
        ]
        with tempfile.TemporaryDirectory(prefix="agent-exam-clean-probe-") as tmp:
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=tmp,
                )
            except (OSError, FileNotFoundError):
                return None

            lines: list[str] = []

            def _reader() -> None:
                for line in proc.stdout:  # type: ignore[union-attr]
                    lines.append(line)
                    if len(lines) >= 20:
                        break

            t = threading.Thread(target=_reader)
            t.start()
            t.join(timeout=10)

            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

            for line in lines:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("type") == "system" and data.get("subtype") == "init":
                    return data.get("skills") or []
            return None

    def preflight(self, cfg=None) -> list[CheckResult]:
        """Binary + version + static blocked-plugin check against
        ~/.claude/settings.json. No LLM call.
        """
        results: list[CheckResult] = []

        try:
            out = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except FileNotFoundError:
            return [
                CheckResult(
                    name="claude binary", status="FAIL", hint="claude not on PATH"
                )
            ]
        if out.returncode != 0:
            return [
                CheckResult(
                    name="claude binary",
                    status="FAIL",
                    hint=f"`claude --version` exited {out.returncode}",
                )
            ]
        results.append(
            CheckResult(name="claude binary", status="OK", hint=out.stdout.strip())
        )

        if cfg is not None:
            results.extend(_blocked_plugins_static_check(cfg))

        global_skills = self.get_global_skills()
        if global_skills is None:
            results.append(
                CheckResult(
                    name="global claude skills",
                    status="WARN",
                    hint="could not run clean claude probe to list global skills",
                )
            )
        else:
            results.append(
                check_global_skills_against_staged(
                    global_skills,
                    cfg,
                    self.name,
                    check_name="global claude skills",
                    normalize=lambda s: s.split(":", 1)[-1],
                )
            )

        return results

    def stage_mcp_config(self, run_tmp_root: Path, cfg, servers=None) -> dict:
        """Render `{"mcpServers": ...}` for `--mcp-config`."""
        return stage_mcp_json(run_tmp_root, cfg, servers)

    def probe_checks(self, probe_result, cfg=None) -> list[CheckResult]:
        """Post-probe checks against the round-trip transcript. Catches
        memory leaks and blocked-plugin loading that the static check
        would miss (e.g. plugins loaded via mechanisms not reflected in
        settings.json)."""
        transcript = probe_result.raw_transcript_path
        results = [hermetic_check(transcript)]
        results.extend(probe_connection_check(probe_result, cfg))
        if cfg is not None:
            cfg_provider = cfg.provider(self.name)
            results.append(
                blocked_plugins_in_probe(transcript, cfg_provider.blocked_plugins)
            )
        return results

    def session_checks(self, run_result, cfg=None) -> list[CheckResult]:
        """Runner-side warnings from the first finished attempt. Today just
        checks that every staged skill was listed with its description.
        """
        if cfg is None:
            return []
        return [
            skill_descriptions_in_session(
                run_result.raw_transcript_path,
                [name for name, _path in discover_skills(cfg.skills_dirs)],
            )
        ]

    def pre_run_warnings(self, cfg=None) -> list[CheckResult]:
        """Runner-side warnings printed before any trial starts. Today
        just checks for enabled blocked plugins in ~/.claude/settings.json.
        """
        if cfg is None:
            return []
        return _blocked_plugins_static_check(cfg, context="run-start")


def _blocked_plugins_static_check(cfg, context: str = "doctor") -> list[CheckResult]:
    """Read `~/.claude/settings.json` and return check results for any
    blocked plugin that's enabled. Used by both doctor preflight and
    runner pre-run-warnings.
    """
    provider_cfg = cfg.provider("claude_code")
    blocked = provider_cfg.blocked_plugins
    if not blocked:
        return []
    settings_path = Path.home() / ".claude" / "settings.json"
    found = enabled_blocked_in_settings(settings_path, blocked)
    if found:
        return [
            CheckResult(
                name="blocked plugins enabled",
                status="WARN",
                hint=(
                    f"{', '.join(found)} enabled in ~/.claude/settings.json; "
                    "they'll load additively into every trial and "
                    "--without-skill can't exclude them"
                ),
            )
        ]
    # Report OK only from doctor (runner stays quiet when nothing's wrong).
    if context == "doctor":
        return [
            CheckResult(
                name="blocked plugins absent",
                status="OK",
                hint=f"none of {blocked} enabled",
            )
        ]
    return []


def _terminate_tree(process: subprocess.Popen) -> None:
    terminate_tree(process)
