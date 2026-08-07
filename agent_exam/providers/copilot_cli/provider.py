from __future__ import annotations

import os
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from ..._models import _StrictModel
from ...errors import FrameworkError, ProviderTimeout
from ...ratelimit import with_retries
from ..base import Provider
from ..process_utils import terminate_tree
from .stream_parser import StreamState, drain_stderr, drain_stream
from .transcripts import build_run_result

if TYPE_CHECKING:
    from pydantic import BaseModel

    from ...schemas import CheckResult, RunResult


class CopilotCliTaskConfig(_StrictModel):
    """The `copilot_cli:` block on a task YAML."""

    allowed_tools: list[str] | None = None


class CopilotCliProvider(Provider):
    name = "copilot_cli"
    skills_rel_path: ClassVar[str] = ".github/skills"
    # Copilot CLI's read-only file inspection tools per GitHub's CLI
    # command reference: view (read files/dirs), glob (find by pattern),
    # grep (search text). All accepted as `--available-tools` /
    # `--allow-tool` values.
    safe_judge_tools = ("view", "glob", "grep")
    task_config_model: ClassVar[type[BaseModel]] = CopilotCliTaskConfig

    def task_options(
        self, task_cfg: CopilotCliTaskConfig | None, framework_cfg, task_kind: str
    ) -> dict:
        opts: dict = {}
        if task_cfg and task_cfg.allowed_tools:
            opts["allowed_tools"] = list(task_cfg.allowed_tools)
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
        """Invoke with transparent rate-limit retry.

        Copilot CLI does not surface HTTP status codes in the JSON stream, so
        rate-limit detection is not currently possible. We still wrap in
        with_retries for consistency and future-proofing.
        """
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
        cmd = ["copilot", "-p", prompt, "--output-format", "json"]
        if model:
            cmd.extend(["--model", model])

        # Per-task tool restriction: if `allowed_tools` is provided, restrict
        # the model to exactly those tools (plus skill + report_intent which are
        # always needed) using --available-tools and --allow-tool.  The model
        # simply cannot call tools outside the list — no prompt, no hang.
        # If no `allowed_tools` is set, fall through to extra_args (the config
        # default includes --allow-all-tools for backward-compatible headless use).
        allowed = list(provider_options.get("allowed_tools") or [])
        if allowed:
            # Precise mode: only the listed tools (plus internal ones) are
            # visible to the model — anything else is hidden entirely.
            tools = list(dict.fromkeys([*allowed, "skill", "report_intent"]))
            cmd.extend(["--available-tools", ",".join(tools)])
            for t in tools:
                cmd.extend(["--allow-tool", t])
        else:
            # Default headless mode: allow all tools (required to avoid
            # interactive permission prompts blocking the subprocess).
            cmd.append("--allow-all-tools")

        cmd.extend(provider_options.get("extra_args") or [])

        env = dict(os.environ)
        for key, value in (provider_options.get("env_overrides") or {}).items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value

        cwd_abs = cwd.resolve()

        # Save the raw JSONL stream for diagnostics. Use a UUID suffix so that
        # concurrent trigger attempts (which share the same cwd) each get their
        # own file rather than trampling each other.
        raw_dir = cwd_abs.parent / ".raw_streams" / cwd_abs.name
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"stream_{uuid.uuid4().hex[:8]}.jsonl"
        raw_fh = raw_path.open("wb")

        started = time.time()
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(cwd_abs),
            env=env,
            start_new_session=True,
        )

        state = StreamState(provider=self)
        if stop_on_first_skill:
            state.skill_detection_enabled = True
            state.target_skill = provider_options.get("target_skill")
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
                process, state, timeout_seconds
            )
        else:
            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                terminate_tree(process)

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
                f"copilot -p timed out after {timeout_seconds}s "
                f"(session_id={state.session_id})",
                partial_run_result=partial,
            )

        if process.returncode != 0 and not killed_on_skill:
            stderr_tail = (
                bytes(state.stderr_tail).decode("utf-8", errors="replace").strip()
            )
            raise FrameworkError(
                f"copilot -p exited with {process.returncode}. "
                f"stderr tail:\n{stderr_tail[-1024:]}"
            )

        return build_run_result(
            state,
            wall_time_seconds=wall_time,
            stream_detected_skill=state.detected_skill,
            raw_transcript_path=raw_path,
            user_prompt=prompt,
        )

    def _wait_with_skill_kill(
        self, process: subprocess.Popen, state: StreamState, timeout_seconds: int
    ) -> tuple[bool, bool]:
        """Poll for early skill fire, natural exit, or wall-clock timeout.

        Returns `(killed_on_skill, timed_out)`.
        """
        deadline = time.time() + timeout_seconds
        while True:
            if process.poll() is not None:
                return False, False
            if state.kill_signal.wait(timeout=0.05):
                terminate_tree(process, sigterm_timeout=1.0)
                return True, False
            if time.time() >= deadline:
                terminate_tree(process)
                return False, True

    def get_global_skills(self) -> list[str]:
        """Discover skills installed globally for Copilot CLI."""
        from ..skill_staging import discover_skills

        copilot_dir = Path.home() / ".copilot"
        skill_roots = [copilot_dir / "skills"]
        installed_plugins = copilot_dir / "installed-plugins"
        if installed_plugins.is_dir():
            skill_roots.extend(installed_plugins.glob("*/*/skills"))

        return [name for name, _ in discover_skills(skill_roots)]

    def preflight(self, cfg=None) -> list[CheckResult]:
        """Binary version check + personal-skills leak warning."""
        from ..skill_staging import check_global_skills_against_staged
        from .doctor_probes import check_binary

        results = [check_binary()]
        if results[0].status == "FAIL":
            return results
        results.append(
            check_global_skills_against_staged(
                self.get_global_skills(),
                cfg,
                self.name,
                check_name="personal skills",
            )
        )
        return results

    def probe_checks(self, probe_result, cfg=None) -> list[CheckResult]:
        """Post-probe: verify the model name was captured."""
        from .doctor_probes import check_probe_model

        return [check_probe_model(probe_result)]

    def pre_run_warnings(self, cfg=None) -> list[CheckResult]:
        """Warn before any trial if personal skills could leak."""
        from ..skill_staging import check_global_skills_against_staged

        result = check_global_skills_against_staged(
            self.get_global_skills(),
            cfg,
            self.name,
            check_name="personal skills",
        )
        # Only surface as a warning at run-start when non-empty.
        if result.status == "WARN":
            return [result]
        return []
