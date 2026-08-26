from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from .._models import _StrictModel
from .skill_staging import stage_skills_into

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import BaseModel

    from ..config import Config, McpServerConfig
    from ..schemas import CheckResult, RunResult


class Provider:
    """Base class for harness adapters (Claude Code, OpenCode, etc).

    Was a ``Protocol`` historically; concrete providers used duck typing.
    Now a regular base class so shared defaults (notably
    :py:meth:`is_same_skill`) can be inherited. Concrete providers
    override the methods that need harness-specific behavior; the rest
    use the inherited defaults.
    """

    name: str = ""

    #: Relative path, under an attempt's runtime cwd, where this provider's
    #: host agent walk-up discovers project skills (e.g. ``.claude/skills``
    #: for Claude Code, ``.github/skills`` for Copilot CLI). Every provider
    #: must set this.
    skills_rel_path: ClassVar[str]

    #: Pydantic model for the harness's task-config section (e.g. the
    #: `claude_code:` block in a task YAML). Subclasses override to
    #: declare their schema; the Task model uses this for validation.
    #: The base class default accepts only empty mappings — providers
    #: with no per-task knobs can inherit it as-is.
    task_config_model: ClassVar[type[BaseModel]] = _StrictModel

    #: Tool names this harness exposes to a ``judge_agent`` assertion by
    #: default. Read-only, cwd-confined inspection tools — typically
    #: this harness's equivalents of read / glob / grep. Empty on the
    #: base class so providers that don't (yet) support ``judge_agent``
    #: cause it to fail fast with a clear error. Names follow each
    #: harness's native casing. Providers whose native permission model
    #: is not an allowlist should override :meth:`judge_agent_options`.
    safe_judge_tools: tuple[str, ...] = ()

    @property
    def supports_mcp(self) -> bool:
        """Whether this harness can attach MCP servers, i.e. whether it
        overrides :meth:`stage_mcp_config`. Derived from the override so the
        preflight can ask without importing the provider registry, which
        imports every provider, which imports the module the preflight lives
        in — and without every provider hand-maintaining a flag that could
        drift from its own override.
        """
        return type(self).stage_mcp_config is not Provider.stage_mcp_config

    reports_mcp_connections: ClassVar[bool] = True
    """Whether this harness announces per-server MCP connection status at
    session start (see :func:`agent_exam.mcp.connection_check`). Meaningless
    when :attr:`supports_mcp` is ``False``. A harness whose event stream
    carries no such signal overrides this to ``False`` so
    :func:`agent_exam.mcp.preflight` can say so, rather than a dead server
    reading as a plain task failure with no indication why.
    """

    #: Human-readable model source used when ``invoke(..., model="")``
    #: intentionally omits the provider's model flag. ``None`` means doctor
    #: should not run LLM probes without an explicit configured model.
    omitted_model_label: ClassVar[str | None] = None

    def judge_agent_options(self) -> dict:
        """Provider options for the cwd-aware ``judge_agent`` assertion.

        Most providers expose read/glob/grep-like tools directly, so the
        default bridge passes those names through the existing per-provider
        tool/permission mapper. Providers with a different native permission
        model can override this.
        """
        return {"allowed_tools": list(self.safe_judge_tools)}

    @staticmethod
    def is_same_skill(detected: str, target: str) -> bool:
        """Return True if a detected skill name refers to the asserted target.

        Skill-name conventions are harness-specific. The default accepts
        an exact match or a match after stripping a single
        ``<plugin>:`` prefix from either side (the shape Claude Code,
        OpenCode, and Copilot CLI all surface today). Override in a
        subclass when a harness uses different rules.
        """
        if detected == target:
            return True
        if ":" in detected and detected.split(":", 1)[1] == target:
            return True
        return bool(":" in target and target.split(":", 1)[1] == detected)

    def get_global_skills(self) -> list[str]:
        """Return skill names loaded globally by this provider.

        Override when the provider has a global skill discovery path
        (e.g. ``~/.claude/skills/``, ``~/.codex/skills/``,
        ``~/.copilot/skills/``).  The
        default returns an empty list.
        """
        return []

    def invoke(
        self,
        prompt: str,
        model: str,
        cwd: Path,
        provider_options: dict,
        stop_on_first_trigger: bool,
        timeout_seconds: int,
    ) -> RunResult: ...

    def preflight(self, cfg: Config) -> list[CheckResult]:
        """Doctor's pre-run static checks. Anything cheap (no LLM call):
        binary/version, config-driven validation, local filesystem state.
        Return an empty list if the provider has nothing to check.
        """

    def probe_checks(self, probe_result: RunResult, cfg: Config) -> list[CheckResult]:
        """Doctor's post-probe checks, evaluated against the round-trip
        probe's RunResult (specifically its `raw_transcript_path`).
        Useful for leak detection, plugin-loading sanity, etc.
        Return an empty list if nothing applies.
        """

    def session_checks(self, run_result: RunResult, cfg: Config) -> list[CheckResult]:
        """Warnings the runner prints once, from the first finished attempt's
        transcript. For what an attempt reveals about the skills under test —
        e.g. whether the agent was shown enough of them to route on — as
        opposed to :meth:`probe_checks`, which is about the developer's setup.
        Providers with nothing to check inherit this.
        """
        return []

    def pre_run_warnings(self, cfg: Config) -> list[CheckResult]:
        """Warnings the runner prints at run start, before any trials.
        Used for things like "you have a conflicting plugin enabled
        globally" that the dev should see before burning tokens.
        Return an empty list if nothing applies.
        """

    def task_options(self, task_cfg: Any, framework_cfg: Any, task_kind: str) -> dict:
        """Map this provider's typed task config + framework provider
        config into the provider-specific additions to the
        `provider_options` dict that `invoke()` consumes.

        `task_cfg` is an instance of this provider's `task_config_model` (or
        None when the task YAML has no section for this provider).

        `task_kind` is the task's kind string (e.g. ``"execute"``,
        ``"trigger"``). Providers may adjust defaults accordingly. For example,
        providers should bypass permission checks for trigger tasks, so that
        trigger evals match real user conditions where our skills may compete
        with other skills and tools.

        Cross-cutting options (env_overrides, extra_args, target_skill,
        target_tool, negative_trigger) are pool.py's concern, not this
        method's.

        Default: no per-provider options.
        """
        return {}

    def stage_run_env(
        self,
        run_tmp_root: Path,
        cfg: Config,
        skills_to_exclude: frozenset[str] = frozenset(),
    ) -> None:
        """Stage the provider's skills into the attempt's runtime cwd.

        Called once per attempt so skills are discovered at the project root
        (cwd) rather than a parent directory — matching how real users set up
        project-specific skills. Copies each skill dir under ``skills_rel_path``
        via :py:func:`skill_staging.stage_skills_into`.
        """
        stage_skills_into(
            run_tmp_root,
            cfg.skills_dirs,
            self.skills_rel_path,
            exclude=skills_to_exclude,
        )

    def stage_mcp_config(
        self,
        run_tmp_root: Path,
        cfg: Config,
        servers: list[str] | None = None,
    ) -> dict:
        """Attach the configured MCP servers and return the provider options
        that do it — merged into the dict :meth:`invoke` consumes.

        *servers* names the subset of ``cfg.mcp_servers`` this task wants;
        ``None`` means all of them. Any file a harness needs is rendered
        under *run_tmp_root*, which is the parent of the attempt's cwd —
        never inside it, since the cwd is archived into the run's artifacts
        and a server block can hold a credential.

        The default attaches nothing, so a harness without MCP support needs
        no code; :py:func:`agent_exam.mcp.preflight` warns when a run
        configures servers such a harness will ignore.
        """
        return {}

    def validate_mcp_servers(self, servers: dict[str, McpServerConfig]) -> list[str]:
        """Provider-specific static checks on the MCP *servers* a run
        attaches, beyond what :py:func:`agent_exam.mcp.preflight` already
        checks generically. Return one problem description per server this
        harness cannot actually attach as configured, so it surfaces before
        the agent runs instead of when the first attempt stages it.

        Default: no extra constraints.
        """
        return []
