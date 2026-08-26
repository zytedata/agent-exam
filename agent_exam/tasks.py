from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import yaml
from pydantic import (
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from ._models import (
    PositiveNumber,
    _ScalarShorthandModel,  # noqa: F401 — re-exported for tests if needed
    _StrictModel,
    render_validation_error,
)
from ._validate import (
    reject_unknown_keys,
    require_str,
    require_str_list,
)
from .errors import UsageError
from .providers.claude_code.provider import ClaudeCodeTaskConfig
from .providers.codex_cli.provider import CodexCliTaskConfig
from .providers.copilot_cli.provider import CopilotCliTaskConfig
from .providers.dummy import DummyTaskConfig
from .providers.opencode.provider import OpenCodeTaskConfig

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

# Recognized provider/harness sections in a task YAML. Each provider
# owns the schema for its own section via `Provider.task_config_model`.
KNOWN_PROVIDER_NAMES = frozenset(
    {"claude_code", "dummy", "opencode", "copilot_cli", "codex_cli"}
)

# Meta-keys allowed alongside an assertion type key in a YAML list item.
# (Assertion entries aren't a fixed-shape pydantic model — the type-as-key
# pattern doesn't fit a single schema — so we keep this parser procedural.)
_ASSERTION_META_KEYS = frozenset({"known_issue", "providers"})


# --- Runtime dataclasses (consumed by pool / runner / scoring) -------------


@dataclass
class Assertion:
    type: str
    # Raw config as loaded from YAML — kept for report serialization
    # (the on-disk reports include `config` verbatim, so it has to be a
    # JSON-friendly value, not a pydantic model).
    config: Any
    # Parsed pydantic model. `check` receives this; `_parse_assertion`
    # populates it. None only when an Assertion is constructed directly
    # (in tests) without going through `_parse_assertion` — `_eval`
    # falls back to `config` in that case.
    parsed_config: Any | None = None
    # Non-restrictive "known issue" marker. When set, the assertion still
    # runs and its pass/fail is reported, but it's EXCLUDED from the
    # task's aggregate verdict — lets you land a failing check without
    # breaking the suite.
    known_issue: str | None = None
    # Provider allowlist. When set, the assertion is evaluated only if
    # the current run's provider is in the list; otherwise it's skipped
    # (no LLM call for judges, no check invocation).
    providers: list[str] | None = None


_FIXTURE_EMPTY_DIR_MARKERS = (".gitkeep",)
"""File names that exist only so git can store an otherwise empty fixture
directory. Stripped when a fixture is staged: the directory is what the fixture
author meant the agent to see, the marker is bookkeeping."""


@dataclass
class Task:
    suite: str
    name: str
    kind: str
    prompt: str
    description: str | None
    assertions: list[Assertion]
    fixture: str | None
    env: dict[str, str | None]
    timeout_seconds: int | float | None
    concurrency_group: str | None
    raw: dict
    source_path: Path
    # Trigger-kind fields. Default values keep execute tasks unchanged.
    stop_on_first_trigger: bool = False
    target_skill: str | None = None
    # Set instead of `target_skill` when the trigger is about a tool — an
    # MCP one, typically — rather than a skill.
    target_tool: str | None = None
    should_trigger: bool | None = None
    # Provider-specific task-config sections. Keyed by provider name;
    # value is a typed pydantic model — each provider's `task_config_model`
    # (e.g. `ClaudeCodeTaskConfig`). pool.py looks up the current-run
    # provider's section when assembling invocation options.
    provider_configs: dict[str, Any] = field(default_factory=dict)
    # Task-level known_issue: the whole task is expected to fail.
    known_issue: str | None = None
    # Sorted union of the task's own tags and its suite's. Every name has to
    # be declared under `tags:` in config.yaml; a tag declared
    # `exclude_by_default` keeps the task out of a wide run — see
    # `select_by_tags`.
    tags: list[str] = field(default_factory=list)
    # Which of the servers declared under `mcp_servers:` in config.yaml to
    # attach. `None` attaches all of them, the way `skills_dirs` stages
    # every skill; an empty list attaches none.
    mcp_servers: list[str] | None = None


# --- Pydantic task-file schema --------------------------------------------
#
# One model per file (with a discriminated union on `kind`). `load_task`
# validates the whole file in one pass; the runtime `Task` dataclass is
# built from the validated model.


class _Setup(_StrictModel):
    fixture: str | None = None
    env: dict[str, str | None] = Field(default_factory=dict)

    @field_validator("env", mode="before")
    @classmethod
    def _coerce_env_values(cls, v: Any) -> Any:
        # YAML may load values as int/float/bool (e.g. `LEVEL: 3`). The
        # subprocess env always wants strings; preserve None for the
        # "deliberately unset" case.
        if not isinstance(v, dict):
            return v
        return {
            str(k): (val if val is None or isinstance(val, str) else str(val))
            for k, val in v.items()
        }


class _TaskCommonModel(_StrictModel):
    """Fields shared by both task kinds."""

    description: str | None = None
    setup: _Setup = Field(default_factory=_Setup)
    timeout_seconds: PositiveNumber | None = None
    concurrency_group: str | None = None
    tags: list[str] = Field(default_factory=list)
    mcp_servers: list[str] | None = None
    # Provider task-config sections — each provider owns its own schema
    # (`<Provider>.task_config_model`). Pydantic validates each present
    # section directly when constructing the Task model.
    claude_code: ClaudeCodeTaskConfig | None = None
    codex_cli: CodexCliTaskConfig | None = None
    opencode: OpenCodeTaskConfig | None = None
    copilot_cli: CopilotCliTaskConfig | None = None
    dummy: DummyTaskConfig | None = None

    def _provider_configs(self) -> dict[str, Any]:
        """Build the {name: typed-model} dict the runtime expects,
        skipping absent providers."""
        out: dict[str, Any] = {}
        for name in KNOWN_PROVIDER_NAMES:
            section = getattr(self, name, None)
            if section is not None:
                out[name] = section
        return out


class _ExecuteTaskModel(_TaskCommonModel):
    kind: Literal["execute"] = "execute"
    prompt: str
    # Parsed in a `mode="before"` validator (assertion entries don't fit
    # a fixed pydantic schema — the type-as-key shape varies). Stored as
    # already-built `Assertion` dataclasses; `list[Any]` keeps pydantic
    # from second-guessing them.
    assertions: list[Any] = Field(default_factory=list)
    known_issue: str | None = None

    @field_validator("prompt")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must be a non-empty string")
        return v

    @field_validator("assertions", mode="before")
    @classmethod
    def _parse_entries(cls, v: Any) -> Any:
        if not isinstance(v, list):
            raise ValueError("must be a list")
        out: list[Assertion] = []
        for i, entry in enumerate(v):
            try:
                out.append(_parse_assertion(entry))
            except UsageError as exc:
                raise ValueError(f"[{i}] {exc}") from exc
        return out


class _TriggerTaskModel(_TaskCommonModel):
    kind: Literal["trigger"] = "trigger"
    skill: str | None = None
    tool: str | None = None
    positive: list[str] = Field(default_factory=list)
    negative: list[str] = Field(default_factory=list)

    @field_validator("skill", "tool")
    @classmethod
    def _non_empty_target(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("must be a non-empty string")
        return v

    @model_validator(mode="after")
    def _exactly_one_target(self) -> _TriggerTaskModel:
        if bool(self.skill) == bool(self.tool):
            raise ValueError(
                "trigger task must declare exactly one of 'skill: <name>' "
                "or 'tool: <name>'"
            )
        return self

    @field_validator("positive", "negative", mode="before")
    @classmethod
    def _list_of_nonempty_strs(cls, v: Any) -> Any:
        if not isinstance(v, list):
            raise ValueError("must be a list")
        for prompt in v:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"each case must be a non-empty string (got {prompt!r})"
                )
        return v

    @model_validator(mode="after")
    def _at_least_one_case(self) -> _TriggerTaskModel:
        if not self.positive and not self.negative:
            raise ValueError("need at least one of 'positive' or 'negative'")
        return self


# Discriminated union of the two kinds. Pydantic dispatches on `kind`
# and the kind-specific allowlist (extra="forbid") falls out for free.
_TaskFile = Annotated[
    _ExecuteTaskModel | _TriggerTaskModel,
    Field(discriminator="kind"),
]
_TaskFileAdapter: TypeAdapter[Any] = TypeAdapter(_TaskFile)


class SuiteConfig(_StrictModel):
    """`suite.yml` schema. Optional file, defaults to all-None."""

    # Skills this suite evaluates. When `--without-skill` is used, these
    # skills are excluded from the bundle. Defaults to the suite name.
    evaluated_skills: list[str] | None = None
    # Tags every task in the suite wears, on top of its own.
    tags: list[str] = Field(default_factory=list)


# --- Assertion entry parser (procedural — see comment on _ASSERTION_META_KEYS)


def _parse_assertion(entry: Any) -> Assertion:
    if not isinstance(entry, dict):
        raise UsageError(f"each assertion must be a mapping (got {entry!r})")
    type_pairs = [(k, v) for k, v in entry.items() if k not in _ASSERTION_META_KEYS]
    if len(type_pairs) != 1:
        raise UsageError(
            f"assertion must have exactly one assertion-type key "
            f"(got {entry!r}); meta keys allowed: "
            f"{sorted(_ASSERTION_META_KEYS)}"
        )
    name, cfg = type_pairs[0]
    from .assertions.registry import parse_assertion_config

    parsed_config = parse_assertion_config(name, cfg)
    known_issue = entry.get("known_issue")
    require_str(known_issue, "assertion known_issue", allow_none=True)
    providers = entry.get("providers")
    if providers is not None:
        require_str_list(providers, "assertion providers", non_empty=True)
        reject_unknown_keys(
            providers,
            KNOWN_PROVIDER_NAMES,
            label="assertion providers",
            noun="harness name",
        )
    return Assertion(
        type=name,
        config=cfg,
        parsed_config=parsed_config,
        known_issue=known_issue,
        providers=providers,
    )


# --- File loading ---------------------------------------------------------


def load_task(path: Path, suite: str, suite_tags: Iterable[str] = ()) -> list[Task]:
    """Load a task YAML, validating it against the file schema, and
    return one or more runtime `Task` dataclasses (triggers fan out).

    *suite_tags* are added to whatever tags the file declares.
    """
    with path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise UsageError(f"{path}: task file must be a YAML mapping")
    # The discriminated union dispatches on `kind`; default to execute
    # for the common bare task.
    if "kind" not in raw:
        raw = {"kind": "execute", **raw}

    try:
        model = _TaskFileAdapter.validate_python(raw)
    except ValidationError as exc:
        raise UsageError(render_validation_error(str(path), exc)) from exc

    if isinstance(model, _ExecuteTaskModel):
        tasks = [_task_from_execute(model, path, suite, raw)]
    else:
        tasks = _tasks_from_trigger(model, path, suite, raw)
    tags = sorted({*model.tags, *suite_tags})
    for task in tasks:
        task.tags = tags
    return tasks


def _task_from_execute(m: _ExecuteTaskModel, path: Path, suite: str, raw: dict) -> Task:
    return Task(
        suite=suite,
        name=path.stem,
        kind="execute",
        prompt=m.prompt,
        description=m.description,
        assertions=m.assertions,
        fixture=m.setup.fixture,
        env=dict(m.setup.env),
        timeout_seconds=m.timeout_seconds,
        concurrency_group=m.concurrency_group,
        raw=raw,
        source_path=path,
        provider_configs=m._provider_configs(),
        known_issue=m.known_issue,
        mcp_servers=m.mcp_servers,
    )


def _tasks_from_trigger(
    m: _TriggerTaskModel, path: Path, suite: str, raw: dict
) -> list[Task]:
    from .assertions.registry import parse_assertion_config

    provider_configs = m._provider_configs()

    if m.tool:
        target = m.tool
        synth_types = ("first_tool", "tool_not_called")
    else:
        target = m.skill
        synth_types = ("first_skill", "skill_not_invoked")

    def _emit(prompt: str, should_trigger: bool, idx: int) -> Task:
        synth_type = synth_types[0] if should_trigger else synth_types[1]
        # Build the typed config the YAML-load path produces, so `_eval`
        # doesn't need a separate code path for trigger-synthesized
        # assertions.
        synthesized = [
            Assertion(
                type=synth_type,
                config=target,
                parsed_config=parse_assertion_config(synth_type, target),
            )
        ]
        return Task(
            suite=suite,
            name=f"{path.stem}-{idx}",
            kind="trigger",
            prompt=prompt,
            description=(
                f"trigger case {idx}: should_trigger={should_trigger} "
                f"(from {path.stem}.yaml)"
            ),
            assertions=synthesized,
            fixture=m.setup.fixture,
            env=dict(m.setup.env),
            timeout_seconds=m.timeout_seconds,
            concurrency_group=m.concurrency_group,
            raw=raw,
            source_path=path,
            stop_on_first_trigger=True,
            target_skill=m.skill,
            target_tool=m.tool,
            should_trigger=should_trigger,
            provider_configs=provider_configs,
            mcp_servers=m.mcp_servers,
        )

    results: list[Task] = []
    idx = 0
    for prompt in m.positive:
        results.append(_emit(prompt, True, idx))
        idx += 1
    for prompt in m.negative:
        results.append(_emit(prompt, False, idx))
        idx += 1
    return results


def list_suites(evals_dir: Path) -> list[str]:
    suites_dir = evals_dir / "suites"
    if not suites_dir.is_dir():
        return []
    return sorted(
        p.name
        for p in suites_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


def list_tasks(evals_dir: Path, suite: str) -> list[str]:
    tasks_dir = evals_dir / "suites" / suite / "tasks"
    if not tasks_dir.is_dir():
        return []
    return sorted(p.stem for p in tasks_dir.glob("*.yaml"))


def load_suite_config(evals_dir: Path, suite: str) -> SuiteConfig:
    """Load optional suite.yml config for the suite.

    Returns a `SuiteConfig` with defaults if no suite.yml exists.
    """
    suites = list_suites(evals_dir)
    if suite not in suites:
        available = ", ".join(suites) or "(none)"
        raise UsageError(f"no suite {suite!r}; available: {available}")
    config_path = evals_dir / "suites" / suite / "suite.yml"
    if not config_path.exists():
        return SuiteConfig()
    with config_path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise UsageError(f"{config_path}: suite config must be a YAML mapping")
    try:
        return SuiteConfig.model_validate(raw)
    except ValidationError as exc:
        raise UsageError(render_validation_error(str(config_path), exc)) from exc


def load_suite(
    evals_dir: Path, suite: str, task_filter: str | None = None
) -> list[Task]:
    suites = list_suites(evals_dir)
    if suite not in suites:
        available = ", ".join(suites) or "(none)"
        raise UsageError(f"no suite {suite!r}; available: {available}")
    tasks_dir = evals_dir / "suites" / suite / "tasks"
    files = sorted(tasks_dir.glob("*.yaml"))
    # A single fanned-out trigger case ("<stem>-<n>", from "<suite>::<task>::<n>")
    # has no file of its own; match its parent file and keep only that case.
    case_name: str | None = None
    if task_filter is not None:
        matched = [f for f in files if f.stem == task_filter]
        if not matched:
            stem, _, idx = task_filter.rpartition("-")
            if idx.isdigit():
                parent = [f for f in files if f.stem == stem]
                if parent:
                    matched, case_name = parent, task_filter
        if not matched:
            available = ", ".join(list_tasks(evals_dir, suite)) or "(none)"
            raise UsageError(
                f"no task {task_filter!r} in suite {suite!r}. Available: {available}"
            )
        files = matched
    suite_tags = load_suite_config(evals_dir, suite).tags
    out: list[Task] = []
    for f in files:
        out.extend(load_task(f, suite, suite_tags))
    if case_name is not None:
        selected = [t for t in out if t.name == case_name]
        if not selected:
            n = len(out)
            raise UsageError(
                f"no trigger case {case_name!r} in suite {suite!r}: "
                f"{files[0].stem!r} has {n} case(s) (indices 0..{n - 1})"
            )
        out = selected
    return out


def expand_specs(
    evals_dir: Path, specs: list[tuple[str, str | None]]
) -> list[tuple[str, str | None]]:
    """Expand wildcard suite specs into concrete (suite, task_filter) pairs.

    A suite value of ``"*"`` is replaced by every available suite that
    contains at least one task matching *task_filter* (or any task when
    *task_filter* is ``None``).  Concrete suite names pass through unchanged.
    Raises ``UsageError`` when a wildcard produces no matches.
    """
    out: list[tuple[str, str | None]] = []
    for suite, task_filter in specs:
        if suite == "*":
            matched = [
                s
                for s in list_suites(evals_dir)
                if task_filter is None or task_filter in list_tasks(evals_dir, s)
            ]
            if not matched:
                hint = (
                    f"no suite has a task {task_filter!r}"
                    if task_filter
                    else "no suites found"
                )
                raise UsageError(f"wildcard spec '*' matched nothing: {hint}")
            out.extend((s, task_filter) for s in matched)
        else:
            out.append((suite, task_filter))
    return out


def select_by_tags(
    tasks: list[Task],
    specs: list[tuple[str, str | None]],
    *,
    default_excluded: Iterable[str],
    suite_tags: Mapping[str, Iterable[str]] | None = None,
    include: Iterable[str] = (),
    exclude: Iterable[str] = (),
    all_tags: bool = False,
) -> tuple[list[Task], dict[str, int]]:
    """Drop the tasks a run's tags exclude, and report what went.

    Returns the surviving tasks plus, per tag, how many tasks it dropped.

    *exclude* (``--exclude-tag``) always applies. *default_excluded* — the
    tags configured ``exclude_by_default`` — is lifted per tag by *include*
    (``--tag``), wholesale by *all_tags*, and for the tasks a spec names.
    A run targeting a single suite also lifts the tags that suite declares in
    its :file:`suite.yml`, taken from *suite_tags*: asking for one suite by
    name asks for what that suite is, while a tag on an individual task still
    keeps that task out.
    """
    by_default = set() if all_tags else set(default_excluded) - set(include)
    suites = {suite for suite, _ in specs}
    if len(suites) == 1:
        by_default -= set((suite_tags or {}).get(next(iter(suites)), ()))
    named = {(suite, task) for suite, task in specs if task is not None}
    # A trigger file fans out into cases named "<stem>-<n>", so a spec that
    # names the file exempts every case it produced; `<suite>::<task>::<n>`
    # arrives as a filter matching one case name.
    exempt = {
        (t.suite, t.name)
        for t in tasks
        if any(
            t.suite == suite and task in (t.source_path.stem, t.name)
            for suite, task in named
        )
    }
    forced = set(exclude)
    dropped: dict[str, int] = {}
    kept: list[Task] = []
    for t in tasks:
        hit = sorted(set(t.tags) & forced)
        if not hit and (t.suite, t.name) not in exempt:
            hit = sorted(set(t.tags) & by_default)
        if hit:
            # One task counts once, under its first tag, so the counts add
            # up to the number of tasks dropped.
            dropped[hit[0]] = dropped.get(hit[0], 0) + 1
        else:
            kept.append(t)
    return kept, dropped


def load_specs(evals_dir: Path, specs: list[tuple[str, str | None]]) -> list[Task]:
    """Load tasks from multiple (suite, task_filter) pairs, deduplicating by suite::name."""
    seen: set[tuple[str, str]] = set()
    out: list[Task] = []
    for suite, task_filter in specs:
        for task in load_suite(evals_dir, suite, task_filter=task_filter):
            key = (task.suite, task.name)
            if key not in seen:
                seen.add(key)
                out.append(task)
    return out
