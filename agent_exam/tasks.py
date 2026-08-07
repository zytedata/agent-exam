from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

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
    stop_on_first_skill: bool = False
    target_skill: str | None = None
    should_trigger: bool | None = None
    # Provider-specific task-config sections. Keyed by provider name;
    # value is a typed pydantic model — each provider's `task_config_model`
    # (e.g. `ClaudeCodeTaskConfig`). pool.py looks up the current-run
    # provider's section when assembling invocation options.
    provider_configs: dict[str, Any] = field(default_factory=dict)
    # Task-level known_issue: the whole task is expected to fail.
    known_issue: str | None = None


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
    skill: str
    positive: list[str] = Field(default_factory=list)
    negative: list[str] = Field(default_factory=list)

    @field_validator("skill")
    @classmethod
    def _non_empty_skill(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("trigger task must declare 'skill: <name>'")
        return v

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


def load_task(path: Path, suite: str) -> list[Task]:
    """Load a task YAML, validating it against the file schema, and
    return one or more runtime `Task` dataclasses (triggers fan out)."""
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
        return [_task_from_execute(model, path, suite, raw)]
    return _tasks_from_trigger(model, path, suite, raw)


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
    )


def _tasks_from_trigger(
    m: _TriggerTaskModel, path: Path, suite: str, raw: dict
) -> list[Task]:
    from .assertions.registry import parse_assertion_config

    provider_configs = m._provider_configs()

    def _emit(prompt: str, should_trigger: bool, idx: int) -> Task:
        synth_type = "first_skill" if should_trigger else "skill_not_invoked"
        # Build the typed config the YAML-load path produces, so `_eval`
        # doesn't need a separate code path for trigger-synthesized
        # assertions.
        synthesized = [
            Assertion(
                type=synth_type,
                config=m.skill,
                parsed_config=parse_assertion_config(synth_type, m.skill),
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
            stop_on_first_skill=True,
            target_skill=m.skill,
            should_trigger=should_trigger,
            provider_configs=provider_configs,
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
    out: list[Task] = []
    for f in files:
        out.extend(load_task(f, suite))
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
