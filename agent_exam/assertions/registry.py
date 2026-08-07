from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from .._models import render_validation_error
from ..errors import UsageError
from ..schemas import AssertionResult, RunResult
from . import (
    file_contains,
    file_exists,
    first_skill,
    judge,
    judge_agent,
    no_permission_errors,
    skill_invoked,
    skill_not_invoked,
    tool_called,
    tool_count,
    tool_not_called,
)

if TYPE_CHECKING:
    from pathlib import Path

CheckFn = Callable[..., AssertionResult]


_REGISTRY: dict[str, CheckFn] = {
    "file_exists": file_exists.check,
    "file_contains": file_contains.check,
    "tool_called": tool_called.check,
    "tool_not_called": tool_not_called.check,
    "tool_count": tool_count.check,
    "first_skill": first_skill.check,
    "judge": judge.check,
    "judge_agent": judge_agent.check,
    "no_permission_errors": no_permission_errors.check,
    "skill_invoked": skill_invoked.check,
    "skill_not_invoked": skill_not_invoked.check,
}

# Each assertion's config is validated by constructing its pydantic
# model. The typed instance is what `check` receives — no preamble, no
# re-extraction.
_MODELS: dict[str, type[BaseModel]] = {
    "file_exists": file_exists.FileExistsConfig,
    "file_contains": file_contains.FileContainsConfig,
    "tool_called": tool_called.ToolCalledConfig,
    "tool_not_called": tool_not_called.ToolNotCalledConfig,
    "tool_count": tool_count.ToolCountConfig,
    "first_skill": first_skill.FirstSkillConfig,
    "judge": judge.JudgeConfig,
    "judge_agent": judge_agent.JudgeAgentConfig,
    "no_permission_errors": no_permission_errors.NoPermissionErrorsConfig,
    "skill_invoked": skill_invoked.SkillInvokedConfig,
    "skill_not_invoked": skill_not_invoked.SkillNotInvokedConfig,
}

# Drift guard: every assertion has a check AND a model. A mismatch is a
# programming error caught at import.
assert set(_REGISTRY) == set(_MODELS), (
    f"assertion registry drift: checks={sorted(_REGISTRY)} models={sorted(_MODELS)}"
)


def get_check(type_name: str) -> CheckFn:
    if type_name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise UsageError(
            f"unknown assertion type {type_name!r}; available: {available}"
        )
    return _REGISTRY[type_name]


def parse_assertion_config(type_name: str, config: Any) -> BaseModel:
    """Validate `config` against `type_name`'s pydantic model and return
    the parsed instance. Raises `UsageError` on a malformed config or an
    unknown type.
    """
    if type_name not in _MODELS:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise UsageError(
            f"unknown assertion type {type_name!r}; available: {available}"
        )
    model_cls = _MODELS[type_name]
    try:
        return model_cls.model_validate(config)
    except ValidationError as exc:
        raise UsageError(render_validation_error(type_name, exc)) from exc


def validate_config(type_name: str, config: Any) -> None:
    """Validate an assertion's config; raise `UsageError` if malformed.

    Same as `parse_assertion_config` but discards the result — for
    callers that only need pass/fail (tests, doctor's suite checks).
    """
    parse_assertion_config(type_name, config)


def register(type_name: str, check_fn: CheckFn, model: type[BaseModel]) -> None:
    """Register a new assertion type (check function + pydantic config
    model)."""
    _REGISTRY[type_name] = check_fn
    _MODELS[type_name] = model


def known_types() -> list[str]:
    return sorted(_REGISTRY)


def call_check(
    fn: CheckFn,
    config: Any,
    result: RunResult,
    cwd: Path,
    context: Any = None,
) -> AssertionResult:
    """Call a check function, passing `context` only if it accepts it.

    Lets most assertions keep the `(config, result, cwd)` shape while the
    judge assertion opts into the scoring context.
    """
    sig = inspect.signature(fn)
    if "context" in sig.parameters:
        return fn(config, result, cwd, context=context)
    return fn(config, result, cwd)
