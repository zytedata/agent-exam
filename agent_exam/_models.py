"""Pydantic foundations for assertion config models.

Each assertion declares a `<Foo>Config` pydantic model; `_parse_assertion`
constructs the model from raw YAML and surfaces a clean UsageError if
the config is malformed. `check` then receives the typed config directly
— no preamble, no re-extraction, types flow.

Two shared base classes cover what most assertions need:

- `_StrictModel`: rejects unknown keys (`extra="forbid"`) — the anti-typo
  default for every config in the codebase.
- `_ScalarShorthandModel`: enables `<assertion>: <scalar>` shorthand
  alongside the full `<assertion>: {<key>: <scalar>}` form. Subclasses
  set `_shorthand_key: ClassVar[str]` to declare which field the scalar
  populates.

`render_validation_error` turns a pydantic `ValidationError` into a flat
CLI message — one line per error, no `pydantic.dev` URL, with the
"Value error," wrapping that pydantic adds to validator-raised
`ValueError`s stripped. Used by the registry to wrap pydantic errors as
`UsageError` so they read like every other config error in the system.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, ClassVar

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

# Strips pydantic's `... or instance of <ModelClass>` suffix from
# type-mismatch messages — internal class names mean nothing to a YAML
# author and just make errors look noisy.
_MODEL_CLASS_SUFFIX = re.compile(r" or instance of _?[A-Z]\w+")


class _StrictModel(BaseModel):
    """Reject unknown keys — every config model in the codebase wants this."""

    model_config = ConfigDict(extra="forbid")


class _ScalarShorthandModel(_StrictModel):
    """Enable the `<assertion>: <scalar>` shorthand form.

    Subclasses set `_shorthand_key: ClassVar[str]` to declare which
    field the scalar value populates. Any non-dict input is treated as
    the shorthand value — `model_validate(x)` becomes
    `model_validate({<key>: x})`. The full mapping form is unchanged.

    The wrap is type-agnostic: if `x` is the wrong type for the field,
    pydantic surfaces a per-field message (e.g. `path: Input should be a
    valid string`) instead of the misleading top-level
    `Input should be a valid dictionary`.
    """

    _shorthand_key: ClassVar[str]

    @model_validator(mode="before")
    @classmethod
    def _apply_shorthand(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return v
        return {cls._shorthand_key: v}


def _reject_bool_and_str(v: Any) -> Any:
    if isinstance(v, bool):
        raise ValueError("must be a number, not a bool")
    if isinstance(v, str):
        raise ValueError("must be a number, not a string")
    return v


# A numeric field that accepts int OR float and rejects bool (otherwise
# an int subclass) and strings (otherwise lax-coerced). Single-error
# message — avoids pydantic's two-branch verbose `StrictInt | StrictFloat`
# union error.
Number = Annotated[int | float, BeforeValidator(_reject_bool_and_str)]

# Number plus a positivity constraint (`> 0`). Type-mismatch errors come
# from the BeforeValidator; sign errors are pydantic's
# "Input should be greater than 0".
PositiveNumber = Annotated[
    int | float, BeforeValidator(_reject_bool_and_str), Field(gt=0)
]


def render_validation_error(label: str, exc: ValidationError) -> str:
    """Convert a pydantic `ValidationError` into a flat CLI message.

    Single error: `<label>: [<loc>: ]<msg>` on one line.
    Multiple errors: `<label>: invalid config (N errors)` header, then
    `<loc>: <msg>` per error.

    `include_url=False` drops pydantic's "see docs at ..." appendage,
    and the "Value error, " prefix pydantic adds around validator-raised
    `ValueError` messages is stripped — both are noise for CLI use.
    """
    errors = exc.errors(include_url=False)
    if len(errors) == 1:
        err = errors[0]
        loc = ".".join(str(x) for x in err["loc"])
        msg = _MODEL_CLASS_SUFFIX.sub("", err["msg"].removeprefix("Value error, "))
        if loc:
            return f"{label}: {loc}: {msg}"
        return f"{label}: {msg}"
    lines = [f"{label}: invalid config ({len(errors)} errors)"]
    for err in errors:
        loc = ".".join(str(x) for x in err["loc"]) or "<root>"
        msg = _MODEL_CLASS_SUFFIX.sub("", err["msg"].removeprefix("Value error, "))
        lines.append(f"  {loc}: {msg}")
    return "\n".join(lines)
