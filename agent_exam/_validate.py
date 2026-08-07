"""Small shared validation primitives.

These collapse the repetitive shape/type checks that recur across the
task loader (`tasks.py`) and the assertion validators (`assertions/*`).
Each raises `UsageError` with a uniform message so callers don't
hand-roll — and drift — their own.

Deliberately not a schema DSL: the genuinely per-field logic (cross-field
rules like `tool_count`'s min/max, shorthand forms, the str-or-dict judge
config) stays as plain Python in the caller. Only the boilerplate lives
here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .errors import UsageError

if TYPE_CHECKING:
    from collections.abc import Iterable


def reject_unknown_keys(
    items: Iterable[Any],
    allowed: Iterable[str],
    *,
    label: str,
    noun: str = "key",
) -> None:
    """Raise UsageError if `items` (a mapping — iterated as its keys — or
    any other iterable) contains an entry outside `allowed`. Catches
    typos that would otherwise be silently ignored.
    """
    allowed_set = set(allowed)
    unknown = sorted(str(k) for k in items if k not in allowed_set)
    if unknown:
        raise UsageError(
            f"{label}: unknown {noun}(s) {unknown}; "
            f"valid: {', '.join(sorted(allowed_set))}"
        )


def require_str(
    value: Any,
    label: str,
    *,
    allow_none: bool = False,
    non_empty: bool = False,
) -> None:
    """Raise UsageError unless `value` is a string — optionally non-empty
    (after strip), optionally None."""
    if value is None and allow_none:
        return
    if not isinstance(value, str) or (non_empty and not value.strip()):
        kind = "a non-empty string" if non_empty else "a string"
        raise UsageError(f"{label} must be {kind} (got {value!r})")


def require_str_list(
    value: Any,
    label: str,
    *,
    allow_none: bool = False,
    non_empty: bool = False,
) -> None:
    """Raise UsageError unless `value` is a list of non-empty strings —
    optionally itself non-empty, optionally None."""
    if value is None and allow_none:
        return
    if (
        not isinstance(value, list)
        or (non_empty and len(value) == 0)
        or not all(isinstance(x, str) and x for x in value)
    ):
        kind = "a non-empty list of strings" if non_empty else "a list of strings"
        raise UsageError(f"{label} must be {kind} (got {value!r})")


def require_int(
    value: Any,
    label: str,
    *,
    allow_none: bool = False,
    minimum: int | None = None,
) -> None:
    """Raise UsageError unless `value` is an integer — optionally >=
    `minimum`, optionally None. `bool` is rejected: it's an `int`
    subclass, but `timeout_seconds: true` is a typo, not a 1."""
    if value is None and allow_none:
        return
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or (minimum is not None and value < minimum)
    ):
        bound = f" >= {minimum}" if minimum is not None else ""
        raise UsageError(f"{label} must be an integer{bound} (got {value!r})")


def require_number(
    value: Any,
    label: str,
    *,
    allow_none: bool = False,
    positive: bool = False,
) -> None:
    """Raise UsageError unless `value` is a number (int or float —
    `bool` rejected), optionally strictly positive, optionally None.
    Use this for fields where sub-integer precision is meaningful
    (e.g. `timeout_seconds: 0.5`); use `require_int` for counts."""
    if value is None and allow_none:
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise UsageError(f"{label} must be a number (got {value!r})")
    if positive and value <= 0:
        raise UsageError(f"{label} must be a positive number (got {value!r})")
