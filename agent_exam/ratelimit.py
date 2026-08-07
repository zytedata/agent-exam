"""Rate-limit retry policy for provider calls.

Exponential backoff 1→2→4→8→16→32, cap 60, max 5 retries per call. Honors
an explicit `Retry-After` value (in seconds) when the harness surfaces one;
otherwise falls back to the exponential schedule.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, TypeVar

from .errors import RateLimitError, RateLimitExhausted

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")

_MAX_RETRIES = 5
_CAP_SECONDS = 60.0
_SCHEDULE = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]


def next_delay(attempt: int, retry_after: float | None) -> float:
    """Return the sleep delay before retry `attempt` (0-indexed).

    Prefer an explicit `Retry-After` if provided. Otherwise use the
    exponential schedule, capped at 60s.
    """
    if retry_after is not None and retry_after > 0:
        return min(retry_after, _CAP_SECONDS)
    idx = min(attempt, len(_SCHEDULE) - 1)
    return min(_SCHEDULE[idx], _CAP_SECONDS)


def with_retries(
    fn: Callable[[], T],
    *,
    max_retries: int = _MAX_RETRIES,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Callable[[int, float, RateLimitError], None] | None = None,
) -> T:
    """Run `fn`, retrying on RateLimitError up to `max_retries` times.

    After max_retries consecutive failures, raises RateLimitExhausted with
    the original error attached.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except RateLimitError as exc:
            if attempt >= max_retries:
                raise RateLimitExhausted(
                    f"rate-limited after {max_retries} retries: {exc}"
                ) from exc
            delay = next_delay(attempt, exc.retry_after)
            if on_retry is not None:
                on_retry(attempt, delay, exc)
            sleep(delay)
            attempt += 1
