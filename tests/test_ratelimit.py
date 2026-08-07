from __future__ import annotations

import pytest

from agent_exam.errors import RateLimitError, RateLimitExhausted
from agent_exam.ratelimit import next_delay, with_retries


def test_exponential_schedule():
    assert next_delay(0, None) == 1.0
    assert next_delay(1, None) == 2.0
    assert next_delay(2, None) == 4.0
    assert next_delay(3, None) == 8.0
    assert next_delay(4, None) == 16.0
    assert next_delay(5, None) == 32.0


def test_cap_at_60_for_high_attempts():
    # Beyond the schedule the cap kicks in.
    assert next_delay(10, None) == 32.0  # schedule plateaus at 32
    # Explicit retry-after above cap is clamped.
    assert next_delay(0, 9999.0) == 60.0


def test_retry_after_wins_over_schedule():
    assert next_delay(3, 5.0) == 5.0
    assert next_delay(0, 0.5) == 0.5


def test_retry_after_zero_falls_back_to_schedule():
    assert next_delay(1, 0) == 2.0
    assert next_delay(1, None) == 2.0


def test_with_retries_returns_on_success():
    calls = []

    def fn():
        calls.append(1)
        return "ok"

    assert with_retries(fn, sleep=lambda _: None) == "ok"
    assert calls == [1]


def test_with_retries_recovers_after_transient():
    attempts = {"n": 0}

    def fn():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RateLimitError("transient", retry_after=0.1)
        return "done"

    slept: list[float] = []
    result = with_retries(fn, sleep=slept.append)
    assert result == "done"
    assert attempts["n"] == 3
    assert slept == [0.1, 0.1]  # two sleeps before the successful 3rd attempt


def test_with_retries_exhausted_raises():
    def fn():
        raise RateLimitError("always", retry_after=0.01)

    with pytest.raises(RateLimitExhausted):
        with_retries(fn, sleep=lambda _: None, max_retries=2)


def test_on_retry_callback_invoked():
    events: list[tuple[int, float]] = []

    def fn():
        raise RateLimitError("x", retry_after=0.5)

    with pytest.raises(RateLimitExhausted):
        with_retries(
            fn,
            sleep=lambda _: None,
            max_retries=2,
            on_retry=lambda attempt, delay, exc: events.append((attempt, delay)),
        )
    assert len(events) == 2
    assert all(delay == 0.5 for _, delay in events)
    assert [a for a, _ in events] == [0, 1]
