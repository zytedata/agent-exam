from __future__ import annotations

import re
from typing import Literal

Verdict = Literal["YES", "NO", "UNCLEAR"]

_VERDICT_RE = re.compile(r"VERDICT\s*:\s*(YES|NO|UNCLEAR)\b", re.IGNORECASE)


def parse_verdict(response: str) -> tuple[Verdict, str]:
    """Parse a judge response into (verdict, reasoning).

    - Looks for `VERDICT:` on the last non-empty line.
    - Everything before that line is reasoning (whitespace-trimmed).
    - If no `VERDICT:` line found, returns UNCLEAR with the raw response as
      the reasoning (for debugging).
    """
    if not response or not response.strip():
        return "UNCLEAR", response or ""

    lines = response.splitlines()
    last_idx: int | None = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip():
            last_idx = i
            break
    if last_idx is None:
        return "UNCLEAR", response.strip()

    match = _VERDICT_RE.search(lines[last_idx])
    if not match:
        return "UNCLEAR", response.strip()

    verdict: Verdict = match.group(1).upper()  # type: ignore[assignment]
    reasoning = "\n".join(lines[:last_idx]).strip()
    return verdict, reasoning
