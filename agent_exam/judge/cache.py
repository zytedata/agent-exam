from __future__ import annotations

import hashlib
import json
import threading
from typing import TYPE_CHECKING

from ..schemas import TextBlock, Turn

if TYPE_CHECKING:
    from pathlib import Path


def _sha16(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def agent_output_hash(trajectory: list[Turn]) -> str:
    """Stable 16-hex-char hash of the last assistant turn's text content.

    Defined independently from the cache key so rescore (step 7) can reuse
    the same hash without re-importing the whole cache module.
    """
    for turn in reversed(trajectory):
        if turn.role != "assistant":
            continue
        parts = [b.text for b in turn.content if isinstance(b, TextBlock)]
        text = "\n".join(p for p in parts if p is not None)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return hashlib.sha256(b"").hexdigest()[:16]


def cwd_hash(cwd: Path) -> str:
    """Stable 16-hex-char hash of a directory's content manifest.

    Walks ``cwd`` in sorted order and folds (relative path, file
    content) pairs into a SHA-256. Empty / missing directories hash to
    the empty input. Symlinks are skipped (their targets may sit
    outside the archive and aren't part of the attempt's recorded
    state). Stable across machines so the same archived attempt
    rescored elsewhere produces the same hash.
    """
    h = hashlib.sha256()
    if cwd.is_dir():
        for path in sorted(cwd.rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            rel = path.relative_to(cwd).as_posix()
            h.update(rel.encode("utf-8"))
            h.update(b"\x00")
            try:
                h.update(hashlib.sha256(path.read_bytes()).digest())
            except OSError:
                # Unreadable file → record the path with an empty
                # content hash so the manifest is still stable.
                h.update(hashlib.sha256(b"").digest())
            h.update(b"\x00")
    return h.hexdigest()[:16]


def tools_signature(tools: tuple[str, ...] | list[str]) -> str:
    """Stable 8-hex-char hash of a tool list. Order-insensitive."""
    return hashlib.sha256(",".join(sorted(tools)).encode("utf-8")).hexdigest()[:8]


def key_for(criterion: str, output_hash: str, judge_model: str) -> str:
    """Cache key. Changing criterion text, output, or judge model all invalidate."""
    return _sha16(criterion, output_hash, judge_model)


def key_for_judge_agent(
    criterion: str,
    output_hash: str,
    cwd_hash_: str,
    tools_sig: str,
    judge_model: str,
) -> str:
    """Cache key for the agentic judge.

    Distinct from :func:`key_for` because the agentic judge's verdict
    depends on the cwd it can read (``cwd_hash_``) and the toolset it
    can call (``tools_sig``) in addition to the inputs the plain judge
    considers. An ``assertion_type`` prefix namespaces the key so
    ``judge``-cache entries cannot collide.
    """
    return _sha16(
        "judge_agent", criterion, output_hash, cwd_hash_, tools_sig, judge_model
    )


class JudgeCache:
    """Additive per-run judge-verdict cache.

    File: `evals/runs/<run-id>/judge-cache.json`. Rewritten in full on every
    write (small file, atomic replace). Entries never evicted — if an old
    criterion is reintroduced during a rescore, its old verdict is still
    there.
    """

    def __init__(self, path: Path):
        self.path = path
        self._entries: dict[str, dict] = {}
        # Serializes concurrent put()s now that scoring dispatches
        # judges in a thread pool. get() reads are safe without the
        # lock (dicts allow concurrent reads during writes in CPython).
        self._lock = threading.Lock()
        if path.exists():
            try:
                self._entries = json.loads(path.read_text())
                if not isinstance(self._entries, dict):
                    self._entries = {}
            except (OSError, json.JSONDecodeError):
                self._entries = {}

    def get(self, key: str) -> dict | None:
        return self._entries.get(key)

    def put(self, key: str, criterion: str, verdict: str, reasoning: str) -> None:
        with self._lock:
            self._entries[key] = {
                "criterion": criterion,
                "verdict": verdict,
                "reasoning": reasoning,
            }
            self._flush()

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w") as fh:
            json.dump(self._entries, fh, indent=2)
            fh.write("\n")
        tmp.replace(self.path)

    def __len__(self) -> int:
        return len(self._entries)
