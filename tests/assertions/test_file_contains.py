from __future__ import annotations

from fixtures.canned_run_result import run_result

from agent_exam.assertions import file_contains
from agent_exam.assertions.file_contains import FileContainsConfig


def _cfg(**kw) -> FileContainsConfig:
    return FileContainsConfig(**kw)


def test_substring_match(cwd):
    (cwd / "a.py").write_text("from x import y\n@dataclass\nclass C: pass\n")
    r = file_contains.check(
        _cfg(path="a.py", pattern="@dataclass"), run_result([]), cwd
    )
    assert r.pass_


def test_substring_missing(cwd):
    (cwd / "a.py").write_text("no match here")
    r = file_contains.check(
        _cfg(path="a.py", pattern="@dataclass"), run_result([]), cwd
    )
    assert not r.pass_


def test_regex(cwd):
    (cwd / "a.py").write_text("class Foo:\n    x = 1\n")
    r = file_contains.check(
        _cfg(path="a.py", pattern=r"class\s+\w+:", regex=True),
        run_result([]),
        cwd,
    )
    assert r.pass_


def test_missing_file(cwd):
    r = file_contains.check(_cfg(path="nope.py", pattern="x"), run_result([]), cwd)
    assert not r.pass_
    assert "missing" in r.reason
