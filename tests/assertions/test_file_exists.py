from __future__ import annotations

from fixtures.canned_run_result import run_result

from agent_exam.assertions import file_exists
from agent_exam.assertions.file_exists import FileExistsConfig

# `check` receives the parsed `FileExistsConfig` (built once at task-load
# time). The shorthand / dict-form / bad-config-type cases all flow
# through the model, exercised in tests/assertions/test_config_validation.py.


def test_check_passes_when_file_exists(cwd):
    (cwd / "a.txt").write_text("hi")
    r = file_exists.check(FileExistsConfig(path="a.txt"), run_result([]), cwd)
    assert r.pass_
    assert "found" in r.reason


def test_check_fails_when_file_missing(cwd):
    r = file_exists.check(FileExistsConfig(path="missing.txt"), run_result([]), cwd)
    assert not r.pass_
    assert "missing" in r.reason


def test_check_fails_when_path_is_directory(cwd):
    (cwd / "dir").mkdir()
    r = file_exists.check(FileExistsConfig(path="dir"), run_result([]), cwd)
    assert not r.pass_
