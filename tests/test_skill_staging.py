"""Tests for the provider-agnostic skill staging helpers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from typing import TYPE_CHECKING

from agent_exam.providers.skill_staging import (
    check_skill_clashes,
    discover_skills,
    stage_skills_into,
)

if TYPE_CHECKING:
    from pathlib import Path

# --- discover_skills --------------------------------------------------------


def test_discover_skills_finds_skills_with_skill_md(tmp_path):
    src = tmp_path / "skills"
    (src / "scrape-codegen").mkdir(parents=True)
    (src / "scrape-codegen" / "SKILL.md").write_text("# skill")
    (src / "not-a-skill").mkdir()

    found = discover_skills([src])
    assert [name for name, _ in found] == ["scrape-codegen"]


def test_discover_skills_respects_exclude(tmp_path):
    src = tmp_path / "skills"
    (src / "a").mkdir(parents=True)
    (src / "a" / "SKILL.md").write_text("")
    (src / "b").mkdir()
    (src / "b" / "SKILL.md").write_text("")

    found = discover_skills([src], exclude=frozenset({"b"}))
    assert [name for name, _ in found] == ["a"]


def test_discover_skills_dedupes_by_name(tmp_path):
    src1 = tmp_path / "dir1"
    src2 = tmp_path / "dir2"
    (src1 / "scrape-codegen").mkdir(parents=True)
    (src1 / "scrape-codegen" / "SKILL.md").write_text("# first")
    (src2 / "scrape-codegen").mkdir(parents=True)
    (src2 / "scrape-codegen" / "SKILL.md").write_text("# second")

    found = discover_skills([src1, src2])
    assert len(found) == 1
    assert found[0][0] == "scrape-codegen"
    # Earlier directory wins.
    assert found[0][1].parent == src1


def test_discover_skills_skips_missing_dirs(tmp_path):
    found = discover_skills([tmp_path / "missing"])
    assert found == []


def test_discover_skills_none_returns_empty_list():
    assert discover_skills(None) == []


# --- check_skill_clashes ----------------------------------------------------


def test_check_skill_clashes_empty_globals():
    result = check_skill_clashes([], {"scrape-codegen"})
    assert result.status == "OK"
    assert "no global skills" in result.hint


def test_check_skill_clashes_exact_match():
    result = check_skill_clashes(
        ["scrape-codegen", "other"],
        {"scrape-codegen"},
    )
    assert result.status == "WARN"
    assert "scrape-codegen" in result.hint


def test_check_skill_clashes_no_match():
    result = check_skill_clashes(
        ["personal-note"],
        {"scrape-codegen"},
    )
    assert result.status == "OK"
    assert "none clash" in result.hint


def test_check_skill_clashes_with_namespace_normalize():
    """Namespaced global skills are compared after stripping the namespace."""
    result = check_skill_clashes(
        ["plugin-a:scrape-codegen", "plugin-b:other"],
        {"scrape-codegen"},
        normalize=lambda s: s.split(":", 1)[-1],
    )
    assert result.status == "WARN"
    # The full namespaced name is shown in the warning.
    assert "plugin-a:scrape-codegen" in result.hint
    assert "plugin-b:other" not in result.hint


def test_check_skill_clashes_multiple_plugins_same_skill():
    """If two plugins ship the same skill name, both are reported."""
    result = check_skill_clashes(
        ["plugin-a:scrape-codegen", "plugin-b:scrape-codegen"],
        {"scrape-codegen"},
        normalize=lambda s: s.split(":", 1)[-1],
    )
    assert result.status == "WARN"
    assert "plugin-a:scrape-codegen" in result.hint
    assert "plugin-b:scrape-codegen" in result.hint


def test_check_skill_clashes_custom_name():
    result = check_skill_clashes(
        ["x"],
        {"x"},
        check_name="my-check",
    )
    assert result.name == "my-check"
    assert result.status == "WARN"


# --- stage_skills_into ------------------------------------------------------


def _make_skills_dir(root: Path, names: list[str]) -> Path:
    src = root / "skills"
    src.mkdir(parents=True, exist_ok=True)
    for name in names:
        (src / name).mkdir(parents=True, exist_ok=True)
        (src / name / "SKILL.md").write_text(f"# {name}")
    return src


def test_stage_skills_into_creates_real_directory_copies(tmp_path):
    src = _make_skills_dir(tmp_path, ["a", "b"])
    run_dir = tmp_path / "run"

    staged = stage_skills_into(run_dir, [src], ".claude/skills")

    assert staged == ["a", "b"]
    target = run_dir / ".claude" / "skills"
    for name in ["a", "b"]:
        skill_dir = target / name
        assert skill_dir.is_dir()
        assert not skill_dir.is_symlink()
        assert (skill_dir / "SKILL.md").read_text() == f"# {name}"


def test_stage_skills_into_copies_nested_files_as_real_and_reachable(tmp_path):
    """Nested skill files (e.g. scripts/) are copied as real files, reachable
    via a recursive walk — the sandbox-traversal property copies buy us over
    symlinks (see `stage_skills_into`).
    """
    src = tmp_path / "skills"
    (src / "a" / "scripts").mkdir(parents=True)
    (src / "a" / "SKILL.md").write_text("# a")
    (src / "a" / "scripts" / "run.py").write_text("# script")
    run_dir = tmp_path / "run"

    stage_skills_into(run_dir, [src], ".claude/skills")

    skill_dir = run_dir / ".claude" / "skills" / "a"
    script = skill_dir / "scripts" / "run.py"
    assert script.is_file()
    assert not script.is_symlink()
    assert any(p.name == "run.py" for p in skill_dir.rglob("*"))


def test_stage_skills_into_idempotent_when_called_twice(tmp_path):
    """Re-staging into the same target with the same skills_dirs is safe —
    `dirs_exist_ok=True` means no FileExistsError on the second copy.
    """
    src = _make_skills_dir(tmp_path, ["a"])
    run_dir = tmp_path / "run"

    stage_skills_into(run_dir, [src], ".claude/skills")
    stage_skills_into(run_dir, [src], ".claude/skills")  # second call must not raise

    assert (run_dir / ".claude" / "skills" / "a" / "SKILL.md").is_file()


def test_stage_skills_into_refreshes_content_on_restage(tmp_path):
    """Re-staging refreshes file content from the (updated) source."""
    src = _make_skills_dir(tmp_path, ["a"])
    run_dir = tmp_path / "run"

    stage_skills_into(run_dir, [src], ".claude/skills")
    (src / "a" / "SKILL.md").write_text("# updated")
    stage_skills_into(run_dir, [src], ".claude/skills")

    staged = run_dir / ".claude" / "skills" / "a" / "SKILL.md"
    assert staged.read_text() == "# updated"


def test_stage_skills_into_is_race_safe_under_parallel_calls(tmp_path):
    """Multiple workers staging the same skills_dirs into the same target
    must not crash. This reproduces the trigger-eval shared-cwd race —
    `dirs_exist_ok=True` keeps concurrent copies from raising FileExistsError.
    """
    src = _make_skills_dir(tmp_path, ["a", "b", "c", "d"])
    run_dir = tmp_path / "run"

    # 16 workers all trying to stage at once. Threads are sufficient — the
    # race is at the filesystem layer; the GIL doesn't serialize syscalls.
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [
            pool.submit(stage_skills_into, run_dir, [src], ".claude/skills")
            for _ in range(16)
        ]
        done, _ = wait(futures)

    # No worker raised — exception propagation through Future.result().
    for f in done:
        assert f.exception() is None, f"worker crashed: {f.exception()!r}"

    # Every skill staged as a real directory copy of the source.
    target = run_dir / ".claude" / "skills"
    for name in ["a", "b", "c", "d"]:
        skill_dir = target / name
        assert skill_dir.is_dir()
        assert not skill_dir.is_symlink()
        assert (skill_dir / "SKILL.md").read_text() == f"# {name}"
