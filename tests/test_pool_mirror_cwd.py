"""_mirror_cwd archives an attempt's cwd, pruning only the skill-discovery
symlink dirs it stages (e.g. `.agents/skills/`) so archives stay small
without silently dropping unrelated fixture/agent content."""

from __future__ import annotations

from agent_exam.pool import _mirror_cwd


def test_mirror_cwd_prunes_only_skills_subdir_of_discovery_dirs(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"

    (src / ".agents" / "skills" / "probe-skill").mkdir(parents=True)
    (src / ".agents" / "skills" / "probe-skill" / "SKILL.md").write_text("# skill")
    (src / ".agents" / "notes.txt").write_text("agent deliverable")
    (src / "output.txt").write_text("agent output")

    _mirror_cwd(src, dst)

    assert not (dst / ".agents" / "skills").exists()
    assert (dst / ".agents" / "notes.txt").read_text() == "agent deliverable"
    assert (dst / "output.txt").read_text() == "agent output"


def test_mirror_cwd_ignores_nested_dirs_named_like_discovery_dirs(tmp_path):
    """Only the top-level `<discovery-dir>/skills` is pruned — a nested
    directory that happens to share a discovery-dir name is fixture
    content, not something the provider staged, and must be archived."""
    src = tmp_path / "src"
    dst = tmp_path / "dst"

    (src / "some_project" / ".claude" / "skills").mkdir(parents=True)
    (src / "some_project" / ".claude" / "skills" / "marker.txt").write_text("keep me")

    _mirror_cwd(src, dst)

    assert (
        dst / "some_project" / ".claude" / "skills" / "marker.txt"
    ).read_text() == "keep me"
