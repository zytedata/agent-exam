from __future__ import annotations

from fixtures.canned_run_result import assistant_turn, text, user_turn

from agent_exam.judge.cache import (
    JudgeCache,
    agent_output_hash,
    cwd_hash,
    key_for,
    key_for_judge_agent,
    tools_signature,
)


def test_key_is_stable_deterministic():
    k1 = key_for("criterion", "outputhash", "claude-haiku-4-5")
    k2 = key_for("criterion", "outputhash", "claude-haiku-4-5")
    assert k1 == k2


def test_key_changes_on_any_field():
    base = key_for("c", "o", "m")
    assert base != key_for("c!", "o", "m")
    assert base != key_for("c", "o!", "m")
    assert base != key_for("c", "o", "m!")


def test_agent_output_hash_uses_last_assistant_text():
    t1 = [
        user_turn("u"),
        assistant_turn(text("first")),
    ]
    t2 = [
        user_turn("u"),
        assistant_turn(text("first")),
        assistant_turn(text("different")),
    ]
    assert agent_output_hash(t1) != agent_output_hash(t2)


def test_agent_output_hash_stable_for_same_text():
    t1 = [assistant_turn(text("hello"))]
    t2 = [user_turn("u"), assistant_turn(text("hello"))]
    assert agent_output_hash(t1) == agent_output_hash(t2)


def test_cache_roundtrip(tmp_path):
    path = tmp_path / "judge-cache.json"
    c = JudgeCache(path)
    c.put("abc123", "criterion text", "YES", "reasoning goes here")
    assert path.exists()
    c2 = JudgeCache(path)
    entry = c2.get("abc123")
    assert entry == {
        "criterion": "criterion text",
        "verdict": "YES",
        "reasoning": "reasoning goes here",
    }


def test_cache_survives_corrupt_file(tmp_path):
    path = tmp_path / "judge-cache.json"
    path.write_text("{not json")
    c = JudgeCache(path)
    assert len(c) == 0
    c.put("k", "c", "YES", "r")
    assert c.get("k") is not None


def test_cache_additive_no_eviction(tmp_path):
    path = tmp_path / "judge-cache.json"
    c = JudgeCache(path)
    c.put("k1", "c1", "YES", "r1")
    c.put("k2", "c2", "NO", "r2")
    assert c.get("k1") is not None
    assert c.get("k2") is not None
    c.put("k1", "c1-updated", "NO", "r1-updated")
    # Same key — overwritten is fine. Other key still there.
    assert c.get("k1")["verdict"] == "NO"
    assert c.get("k2") is not None


def test_cwd_hash_empty_dir(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    h_empty = cwd_hash(d)
    h_missing = cwd_hash(tmp_path / "does-not-exist")
    assert h_empty == h_missing  # both fold zero entries


def test_cwd_hash_changes_with_content(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    h1 = cwd_hash(tmp_path)
    (tmp_path / "a.txt").write_text("world")
    h2 = cwd_hash(tmp_path)
    assert h1 != h2


def test_cwd_hash_changes_with_new_file(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    h1 = cwd_hash(tmp_path)
    (tmp_path / "b.txt").write_text("y")
    h2 = cwd_hash(tmp_path)
    assert h1 != h2


def test_cwd_hash_stable_across_calls(tmp_path):
    (tmp_path / "nested" / "deep").mkdir(parents=True)
    (tmp_path / "nested" / "deep" / "f.txt").write_text("content")
    (tmp_path / "top.txt").write_text("top")
    assert cwd_hash(tmp_path) == cwd_hash(tmp_path)


def test_cwd_hash_skips_symlinks(tmp_path):
    (tmp_path / "real.txt").write_text("data")
    (tmp_path / "link.txt").symlink_to(tmp_path / "real.txt")
    h_with_link = cwd_hash(tmp_path)
    (tmp_path / "link.txt").unlink()
    h_without_link = cwd_hash(tmp_path)
    assert h_with_link == h_without_link


def test_tools_signature_order_insensitive():
    assert tools_signature(("read", "glob", "grep")) == tools_signature(
        ("grep", "read", "glob")
    )


def test_tools_signature_differs_on_membership():
    assert tools_signature(("read", "glob")) != tools_signature(
        ("read", "glob", "grep")
    )


def test_key_for_judge_agent_namespaced_from_judge():
    """Judge and judge_agent keys must not collide on identical inputs."""
    plain = key_for("crit", "out", "model")
    agent = key_for_judge_agent("crit", "out", "anycwd", "anytools", "model")
    assert plain != agent


def test_key_for_judge_agent_varies_on_cwd_and_tools():
    base = key_for_judge_agent("c", "o", "cwd1", "tools1", "m")
    assert base != key_for_judge_agent("c", "o", "cwd2", "tools1", "m")
    assert base != key_for_judge_agent("c", "o", "cwd1", "tools2", "m")
