from __future__ import annotations

from agent_exam.providers.claude_code.skill_detect import (
    detect_from_input,
    detect_from_partial,
)


def test_partial_detects_skill_tool():
    assert (
        detect_from_partial("Skill", '{"skill":"scrape-codegen"}') == "scrape-codegen"
    )
    assert detect_from_partial("Skill", '{"name":"scrape"') == "scrape"


def test_partial_detects_read_skill_md():
    assert (
        detect_from_partial("Read", '{"file_path":"/x/skills/scrape/SKILL.md"}')
        == "scrape"
    )


def test_partial_misses_on_non_skill_tools():
    assert detect_from_partial("Bash", '{"skill":"x"}') is None


def test_partial_handles_incomplete_json():
    # Mid-stream: JSON isn't closed yet.
    assert detect_from_partial("Skill", '{"skill":"scrape-cod') is None
    assert detect_from_partial("Skill", '{"skill":"scrape-codegen"') == "scrape-codegen"


def test_input_detects_skill_tool():
    skill, kind = detect_from_input("Skill", {"skill": "foo"})
    assert skill == "foo"
    assert kind == "skill_tool"


def test_input_detects_read_skill_md():
    skill, kind = detect_from_input(
        "Read", {"file_path": "/plugin/skills/foo/SKILL.md"}
    )
    assert skill == "foo"
    assert kind == "skill_md_read"


def test_input_misses_on_unrelated_read():
    skill, kind = detect_from_input("Read", {"file_path": "/plugin/README.md"})
    assert skill is None
    assert kind is None


def test_input_handles_non_dict_input():
    assert detect_from_input("Skill", None) == (None, None)
