"""Tests for blocked-plugin detection helpers."""

from __future__ import annotations

import json

from agent_exam.providers.claude_code.blocked_plugins import (
    blocked_skills_in_listing,
    enabled_blocked_in_settings,
)

# --- enabled_blocked_in_settings --------------------------------------------


def test_finds_single_blocked_enabled(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text(
        json.dumps(
            {
                "enabledPlugins": {
                    "skill-creator@claude-plugins-official": True,
                    "scraping-agent@some-market": True,
                }
            }
        )
    )
    found = enabled_blocked_in_settings(p, ["scraping-agent"])
    assert found == ["scraping-agent"]


def test_ignores_other_plugins(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"enabledPlugins": {"skill-creator@official": True}}))
    assert enabled_blocked_in_settings(p, ["scraping-agent"]) == []


def test_ignores_disabled(tmp_path):
    """An `enabledPlugins` entry with value False should not count as enabled."""
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"enabledPlugins": {"scraping-agent@m": False}}))
    assert enabled_blocked_in_settings(p, ["scraping-agent"]) == []


def test_dedupes_multiple_marketplace_entries(tmp_path):
    """Same plugin name under two marketplaces should report once."""
    p = tmp_path / "settings.json"
    p.write_text(
        json.dumps(
            {
                "enabledPlugins": {
                    "scraping-agent@market1": True,
                    "scraping-agent@market2": True,
                }
            }
        )
    )
    assert enabled_blocked_in_settings(p, ["scraping-agent"]) == ["scraping-agent"]


def test_missing_file_returns_empty(tmp_path):
    assert enabled_blocked_in_settings(tmp_path / "missing.json", ["x"]) == []


def test_malformed_json_returns_empty(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text("{not json")
    assert enabled_blocked_in_settings(p, ["x"]) == []


def test_empty_blocklist_returns_empty(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"enabledPlugins": {"anything@m": True}}))
    assert enabled_blocked_in_settings(p, []) == []


def test_missing_enabledPlugins_key(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"permissions": {}}))
    assert enabled_blocked_in_settings(p, ["x"]) == []


# --- blocked_skills_in_listing ---------------------------------------------


SAMPLE_LISTING = """\
- scrape-codegen: Generate web-poet page object code from an extraction spec
- scrape-scrapy-cloud: Scrapy Cloud deploy/schedule skill
- skill-creator:skill-creator: Create new skills, modify existing ones
- claude-md-management:claude-md-improver: Audit CLAUDE.md files
- scraping-agent:scrape-codegen: Overrides the staged copy — bad news
"""


def test_finds_blocked_prefixed_skills():
    hits = blocked_skills_in_listing(SAMPLE_LISTING, ["scraping-agent"])
    assert hits == ["scraping-agent:scrape-codegen"]


def test_ignores_unprefixed_skills():
    """Skills without `<plugin>:` prefix are our staged ones — always fine."""
    assert blocked_skills_in_listing(SAMPLE_LISTING, ["scrape-codegen"]) == []


def test_finds_multiple_plugins():
    hits = blocked_skills_in_listing(
        SAMPLE_LISTING, ["scraping-agent", "skill-creator"]
    )
    assert hits == [
        "scraping-agent:scrape-codegen",
        "skill-creator:skill-creator",
    ]


def test_empty_blocklist_short_circuits():
    assert blocked_skills_in_listing(SAMPLE_LISTING, []) == []


def test_empty_listing():
    assert blocked_skills_in_listing("", ["x"]) == []
