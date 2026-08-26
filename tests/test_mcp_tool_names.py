"""Canonical MCP tool names: one `tool_called:` line grades on any harness."""

from __future__ import annotations

import pytest

from agent_exam.mcp import canonical_tool_name, canonicalize_tool_names
from agent_exam.schemas import ToolCallBlock, Turn


def _call(name: str, subagent=None) -> ToolCallBlock:
    return ToolCallBlock(
        tool_use_id="1",
        name=name,
        input={},
        status="ok",
        result="",
        subagent=subagent,
    )


@pytest.mark.parametrize(
    "spelled",
    ["mcp__files__read", "files__read", "files_read", "files-read", "mcp_files_read"],
)
def test_every_harness_spelling_reaches_the_same_name(spelled):
    assert canonical_tool_name(spelled, ["files", "remote"]) == "mcp__files__read"


def test_a_tool_of_no_configured_server_is_left_alone():
    assert canonical_tool_name("Read", ["files"]) == "Read"
    assert canonical_tool_name("files_read", []) == "files_read"


def test_a_native_tool_that_merely_starts_with_a_server_name_is_left_alone():
    """The heuristic only fires on a separator, so OpenCode's own
    `todowrite` survives a server named `todo`."""
    assert canonical_tool_name("todowrite", ["todo"]) == "todowrite"
    assert canonical_tool_name("webfetch", ["web"]) == "webfetch"


def test_the_longest_matching_server_name_wins():
    assert (
        canonical_tool_name("files-archive_read", ["files", "files-archive"])
        == "mcp__files-archive__read"
    )


def test_renaming_reaches_subagent_calls():
    trajectory = [
        Turn(
            role="assistant",
            content=[
                _call("files_read"),
                _call(
                    "task",
                    subagent=[Turn(role="assistant", content=[_call("remote-search")])],
                ),
            ],
        )
    ]

    canonicalize_tool_names(trajectory, ["files", "remote"])

    parent = trajectory[0].content
    assert parent[0].name == "mcp__files__read"
    assert parent[1].name == "task"
    assert parent[1].subagent[0].content[0].name == "mcp__remote__search"
