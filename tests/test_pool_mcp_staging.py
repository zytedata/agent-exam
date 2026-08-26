"""Per-worker-process memoization of a run's MCP config: staged once per
(provider, run tmp root, server set) regardless of the order the servers were
asked for, and forgotten once the run ends so a process that drives the pool
more than once doesn't accumulate one dead entry per run forever.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_exam.pool import _MCP_STAGING, _mcp_options, forget_mcp_staging


@dataclass
class _FakeProvider:
    name: str = "fake"
    calls: list[list[str] | None] = field(default_factory=list)

    def stage_mcp_config(self, run_tmp_root, cfg, servers):
        self.calls.append(servers)
        return {"mcp_server_names": sorted(servers) if servers else []}


def test_the_same_server_set_in_a_different_order_hits_the_cache(tmp_path):
    provider = _FakeProvider()

    _mcp_options(provider, tmp_path, None, ["a", "b"])
    _mcp_options(provider, tmp_path, None, ["b", "a"])

    assert len(provider.calls) == 1
    forget_mcp_staging(tmp_path)


def test_forget_mcp_staging_drops_only_that_run(tmp_path):
    provider = _FakeProvider()
    other_root = tmp_path / "other"
    other_root.mkdir()

    _mcp_options(provider, tmp_path, None, ["a"])
    _mcp_options(provider, other_root, None, ["a"])

    forget_mcp_staging(tmp_path)

    assert not any(key[1] == tmp_path for key in _MCP_STAGING)
    assert any(key[1] == other_root for key in _MCP_STAGING)

    forget_mcp_staging(other_root)
