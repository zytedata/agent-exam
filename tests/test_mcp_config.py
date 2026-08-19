"""`mcp_servers:` in config.yaml and in a task file: parsing, `${VAR}`
expansion, and the checks that refuse a run the servers cannot serve.
"""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

import pytest

from agent_exam.config import McpHttpServer, McpStdioServer, load_config
from agent_exam.errors import UsageError
from agent_exam.mcp import preflight, resolve_servers
from agent_exam.providers import get_provider
from agent_exam.tasks import load_task
from agent_exam.validation import validate_suite

if TYPE_CHECKING:
    from pathlib import Path


def _project(tmp_path: Path, config: str) -> Path:
    root = tmp_path / "proj"
    (root / "evals" / "suites" / "s" / "tasks").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[tool.agent-exam]\nevals_dir = "evals"\n')
    (root / "evals" / "config.yaml").write_text(dedent(config))
    return root


_CONFIG = """\
default_harness: dummy
mcp_servers:
  files:
    command: mcp-files
    args: ["--root", "."]
    env:
      TOKEN: "${MCP_TOKEN}"
  remote:
    type: http
    url: https://example.test/mcp
    headers:
      Authorization: "Bearer ${MCP_TOKEN}"
"""


def test_config_parses_both_server_shapes(tmp_path):
    cfg = load_config(_project(tmp_path, _CONFIG))

    assert isinstance(cfg.mcp_servers["files"], McpStdioServer)
    assert cfg.mcp_servers["files"].args == ["--root", "."]
    assert isinstance(cfg.mcp_servers["remote"], McpHttpServer)
    assert cfg.mcp_servers["remote"].url == "https://example.test/mcp"


def test_config_defaults_a_url_only_server_to_http(tmp_path):
    root = _project(
        tmp_path,
        """\
        mcp_servers:
          remote:
            url: https://example.test/mcp
        """,
    )
    cfg = load_config(root)

    assert cfg.mcp_servers["remote"] == McpHttpServer(url="https://example.test/mcp")
    assert resolve_servers(cfg)["remote"]["type"] == "http"


def test_config_rejects_unknown_server_key(tmp_path):
    root = _project(
        tmp_path,
        """\
        mcp_servers:
          files:
            command: mcp-files
            cwd: /tmp
        """,
    )
    with pytest.raises(UsageError):
        load_config(root)


def test_config_rejects_a_server_name_with_a_dot(tmp_path):
    """The name is half of an `mcp__<server>__<tool>` tool name and a TOML key
    path in the config codex_cli renders, where a dot would nest instead."""
    root = _project(
        tmp_path,
        """\
        mcp_servers:
          foo.bar:
            command: mcp-files
        """,
    )
    with pytest.raises(UsageError):
        load_config(root)


def test_config_rejects_a_server_name_with_a_doubled_underscore(tmp_path):
    """`mcp__a__b__search` would read as a call to `b` on server `a`."""
    root = _project(
        tmp_path,
        """\
        mcp_servers:
          a__b:
            command: mcp-files
        """,
    )
    with pytest.raises(UsageError):
        load_config(root)


def test_resolve_expands_env_refs(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_TOKEN", "s3cret")
    cfg = load_config(_project(tmp_path, _CONFIG))

    resolved = resolve_servers(cfg)

    assert resolved["files"]["env"] == {"TOKEN": "s3cret"}
    assert resolved["remote"]["headers"] == {"Authorization": "Bearer s3cret"}
    # `type` is dropped for stdio — harnesses that take the MCP JSON verbatim
    # accept it, but some of their own config schemas reject it.
    assert "type" not in resolved["files"]
    assert resolved["remote"]["type"] == "http"


def test_resolve_selects_a_subset(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_TOKEN", "s3cret")
    cfg = load_config(_project(tmp_path, _CONFIG))

    assert sorted(resolve_servers(cfg, ["files"])) == ["files"]
    assert resolve_servers(cfg, []) == {}


def test_resolve_reports_a_missing_variable(tmp_path, monkeypatch):
    monkeypatch.delenv("MCP_TOKEN", raising=False)
    cfg = load_config(_project(tmp_path, _CONFIG))

    with pytest.raises(UsageError, match=r"MCP_TOKEN"):
        resolve_servers(cfg)


def test_preflight_reports_missing_command_and_variable(tmp_path, monkeypatch):
    monkeypatch.delenv("MCP_TOKEN", raising=False)
    cfg = load_config(_project(tmp_path, _CONFIG))

    results = preflight(cfg, get_provider("claude_code"))

    by_name = {r.name: r for r in results}
    assert by_name["mcp server commands"].status == "FAIL"
    assert "files" in by_name["mcp server commands"].hint
    assert by_name["mcp server environment"].status == "FAIL"
    assert "MCP_TOKEN" in by_name["mcp server environment"].hint


def test_preflight_warns_when_the_harness_ignores_the_config(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_TOKEN", "s3cret")
    cfg = load_config(_project(tmp_path, _CONFIG))

    results = preflight(cfg, get_provider("dummy"))

    warn = next(r for r in results if r.name == "mcp servers supported")
    assert warn.status == "WARN"
    assert "dummy" in warn.hint
    # A harness that ignores mcp_servers entirely never reaches the
    # command/env checks below — they'd only ever fail over servers it
    # was never going to attach in the first place.
    assert [r.name for r in results] == ["mcp servers supported"]


def test_preflight_warns_when_the_harness_reports_no_connection_status(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MCP_TOKEN", "s3cret")
    cfg = load_config(_project(tmp_path, _CONFIG))

    results = preflight(cfg, get_provider("codex_cli"))

    warn = next(r for r in results if r.name == "mcp connection status")
    assert warn.status == "WARN"
    assert "codex_cli" in warn.hint


def test_preflight_is_silent_without_servers(tmp_path):
    cfg = load_config(_project(tmp_path, "default_harness: dummy\n"))

    assert preflight(cfg, get_provider("dummy")) == []


def test_task_selects_servers(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(
        dedent(
            """\
            kind: execute
            prompt: x
            mcp_servers: [files]
            assertions: []
            """
        )
    )
    assert load_task(p, "s")[0].mcp_servers == ["files"]


def test_task_defaults_to_every_server(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("kind: execute\nprompt: x\nassertions: []\n")

    assert load_task(p, "s")[0].mcp_servers is None


def test_validation_rejects_an_undeclared_server(tmp_path):
    root = _project(tmp_path, _CONFIG)
    (root / "evals" / "suites" / "s" / "tasks" / "t.yaml").write_text(
        dedent(
            """\
            kind: execute
            prompt: x
            mcp_servers: [flies]
            assertions: []
            """
        )
    )
    cfg = load_config(root)

    fails = [c for c in validate_suite(cfg, "s") if c.status == "FAIL"]

    assert [c.name for c in fails] == ["s: mcp servers declared"]
    assert "flies" in fails[0].hint


_SCOPED_CONFIG = """\
default_harness: claude_code
mcp_servers:
  local:
    command: sh
  remote:
    type: http
    url: https://example.test/mcp
    headers:
      Authorization: "Bearer ${MCP_TOKEN}"
"""


def _task(tmp_path: Path, servers: str, name: str = "t"):
    p = tmp_path / f"{name}.yaml"
    p.write_text(f"kind: execute\nprompt: x\nmcp_servers: {servers}\nassertions: []\n")
    return load_task(p, "s")[0]


def test_preflight_skips_servers_no_task_selects(tmp_path, monkeypatch):
    monkeypatch.delenv("MCP_TOKEN", raising=False)
    cfg = load_config(_project(tmp_path, _SCOPED_CONFIG))
    tasks = [_task(tmp_path, "[local]")]

    results = preflight(cfg, get_provider("claude_code"), tasks)

    assert [r.status for r in results] == ["OK"]


def test_preflight_covers_every_server_a_task_selects(tmp_path, monkeypatch):
    monkeypatch.delenv("MCP_TOKEN", raising=False)
    cfg = load_config(_project(tmp_path, _SCOPED_CONFIG))
    tasks = [_task(tmp_path, "[local]"), _task(tmp_path, "[remote]", "b")]

    results = preflight(cfg, get_provider("claude_code"), tasks)

    by_name = {r.name: r for r in results}
    assert by_name["mcp server environment"].status == "FAIL"


def test_preflight_checks_everything_for_a_task_without_a_subset(tmp_path, monkeypatch):
    monkeypatch.delenv("MCP_TOKEN", raising=False)
    cfg = load_config(_project(tmp_path, _SCOPED_CONFIG))
    p = tmp_path / "all.yaml"
    p.write_text("kind: execute\nprompt: x\nassertions: []\n")
    tasks = [_task(tmp_path, "[local]"), load_task(p, "s")[0]]

    results = preflight(cfg, get_provider("claude_code"), tasks)

    assert any(r.status == "FAIL" for r in results)


def test_preflight_is_silent_when_no_task_selects_a_server(tmp_path):
    cfg = load_config(_project(tmp_path, _SCOPED_CONFIG))

    assert preflight(cfg, get_provider("claude_code"), [_task(tmp_path, "[]")]) == []


def test_preflight_rejects_a_codex_header_it_cannot_send_before_the_run(
    tmp_path, monkeypatch
):
    """codex_cli's own header-shape constraint surfaces at preflight, not
    only when the first attempt stages the config."""
    monkeypatch.setenv("MCP_TOKEN", "s3cret")
    root = _project(
        tmp_path,
        _SCOPED_CONFIG.replace('Authorization: "Bearer ${MCP_TOKEN}"', "X-Key: k"),
    )
    cfg = load_config(root)

    results = preflight(cfg, get_provider("codex_cli"))

    by_name = {r.name: r for r in results}
    assert by_name["mcp server configuration"].status == "FAIL"
    assert "remote" in by_name["mcp server configuration"].hint


def test_preflight_rejects_a_codex_sse_server_before_the_run(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_TOKEN", "s3cret")
    root = _project(tmp_path, _SCOPED_CONFIG.replace("type: http", "type: sse"))
    cfg = load_config(root)

    results = preflight(cfg, get_provider("codex_cli"))

    by_name = {r.name: r for r in results}
    assert by_name["mcp server configuration"].status == "FAIL"
    assert "sse" in by_name["mcp server configuration"].hint
