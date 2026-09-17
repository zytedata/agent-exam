"""The minimal MCP client behind the trigger tool check."""

from __future__ import annotations

import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from agent_exam.mcp_client import McpClientError, list_tools

_FAKE = Path(__file__).parent / "fixtures" / "fake_mcp_server.py"


def _stdio(*modes: str) -> dict:
    return {"command": sys.executable, "args": [str(_FAKE), *modes]}


def test_lists_the_tools_of_a_stdio_server():
    assert list_tools(_stdio()) == ["search", "files_search", "read"]


def test_follows_the_pages_of_a_long_listing():
    assert list_tools(_stdio("pages")) == ["search", "files_search", "read"]


def test_skips_what_a_server_logs_to_stdout():
    assert list_tools(_stdio("noise")) == ["search", "files_search", "read"]


def test_reports_a_command_that_does_not_exist():
    with pytest.raises(McpClientError, match="cannot start no-such-mcp-server"):
        list_tools({"command": "no-such-mcp-server", "args": []})


def test_reports_a_server_that_dies_with_what_it_said():
    with pytest.raises(McpClientError) as excinfo:
        list_tools(_stdio("exit"))

    assert "exited before answering initialize" in str(excinfo.value)
    assert "boom: cannot start" in str(excinfo.value)


def test_reports_a_server_that_never_answers():
    with pytest.raises(
        McpClientError, match=re.escape("no answer to initialize within 0.5s")
    ):
        list_tools(_stdio("hang"), timeout=0.5)


def test_reports_a_json_rpc_error():
    with pytest.raises(McpClientError, match="tools/list failed: listing is broken"):
        list_tools(_stdio("error"))


def test_the_legacy_sse_transport_is_declined():
    with pytest.raises(McpClientError, match="SSE transport is not supported"):
        list_tools({"type": "sse", "url": "http://127.0.0.1:1/sse"})


# --- Streamable HTTP ----------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    """A Streamable HTTP server: a session id on ``initialize``, ``202`` to
    notifications, and ``tools/list`` as JSON or as an SSE stream that
    carries a server notification ahead of the reply."""

    stream = True
    token: str | None = "secret"
    seen: list[dict] = []

    def log_message(self, *args: object) -> None:
        pass

    def do_DELETE(self) -> None:
        self.send_response(200)
        self.end_headers()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        # Header names as urllib sends them differ in case from the spelling
        # in the code; HTTP does not care, so neither does the record.
        headers = {name.lower(): value for name, value in self.headers.items()}
        self.seen.append({"headers": headers, "body": body})
        if self.token and self.headers.get("Authorization") != f"Bearer {self.token}":
            self.send_response(401)
            self.end_headers()
            return
        if "id" not in body:
            self.send_response(202)
            self.end_headers()
            return
        reply: dict = {"jsonrpc": "2.0", "id": body["id"]}
        if body["method"] == "initialize":
            reply["result"] = {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake", "version": "0"},
            }
            self._json(reply, session_id="s1")
            return
        if self.headers.get("Mcp-Session-Id") != "s1":
            self.send_response(400)
            self.end_headers()
            return
        reply["result"] = {"tools": [{"name": "search"}, {"name": "files_search"}]}
        if self.stream:
            self._sse(reply)
        else:
            self._json(reply)

    def _json(self, reply: dict, session_id: str | None = None) -> None:
        payload = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        if session_id:
            self.send_header("Mcp-Session-Id", session_id)
        self.end_headers()
        self.wfile.write(payload)

    def _sse(self, reply: dict) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/message",
            "params": {},
        }
        self.wfile.write(b": keep-alive\n\n")
        self.wfile.write(
            f"event: message\ndata: {json.dumps(notification)}\n\n".encode()
        )
        self.wfile.write(f"event: message\ndata: {json.dumps(reply)}\n\n".encode())


@pytest.fixture
def http_server(request):
    _Handler.seen = []
    _Handler.stream = getattr(request, "param", True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/mcp"
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(
    "http_server", [True, False], indirect=True, ids=["sse", "json"]
)
def test_lists_the_tools_of_an_http_server(http_server):
    server = {"url": http_server, "headers": {"Authorization": "Bearer secret"}}

    assert list_tools(server) == ["search", "files_search"]

    methods = [s["body"]["method"] for s in _Handler.seen]
    assert methods == ["initialize", "notifications/initialized", "tools/list"]
    listing = _Handler.seen[-1]["headers"]
    # The session the server opened, and the version it negotiated, ride
    # along on every later request.
    assert listing["mcp-session-id"] == "s1"
    assert listing["mcp-protocol-version"] == "2025-03-26"
    assert listing["accept"] == "application/json, text/event-stream"


def test_reports_an_http_error_status(http_server):
    with pytest.raises(McpClientError, match="initialize: HTTP 401"):
        list_tools({"url": http_server, "headers": {}})


def test_reports_a_server_that_cannot_be_reached():
    with pytest.raises(
        McpClientError, match=re.escape("cannot reach http://127.0.0.1:1/mcp")
    ):
        list_tools({"url": "http://127.0.0.1:1/mcp"}, timeout=2)
