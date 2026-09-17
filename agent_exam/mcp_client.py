"""A minimal MCP client: enough to ask a server which tools it serves.

agent-exam otherwise never speaks MCP itself — the harness under evaluation
attaches the servers. Doctor and the runner use this to check that the tools
trigger tasks target exist, against each server's own ``tools/list`` rather
than a guess from the name (see :func:`agent_exam.validation.check_trigger_tools`).

Supports the stdio and Streamable HTTP transports. The legacy SSE transport
(``type: sse``) is not: such a server reports as unlistable, and the targets
it might serve go unchecked.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from http.client import HTTPResponse
    from typing import IO

PROTOCOL_VERSION = "2025-06-18"

try:
    _VERSION = version("agent-exam")
except PackageNotFoundError:  # pragma: no cover - a checkout that isn't installed
    _VERSION = "0"


class McpClientError(Exception):
    """The server could not be asked for its tools; the message says why."""


def list_tools(server: dict, *, timeout: float = 20.0) -> list[str]:
    """The names of the tools *server* advertises.

    *server* is one resolved ``mcp_servers`` entry, as
    :func:`agent_exam.mcp.resolve_servers` returns it: ``command``, ``args``
    and ``env`` for a stdio server, ``url`` and ``headers`` for an HTTP one,
    every ``${VAR}`` already expanded. *timeout* bounds each exchange with
    the server. Raises :class:`McpClientError` when the server cannot be
    started or reached, does not answer in time, or answers with an error.
    """
    if "command" in server:
        session: _Session = _StdioSession(server, timeout)
    elif server.get("type", "http") == "sse":
        raise McpClientError("the SSE transport is not supported for listing tools")
    else:
        session = _HttpSession(server, timeout)
    session.open()
    try:
        session.initialize()
        return session.list_tools()
    finally:
        session.close()


def _unwrap(reply: dict, method: str) -> dict:
    """The ``result`` of a JSON-RPC *reply* to *method*, or the error it carries."""
    error = reply.get("error")
    if error is not None:
        detail = error.get("message", error) if isinstance(error, dict) else error
        raise McpClientError(f"{method} failed: {detail}")
    result = reply.get("result")
    if not isinstance(result, dict):
        raise McpClientError(f"{method} answered without a result")
    return result


def _reply_to(parsed: object, wanted_id: int) -> dict | None:
    """The JSON-RPC message in *parsed* (one message, or a batch) answering
    request *wanted_id*, if it is there."""
    messages = parsed if isinstance(parsed, list) else [parsed]
    for message in messages:
        if isinstance(message, dict) and message.get("id") == wanted_id:
            return message
    return None


class _Session:
    """One MCP session: the initialize handshake, then requests.

    Transports differ in how a message goes out and how its reply comes
    back; everything about what to say is here.
    """

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout
        self.protocol_version: str | None = None
        self._next_id = 0

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def _request(self, method: str, params: dict) -> dict:
        raise NotImplementedError

    def _notify(self, method: str) -> None:
        raise NotImplementedError

    def _message(self, method: str, params: dict | None, *, with_id: bool) -> dict:
        message: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        if with_id:
            self._next_id += 1
            message["id"] = self._next_id
        return message

    def initialize(self) -> None:
        result = self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "agent-exam", "version": _VERSION},
            },
        )
        negotiated = result.get("protocolVersion")
        self.protocol_version = (
            negotiated if isinstance(negotiated, str) else PROTOCOL_VERSION
        )
        self._notify("notifications/initialized")

    def list_tools(self) -> list[str]:
        names: list[str] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        while True:
            result = self._request("tools/list", {"cursor": cursor} if cursor else {})
            tools = result.get("tools")
            if not isinstance(tools, list):
                raise McpClientError("tools/list answered without a tools list")
            names.extend(
                tool["name"]
                for tool in tools
                if isinstance(tool, dict) and isinstance(tool.get("name"), str)
            )
            cursor = result.get("nextCursor")
            # A server that hands the same cursor back would page forever.
            if not isinstance(cursor, str) or not cursor or cursor in seen_cursors:
                return names
            seen_cursors.add(cursor)


class _StdioSession(_Session):
    """The server is a child process; messages are lines on its stdin/stdout."""

    def __init__(self, server: dict, timeout: float) -> None:
        super().__init__(timeout)
        self._server = server
        self._proc: subprocess.Popen[str] | None = None
        self._stdin: IO[str] | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=20)
        self._stderr_pump: threading.Thread | None = None

    def open(self) -> None:
        command = [self._server["command"], *self._server.get("args", [])]
        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, **self._server.get("env", {})},
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            raise McpClientError(f"cannot start {command[0]}: {exc}") from exc
        self._proc, self._stdin = proc, proc.stdin
        threading.Thread(
            target=self._pump_stdout, args=(proc.stdout,), daemon=True
        ).start()
        self._stderr_pump = threading.Thread(
            target=self._pump_stderr, args=(proc.stderr,), daemon=True
        )
        self._stderr_pump.start()

    def _pump_stdout(self, stdout: IO[str]) -> None:
        for line in stdout:
            self._lines.put(line)
        self._lines.put(None)

    def _pump_stderr(self, stderr: IO[str]) -> None:
        for line in stderr:
            self._stderr.append(line.rstrip("\r\n"))

    def close(self) -> None:
        proc = self._proc
        if proc is None:
            return
        if self._stdin is not None:
            with contextlib.suppress(OSError):
                self._stdin.close()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    def _exited(self, what: str) -> str:
        # A process that has just died may still have its last words in
        # flight; the pump ends once its stderr closes.
        if self._stderr_pump is not None:
            self._stderr_pump.join(timeout=0.5)
        tail = "\n".join(self._stderr)
        return f"{what}; its stderr ends with:\n{tail}" if tail else what

    def _send(self, message: dict) -> None:
        if self._stdin is None:
            raise McpClientError("the server was never started")
        try:
            self._stdin.write(json.dumps(message) + "\n")
            self._stdin.flush()
        except (OSError, ValueError) as exc:
            raise McpClientError(
                self._exited(f"the server stopped reading before {message['method']}")
            ) from exc

    def _notify(self, method: str) -> None:
        self._send(self._message(method, None, with_id=False))

    def _request(self, method: str, params: dict) -> dict:
        message = self._message(method, params, with_id=True)
        self._send(message)
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise McpClientError(f"no answer to {method} within {self.timeout:g}s")
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                raise McpClientError(
                    self._exited(f"the server exited before answering {method}")
                )
            try:
                parsed = json.loads(line)
            except ValueError:
                continue  # a log line on stdout, not a message
            reply = _reply_to(parsed, message["id"])
            if reply is not None:
                return _unwrap(reply, method)


class _HttpSession(_Session):
    """Streamable HTTP: every message is a POST; a reply comes back as JSON
    or as an SSE stream carrying it."""

    def __init__(self, server: dict, timeout: float) -> None:
        super().__init__(timeout)
        self._url = server["url"]
        self._headers = dict(server.get("headers", {}))
        self._session_id: str | None = None

    def open(self) -> None:
        if urllib.parse.urlsplit(self._url).scheme not in ("http", "https"):
            raise McpClientError(f"{self._url} is not an http(s) URL")

    def close(self) -> None:
        if self._session_id is None:
            return
        # Servers may answer 405; ending the session is a courtesy either way.
        request = urllib.request.Request(
            self._url, headers=self._request_headers(), method="DELETE"
        )
        with (
            contextlib.suppress(urllib.error.URLError, OSError),
            urllib.request.urlopen(request, timeout=self.timeout),
        ):
            pass

    def _request_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **self._headers,
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        if self.protocol_version:
            headers["MCP-Protocol-Version"] = self.protocol_version
        return headers

    def _notify(self, method: str) -> None:
        self._post(self._message(method, None, with_id=False))

    def _request(self, method: str, params: dict) -> dict:
        message = self._message(method, params, with_id=True)
        reply = self._post(message)
        assert reply is not None
        return _unwrap(reply, method)

    def _post(self, message: dict) -> dict | None:
        method = message["method"]
        request = urllib.request.Request(
            self._url,
            data=json.dumps(message).encode("utf-8"),
            headers=self._request_headers(),
            method="POST",
        )
        try:
            response = urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            raise McpClientError(
                f"{method}: HTTP {exc.code} {exc.reason} from {self._url}"
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            reason = getattr(exc, "reason", None) or exc
            raise McpClientError(f"cannot reach {self._url}: {reason}") from exc
        with response:
            session_id = response.headers.get("Mcp-Session-Id")
            if session_id:
                self._session_id = session_id
            if "id" not in message:
                return None  # a notification: 202 Accepted, nothing to read
            try:
                if "text/event-stream" in response.headers.get("Content-Type", ""):
                    return self._read_event_stream(response, message["id"], method)
                body = response.read()
            except OSError as exc:
                raise McpClientError(
                    f"{method}: reading the answer from {self._url} failed: {exc}"
                ) from exc
        try:
            parsed = json.loads(body)
        except ValueError:
            raise McpClientError(
                f"{method}: {self._url} answered with something other than JSON"
            ) from None
        reply = _reply_to(parsed, message["id"])
        if reply is None:
            raise McpClientError(f"{method}: {self._url} answered another request")
        return reply

    def _read_event_stream(
        self, response: HTTPResponse, wanted_id: int, method: str
    ) -> dict:
        """The reply to *wanted_id* from an SSE *response*, read only as far
        as that reply: a server may keep the stream open afterwards."""
        data: list[str] = []
        for raw in response:
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if line:
                if line.startswith("data:"):
                    data.append(line[5:].removeprefix(" "))
                continue
            # A blank line ends an event. Other messages on the stream —
            # server notifications, requests to the client — are skipped.
            if data:
                try:
                    parsed = json.loads("\n".join(data))
                except ValueError:
                    parsed = None
                data = []
                reply = _reply_to(parsed, wanted_id)
                if reply is not None:
                    return reply
        raise McpClientError(
            f"{method}: the event stream from {self._url} ended without an answer"
        )
