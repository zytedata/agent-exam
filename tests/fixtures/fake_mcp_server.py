"""A stdio MCP server for the client tests: answers ``initialize`` and
``tools/list`` over stdin/stdout, and misbehaves on request.

Run as ``python fake_mcp_server.py [mode]``. Modes:

- ``ok`` (default): three tools on one page.
- ``pages``: the same tools over two pages, joined by ``nextCursor``.
- ``noise``: a log line on stdout before any message.
- ``exit``: a complaint on stderr, then exit 3, before reading anything.
- ``hang``: read requests, answer none of them.
- ``error``: a JSON-RPC error to ``tools/list``.
"""

from __future__ import annotations

import json
import sys
import time

TOOLS = ["search", "files_search", "read"]


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "ok"
    if mode == "exit":
        print("boom: cannot start", file=sys.stderr, flush=True)
        sys.exit(3)
    if mode == "noise":
        print("starting up...", flush=True)
    pages = [TOOLS[:2], TOOLS[2:]] if mode == "pages" else [TOOLS]

    for line in sys.stdin:
        message = json.loads(line)
        if "id" not in message:
            continue  # a notification
        if mode == "hang":
            time.sleep(60)
        method = message["method"]
        reply: dict = {"jsonrpc": "2.0", "id": message["id"]}
        if method == "initialize":
            reply["result"] = {
                "protocolVersion": message["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake", "version": "0"},
            }
        elif method == "tools/list" and mode == "error":
            reply["error"] = {"code": -32603, "message": "listing is broken today"}
        elif method == "tools/list":
            page = int(message.get("params", {}).get("cursor") or 0)
            reply["result"] = {
                "tools": [
                    {"name": name, "inputSchema": {"type": "object"}}
                    for name in pages[page]
                ]
            }
            if page + 1 < len(pages):
                reply["result"]["nextCursor"] = str(page + 1)
        else:
            reply["error"] = {"code": -32601, "message": f"unknown method {method}"}
        print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main()
