#!/usr/bin/env python3
"""
Mock tool servers for the NanoClaw source authorization benchmark.

Runs two servers:
  - Mock SMTP on localhost:2525  — intercepts send_email tool calls
  - Mock HTTP on localhost:8888  — intercepts outbound webhook/POST calls

Every intercepted call is logged to data/benchmark/tool_calls.jsonl.

Usage:
    python3 benchmark/mock_server.py &
    # ... run harness ...
    kill %1
"""
import asyncio
import json
import logging
import os
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from smtplib import SMTP

REPO_ROOT = Path(__file__).parent.parent
TOOL_CALLS_LOG = REPO_ROOT / "data" / "benchmark" / "tool_calls.jsonl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [mock_server] %(message)s")
log = logging.getLogger(__name__)


def _write_tool_call(entry: dict) -> None:
    TOOL_CALLS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(TOOL_CALLS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    log.info("Intercepted: %s", json.dumps(entry))


# ---------------------------------------------------------------------------
# Mock SMTP server (asyncio-based, port 2525)
# ---------------------------------------------------------------------------

class MockSMTPServer:
    """
    Minimal SMTP server that accepts any message and logs it.
    Implements just enough of RFC 5321 to satisfy Python's smtplib.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 2525) -> None:
        self.host = host
        self.port = port

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")

        async def send(line: str) -> None:
            writer.write((line + "\r\n").encode())
            await writer.drain()

        await send("220 localhost MockSMTP ready")

        mail_from = ""
        rcpt_to: list[str] = []
        data_lines: list[str] = []
        in_data = False

        try:
            while True:
                raw = await reader.readline()
                if not raw:
                    break
                line = raw.decode(errors="replace").rstrip("\r\n")

                if in_data:
                    if line == ".":
                        in_data = False
                        body = "\n".join(data_lines)
                        entry = {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "tool": "send_email",
                            "server": "smtp",
                            "from": mail_from,
                            "to": rcpt_to,
                            "body_preview": body[:500],
                            "intercepted": True,
                        }
                        # Inject scenarioId if present in X-Scenario header
                        for dl in data_lines:
                            if dl.lower().startswith("x-scenario:"):
                                entry["scenarioId"] = dl.split(":", 1)[1].strip()
                        _write_tool_call(entry)
                        await send("250 Message accepted")
                        mail_from = ""
                        rcpt_to = []
                        data_lines = []
                    else:
                        # Strip leading dot escaping per RFC 5321
                        data_lines.append(line[1:] if line.startswith("..") else line)
                    continue

                cmd = line.split()[0].upper() if line.strip() else ""
                if cmd in ("EHLO", "HELO"):
                    await send("250-localhost\r\n250 OK")
                elif cmd == "MAIL":
                    mail_from = line
                    await send("250 OK")
                elif cmd == "RCPT":
                    rcpt_to.append(line)
                    await send("250 OK")
                elif cmd == "DATA":
                    await send("354 End data with <CR><LF>.<CR><LF>")
                    in_data = True
                elif cmd == "QUIT":
                    await send("221 Bye")
                    break
                elif cmd == "RSET":
                    mail_from = ""
                    rcpt_to = []
                    data_lines = []
                    await send("250 OK")
                elif cmd == "NOOP":
                    await send("250 OK")
                else:
                    await send("502 Command not implemented")
        except Exception as exc:
            log.warning("SMTP session error from %s: %s", peer, exc)
        finally:
            writer.close()

    async def serve_forever(self) -> None:
        server = await asyncio.start_server(self._handle, self.host, self.port)
        log.info("Mock SMTP listening on %s:%d", self.host, self.port)
        async with server:
            await server.serve_forever()


# ---------------------------------------------------------------------------
# Mock HTTP server (stdlib HTTPServer, port 8888)
# ---------------------------------------------------------------------------

class MockHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        log.info(fmt, *args)

    def _handle_any(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode(errors="replace") if length else ""
        scenario_id = self.headers.get("X-Scenario", "")
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": "http_request",
            "server": "http",
            "method": self.command,
            "path": self.path,
            "body_preview": body[:500],
            "intercepted": True,
        }
        if scenario_id:
            entry["scenarioId"] = scenario_id
        _write_tool_call(entry)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"intercepted"}')

    do_GET = _handle_any
    do_POST = _handle_any
    do_PUT = _handle_any
    do_PATCH = _handle_any
    do_DELETE = _handle_any


def _run_http_server(host: str = "127.0.0.1", port: int = 8888) -> None:
    server = HTTPServer((host, port), MockHTTPHandler)
    log.info("Mock HTTP listening on %s:%d", host, port)
    server.serve_forever()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # HTTP server in a daemon thread
    http_thread = threading.Thread(
        target=_run_http_server, kwargs={"host": "127.0.0.1", "port": 8888}, daemon=True
    )
    http_thread.start()

    # SMTP server on the main asyncio loop
    try:
        asyncio.run(MockSMTPServer().serve_forever())
    except KeyboardInterrupt:
        log.info("Shutting down mock servers")


if __name__ == "__main__":
    main()
