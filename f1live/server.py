"""Dashboard web server: static page, live Server-Sent Events stream, and a small JSON API."""

from __future__ import annotations

import ipaddress
import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

from . import engineer as ENG
from .app import Pipeline

WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
SAFE = re.compile(r"^[A-Za-z0-9_\-]+$")


def make_handler(pipe: Pipeline, cfg: dict):
    allow_remote_control = bool(cfg.get("web", {}).get("allow_remote_control", False))

    class Handler(BaseHTTPRequestHandler):
        server_version = "f1live"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # keep the console for race messages
            pass

        # ----------------------------------------------------------- helpers
        def _send(self, code: int, body: bytes, ctype: str = "application/json", extra: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, default=str).encode())

        def _file(self, path: str, ctype: str) -> None:
            if not os.path.isfile(path):
                return self._json({"error": "not found"}, 404)
            with open(path, "rb") as f:
                self._send(200, f.read(), ctype)

        def _is_local(self) -> bool:
            try:
                return ipaddress.ip_address(self.client_address[0]).is_loopback
            except ValueError:
                return False

        def _session_path(self, sid: str) -> str | None:
            return pipe.store.path_of(sid) if SAFE.match(sid or "") else None

        # --------------------------------------------------------------- GET
        def do_GET(self) -> None:
            url = urlparse(self.path)
            path = unquote(url.path)
            parts = [p for p in path.split("/") if p]
            query = parse_qs(url.query)
            if path in ("/", "/index.html"):
                return self._file(os.path.join(WEB_DIR, "index.html"), "text/html; charset=utf-8")
            if path == "/api/state":
                return self._send(200, pipe.snapshot_bytes)
            if path == "/api/detail":
                return self._send(200, pipe.detail_bytes)
            if path == "/api/stream":
                return self._stream(_engineer_args(query))
            if path == "/api/engineer":
                args = _engineer_args(query)
                if not args:
                    return self._json({"error": "driver required"}, 400)
                return self._send(200, _engineer_json(pipe.engineer_inputs, *args))
            if path == "/api/sessions":
                return self._json({"sessions": pipe.store.list(), "can_control": self._can_control()})
            if len(parts) >= 3 and parts[:2] == ["api", "sessions"]:
                sp = self._session_path(parts[2])
                if not sp:
                    return self._json({"error": "unknown session"}, 404)
                if len(parts) == 4 and parts[3] == "summary":
                    return self._file(os.path.join(sp, "summary.json"), "application/json")
                if len(parts) == 4 and parts[3] == "summary.md":
                    return self._file(os.path.join(sp, "summary.md"), "text/markdown; charset=utf-8")
            return self._json({"error": "not found"}, 404)

        def _can_control(self) -> bool:
            return allow_remote_control or self._is_local()

        def _stream(self, engineer: tuple[str, list[str]] | None = None) -> None:
            """Live snapshots. With a driver (the engineer page), each one also carries that driver's
            race engineer analysis, with the viewer's handling feedback applied."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            seen = -1
            try:
                while True:
                    with pipe.cond:
                        pipe.cond.wait_for(lambda: pipe.snapshot_version != seen, timeout=15)
                        version, data, inputs = pipe.snapshot_version, pipe.snapshot_bytes, pipe.engineer_inputs
                    if version == seen:
                        self.wfile.write(b": keepalive\n\n")
                    else:
                        seen = version
                        if engineer:   # splice it into the snapshot object
                            data = data[:-1] + b',"engineer":' + _engineer_json(inputs, *engineer) + b"}"
                        self.wfile.write(b"data: " + data + b"\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                return

        # -------------------------------------------------------------- POST
        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if not self._can_control():
                return self._json({"error": "controls are only available on the recording computer "
                                            "(set web.allow_remote_control = true to allow others)"}, 403)
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            except ValueError:
                return self._json({"error": "bad json"}, 400)
            if path == "/api/flag":
                pipe.commands.put(("flag", str(body.get("note", ""))[:200]))
                return self._json({"ok": True})
            if path == "/api/keep":
                sid = str(body.get("session", ""))
                if not self._session_path(sid):
                    return self._json({"error": "unknown session"}, 404)
                pipe.commands.put(("keep", {"session": sid, "keep": bool(body.get("keep", True))}))
                return self._json({"ok": True})
            if path == "/api/hammer":
                pipe.commands.put(("hammer", bool(body.get("on", True))))
                return self._json({"ok": True})
            if path == "/api/keep-current":
                pipe.commands.put(("keep_current", bool(body.get("keep", True))))
                return self._json({"ok": True})
            if path == "/api/keep-all-pending":
                pipe.commands.put(("keep_all_pending", None))
                return self._json({"ok": True})
            if path == "/api/delete-raw":
                sid = str(body.get("session", ""))
                if not self._session_path(sid):
                    return self._json({"error": "unknown session"}, 404)
                pipe.commands.put(("delete_raw", sid))
                return self._json({"ok": True})
            return self._json({"error": "not found"}, 404)

    return Handler


def _engineer_args(query: dict) -> tuple[str, list[str]] | None:
    """?engineer=<driver name>&fb=<feedback id>,<feedback id>  (driver= also accepted)"""
    name = (query.get("engineer") or query.get("driver") or [""])[0].strip()[:64]
    if not name:
        return None
    fb = [f for f in ",".join(query.get("fb", [])).split(",") if f]
    return name, fb


def _engineer_json(inputs: dict, name: str, feedback: list[str]) -> bytes:
    return json.dumps(ENG.payload(inputs.get(name.lower()), name, feedback), separators=(",", ":")).encode()


class QuietServer(ThreadingHTTPServer):
    """Browsers drop connections all the time (tab closed, page reloaded); don't print tracebacks for it."""

    def handle_error(self, request, client_address):
        import sys
        if isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            return
        super().handle_error(request, client_address)


def serve(pipe: Pipeline, cfg: dict) -> ThreadingHTTPServer:
    web = cfg.get("web", {})
    host, port = web.get("host", "0.0.0.0"), int(web.get("port", 8020))
    httpd = QuietServer((host, port), make_handler(pipe, cfg))
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, name="web", daemon=True).start()
    shown = "localhost" if host in ("0.0.0.0", "::") else host
    pipe._log(f"dashboard: http://{shown}:{port}" + ("  (also reachable from other machines on your network)" if host in ("0.0.0.0", "::") else ""))
    return httpd
