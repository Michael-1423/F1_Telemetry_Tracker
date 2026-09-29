"""f1live: live F1 2020 telemetry tracker for online lobbies.

    python -m f1live                       start listening + dashboard (http://localhost:8020)
    python -m f1live replay FILE [--speed 10|max] [--no-server]
    python -m f1live send FILE [--to HOST:PORT] [--speed 1]   test a live setup with a recording
    python -m f1live sessions              list recorded sessions
    python -m f1live keep SESSION [--off]  keep (or stop keeping) a session's raw data
    python -m f1live prompt INCIDENT_DIR   rebuild prompt.md for a saved incident
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time

from .app import Pipeline, run_live, run_replay
from .storage import Retention, SessionStore

DEFAULT_CONFIG = "f1live.toml"


def load_config(path: str | None) -> dict:
    path = path or (DEFAULT_CONFIG if os.path.exists(DEFAULT_CONFIG) else None)
    if not path:
        return {}
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    try:
        import tomllib
    except ImportError:  # Python < 3.11
        sys.exit("Reading .toml config needs Python 3.11+; use a .json config instead.")
    with open(path, "rb") as f:
        return tomllib.load(f)


def _mb(n: int) -> str:
    return f"{n / 1e6:,.1f} MB"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m f1live", description="Live F1 2020 telemetry tracker")
    ap.add_argument("--config", help="config file (default: f1live.toml if present)")
    ap.add_argument("--data", help="data directory (default: data/races)")
    sub = ap.add_subparsers(dest="cmd")

    lv = sub.add_parser("live", help="listen for telemetry (default)")
    for p in (ap, lv):
        p.add_argument("--port", type=int, help="UDP port the games send to (default 20777)")
        p.add_argument("--web-port", type=int, help="dashboard port (default 8020)")
        p.add_argument("--no-raw", action="store_true", help="don't record raw packets for this run")
        p.add_argument("--keep-raw", action="store_true", help="keep the raw data of the race recorded now")
        p.add_argument("--keep-previous-raw", action="store_true",
                       help="keep the raw data of the most recent earlier session (it would otherwise be deleted when this race starts)")
        p.add_argument("--forward", action="append", default=None, metavar="HOST:PORT",
                       help="also forward every packet to another tool (repeatable)")

    rp = sub.add_parser("replay", help="replay a recording (.f1raw.gz or old .jsonl capture)")
    rp.add_argument("file")
    rp.add_argument("--speed", default="10", help="playback speed multiplier, or 'max' (default 10)")
    rp.add_argument("--start", type=float, default=0.0, help="skip the first N seconds of the recording")
    rp.add_argument("--no-server", action="store_true", help="don't start the dashboard")
    rp.add_argument("--record-raw", action="store_true", help="write a raw recording of the replayed session (converts old captures)")
    rp.add_argument("--web-port", type=int)

    sd = sub.add_parser("send", help="send a recording to a running tracker over UDP (setup test)")
    sd.add_argument("file")
    sd.add_argument("--to", default="127.0.0.1:20777", metavar="HOST:PORT")
    sd.add_argument("--speed", default="1", help="speed multiplier, or 'max' (default 1 = real time)")
    sd.add_argument("--start", type=float, default=0.0, help="skip the first N seconds")

    sub.add_parser("sessions", help="list recorded sessions")
    kp = sub.add_parser("keep", help="keep a session's raw data from automatic deletion")
    kp.add_argument("session")
    kp.add_argument("--off", action="store_true", help="allow it to be deleted again")
    pp = sub.add_parser("prompt", help="rebuild prompt.md for an incident folder")
    pp.add_argument("incident_dir")

    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.data:
        cfg.setdefault("storage", {})["dir"] = a.data
    cmd = a.cmd or "live"

    if cmd == "sessions":
        store = SessionStore(cfg.get("storage", {}).get("dir", "data/races"))
        for s in store.list():
            print(f"{s['id']:<60} {s.get('status', ''):<9} raw {_mb(s['raw_bytes']):>10}{' (kept)' if s['keep_raw'] else '':<8} "
                  f"incidents {s['incidents']:>3}  total {_mb(s['total_bytes'])}")
        return
    if cmd == "keep":
        store = SessionStore(cfg.get("storage", {}).get("dir", "data/races"))
        ok = Retention(store).keep(a.session, not a.off)
        print("ok" if ok else f"unknown session {a.session}")
        return
    if cmd == "send":
        from .app import send_udp
        host, _, port = a.to.rpartition(":")
        stop = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        n = send_udp(a.file, (host or "127.0.0.1", int(port)), None if a.speed == "max" else float(a.speed), stop, a.start)
        print(f"sent {n} packets")
        return
    if cmd == "prompt":
        import gzip
        from .incidents import build_prompt
        with gzip.open(os.path.join(a.incident_dir, "incident.json.gz"), "rt", encoding="utf-8") as f:
            meta = json.load(f)
        with open(os.path.join(a.incident_dir, "prompt.md"), "w", encoding="utf-8") as f:
            f.write(build_prompt(meta))
        print(os.path.join(a.incident_dir, "prompt.md"))
        return

    if getattr(a, "port", None):
        cfg.setdefault("udp", {})["port"] = a.port
    if getattr(a, "web_port", None):
        cfg.setdefault("web", {})["port"] = a.web_port
    if getattr(a, "forward", None):
        cfg.setdefault("udp", {})["forward"] = a.forward
    if getattr(a, "no_raw", False):
        cfg.setdefault("storage", {})["record_raw"] = False
    if getattr(a, "keep_raw", False):
        cfg["_keep_raw"] = True

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *_: stop.set())

    if cmd == "replay":
        cfg.setdefault("storage", {}).setdefault("record_raw", False)
        if a.record_raw:
            cfg["storage"]["record_raw"] = True
        cfg["storage"].setdefault("raw_retention", "keep")
        pipe = Pipeline(cfg, mode="replay")
        httpd = None
        if not a.no_server:
            from .server import serve
            httpd = serve(pipe, cfg)
        speed = None if a.speed == "max" else float(a.speed)
        worker = threading.Thread(target=run_replay, args=(pipe, a.file, speed, stop, a.start), daemon=True)
        worker.start()
        while worker.is_alive():
            worker.join(0.3)
        if httpd and not stop.is_set():
            pipe._log("replay done; dashboard still running (Ctrl+C to quit)")
            while not stop.is_set():
                time.sleep(0.3)
        return

    pipe = Pipeline(cfg, mode="live")
    if a.keep_previous_raw:
        prev = pipe.store.list()
        if prev:
            pipe.retention.keep(prev[0]["id"], True)
            pipe._log(f"keeping raw data of {prev[0]['id']}")
    from .server import serve
    serve(pipe, cfg)
    run_live(pipe, cfg, stop)


if __name__ == "__main__":
    main()
