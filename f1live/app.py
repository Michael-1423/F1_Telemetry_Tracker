"""Pipeline wiring: receive -> decode -> model/detectors -> incidents -> storage -> dashboard."""

from __future__ import annotations

import json
import os
import queue
import socket
import threading
import time
from typing import Iterable

from . import packets as P
from .incidents import IncidentRecorder
from .model import Session
from .storage import (RAW_NAME, RawWriter, Retention, SessionStore, dir_size, free_disk_bytes, read_any,
                      write_json)
from .summary import build_summary, driver_detail, snapshot, summary_markdown

VERSION = "1.0"


class SessionCtx:
    def __init__(self, session: Session, recorder: IncidentRecorder):
        self.session = session
        self.recorder = recorder
        self.raw: RawWriter | None = None
        self.pending: list[tuple[float, str, bytes]] = []   # packets before the folder exists
        self.events_file = None
        self.finalized = False
        self.keep_raw_on_create = False


class Pipeline:
    """Owns all mutable race state. Only the processing thread calls ingest()/tick()/command handlers."""

    def __init__(self, cfg: dict, mode: str = "live"):
        self.cfg = cfg
        self.mode = mode
        st = cfg.get("storage", {})
        self.store = SessionStore(st.get("dir", "data/races"))
        self.retention = Retention(self.store, st.get("raw_retention", "delete_on_next_race"),
                                   float(st.get("delete_grace_seconds", 120)), st.get("delete_trigger", "race"))
        self.record_raw = bool(st.get("record_raw", True))
        # which session kinds keep raw packets / incident packages (practice & quali: summaries only)
        self.raw_kinds = set(st.get("record_raw_for", ["race"]))
        self.incident_kinds = set(cfg.get("incidents", {}).get("sessions", ["race"]))
        self.end_timeout = float(cfg.get("session", {}).get("end_timeout_seconds", 90))
        self.sessions: dict[int, SessionCtx] = {}
        self.current: SessionCtx | None = None
        self.keep_next_raw = bool(cfg.get("_keep_raw", False))
        self.commands: queue.Queue = queue.Queue()
        self.snapshot_bytes = b'{"session": null}'
        self.detail_bytes = b'{"drivers": []}'
        self.detail_version = 0
        self._detail_sig = None
        self.snapshot_version = 0
        self.cond = threading.Condition()
        self.stats = {"packets": 0, "bad": 0, "other_format": 0, "started": time.time()}
        self.last_packet_wall = 0.0
        self.log: list[str] = []

    # ---------------------------------------------------------------- ingest
    def ingest(self, wall: float, key: str, payload: bytes) -> None:
        self.stats["packets"] += 1
        self.last_packet_wall = wall
        try:
            h = P.peek_header(payload)
            if h["packetFormat"] != 2020:
                self.stats["other_format"] += 1
                return
            pkt = P.decode(payload)
        except P.DecodeError:
            self.stats["bad"] += 1
            return
        uid = h["sessionUID"]
        if not uid:
            return
        ctx = self.sessions.get(uid)
        if ctx is None:
            ctx = self._new_ctx(uid, wall)
        if ctx.finalized:
            return
        s = ctx.session
        s.handle(pkt, key, wall)
        if s.path is None:
            ctx.pending.append((wall, key, payload))
            if s.track_id is not None and s.num_active:
                self._open_folder(ctx, wall)
        elif ctx.raw:
            ctx.raw.write(wall, key, payload)
        ctx.recorder.feed(wall, key, payload)   # rolling buffer; packages are only written when triggered
        if self.current is None or self.current.finalized or self.current.session.started_wall <= s.started_wall:
            self.current = ctx

    def _new_ctx(self, uid: int, wall: float) -> SessionCtx:
        recorder = IncidentRecorder(self.cfg)
        s = Session(uid, wall, self.cfg, emit_hook=self._on_event)
        ctx = SessionCtx(s, recorder)
        ctx.keep_raw_on_create = self.keep_next_raw
        self.sessions[uid] = ctx
        self._log(f"new session {uid:x}")
        return ctx

    def _open_folder(self, ctx: SessionCtx, wall: float) -> None:
        s = ctx.session
        s.path = self.store.create(s.started_wall, s.track_name, s.type_name, s.uid)
        sid = os.path.basename(s.path)
        if ctx.keep_raw_on_create:
            self.retention.keep(sid, True)
        if self.record_raw and s.kind in self.raw_kinds:
            ctx.raw = RawWriter(os.path.join(s.path, RAW_NAME))
            for item in ctx.pending:
                ctx.raw.write(*item)
        else:
            self._log(f"{s.type_name}: summary only, no raw data recorded")
        ctx.pending = []
        ctx.events_file = open(os.path.join(s.path, "events.jsonl"), "a", encoding="utf-8")
        for ev in s.events:
            ctx.events_file.write(json.dumps(ev) + "\n")
        self._write_meta(ctx, "live")
        self.retention.on_session_start(sid, s.is_race, wall)
        self._log(f"recording {s.type_name} at {s.track_name} -> {s.path}")

    def _on_event(self, s: Session, ev: dict) -> None:
        ctx = self.sessions.get(s.uid)
        if ctx is None:
            return
        if s.kind in self.incident_kinds or ev["kind"] == "manual":
            ctx.recorder.trigger(s, ev)
        if ctx.events_file:
            ctx.events_file.write(json.dumps(ev) + "\n")

    def _write_meta(self, ctx: SessionCtx, status: str) -> None:
        s = ctx.session
        if not s.path:
            return
        write_json(os.path.join(s.path, "session.json"), {
            "uid": str(s.uid), "track": s.track_name, "track_id": s.track_id, "type": s.type_name, "kind": s.kind,
            "is_race": s.is_race, "total_laps": s.total_laps, "started": s.started_wall, "last_packet": s.last_wall,
            "status": status, "humans": [s.display_name(c.idx) for c in s.humans()],
            "sources": {k: v.player_car for k, v in s.sources.items()}, "app_version": VERSION,
        })

    # ------------------------------------------------------------------ tick
    def tick(self, wall: float) -> None:
        """Housekeeping driven by the clock (real time live, packet time in replay)."""
        for uid, ctx in list(self.sessions.items()):
            if ctx.finalized:
                continue
            s = ctx.session
            ctx.recorder.tick(s, wall)
            quiet = wall - s.last_wall
            newer = any(o is not ctx and not o.finalized and o.session.started_wall > s.started_wall
                        for o in self.sessions.values())
            if quiet > self.end_timeout or (newer and quiet > 10) or (s.all_sources_ended() and quiet > 30):
                self.finalize(ctx)
            elif ctx.events_file:
                ctx.events_file.flush()
        self.retention.tick(wall if self.mode == "live" else time.time())
        self._drain_commands()

    def finalize(self, ctx: SessionCtx) -> None:
        if ctx.finalized:
            return
        s = ctx.session
        s.ended = True
        ctx.recorder.flush(s)
        ctx.finalized = True
        if ctx.raw:
            ctx.raw.close()
        if s.path:
            sm = build_summary(s, ctx.recorder.done)
            write_json(os.path.join(s.path, "summary.json"), sm)
            with open(os.path.join(s.path, "summary.md"), "w", encoding="utf-8") as f:
                f.write(summary_markdown(sm))
            self._write_meta(ctx, "finished")
            if ctx.events_file:
                ctx.events_file.close()
                ctx.events_file = None
            self._log(f"session finished: {s.path}")

    def finalize_all(self) -> None:
        for ctx in list(self.sessions.values()):
            self.finalize(ctx)

    # -------------------------------------------------------------- commands
    def _drain_commands(self) -> None:
        while True:
            try:
                cmd, arg = self.commands.get_nowait()
            except queue.Empty:
                return
            try:
                if cmd == "flag":
                    self.flag(arg or "")
                elif cmd == "keep":
                    self.retention.keep(arg["session"], bool(arg.get("keep", True)))
                elif cmd == "keep_all_pending":
                    self.retention.keep_all_pending()
                elif cmd == "delete_raw":
                    cur = self.current.session.path if self.current else None
                    if not (cur and os.path.basename(cur) == arg):
                        self.retention.delete_now(arg)
                elif cmd == "keep_current":
                    if self.current and self.current.session.path:
                        self.retention.keep(os.path.basename(self.current.session.path), bool(arg))
                    elif self.current:
                        self.current.keep_raw_on_create = bool(arg)
            except Exception as e:  # never let a UI action kill the pipeline
                self._log(f"command {cmd} failed: {e}")

    def flag(self, note: str) -> dict | None:
        ctx = self.current
        if not ctx or ctx.finalized:
            return None
        s = ctx.session
        cars = [c.idx for c in s.humans()]
        return s.emit("manual", "major", cars, "Flagged manually" + (f": {note}" if note else ""), capture=True)

    # -------------------------------------------------------------- snapshot
    def build_snapshot(self, now: float) -> None:
        ctx = self.current
        s = ctx.session if ctx else None
        raw_bytes = ctx.raw.size() if (ctx and ctx.raw) else 0
        extra = {
            "app": {"version": VERSION, "mode": self.mode, "now": now,
                    "udp_port": int(self.cfg.get("udp", {}).get("port", 20777)),
                    "packets": self.stats["packets"], "bad_packets": self.stats["bad"],
                    "other_format": self.stats["other_format"],
                    "last_packet_age": round(now - self.last_packet_wall, 1) if (self.last_packet_wall and self.mode == "live") else None,
                    "recording_raw": bool(ctx and ctx.raw and not ctx.raw.closed),
                    "raw_bytes": raw_bytes, "data_dir": self.store.base,
                    "free_disk": free_disk_bytes(self.store.base),
                    "keep_current_raw": bool(ctx and ctx.session.path and os.path.exists(os.path.join(ctx.session.path, "KEEP_RAW")))
                    or bool(ctx and ctx.keep_raw_on_create),
                    "log": self.log[-8:]},
            "retention": self.retention.status(time.time()),
        }
        if s is not None:
            sig = (s.uid, s.zones_version, tuple((c.idx, len(c.laps), bool(c.setup)) for c in s.humans()))
            if sig != self._detail_sig:
                self._detail_sig = sig
                self.detail_bytes = json.dumps(driver_detail(s), default=_default, separators=(",", ":")).encode()
                self.detail_version += 1
        extra["app"]["detail_version"] = self.detail_version
        snap = snapshot(s, ctx.recorder.done if ctx else [], extra)
        data = json.dumps(snap, default=_default, separators=(",", ":")).encode()
        with self.cond:
            self.snapshot_bytes = data
            self.snapshot_version += 1
            self.cond.notify_all()

    def _log(self, msg: str) -> None:
        line = time.strftime("%H:%M:%S ") + msg
        self.log.append(line)
        print(line, flush=True)


def _default(o):
    if isinstance(o, set):
        return sorted(o)
    return str(o)


# ------------------------------------------------------------------------ runners
def run_live(pipe: Pipeline, cfg: dict, stop: threading.Event) -> None:
    udp = cfg.get("udp", {})
    host, port = udp.get("host", "0.0.0.0"), int(udp.get("port", 20777))
    forwards = []
    for f in udp.get("forward", []):
        h, _, p = f.rpartition(":")
        forwards.append((h or "127.0.0.1", int(p)))
    q: queue.Queue = queue.Queue(maxsize=200000)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    except OSError:
        pass
    sock.bind((host, port))
    sock.settimeout(0.5)
    fwd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if forwards else None
    pipe._log(f"listening for F1 2020 telemetry on UDP {host}:{port}" + (f", forwarding to {forwards}" if forwards else ""))

    def receive() -> None:
        while not stop.is_set():
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                q.put_nowait((time.time(), f"{addr[0]}:{addr[1]}", data))
            except queue.Full:
                pipe.stats["bad"] += 1
            for target in forwards:
                try:
                    fwd_sock.sendto(data, target)
                except OSError:
                    pass

    threading.Thread(target=receive, name="udp-receiver", daemon=True).start()
    _process(pipe, stop, lambda: q.get(timeout=0.2), live=True)
    sock.close()


def run_replay(pipe: Pipeline, path: str, speed: float | None, stop: threading.Event, start_at: float = 0.0) -> None:
    """Feed a recording through the pipeline. speed=None means as fast as possible."""
    pipe._log(f"replaying {path} at {'max' if not speed else f'{speed:g}x'} speed")
    it: Iterable = read_any(path)
    t0_rec = None
    t0_real = time.monotonic()
    last_snap = 0.0
    last_tick = None
    n = 0
    for wall, key, payload in it:
        if stop.is_set():
            break
        if t0_rec is None:
            t0_rec = wall
        if wall - t0_rec < start_at:
            continue
        if speed:
            target = t0_real + (wall - t0_rec - start_at) / speed
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(min(delay, 1.0))
        pipe.ingest(wall, key, payload)
        n += 1
        if last_tick is None or wall - last_tick >= 0.5:
            last_tick = wall
            pipe.tick(wall)
        now = time.monotonic()
        if now - last_snap > 0.5:
            last_snap = now
            pipe.build_snapshot(wall)
    if last_tick is not None:
        pipe.tick(last_tick + 1)
    pipe.finalize_all()
    pipe.build_snapshot(time.time())
    pipe._log(f"replay finished: {n} packets")


def send_udp(path: str, target: tuple[str, int], speed: float | None, stop: threading.Event,
             start_at: float = 0.0, log=print) -> int:
    """Send a recording to a running tracker over UDP, one socket per original game, so the
    receiver sees several players exactly like on race night. Useful to test the setup."""
    socks: dict[str, socket.socket] = {}
    t0_rec = None
    t0_real = time.monotonic()
    n = 0
    for wall, key, payload in read_any(path):
        if stop.is_set():
            break
        if t0_rec is None:
            t0_rec = wall
        if wall - t0_rec < start_at:
            continue
        if speed:
            delay = t0_real + (wall - t0_rec - start_at) / speed - time.monotonic()
            if delay > 0:
                time.sleep(min(delay, 1.0))
        s = socks.get(key)
        if s is None:
            s = socks[key] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            log(f"sending as game {len(socks)} (originally {key})")
        s.sendto(payload, target)
        n += 1
    for s in socks.values():
        s.close()
    return n


def _process(pipe: Pipeline, stop: threading.Event, get, live: bool) -> None:
    last_snap = 0.0
    last_tick = 0.0
    while not stop.is_set():
        try:
            wall, key, payload = get()
            pipe.ingest(wall, key, payload)
        except queue.Empty:
            pass
        now = time.time()
        if now - last_tick >= 0.5:
            last_tick = now
            pipe.tick(now)
        if now - last_snap >= 0.5:
            last_snap = now
            pipe.build_snapshot(now)
    pipe.finalize_all()


def data_usage(pipe: Pipeline) -> dict:
    return {"dir": pipe.store.base, "bytes": dir_size(pipe.store.base), "free": free_disk_bytes(pipe.store.base)}
