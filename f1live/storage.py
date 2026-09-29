"""Session folders, raw packet recording/reading, and raw-data retention."""

from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import struct
import threading
import time
from datetime import datetime
from typing import Iterator

from . import packets as P

RAW_MAGIC = b"F1LIVE-RAW1\n"
RAW_NAME = "raw.f1raw.gz"
KEEP_MARKER = "KEEP_RAW"
_SRC = struct.Struct("<cHH")      # b"S", source id, name length
_PKT = struct.Struct("<cdHH")     # b"P", wall time, source id, payload length


class RawWriter:
    """Append-only gzip stream of (wall time, source, UDP payload) records.

    The stream is flushed every few seconds with a sync flush so a crash loses at most
    a few seconds of data and the file stays readable.
    """

    def __init__(self, path: str, flush_every: float = 3.0, level: int = 3):
        self.path = path
        self._f = gzip.open(path, "wb", compresslevel=level)
        self._f.write(RAW_MAGIC)
        self._ids: dict[str, int] = {}
        self._flush_every = flush_every
        self._last_flush = time.monotonic()
        self.packets = 0
        self.closed = False

    def write(self, wall: float, source: str, payload: bytes) -> None:
        if self.closed:
            return
        sid = self._ids.get(source)
        if sid is None:
            sid = self._ids[source] = len(self._ids)
            name = source.encode()
            self._f.write(_SRC.pack(b"S", sid, len(name)) + name)
        self._f.write(_PKT.pack(b"P", wall, sid, len(payload)) + payload)
        self.packets += 1
        now = time.monotonic()
        if now - self._last_flush > self._flush_every:
            self._f.flush()
            self._last_flush = now

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self._f.close()

    def size(self) -> int:
        try:
            return os.path.getsize(self.path)
        except OSError:
            return 0


def read_raw(path: str) -> Iterator[tuple[float, str, bytes]]:
    """Yield (wall time, source key, payload) from a raw recording. Tolerates a truncated tail."""
    names: dict[int, str] = {}
    with gzip.open(path, "rb") as f:
        if f.read(len(RAW_MAGIC)) != RAW_MAGIC:
            raise ValueError(f"{path} is not an f1live raw recording")
        while True:
            try:
                kind = f.read(1)
                if not kind:
                    return
                if kind == b"S":
                    sid, n = struct.unpack("<HH", f.read(4))
                    names[sid] = f.read(n).decode()
                elif kind == b"P":
                    wall, sid, n = struct.unpack("<dHH", f.read(12))
                    payload = f.read(n)
                    if len(payload) < n:
                        return
                    yield wall, names.get(sid, f"source{sid}"), payload
                else:
                    return
            except (EOFError, struct.error, OSError):
                return


def read_legacy_jsonl(path: str) -> Iterator[tuple[float, str, bytes]]:
    """Yield records from captures made by the old f1_capture.py (JSON lines, optionally .gz)."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                d = r["data"]
                if (d.get("header", {}).get("packetId") == P.EVENT and d.get("eventStringCode") in P.EVENT_DETAILS
                        and not isinstance(d.get("eventDetails"), dict)):
                    continue  # the old capture script lost event details; don't invent them
                wall = datetime.fromisoformat(r["capture_time"]).timestamp()
                payload = P.encode(d)
            except (ValueError, KeyError, P.DecodeError):
                continue
            yield wall, f"{r.get('source_ip', '?')}:{r.get('source_port', 0)}", payload


def read_any(path: str) -> Iterator[tuple[float, str, bytes]]:
    if path.endswith(".f1raw.gz") or path.endswith(".f1raw"):
        return read_raw(path)
    return read_legacy_jsonl(path)


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")[:40] or "x"


def dir_size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return total


class SessionStore:
    """Owns the data directory: one folder per game session."""

    def __init__(self, base: str):
        self.base = os.path.abspath(base)
        os.makedirs(self.base, exist_ok=True)
        self._lock = threading.Lock()

    def create(self, started_wall: float, track: str, session_type: str, uid: int) -> str:
        stamp = datetime.fromtimestamp(started_wall).strftime("%Y-%m-%d_%H%M")
        name = f"{stamp}_{_slug(track)}_{_slug(session_type)}_{uid & 0xFFFFFF:06x}"
        path = os.path.join(self.base, name)
        with self._lock:
            os.makedirs(os.path.join(path, "incidents"), exist_ok=True)
        return path

    def list(self) -> list[dict]:
        out = []
        for name in sorted(os.listdir(self.base), reverse=True):
            path = os.path.join(self.base, name)
            meta_path = os.path.join(path, "session.json")
            if not os.path.isfile(meta_path):
                continue
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
            except (OSError, ValueError):
                continue
            raw = os.path.join(path, RAW_NAME)
            meta.update({
                "id": name,
                "raw_bytes": os.path.getsize(raw) if os.path.exists(raw) else 0,
                "keep_raw": os.path.exists(os.path.join(path, KEEP_MARKER)),
                "total_bytes": dir_size(path),
                "incidents": len([d for d in os.listdir(os.path.join(path, "incidents"))
                                  if os.path.isdir(os.path.join(path, "incidents", d))])
                if os.path.isdir(os.path.join(path, "incidents")) else 0,
                "incident_bytes": dir_size(os.path.join(path, "incidents")),
                "has_summary": os.path.exists(os.path.join(path, "summary.json")),
            })
            out.append(meta)
        return out

    def path_of(self, session_id: str) -> str | None:
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", session_id or ""):
            return None
        p = os.path.join(self.base, session_id)
        return p if os.path.isdir(p) else None


def write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, float) and (o != o):
        return None
    if isinstance(o, set):
        return sorted(o)
    return str(o)


class Retention:
    """Deletes raw recordings (and, by default, incident packages) of earlier sessions once the next
    race starts.

    When a new race begins, every older session that has raw data or incident packages and no KEEP_RAW
    marker is scheduled for deletion after a grace period. During the grace period the dashboard
    shows a banner with a Keep button, which keeps both. Summaries and event logs are never deleted.
    """

    def __init__(self, store: SessionStore, mode: str = "delete_on_next_race", grace_s: float = 120.0,
                 trigger: str = "race", incidents: bool = True):
        self.store = store
        self.mode = mode
        self.grace_s = grace_s
        self.trigger = trigger
        self.incidents = incidents            # delete incident packages along with the raw data
        self.pending: dict[str, float] = {}   # session id -> wall time of deletion
        self.deleted: list[dict] = []
        self._lock = threading.Lock()

    def on_session_start(self, new_session_id: str, is_race: bool, now: float) -> None:
        if self.mode != "delete_on_next_race":
            return
        if self.trigger == "race" and not is_race:
            return
        with self._lock:
            for s in self.store.list():
                if s["id"] == new_session_id or s["keep_raw"] or not self._deletable_bytes(s):
                    continue
                self.pending.setdefault(s["id"], now + self.grace_s)

    def keep(self, session_id: str, keep: bool = True) -> bool:
        path = self.store.path_of(session_id)
        if not path:
            return False
        marker = os.path.join(path, KEEP_MARKER)
        with self._lock:
            if keep:
                with open(marker, "w") as f:
                    f.write(datetime.now().isoformat())
                self.pending.pop(session_id, None)
            elif os.path.exists(marker):
                os.remove(marker)
        return True

    def keep_all_pending(self) -> None:
        for sid in list(self.pending):
            self.keep(sid, True)

    def _deletable_bytes(self, s: dict) -> int:
        return s["raw_bytes"] + (s.get("incident_bytes", 0) if self.incidents else 0)

    def delete_now(self, session_id: str) -> bool:
        path = self.store.path_of(session_id)
        if not path:
            return False
        size = 0
        raw = os.path.join(path, RAW_NAME)
        if os.path.exists(raw):
            size += os.path.getsize(raw)
            os.remove(raw)
        inc = os.path.join(path, "incidents")
        if self.incidents and os.path.isdir(inc):
            for name in os.listdir(inc):
                p = os.path.join(inc, name)
                if os.path.isdir(p):
                    size += dir_size(p)
                    shutil.rmtree(p, ignore_errors=True)
        if size:
            self.deleted.append({"id": session_id, "bytes": size, "at": time.time()})
        with self._lock:
            self.pending.pop(session_id, None)
        return True

    def tick(self, now: float) -> None:
        with self._lock:
            due = [sid for sid, t in self.pending.items() if t <= now]
        for sid in due:
            path = self.store.path_of(sid)
            if path and os.path.exists(os.path.join(path, KEEP_MARKER)):
                with self._lock:
                    self.pending.pop(sid, None)
                continue
            self.delete_now(sid)

    def status(self, now: float) -> dict:
        with self._lock:
            pend = [{"id": sid, "delete_in_s": max(0, round(t - now))} for sid, t in self.pending.items()]
        for p in pend:
            path = self.store.path_of(p["id"])
            raw = os.path.join(path, RAW_NAME) if path else None
            p["raw_bytes"] = os.path.getsize(raw) if raw and os.path.exists(raw) else 0
            p["incident_bytes"] = dir_size(os.path.join(path, "incidents")) if (path and self.incidents) else 0
        return {"mode": self.mode, "incidents": self.incidents, "pending": pend, "recently_deleted": self.deleted[-5:]}


def free_disk_bytes(path: str) -> int:
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return 0
