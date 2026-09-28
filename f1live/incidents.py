"""Incident capture: keeps a rolling window of raw packets and, when something happens
(collision, spin, heavy damage, penalty, retirement, or a manual flag), saves the data
around it as a self-contained package for later review:

    incidents/<nnn>_<lap>_<kind>_<drivers>/
        incident.json.gz   metadata, context at the time, per-car traces, relative geometry
        packets.f1raw.gz   every raw packet in the window (re-decodable, replayable)
        prompt.md          ready-to-paste brief for an LLM steward, with 2020 rules excerpts
"""

from __future__ import annotations

import copy
import gzip
import json
import math
import os
import queue
import re
import threading
from collections import deque

from . import lookups as L
from . import packets as P
from .model import Session, fmt_clock, fmt_time
from .rules import RULES_2020
from .storage import RawWriter, write_json


class Incident:
    def __init__(self, n: int, wall: float, pre: float, post: float, events: list[dict], cars: list[int],
                 packets: list, context: dict):
        self.n = n
        self.id = f"{n:03d}"
        self.trigger_wall = wall
        self.start = wall - pre
        self.end = wall + post
        self.events = events
        self.cars = cars
        self.packets = packets
        self.context = context
        self.status = "recording"
        self.last_st = events[0]["st"]
        self.path: str | None = None
        self.title = ""


class IncidentRecorder:
    def __init__(self, cfg: dict):
        icfg = cfg.get("incidents", {})
        self.pre = float(icfg.get("pre_seconds", 20))
        self.post = float(icfg.get("post_seconds", 10))
        self.max_len = float(icfg.get("max_seconds", 90))
        self.kinds = set(icfg.get("capture", ["collision", "spin", "damage", "penalty", "retirement", "manual"]))
        self.include_ai_only = bool(icfg.get("include_ai_only", False))
        self.ring: deque = deque()
        self.active: list[Incident] = []
        self.done: list[dict] = []
        self._n = 0
        self._q: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._work, name="incident-writer", daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------ live
    def feed(self, wall: float, key: str, payload: bytes) -> None:
        item = (wall, key, payload)
        self.ring.append(item)
        cutoff = wall - self.pre - 1.0
        while self.ring and self.ring[0][0] < cutoff:
            self.ring.popleft()
        for inc in self.active:
            inc.packets.append(item)

    def wants(self, s: Session, ev: dict) -> bool:
        if ev["kind"] == "manual":
            return True
        if not ev.get("capture") or ev["kind"] not in self.kinds:
            return False
        return self.include_ai_only or any(s.cars[i].human for i in ev["cars"] if 0 <= i < 22)

    def trigger(self, s: Session, ev: dict) -> Incident | None:
        if not self.wants(s, ev):
            return None
        wall = ev["wall"]
        cars = [i for i in ev["cars"] if 0 <= i < 22] or [c.idx for c in s.humans()]
        for inc in self.active:
            # merge only if it is the same moment in the race (session time), not just close in arrival time
            same_moment = ev["st"] - inc.last_st <= self.post + 2.0
            if same_moment and wall <= inc.end + 2.0 and (set(cars) & set(inc.cars) or ev["kind"] == "manual"):
                inc.events.append(ev)
                inc.last_st = max(inc.last_st, ev["st"])
                for i in cars:
                    if i not in inc.cars:
                        inc.cars.append(i)
                inc.end = min(max(inc.end, wall + self.post), inc.start + self.max_len)
                ev["incident"] = inc.id
                return inc
        self._n += 1
        inc = Incident(self._n, wall, self.pre, self.post, [ev], list(cars), list(self.ring),
                       context_snapshot(s, cars, wall))
        inc.title = ev["text"]
        ev["incident"] = inc.id
        self.active.append(inc)
        self.done.append(self.describe(inc))
        return inc

    def tick(self, s: Session, wall: float) -> None:
        for inc in [i for i in self.active if wall >= i.end]:
            self._finish(s, inc)

    def flush(self, s: Session) -> None:
        for inc in list(self.active):
            self._finish(s, inc)
        self._q.join()

    def _finish(self, s: Session, inc: Incident) -> None:
        self.active.remove(inc)
        inc.context["after"] = context_snapshot(s, inc.cars, inc.end)
        inc.status = "writing"
        self._update(inc)
        if s.path:
            self._q.put((inc, s.path))
        else:
            inc.status = "discarded (no session folder)"
            self._update(inc)

    def describe(self, inc: Incident) -> dict:
        return {"id": inc.id, "title": inc.title, "wall": inc.trigger_wall, "cars": list(inc.cars),
                "events": [e["id"] for e in inc.events], "status": inc.status,
                "lap": inc.events[0].get("lap"), "clock": inc.events[0].get("clock"),
                "kind": inc.events[0]["kind"], "path": inc.path}

    def _update(self, inc: Incident) -> None:
        for i, d in enumerate(self.done):
            if d["id"] == inc.id:
                self.done[i] = self.describe(inc)

    def _work(self) -> None:
        while True:
            inc, session_path = self._q.get()
            try:
                write_package(inc, session_path)
                inc.status = "saved"
            except Exception as e:  # keep recording even if one package fails
                inc.status = f"failed: {e}"
            finally:
                inc.packets = []
                self._update(inc)
                self._q.task_done()


# ---------------------------------------------------------------------------- context
def context_snapshot(s: Session, cars: list[int], wall: float) -> dict:
    """Everything the package needs from the live model, copied at trigger time."""
    focus = [s.cars[i] for i in cars if 0 <= i < 22]
    near: list[int] = []
    if focus and s.primary in s.last_positions:
        pts = s.last_positions[s.primary][1]
        for c in s.cars:
            if c.active and c.idx not in cars:
                if any(math.hypot(pts[c.idx][0] - pts[f.idx][0], pts[c.idx][1] - pts[f.idx][1]) < 60 for f in focus):
                    near.append(c.idx)
    gaps = s.gaps()
    lapd = focus[0].lap.get("lapDistance", 0.0) if focus else 0.0
    corner = s.corner(lapd)
    seg = lambda tr, step: {k: v for k, v in tr.items() if abs(k * step - lapd) <= 500}
    cars_ctx = {}
    for idx in list(cars) + near:
        c = s.cars[idx]
        cs = s.corner_stats(c)
        cars_ctx[idx] = {
            "idx": idx, "name": s.display_name(idx), "team": c.team, "human": c.human, "own_data": c.own,
            "involved": idx in cars, "position": c.position, "lap": c.lap_num,
            "lap_distance": round(c.lap.get("lapDistance", 0.0), 1),
            "total_distance": round(c.lap.get("totalDistance", 0.0), 1),
            "gap_to_leader": gaps.get(idx, {}).get("gap"), "interval": gaps.get(idx, {}).get("interval"),
            "laps_down": gaps.get(idx, {}).get("laps_down"),
            "penalties_s": c.lap.get("penalties", 0), "warnings": c.warnings,
            "tyre": L.tyre_label(c.status.get("actualTyreCompound"), c.status.get("visualTyreCompound")),
            "tyre_age": c.status.get("tyresAgeLaps"), "tyre_wear": list(c.status.get("tyresWear", [])),
            "damage": {k: c.status.get(k, 0) for k in ("frontLeftWingDamage", "frontRightWingDamage",
                                                         "rearWingDamage", "engineDamage", "gearBoxDamage")},
            "fia_flag": L.FIA_FLAGS.get(c.status.get("vehicleFiaFlags", 0)),
            "pit_status": c.lap.get("pitStatus", 0), "last_lap": c.laps[-1]["time"] if c.laps else None,
            "best_lap": c.best_lap(),
            "history": list(c.hist)[-150:],
            "prev_lap_trace": seg(c.prev_trace, 5),
            "typical_corner": cs.get(corner),
        }
    return copy.deepcopy({
        "wall": wall, "session_time": s.st, "race_clock": s.race_clock(), "track": s.track_name,
        "track_id": s.track_id, "track_length": s.track_length, "session_type": s.type_name,
        "total_laps": s.total_laps, "leader_lap": s.leader_lap(),
        "safety_car": L.SAFETY_CAR.get(s.sc_status), "weather": L.WEATHER.get(s.weather),
        "corner": corner, "cars": cars_ctx, "near": near,
        "authority": {i: s.auth[i] for i in list(cars) + near},
        "ref_line": seg(s.ref_line, 5), "race_start_st": s.race_start_st,
    })


# ---------------------------------------------------------------------------- package
def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")[:48]


def build_traces(inc: Incident) -> dict:
    ctx = inc.context
    wanted = set(ctx["cars"].keys())
    auth = ctx["authority"]
    latest: dict[str, dict] = {}   # per source: latest decoded lap/status/motion
    rows: dict[int, list] = {i: [] for i in wanted}
    geometry: list[dict] = []
    involved = [i for i in inc.cars if i in wanted]
    pairs = [(a, b) for k, a in enumerate(involved) for b in involved[k + 1:]]
    for wall, key, payload in inc.packets:
        try:
            p = P.decode(payload)
        except P.DecodeError:
            continue
        pid = p["id"]
        st = p["header"]["sessionTime"]
        cache = latest.setdefault(key, {})
        if pid == P.LAP:
            cache["lap"] = p["lapData"]
        elif pid == P.STATUS:
            cache["status"] = p["carStatusData"]
        elif pid == P.MOTION:
            cache["motion"] = p["carMotionData"]
            cm = p["carMotionData"]
            for a, b in pairs:
                if auth.get(a) != key:
                    continue
                ma, mb = cm[a], cm[b]
                dx, dz = mb["worldPositionX"] - ma["worldPositionX"], mb["worldPositionZ"] - ma["worldPositionZ"]
                y = ma["yaw"]
                geometry.append({"st": round(st, 2), "a": a, "b": b,
                                 "b_ahead_m": round(dx * math.sin(y) + dz * math.cos(y), 2),
                                 "b_right_m": round(-dx * math.cos(y) + dz * math.sin(y), 2),
                                 "dist_m": round(math.hypot(dx, dz), 2)})
        elif pid == P.TELEMETRY:
            for i in wanted:
                if auth.get(i) != key:
                    continue
                t = p["carTelemetryData"][i]
                ld = (cache.get("lap") or [{}] * 22)[i]
                cs = (cache.get("status") or [{}] * 22)[i]
                m = (cache.get("motion") or [{}] * 22)[i]
                slip = None
                if m:
                    vx, vz = m.get("worldVelocityX", 0), m.get("worldVelocityZ", 0)
                    if math.hypot(vx, vz) > 1.5:
                        ang = math.atan2(vx, vz) - m.get("yaw", 0)
                        slip = round(math.degrees(math.atan2(math.sin(ang), math.cos(ang))), 1)
                rows[i].append({
                    "st": round(st, 2), "lap": ld.get("currentLapNum"), "d": round(ld.get("lapDistance", 0.0), 1),
                    "pos": ld.get("carPosition"), "speed": t["speed"], "thr": round(t["throttle"], 3),
                    "brk": round(t["brake"], 3), "steer": round(t["steer"], 3), "gear": t["gear"], "drs": t["drs"],
                    "off_wheels": sum(1 for x in t["surfaceType"] if x in L.OFF_SURFACES),
                    "x": round(m.get("worldPositionX", 0.0), 2), "z": round(m.get("worldPositionZ", 0.0), 2),
                    "yaw": round(m.get("yaw", 0.0), 4), "glat": round(m.get("gForceLateral", 0.0), 2),
                    "glon": round(m.get("gForceLongitudinal", 0.0), 2), "slip_deg": slip,
                    "lat_m": _lateral(ctx.get("ref_line") or {}, m.get("worldPositionX"), m.get("worldPositionZ")),
                    "fl": cs.get("frontLeftWingDamage"), "fr": cs.get("frontRightWingDamage"),
                    "rw": cs.get("rearWingDamage"), "flag": L.FIA_FLAGS.get(cs.get("vehicleFiaFlags", 0)),
                    "pit": ld.get("pitStatus"), "pen": ld.get("penalties"),
                })
    trig = inc.events[0]["st"]
    for i in wanted - set(involved):   # nearby cars: only the seconds around the trigger
        rows[i] = [r for r in rows[i] if abs(r["st"] - trig) <= 6]
    return {"traces": {str(k): v for k, v in rows.items()}, "geometry": geometry}


def _lateral(ref: dict, x: float | None, z: float | None) -> float | None:
    """Signed distance (m) from the session's reference racing line; + = left of the line."""
    if not ref or x is None:
        return None
    keys = sorted(ref)
    best, bd = None, 1e18
    for k in keys:
        rx, rz = ref[k]
        d = (rx - x) ** 2 + (rz - z) ** 2
        if d < bd:
            best, bd = k, d
    if best is None or bd > 40 ** 2:
        return None
    i = keys.index(best)
    k0, k1 = keys[max(0, i - 1)], keys[min(len(keys) - 1, i + 1)]
    tx, tz = ref[k1][0] - ref[k0][0], ref[k1][1] - ref[k0][1]
    n = math.hypot(tx, tz)
    if n == 0:
        return None
    tx, tz = tx / n, tz / n
    dx, dz = x - ref[best][0], z - ref[best][1]
    return round(dx * tz - dz * tx, 2)


def write_package(inc: Incident, session_path: str) -> None:
    first = inc.events[0]
    names = "_".join(_slug(n) for n in first["names"][:2]) or "field"
    folder = f"{inc.id}_lap{first.get('lap') or 0}_{first['kind']}_{names}"
    path = os.path.join(session_path, "incidents", folder)
    os.makedirs(path, exist_ok=True)
    inc.path = path
    # Only the games that are authoritative for the cars in the package (their own feed, or the
    # primary source that everyone else is read from). Other players' copies are redundant.
    relevant = set(inc.context.get("authority", {}).values())
    raw = RawWriter(os.path.join(path, "packets.f1raw.gz"), level=6)
    for wall, key, payload in inc.packets:
        if not relevant or key in relevant:
            raw.write(wall, key, payload)
    raw.close()
    data = build_traces(inc)
    meta = {
        "id": inc.id, "title": inc.title, "trigger_wall": inc.trigger_wall,
        "window": [inc.start, inc.end], "cars": inc.cars,
        "events": inc.events, "context": inc.context, **data,
    }
    with gzip.open(os.path.join(path, "incident.json.gz"), "wt", encoding="utf-8", compresslevel=6) as f:
        json.dump(meta, f, default=str)
    ctx = inc.context
    write_json(os.path.join(path, "meta.json"), {
        "id": inc.id, "title": inc.title, "kind": first["kind"], "lap": first.get("lap"),
        "clock": first.get("clock"), "corner": ctx.get("corner"), "wall": inc.trigger_wall,
        "drivers": [ctx["cars"][i]["name"] for i in inc.cars if i in ctx["cars"]],
        "events": [{"clock": e.get("clock"), "text": e["text"], "names": e["names"]} for e in inc.events],
        "seconds": round(inc.end - inc.start, 1),
    })
    with open(os.path.join(path, "prompt.md"), "w", encoding="utf-8") as f:
        f.write(build_prompt(meta))


# ---------------------------------------------------------------------------- prompt
def _opt(v, spec: str) -> str:
    return "" if v is None else format(v, spec)


def _clock(ctx: dict, st: float) -> str:
    rs = ctx.get("race_start_st")
    return fmt_clock(st - rs) if rs is not None else f"t={st:.1f}s"


def build_prompt(meta: dict) -> str:
    ctx = meta["context"]
    cars = ctx["cars"]
    involved = [str(i) for i in meta["cars"] if str(i) in cars or i in cars]
    cget = lambda i: cars.get(i) or cars.get(int(i)) or cars.get(str(i))
    trig_st = meta["events"][0]["st"]
    out: list[str] = []
    w = out.append
    w(f"# Incident {meta['id']}: {meta['title']}\n")
    w("You are acting as an FIA Formula One steward reviewing an incident from an online F1 2020 race. "
      "Use only the telemetry below. Establish the facts, decide who (if anyone) is wholly or predominantly "
      "to blame or whether it is a racing incident, cite the applicable 2020 regulations, and recommend the "
      "penalty a 2020 panel would typically give. State any limits of the data that affect your conclusion.\n")
    w("## Session\n")
    w(f"- Track: {ctx['track']} ({ctx['track_length']:.0f} m) · {ctx['session_type']} · lap {ctx['leader_lap']}/{ctx['total_laps']} (leader)")
    w(f"- Race clock at trigger: {_clock(ctx, trig_st)} · location: {ctx['corner']}")
    w(f"- Safety car: {ctx['safety_car'] or 'none'} · weather: {ctx['weather']}")
    w("")
    w("## What the telemetry flagged\n")
    for e in meta["events"]:
        w(f"- {_clock(ctx, e['st'])} · {', '.join(e['names'])}: {e['text']}")
    w("")
    w("## Cars\n")
    w("| Car | Driver | Team | Human | Data | Pos | Lap | Gap to leader | Penalties | Tyre | Wing dmg FL/FR/RW | Flag |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key in list(involved) + [str(i) for i in ctx.get("near", [])]:
        c = cget(key)
        if not c:
            continue
        d = c["damage"]
        gap = "leader" if c["position"] == 1 else (f"+{c['laps_down']} lap" if c.get("laps_down") else
                                                   (f"+{c['gap_to_leader']:.1f}s" if c.get("gap_to_leader") is not None else "-"))
        w(f"| #{c['idx']}{'' if c['involved'] else ' (nearby)'} | {c['name']} | {c['team']} | {'yes' if c['human'] else 'AI'} "
          f"| {'own game' if c['own_data'] else 'remote'} | P{c['position']} | {c['lap']} | {gap} | {c['penalties_s']}s, {c['warnings']} warn "
          f"| {c['tyre']} ({c['tyre_age']} laps) | {d['frontLeftWingDamage']}/{d['frontRightWingDamage']}/{d['rearWingDamage']}% | {c['fia_flag'] or '-'} |")
    after = ctx.get("after", {}).get("cars", {})
    if after:
        w("\nDamage after the incident window: " + "; ".join(
            f"{a['name']} FL/FR/RW {a['damage']['frontLeftWingDamage']}/{a['damage']['frontRightWingDamage']}/{a['damage']['rearWingDamage']}%"
            for k, a in after.items() if a.get("involved")))
    w("")
    # gap & flag history between the two main cars
    if len(involved) >= 2:
        a, b = cget(involved[0]), cget(involved[1])
        ha, hb = {round(h[0]): h for h in a["history"]}, {round(h[0]): h for h in b["history"]}
        common = sorted(set(ha) & set(hb))[-120:]
        if common:
            w(f"## Road gap history, last {len(common)} s ({b['name']} relative to {a['name']})\n")
            w("Positive = " + b["name"] + " ahead on the road. Flags are the FIA flag shown to each car.\n")
            w("| Race clock | Gap (m) | " + a["name"] + " lap / flag | " + b["name"] + " lap / flag |")
            w("|---|---|---|---|")
            L_ = ctx["track_length"] or 1
            for t in common[::5]:
                g = hb[t][1] - ha[t][1]
                g = (g + L_ / 2) % L_ - L_ / 2 if abs(g) > L_ / 2 else g
                fl = lambda h: L.FIA_FLAGS.get(h[4]) or "-"
                w(f"| {_clock(ctx, t)} | {g:+.0f} | {ha[t][3]} / {fl(ha[t])} | {hb[t][3]} / {fl(hb[t])} |")
            w("")
    # inputs around the trigger
    tr = meta["traces"]
    w("## Driver inputs around the trigger (every 0.2 s, from 6 s before to 3 s after)\n")
    w("Steer: −1 full left lock, +1 full right. Throttle/brake 0–1. Lateral = metres from the session's fastest "
      "racing line, + left. Off = wheels on grass/gravel.\n")
    for key in involved:
        c = cget(key)
        rows = [r for r in tr.get(str(key), []) if trig_st - 6 <= r["st"] <= trig_st + 3]
        if not rows:
            continue
        w(f"### {c['name']} ({'own game' if c['own_data'] else 'remote data'})\n")
        w("| Clock | Dist m | km/h | Thr | Brk | Steer | Gear | Lateral m | Slip ° | Off | Wing FL/FR |")
        w("|---|---|---|---|---|---|---|---|---|---|---|")
        last = -1e9
        for r in rows:
            if r["st"] - last < 0.19:
                continue
            last = r["st"]
            w(f"| {_clock(ctx, r['st'])} | {r['d']:.0f} | {r['speed']} | {r['thr']:.2f} | {r['brk']:.2f} | {r['steer']:+.2f} "
              f"| {r['gear']} | {_opt(r['lat_m'], '+.1f')} | {_opt(r['slip_deg'], '+.0f')} "
              f"| {r['off_wheels'] or ''} | {r['fl']}/{r['fr']} |")
        tc = c.get("typical_corner")
        brake_rows = [r for r in rows if r["brk"] > 0.5 and r["st"] <= trig_st + 1]
        this_bp = f"{brake_rows[0]['d']:.0f} m at {brake_rows[0]['speed']} km/h" if brake_rows else "no braking above 50% in this window"
        if tc and tc.get("brake_at") is not None:
            w(f"\nBraking at {ctx['corner']}: this time {this_bp}; usually {tc['brake_at']} m "
              f"(range {tc['brake_range'][0]}–{tc['brake_range'][1]} m over {tc['laps']} clean laps), usual minimum speed {tc['min_speed']} km/h.\n")
        else:
            w(f"\nBraking in this window: {this_bp}.\n")
    geo = [g for g in meta["geometry"] if trig_st - 4 <= g["st"] <= trig_st + 2]
    if geo:
        a, b = geo[0]["a"], geo[0]["b"]
        w(f"## Relative position of {cget(b)['name']} seen from {cget(a)['name']}'s car\n")
        w("Ahead = metres in front of the car's nose direction (car length is about 5.6 m, width about 2 m); right = metres to its right.\n")
        w("| Clock | Ahead m | Right m | Distance m |")
        w("|---|---|---|---|")
        last = -1e9
        for g in geo:
            if g["a"] != a or g["b"] != b or g["st"] - last < 0.19:
                continue
            last = g["st"]
            w(f"| {_clock(ctx, g['st'])} | {g['b_ahead_m']:+.1f} | {g['b_right_m']:+.1f} | {g['dist_m']:.1f} |")
        w("")
    w("## Data notes\n")
    w("- Telemetry is sampled at the game's UDP rate (typically 10–20 Hz); 'remote' cars are seen through another "
      "player's game and are network-smoothed and coarser than a driver's own feed.")
    w("- F1 2020 has no collision event; contacts are inferred from g-force spikes with another car within ~4.5 m, "
      "wing damage appearing with a car close by, or the game's own collision penalties.")
    w("- Full data: incident.json (all traces) and packets.f1raw.gz (raw packets) in this folder.\n")
    w(RULES_2020)
    return "\n".join(out)
