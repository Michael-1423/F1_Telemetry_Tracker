"""Live race model: merges every player's telemetry feed into one picture of the session
and runs the real-time detectors (mistakes, contacts, collisions, penalties, pit stops...).

Each player's game sends data for all 22 cars. For each car we trust the car's own game
when that player is streaming ("own" data, full resolution, includes wheel slip and setup);
otherwise we use one stable primary source ("remote" data, network-smoothed).
"""

from __future__ import annotations

import math
import statistics
import time
from collections import Counter, deque
from typing import Any, Callable

from . import engineer as ENG
from . import laps as LP
from . import lookups as L
from . import packets as P

DETECT_DEFAULTS: dict[str, float] = {
    "offtrack_min_wheels": 3,      # wheels on grass/gravel/etc. to count as an excursion
    "offtrack_min_s": 0.3,         # ...for at least this long
    "offtrack_major_s": 2.0,       # excursions this long (4 wheels) are major
    "track_limits_merge_s": 1.0,   # bounces closer together than this are one excursion
    "ers_deploy_w": 10000.0,       # ERS counts as deploying above this rate (watts)
    "slide_deg": 30.0,             # slip angle for a "big slide"
    "spin_deg": 90.0,              # slip angle for a spin
    "slide_min_kmh": 50.0,
    "reverse_deg": 150.0,          # car moving backwards relative to its heading
    "reverse_min_s": 0.6,
    "contact_dist_m": 4.5,         # centre-to-centre distance for a possible contact
    "contact_dglat": 2.5,          # g-force jump between frames (lateral)
    "contact_dglon": 3.0,          # g-force jump between frames (longitudinal)
    "collision_dg": 6.0,           # g-force jump that counts as a collision on its own
    "damage_min_pct": 3,           # wing/engine/gearbox damage increase to report
    "damage_collision_dist_m": 7.0,
    "blue_flag_warn_s": 8.0,       # human held under blue flags this long -> warning
    "slow_lap_factor": 1.04,       # lap slower than best clean lap x factor -> "slow lap"
    "clean_lap_factor": 1.07,
    "lockup_slip": -0.25,          # own-car only (wheel slip ratio)
    "wheelspin_slip": 0.25,
}

DAMAGE_PARTS = [("frontLeftWingDamage", "front-left wing"), ("frontRightWingDamage", "front-right wing"),
                ("rearWingDamage", "rear wing"), ("engineDamage", "engine"), ("gearBoxDamage", "gearbox")]


def fmt_time(s: float | None) -> str:
    if s is None or s <= 0 or s != s:
        return "-"
    m = int(s // 60)
    return f"{m}:{s - 60 * m:06.3f}" if m else f"{s:.3f}"


def fmt_clock(s: float | None) -> str:
    if s is None:
        return "-"
    s = max(0.0, s)
    return f"{int(s // 60)}:{s % 60:04.1f}"


class Source:
    """One game sending telemetry to us (one player's PC/console)."""

    def __init__(self, key: str, wall: float):
        self.key = key
        self.first_seen = wall
        self.last_seen = wall
        self.player_car: int | None = None
        self.counts: Counter = Counter()
        self.packets = 0
        self.bytes = 0
        self._rate_t = wall
        self._rate_n = 0
        self.rate = 0.0
        self.session_uid = 0
        self.ended = False

    def seen(self, wall: float, pid: int, size: int) -> None:
        self.last_seen = wall
        self.packets += 1
        self.bytes += size
        self.counts[pid] += 1
        self._rate_n += 1
        if wall - self._rate_t >= 2.0:
            self.rate = self._rate_n / (wall - self._rate_t)
            self._rate_t, self._rate_n = wall, 0

    def live(self, wall: float, timeout: float = 3.0) -> bool:
        return wall - self.last_seen < timeout


class Car:
    """Everything we know and accumulate about one car in one session."""

    def __init__(self, idx: int):
        self.idx = idx
        self.name = ""
        self.team = ""
        self.race_number = 0
        self.human = False
        self.your_telemetry = 1
        self.active = False
        self.own = False                  # data comes from this driver's own game
        self.lap: dict = {}
        self.tel: dict = {}
        self.status: dict = {}
        self.motion: dict = {}
        self.extra: dict = {}             # own-car only motion data (wheel slip, suspension)
        self.setup: dict = {}
        self.laps: list[dict] = []
        self.lap_positions: list[int] = []
        self.grid = 0
        self.pit_stops: list[dict] = []
        self.stints: list[dict] = []
        self.counts: Counter = Counter()
        self.offtrack_where: Counter = Counter()
        self.penalty_s = 0
        self.warnings = 0
        self.speed_trap = 0.0
        self.result = None
        self.finish_st: float | None = None
        self.hist: deque = deque(maxlen=4000)      # 1 Hz: (st, totalDist, pos, lap, fiaFlag, pit)
        self.trail: list[float] = []                # session time at each 10 m of total distance
        self.cur_trace: dict[int, tuple] = {}       # 5 m bin -> (speed, thr, brk, steer, gear)
        self.prev_trace: dict[int, tuple] = {}
        self.cur_onsets: list[tuple[float, int]] = []   # (lap distance, speed) where brake crossed 50%
        self.lap_traces: dict[int, tuple] = {}      # humans: lap -> (trace, onsets), for re-analysis
        self.cur_line: dict[int, tuple] = {}        # 5 m bin -> (x, z)
        self.tl_count = 0                           # all four wheels off the white lines
        self.tl_where: Counter = Counter()
        self.cur_tl = 0
        self.cur_deploy_bins: set[int] = set()
        self.deploy_until = -1e9
        self.ers_lap_deploy = 0.0
        self.ers_lap_harvest = 0.0
        self.cur_mix: Counter = Counter()
        self.assists: Counter = Counter()            # (TC, ABS, brake bias) samples while racing
        self.initial_pressures: list | None = None
        self._fuel_lap_start: float | None = None
        self._ers: tuple | None = None
        self.excursions: list[dict] = []     # 2+ wheels beyond the kerbs; later classed as corner cut / ran wide
        self.cur_exc: list[int] = []
        self._exc: dict | None = None
        self._exc_last_end = -1e9
        self._off1: tuple | None = None
        self.game_cuts = 0                   # the game's own corner-cutting warnings / invalidations
        self.runs: list[dict] = []           # each time the car leaves the pits (practice/quali runs)
        self.cur_traffic_s = 0.0             # time this lap within 1 s behind another car
        self._traffic_t: float | None = None
        self.traffic_now: dict | None = None
        self.cur_status: Counter = Counter()
        self.cur_start_pit = True
        # Practice/qualifying: the game only counts timed laps (an out lap, or a lap abandoned into the
        # pits, never moves currentLapNum), so we number every lap driven ourselves.
        self.count_laps = False
        self.cur_time_bins: dict[int, float] = {}   # 10 m bin -> current lap time (for the live delta)
        self.best_time_bins: dict[int, float] = {}
        self.best_valid_time: float | None = None
        self._prev_brk = 0.0
        self.cur_style = Counter()
        self.style = Counter()
        self.cur_lap_flags: set[str] = set()
        self.cur_lap_events: list[int] = []
        self.recent_contact: dict[int, float] = {}
        # detector state
        self._off: dict | None = None
        self._slide: dict | None = None
        self._rev: dict | None = None
        self._blue: dict | None = None
        self._pit: dict | None = None
        self._g: tuple | None = None
        self._lock_on = False
        self._last_spin_end = -1e9
        self._spin_on = False
        self._prev_steer: float | None = None
        self._last_hist = -1e9
        self._prev_lap: dict = {}
        self._prev_status: dict = {}

    def rebaseline(self) -> None:
        """Forget frame-to-frame detector state when this car's data source changes."""
        self.status = {}
        self._g = None
        self._off = None
        self._slide = None
        self._rev = None
        self._prev_steer = None

    # convenient views -------------------------------------------------------
    @property
    def lap_num(self) -> int:
        if self.count_laps:
            return len(self.laps) + 1
        return int(self.lap.get("currentLapNum", 0) or 0)

    @property
    def position(self) -> int:
        return int(self.lap.get("carPosition", 0) or 0)

    def best_lap(self) -> float | None:
        ts = [l["time"] for l in self.laps if l["time"] > 0 and l.get("timed", True) and not l.get("standing")]
        return min(ts) if ts else None

    def best_clean(self) -> float | None:
        ts = [l["time"] for l in self.laps if l.get("clean")]
        return min(ts) if ts else None

    def trail_time_at(self, total_distance: float) -> float | None:
        b = int(total_distance // 10)
        if 0 <= b < len(self.trail):
            t = self.trail[b]
            return t if t >= 0 else None
        return None


class Session:
    """One game session (practice, qualifying or race), identified by its sessionUID."""

    def __init__(self, uid: int, wall: float, cfg: dict, emit_hook: Callable[["Session", dict], None] | None = None):
        self.uid = uid
        self.cfg = cfg
        self.det = {**DETECT_DEFAULTS, **cfg.get("detect", {})}
        self.aliases: dict[str, str] = cfg.get("drivers", {}).get("aliases", {})
        self.custom_corners = {int(k): [tuple(x) for x in v] for k, v in cfg.get("corners", {}).items()}
        self.emit_hook = emit_hook
        self.started_wall = wall
        self.last_wall = wall
        self.sources: dict[str, Source] = {}
        self.cars = [Car(i) for i in range(22)]
        self.info: dict[str, Any] = {}
        self.track_id: int | None = None
        self.track_length = 0.0
        self.session_type = 0
        self.total_laps = 0
        self.sc_status = 0
        self.weather = 0
        self.paused = False
        self.st = 0.0
        self.race_start_st: float | None = None
        self._first_move_st: float | None = None
        self.num_active = 0
        self.events: list[dict] = []
        self._eid = 0
        self._recent_game_events: deque = deque(maxlen=200)
        self.final: dict | None = None
        self.auth: list[str | None] = [None] * 22
        self._auth_t = -1e9
        self.primary: str | None = None
        self.last_positions: dict[str, tuple[float, list]] = {}
        self.ref_line: dict[int, tuple] = {}
        self.ref_line_time = 1e9
        self.zones: list[dict] = []
        self.zones_auto = False
        self.ref_trace: dict[int, tuple] = {}
        self.ref_trace_time = 1e9
        self.zones_version = 0
        self.fastest_lap: tuple | None = None
        self.ended = False
        self.chequered = False
        self.path: str | None = None
        self._pair_events: dict[tuple, float] = {}
        self.engineers: dict[int, ENG.EngineerFeed] = {}   # car index -> race engineer, fed by that driver's own game
        self.engineer_th = cfg.get("engineer", {})

    # ------------------------------------------------------------------ helpers
    @property
    def is_race(self) -> bool:
        return self.session_type in L.RACE_SESSION_TYPES

    @property
    def kind(self) -> str:
        return L.SESSION_KIND.get(self.session_type, "practice")

    @property
    def track_name(self) -> str:
        return L.TRACKS.get(self.track_id, f"Track {self.track_id}") if self.track_id is not None else "Unknown track"

    @property
    def type_name(self) -> str:
        return L.SESSION_TYPES.get(self.session_type, "Session")

    def race_clock(self, st: float | None = None) -> float | None:
        st = self.st if st is None else st
        if self.race_start_st is None:
            return None
        return st - self.race_start_st

    def corner(self, lap_distance: float | None) -> str:
        if lap_distance is not None and self.zones_auto and self.zones:
            for z in self.zones:
                if z["start"] <= lap_distance < z["end"]:
                    return z["name"]
        return L.corner_name(self.track_id, lap_distance, self.track_length, self.custom_corners)

    def display_name(self, idx: int) -> str:
        c = self.cars[idx] if 0 <= idx < 22 else None
        if not c or not c.name:
            return f"Car {idx}"
        return self.aliases.get(c.name, c.name.title() if c.name.isupper() else c.name)

    def humans(self) -> list[Car]:
        return [c for c in self.cars if c.human and c.active]

    def leader_lap(self) -> int:
        laps = [c.lap_num for c in self.cars if c.active and c.position == 1]
        return laps[0] if laps else max((c.lap_num for c in self.cars if c.active), default=0)

    def emit(self, kind: str, severity: str, cars: list[int], text: str, st: float | None = None,
             wall: float | None = None, data: dict | None = None, capture: bool = False,
             lap_distance: float | None = None) -> dict:
        st = self.st if st is None else st
        if cars and severity in ("major", "minor") and not any(self.cars[i].human for i in cars if 0 <= i < 22):
            severity, capture = "ai", False
        self._eid += 1
        main = self.cars[cars[0]] if cars else None
        ev = {
            "id": self._eid, "wall": wall or self.last_wall, "st": round(st, 2),
            "clock": fmt_clock(self.race_clock(st)) if self.race_start_st is not None else None,
            "lap": main.lap_num if main else self.leader_lap(),
            "kind": kind, "severity": severity, "cars": cars,
            "names": [self.display_name(i) for i in cars],
            "humans": [i for i in cars if self.cars[i].human],
            "corner": self.corner(lap_distance if lap_distance is not None else (main.lap.get("lapDistance") if main else None)) if (main or lap_distance is not None) else None,
            "text": text, "data": data or {}, "capture": capture, "incident": None,
        }
        self.events.append(ev)
        for i in cars:
            if 0 <= i < 22:
                self.cars[i].cur_lap_events.append(ev["id"])
                if severity == "major":
                    self.cars[i].cur_lap_flags.add("incident")
        if self.emit_hook:
            self.emit_hook(self, ev)
        return ev

    def _pair_recent(self, key: tuple, st: float, window: float) -> bool:
        """True if `key` was reported within `window` seconds; otherwise records it now."""
        if self._pair_seen(key, st, window):
            return True
        self._pair_events[key] = st
        return False

    def _pair_seen(self, key: tuple, st: float, window: float) -> bool:
        last = self._pair_events.get(key)
        return last is not None and st - last < window

    # ----------------------------------------------------------------- sources
    def source(self, key: str, wall: float) -> Source:
        s = self.sources.get(key)
        if s is None:
            s = self.sources[key] = Source(key, wall)
            s.session_uid = self.uid
        return s

    def update_authority(self, wall: float, force: bool = False) -> None:
        if not force and wall - self._auth_t < 1.0:
            return
        self._auth_t = wall
        live = [s for s in self.sources.values() if s.live(wall)] or list(self.sources.values())
        if not live:
            return
        live.sort(key=lambda s: s.first_seen)
        # keep the current primary while it is alive, so the reference frame doesn't flap
        if not (self.primary and self.primary in self.sources and self.sources[self.primary].live(wall)):
            self.primary = live[0].key
        own = {s.player_car: s.key for s in live if s.player_car is not None and s.player_car < 22}
        for i in range(22):
            new = own.get(i, self.primary)
            if self.auth[i] is not None and new != self.auth[i]:
                self.cars[i].rebaseline()   # different game = different view; don't diff across it
            self.auth[i] = new
            self.cars[i].own = i in own

    # ------------------------------------------------------------------ ingest
    def handle(self, pkt: dict, key: str, wall: float) -> None:
        h = pkt["header"]
        pid = pkt["id"]
        src = self.source(key, wall)
        src.player_car = h["playerCarIndex"]
        self.last_wall = wall
        if self.primary is None or self._auth_t < 0:
            self.update_authority(wall, force=True)
        else:
            self.update_authority(wall)
        is_primary = key == self.primary
        if is_primary or self.primary is None:
            self.st = h["sessionTime"]
        pc = h["playerCarIndex"]
        if pc < 22 and pid in ENG.FEED_PACKETS:
            eng = self.engineers.get(pc)
            if eng is None:
                eng = self.engineers[pc] = ENG.EngineerFeed(self.engineer_th)
            eng.feed(pkt, pc)

        if pid == P.LAP:
            for i, ld in enumerate(pkt["lapData"]):
                if self.auth[i] == key:
                    c = self.cars[i]
                    if self._out(c):
                        c.lap = ld  # retired cars: keep state, no more detection
                        continue
                    self._on_lap(c, ld, h["sessionTime"], wall)
                    if c.human:
                        self._traffic(c, pkt["lapData"], h["sessionTime"])
            if is_primary:
                self._detect_race_start(h["sessionTime"])
        elif pid == P.TELEMETRY:
            for i, td in enumerate(pkt["carTelemetryData"]):
                if self.auth[i] == key:
                    if self._racing(self.cars[i]):
                        self._on_telemetry(self.cars[i], td, h["sessionTime"], wall)
                    else:
                        self.cars[i].tel = td
        elif pid == P.STATUS:
            for i, cs in enumerate(pkt["carStatusData"]):
                if self.auth[i] == key:
                    if self._racing(self.cars[i]):
                        self._on_status(self.cars[i], cs, key, h["sessionTime"], wall)
                    else:
                        self.cars[i].status = cs
        elif pid == P.MOTION:
            cm = pkt["carMotionData"]
            self.last_positions[key] = (h["sessionTime"], [(m["worldPositionX"], m["worldPositionZ"]) for m in cm])
            pc = h["playerCarIndex"]
            if 0 <= pc < 22:
                self.cars[pc].extra = {k: pkt[k] for k in ("wheelSlip", "wheelSpeed", "suspensionPosition",
                                                             "frontWheelsAngle", "localVelocityX")}
            for i, m in enumerate(cm):
                if self.auth[i] == key:
                    if self._racing(self.cars[i]):
                        self._on_motion(self.cars[i], m, cm, key, h["sessionTime"], wall)
                    else:
                        self.cars[i].motion = m
        elif pid == P.SESSION:
            if is_primary:
                self._on_session(pkt, wall)
        elif pid == P.PARTICIPANTS:
            if is_primary or not self.num_active:
                self._on_participants(pkt)
        elif pid == P.EVENT:
            self._on_event(pkt, src, wall)
        elif pid == P.SETUPS:
            pc = h["playerCarIndex"]
            if 0 <= pc < 22:
                self.cars[pc].setup = pkt["carSetups"][pc]
        elif pid == P.FINAL:
            self.final = pkt
            for i, cd in enumerate(pkt["classificationData"]):
                if i < 22 and cd.get("position"):
                    self.cars[i].result = cd

    def _racing(self, c: "Car") -> bool:
        """Detectors only run for cars still racing (not finished, retired or disqualified). In practice and
        qualifying they always run: the game marks a car "retired" after terminal damage (or "finished"
        part-way through) and keeps that status after it sends the car back out from the garage."""
        return self.kind != "race" or c.lap.get("resultStatus", 2) < 3 or not c.lap

    def _out(self, c: "Car") -> bool:
        """Out of the race for good (race sessions only; see _racing)."""
        return self.kind == "race" and c.lap.get("resultStatus", 2) in L.OUT_OF_RACE

    # ------------------------------------------------------------ packet types
    def _auto_zones(self) -> None:
        """No corner map for this track: detect corners from the fastest clean lap so far, then
        work out corner metrics for every lap already recorded."""
        zones = LP.auto_zones(self.ref_trace, self.track_length)
        if not zones:
            return
        self.zones, self.zones_auto = zones, True
        self.zones_version += 1
        for c in self.cars:
            for lap in c.laps:
                tr = c.lap_traces.get(lap["lap"])
                if tr:
                    lap["corners"] = LP.corner_metrics(tr[0], tr[1], zones)

    def _on_session(self, p: dict, wall: float) -> None:
        first = self.track_id is None
        self.track_id = p["trackId"]
        self.track_length = float(p["trackLength"])
        if first:
            self.zones = LP.builtin_zones(self.track_id, self.custom_corners)
            self.zones_version += 1
        self.session_type = p["sessionType"]
        self.total_laps = p["totalLaps"]
        for c in self.cars:
            c.count_laps = self.kind != "race"
        self.info = {k: p[k] for k in ("weather", "trackTemperature", "airTemperature", "totalLaps",
                                         "trackLength", "sessionType", "trackId", "sessionTimeLeft",
                                         "sessionDuration", "pitSpeedLimit", "networkGame", "formula")}
        self.info["forecast"] = p["weatherForecastSamples"][: p["numWeatherForecastSamples"]]
        self.info["marshalFlags"] = [z["zoneFlag"] for z in p["marshalZones"][: p["numMarshalZones"]]]
        if not first:
            if p["safetyCarStatus"] != self.sc_status:
                old, new = self.sc_status, p["safetyCarStatus"]
                if new:
                    self.emit("race_control", "race", [], f"{L.SAFETY_CAR[new]} deployed")
                    for c in self.cars:
                        c.cur_lap_flags.add("sc")
                else:
                    self.emit("race_control", "race", [], f"{L.SAFETY_CAR[old]} ending, green flag")
            if p["weather"] != self.weather:
                self.emit("race_control", "race", [], f"Weather now {L.WEATHER.get(p['weather'], p['weather'])}")
            if bool(p["gamePaused"]) != self.paused:
                self.emit("race_control", "info", [], "Game paused" if p["gamePaused"] else "Game resumed")
        self.sc_status = p["safetyCarStatus"]
        if self.sc_status:
            for c in self.cars:
                c.cur_lap_flags.add("sc")
        self.weather = p["weather"]
        self.paused = bool(p["gamePaused"])

    def _on_participants(self, p: dict) -> None:
        self.num_active = p["numActiveCars"]
        for i, pd in enumerate(p["participants"]):
            c = self.cars[i]
            if pd["name"]:
                if not c.name:
                    c.active = True
                c.name = pd["name"]
                c.team = L.TEAMS.get(pd["teamId"], f"Team {pd['teamId']}")
                c.race_number = pd["raceNumber"]
                c.human = not pd["aiControlled"]
                c.your_telemetry = pd["yourTelemetry"]

    def _detect_race_start(self, st: float) -> None:
        if self.race_start_st is not None or not self.is_race:
            return
        moving = [c for c in self.cars if c.active and (c.tel.get("speed") or 0) > 1 and c.lap_num <= 1]
        if moving and self._first_move_st is None:
            self._first_move_st = st
        elif not moving:
            self._first_move_st = None
        movers = [c for c in moving if (c.tel.get("speed") or 0) > 8]
        if len(movers) >= max(1, len([c for c in self.cars if c.active]) // 2):
            self.race_start_st = self._first_move_st if self._first_move_st is not None else st - 0.5
            self.emit("race_control", "race", [], "Lights out, race started", st=st)

    def _on_event(self, p: dict, src: Source, wall: float) -> None:
        code = p["eventStringCode"]
        det = p.get("eventDetails") or {}
        st = p["header"]["sessionTime"]
        if code in ("SSTA", "SEND"):
            if code == "SEND":
                src.ended = True
            return
        # every player's game reports the same event; keep the first copy
        sig = (code, tuple(sorted(det.items())))
        window = {"DRSE": 120.0, "DRSD": 120.0, "CHQF": 900.0, "RCWN": 900.0}.get(code, 5.0)
        for s, w in self._recent_game_events:
            if s == sig and abs(w - wall) < window:
                return
        self._recent_game_events.append((sig, wall))
        v = det.get("vehicleIdx")
        if code == "PENA":
            self._on_penalty_event(det, st, wall)
        elif code == "FTLP" and v is not None and v < 22:
            self.fastest_lap = (v, det.get("lapTime"))
            self.emit("fastest_lap", "info", [v], f"Fastest lap {fmt_time(det.get('lapTime'))}", st=st, wall=wall)
        elif code == "RTMT" and v is not None and v < 22:
            if self.kind == "race":
                self.emit("retirement", "major", [v], "Retired", st=st, wall=wall, capture=True)
            else:
                self.emit("race_control", "minor", [v], f"{self.display_name(v)}: terminal damage, back to the garage",
                          st=st, wall=wall)
        elif code == "RCWN" and v is not None and v < 22:
            self.emit("race_control", "race", [v], "Race winner", st=st, wall=wall)
        elif code == "CHQF":
            self.chequered = True
            self.emit("race_control", "race", [], "Chequered flag", st=st, wall=wall)
        elif code == "DRSE":
            self.emit("race_control", "race", [], "DRS enabled", st=st, wall=wall)
        elif code == "DRSD":
            self.emit("race_control", "race", [], "DRS disabled", st=st, wall=wall)
        elif code == "TMPT" and v is not None and v < 22:
            pass
        elif code == "SPTP" and v is not None and v < 22:
            c = self.cars[v]
            c.speed_trap = max(c.speed_trap, float(det.get("speed") or 0))

    def _on_penalty_event(self, d: dict, st: float, wall: float) -> None:
        v, o = d.get("vehicleIdx", 255), d.get("otherVehicleIdx", 255)
        if v is None or v >= 22:
            return
        pt, it = d.get("penaltyType"), d.get("infringementType")
        pname = L.PENALTY_TYPES.get(pt, f"penalty #{pt}")
        iname = L.INFRINGEMENTS.get(it, f"infringement #{it}")
        cars = [v] + ([o] if o is not None and o < 22 and o != v else [])
        c = self.cars[v]
        data = {"penaltyType": pt, "penalty": pname, "infringementType": it, "infringement": iname,
                "time": d.get("time"), "lapNum": d.get("lapNum"), "placesGained": d.get("placesGained")}
        if pt == 5:
            c.warnings += 1
        if it in L.CUT_INFRINGEMENTS:
            c.game_cuts += 1
        # lap invalidation reasons: annotate the lap-invalidated event we already raised
        if pt in (10, 11, 12, 13, 14, 15):
            for ev in reversed(self.events[-30:]):
                if ev["kind"] == "lap_invalid" and ev["cars"][:1] == [v] and st - ev["st"] < 4:
                    ev["text"] = f"Lap invalidated: {iname.replace('Lap invalidated: ', '')}"
                    ev["data"].update(data)
                    return
            self.emit("lap_invalid", "minor", [v], f"Lap invalidated: {iname}", st=st, wall=wall, data=data)
            return
        collision = it in L.COLLISION_INFRINGEMENTS
        if pt in L.SERIOUS_PENALTIES:
            what = pname + (f" +{d.get('time')}s" if pt == 4 and d.get("time") not in (None, 255) else "")
            text = f"{what}: {iname}"
            sev = "major"
        else:
            text = f"{pname}: {iname}"
            sev = "major" if collision else "minor"
        if len(cars) > 1:
            text += f" (involving {self.display_name(cars[1])})"
        self.emit("collision" if collision else "penalty", sev, cars, text, st=st, wall=wall, data=data,
                  capture=collision or pt in L.SERIOUS_PENALTIES)

    # ---------------------------------------------------------------- lap data
    def _on_lap(self, c: Car, ld: dict, st: float, wall: float) -> None:
        prev = c.lap
        c.lap = ld
        if ld.get("resultStatus", 0) >= 2 and not c.active and c.name:
            c.active = True
        if not prev:
            c.grid = ld.get("gridPosition", 0)
            c._prev_lap = ld
            c.cur_start_pit = bool(ld.get("pitStatus")) or ld.get("driverStatus") in (0, 3)
            self._trail(c, ld, st)
            return
        lapn, prevn = ld["currentLapNum"], prev.get("currentLapNum", 0)
        ds, pds = ld.get("driverStatus", 4), prev.get("driverStatus", 4)
        c.cur_status[ds] += 1
        if ds == 0 and pds != 0:
            # back to the garage: this lap will never be completed. In practice/qualifying keep it as an
            # untimed in-lap (often a flying lap abandoned after it was invalidated).
            if c.count_laps and len(c.cur_trace) * 5 > 0.1 * (self.track_length or 5000):
                self._close_lap(c, prev, ld, st, wall, untimed=float(prev.get("currentLapTime") or 0.0),
                                kind="out" if c.cur_start_pit else "in")
            self._abandon_lap(c)
        if prev.get("pitStatus", 0) > 0 and ld.get("pitStatus", 0) == 0:
            c.runs.append({"n": len(c.runs) + 1, "lap": c.lap_num, "st": round(st, 1),
                           "fuel_out": round(float(c.status.get("fuelInTank", 0) or 0), 2),
                           "tyre": L.tyre_label(c.status.get("actualTyreCompound"), c.status.get("visualTyreCompound")),
                           "tyre_age": c.status.get("tyresAgeLaps"), "mix": L.FUEL_MIX.get(c.status.get("fuelMix"))})
        if ld.get("lapDistance", -1) >= 0 and ld.get("currentLapTime"):
            c.cur_time_bins.setdefault(int(ld["lapDistance"] // 10), ld["currentLapTime"])
        if ld.get("pitStatus"):
            c.cur_lap_flags.add("pit")
        if lapn > prevn and prevn > 0:
            self._close_lap(c, prev, ld, st, wall)
        elif c.count_laps and lapn == prevn and ds != 0 and self._crossed_line(c, prev, ld):
            # crossed the line without the game counting a lap: the end of an out lap
            self._close_lap(c, prev, ld, st, wall, untimed=float(prev.get("currentLapTime") or 0.0))
        if ld["penalties"] > prev.get("penalties", 0):
            added = ld["penalties"] - prev.get("penalties", 0)
            c.penalty_s = ld["penalties"]
            self.emit("penalty", "major", [c.idx], f"+{added}s time penalty (total {ld['penalties']}s)",
                      st=st, wall=wall, data={"added": added, "total": ld["penalties"]}, capture=False)
        elif ld["penalties"] < prev.get("penalties", 0):
            c.penalty_s = ld["penalties"]
            self.emit("penalty", "info", [c.idx], f"Penalty served (remaining {ld['penalties']}s)", st=st, wall=wall)
        if ld["currentLapInvalid"] and not prev.get("currentLapInvalid") and ld.get("pitStatus", 0) == 0 and lapn == prevn:
            c.cur_lap_flags.add("invalid")
            self.emit("lap_invalid", "minor", [c.idx], f"Lap invalidated at {self.corner(ld['lapDistance'])}",
                      st=st, wall=wall, lap_distance=ld["lapDistance"])
        self._pit_logic(c, prev, ld, st, wall)
        rs, prs = ld.get("resultStatus", 0), prev.get("resultStatus", 0)
        if rs != prs:
            if rs in L.OUT_OF_RACE and self.kind == "race":
                self.emit("retirement", "major", [c.idx], L.RESULT_STATUS[rs].capitalize(), st=st, wall=wall,
                          capture=True)
            elif rs == 3 and (self.kind == "race" or c.finish_st is None):
                c.finish_st = st
                self.emit("finish", "info", [c.idx], f"Finished P{ld['carPosition']}", st=st, wall=wall)
        self._trail(c, ld, st)
        if st - c._last_hist >= 1.0:
            c._last_hist = st
            c.hist.append((round(st, 1), round(ld["totalDistance"], 1), ld["carPosition"], lapn,
                           c.status.get("vehicleFiaFlags", 0), ld.get("pitStatus", 0)))

    def _trail(self, c: Car, ld: dict, st: float) -> None:
        td = ld.get("totalDistance", 0.0)
        if td <= 0:
            return
        b = int(td // 10)
        tr = c.trail
        if b >= len(tr):
            tr.extend([-1.0] * (b + 1 - len(tr)))
        if tr[b] < 0:
            tr[b] = st

    def _crossed_line(self, c: Car, prev: dict, ld: dict) -> bool:
        """The car crossed the start/finish line: the lap timer restarted, or the lap distance wrapped,
        after covering most of a lap."""
        if len(c.cur_trace) * 5 < 0.5 * (self.track_length or 5000):
            return False
        t0, t1 = prev.get("currentLapTime") or 0.0, ld.get("currentLapTime") or 0.0
        if t0 > 20 and t1 < 5 and t0 - t1 > 15:
            return True
        tl = self.track_length or 0
        d0, d1 = prev.get("lapDistance", -1.0), ld.get("lapDistance", -1.0)
        return bool(tl and d0 > tl - 150 and 0 <= d1 < 150)

    def _close_lap(self, c: Car, prev: dict, ld: dict, st: float, wall: float,
                   untimed: float | None = None, kind: str | None = None) -> None:
        """Record the lap just completed. `untimed` is set for laps the game doesn't time (practice and
        qualifying out laps and in-laps): it holds our own measurement of the lap time."""
        timed = untimed is None
        lt = float(ld.get("lastLapTime") or 0.0) if timed else untimed
        s1 = (prev.get("sector1TimeInMS") or 0) / 1000 if timed else 0.0
        s2 = (prev.get("sector2TimeInMS") or 0) / 1000 if timed else 0.0
        s3 = lt - s1 - s2 if lt and s1 and s2 else 0.0
        flags = set(c.cur_lap_flags)
        if prev.get("currentLapInvalid") and timed:   # untimed laps: only if invalidated during this lap
            flags.add("invalid")
        game_lap = prev.get("currentLapNum", 0)
        lap = {
            "lap": c.lap_num if c.count_laps else game_lap, "game_lap": game_lap, "timed": timed,
            "standing": not c.count_laps and game_lap == 1,
            "time": round(lt, 3), "s1": round(s1, 3), "s2": round(s2, 3),
            "s3": round(s3, 3) if s3 > 0 else 0.0, "valid": "invalid" not in flags, "pit": "pit" in flags,
            "sc": "sc" in flags, "incident": "incident" in flags, "position": ld.get("carPosition"),
            "tyre": L.tyre_label(c.status.get("actualTyreCompound"), c.status.get("visualTyreCompound")),
            "tyre_age": c.status.get("tyresAgeLaps"), "events": list(c.cur_lap_events),
            "fuel": round(float(c.status.get("fuelInTank", 0) or 0), 2),
            "wear": list(c.status.get("tyresWear", [])),
        }
        wear = c.status.get("tyresWear") or []
        fuel_now = float(c.status.get("fuelInTank", 0) or 0)
        ended_in_pit = bool(ld.get("pitStatus")) or bool(prev.get("pitStatus")) or c.cur_status.get(2, 0) > c.cur_status.get(1, 0)
        if kind:
            lap["kind"] = kind
        elif c.count_laps and timed:        # the game never times an out lap
            lap["kind"] = "in" if ended_in_pit else "flying"
        else:
            lap["kind"] = "out" if c.cur_start_pit else ("in" if ended_in_pit else "flying")
        lap["fuel_start"] = round(c._fuel_lap_start, 2) if c._fuel_lap_start is not None else None
        lap["traffic_s"] = round(c.cur_traffic_s, 1)
        lap["traffic_pct"] = round(100 * c.cur_traffic_s / lt) if lt > 0 else None
        lap["excursions"] = list(c.cur_exc)
        lap["run"] = len(c.runs)
        vmax = max((v[0] for v in c.cur_trace.values()), default=None)
        lap.update({
            "wear_avg": round(sum(wear) / len(wear), 1) if wear else None,
            "fuel_used": round(c._fuel_lap_start - fuel_now, 2) if c._fuel_lap_start is not None else None,
            "mix": L.FUEL_MIX.get(c.cur_mix.most_common(1)[0][0]) if c.cur_mix else None,
            "ers_deployed_mj": round(c.ers_lap_deploy / 1e6, 2), "ers_harvested_mj": round(c.ers_lap_harvest / 1e6, 2),
            "deploy_zones": LP.segments(c.cur_deploy_bins), "track_limits": c.cur_tl,
            "avg_speed": round(self.track_length / lt * 3.6, 1) if (lt > 0 and self.track_length and timed and not lap["standing"]) else None,
            "vmax": vmax, "corners": LP.corner_metrics(c.cur_trace, c.cur_onsets, self.zones) if self.zones else {},
        })
        best_clean = c.best_clean()
        lap["clean"] = bool(lt > 0 and timed and not lap["standing"] and lap["valid"] and not lap["pit"] and not lap["sc"]
                            and not lap["incident"] and (best_clean is None or lt < best_clean * self.det["clean_lap_factor"]))
        # slow-lap detector
        ref = best_clean or c.best_lap()
        if (ref and timed and lt > ref * self.det["slow_lap_factor"] and not lap["standing"] and not lap["pit"] and not lap["sc"]
                and c.human and self._racing(c)):
            reasons = [e["text"] for e in self.events[-300:] if e["id"] in c.cur_lap_events and e["severity"] in ("major", "minor")]
            lost = lt - ref
            self.emit("slow_lap", "minor", [c.idx], f"Slow lap {lap['lap']}: {fmt_time(lt)} (+{lost:.1f}s)"
                      + (f" after {reasons[0].lower()}" if reasons else ""), st=st, wall=wall,
                      data={"lap": lap["lap"], "time": lt, "lost": round(lost, 2), "reasons": reasons[:4]})
        c.laps.append(lap)
        c.lap_positions.append(ld.get("carPosition", 0))
        if lap["valid"] and timed and lt > 0 and lap["kind"] != "out" and (c.best_valid_time is None or lt < c.best_valid_time):
            c.best_valid_time = lt
            c.best_time_bins = c.cur_time_bins
        if c.human:
            c.lap_traces[lap["lap"]] = (c.cur_trace, c.cur_onsets)
        if lap["clean"]:
            c.style.update(c.cur_style)
            if lt < self.ref_line_time and len(c.cur_line) > 50:
                self.ref_line_time = lt
                self.ref_line = dict(c.cur_line)
            if lt < self.ref_trace_time and len(c.cur_trace) > 200:
                self.ref_trace_time = lt
                self.ref_trace = dict(c.cur_trace)
                if not self.zones and self.track_length:
                    self._auto_zones()
        c.prev_trace = c.cur_trace
        c.cur_trace, c.cur_line, c.cur_onsets = {}, {}, []
        c.cur_tl = 0
        c.cur_deploy_bins = set()
        c.ers_lap_deploy = c.ers_lap_harvest = 0.0
        c.cur_mix = Counter()
        c._fuel_lap_start = fuel_now
        c.cur_style = Counter()
        c.cur_lap_flags = {"sc"} if self.sc_status else set()
        if ld.get("pitStatus"):
            c.cur_lap_flags.add("pit")
        c.cur_lap_events = []
        c.cur_exc = []
        c.cur_traffic_s = 0.0
        c.cur_status = Counter()
        c.cur_time_bins = {}
        # (the game may still say "out lap" for a moment after an out lap ends at the line)
        c.cur_start_pit = timed and (bool(ld.get("pitStatus")) or ld.get("driverStatus") == 3)

    def _abandon_lap(self, c: Car) -> None:
        """The driver returned to the garage mid-lap: drop what was collected for this lap."""
        c.cur_trace, c.cur_line, c.cur_onsets = {}, {}, []
        c.cur_tl = 0
        c.cur_deploy_bins = set()
        c.ers_lap_deploy = c.ers_lap_harvest = 0.0
        c.cur_mix = Counter()
        c.cur_style = Counter()
        c.cur_exc = []
        c.cur_traffic_s = 0.0
        c.cur_time_bins = {}
        c.cur_lap_flags = {"pit"}
        c.cur_start_pit = True
        c._exc = c._off = None
        c._fuel_lap_start = None

    def _traffic(self, c: Car, laps: list[dict], st: float) -> None:
        """Nearest car ahead on track, as a time gap. Time within 1 s counts as running in traffic."""
        ld = c.lap
        v = (c.tel.get("speed") or 0) / 3.6
        dt = st - c._traffic_t if c._traffic_t is not None else 0.0
        c._traffic_t = st
        d = ld.get("lapDistance", -1.0)
        if ld.get("pitStatus") or v < 15 or d < 0 or not self.track_length:
            c.traffic_now = None
            return
        best = None
        for j, o in enumerate(laps):
            if j == c.idx or not self.cars[j].active or o.get("pitStatus") or o.get("driverStatus") == 0 \
                    or (o.get("resultStatus", 2) != 2 if self.kind == "race" else o.get("resultStatus", 2) < 2):
                continue
            gap = (o.get("lapDistance", 0.0) - d) % self.track_length
            if gap > 0 and (best is None or gap < best[0]):
                best = (gap, j)
        gap_s = best[0] / v if best else None
        if gap_s is not None and gap_s < 1.0 and 0 < dt < 0.5:
            c.cur_traffic_s += dt
        c.traffic_now = {"gap_s": round(gap_s, 2), "car": best[1]} if (gap_s is not None and gap_s < 3.0) else None

    def live_delta(self, c: Car) -> float | None:
        """Current lap time minus the personal-best lap's time at the same point on the lap."""
        d, t = c.lap.get("lapDistance", -1.0), c.lap.get("currentLapTime")
        if not c.best_time_bins or d < 0 or not t or c.cur_start_pit:
            return None
        b = int(d // 10)
        for k in (b, b - 1, b + 1, b - 2):
            if k in c.best_time_bins:
                return round(t - c.best_time_bins[k], 3)
        return None

    LEFT_WHEELS, RIGHT_WHEELS = {0, 2}, {1, 3}   # wheel order in the game: RL, RR, FL, FR

    def _finish_excursion(self, c: Car, e: dict) -> None:
        if c._slide is not None or e["last"] - c._last_spin_end < 5.0:
            return                                      # part of a spin, reported as such
        steer = sum(e["steer"]) / len(e["steer"])
        turn = "left" if steer < -0.05 else "right" if steer > 0.05 else None
        nl, nr = len(e["first"] & self.LEFT_WHEELS), len(e["first"] & self.RIGHT_WHEELS)
        side = "left" if nl > nr else "right" if nr > nl else None
        kind = "cut" if (turn and side == turn) else "wide"
        all4 = len(e["union"]) == 4
        where = self.corner(e["d"])
        prev = c.excursions[-1] if c.excursions else None
        if prev and prev["lap"] == e["lap"] and e["st"] - c._exc_last_end < self.det["track_limits_merge_s"]:
            prev["dur"] = round(e["last"] - prev["st"] + 0.1, 2)   # a bounce: same excursion
            prev["wheels"] = max(prev["wheels"], e["max"])
            newly4 = all4 and not prev["all4"]
            prev["all4"] = prev["all4"] or all4
        else:
            prev = {"st": round(e["st"], 2), "lap": e["lap"], "d": round(e["d"], 1), "corner": where,
                    "kind": kind, "side": side, "turn": turn, "wheels": e["max"], "all4": all4,
                    "dur": round(e["last"] - e["st"] + 0.1, 2)}
            c.excursions.append(prev)
            c.cur_exc.append(len(c.excursions) - 1)
            newly4 = all4
        c._exc_last_end = e["last"]
        if newly4:
            c.tl_count += 1
            c.cur_tl += 1
            c.tl_where[prev["corner"]] += 1
            if c.human:
                self.emit("track_limits", "minor", [c.idx],
                          f"All four wheels off at {prev['corner']} ({'/'.join(sorted(e['surf']))}"
                          + (", cutting the corner" if prev["kind"] == "cut" else "") + f") · #{c.tl_count}",
                          st=e["st"], wall=e["wall"], lap_distance=e["d"], data={"kind": prev["kind"], "count": c.tl_count})

    def cut_counts(self, c: Car) -> dict:
        cuts = [e for e in c.excursions if e["kind"] == "cut"]
        return {"cuts": len(cuts), "wides": sum(1 for e in c.excursions if e["kind"] == "wide"),
                "cuts_where": dict(Counter(e["corner"] for e in cuts).most_common()), "game_cuts": c.game_cuts}

    def _pit_logic(self, c: Car, prev: dict, ld: dict, st: float, wall: float) -> None:
        ps, pps = ld.get("pitStatus", 0), prev.get("pitStatus", 0)
        if ps == pps:
            return
        if pps == 0 and ps > 0:
            c._pit = {"lap": c.lap_num, "entry_st": st, "box_st": None, "box_s": 0.0,
                      "tyre_before": L.tyre_label(c.status.get("actualTyreCompound"), c.status.get("visualTyreCompound")),
                      "age_before": c.status.get("tyresAgeLaps"),
                      "damage_before": {k: c.status.get(k, 0) for k, _ in DAMAGE_PARTS}}
        if c._pit is None:
            return
        if ps == 2 and pps != 2:
            c._pit["box_st"] = st
        if pps == 2 and ps != 2 and c._pit.get("box_st") is not None:
            c._pit["box_s"] += st - c._pit["box_st"]
            c._pit["box_st"] = None
        if ps == 0:
            pit = c._pit
            c._pit = None
            pit["lane_s"] = round(st - pit["entry_st"], 1)
            pit["box_s"] = round(pit["box_s"], 1)
            pit["tyre_after"] = L.tyre_label(c.status.get("actualTyreCompound"), c.status.get("visualTyreCompound"))
            pit["new_tyres"] = (c.status.get("tyresAgeLaps", 99) or 0) < (pit.get("age_before") or 0) or pit["tyre_after"] != pit["tyre_before"]
            repaired = [label for k, label in DAMAGE_PARTS
                        if (pit["damage_before"].get(k, 0) or 0) - (c.status.get(k, 0) or 0) >= 5]
            pit["repaired"] = repaired
            c.pit_stops.append(pit)
            parts = [f"pit lane {pit['lane_s']:.1f}s"]
            if pit["box_s"]:
                parts.append(f"stationary {pit['box_s']:.1f}s")
            if pit["new_tyres"]:
                parts.append(f"{pit['tyre_before']} → {pit['tyre_after']}")
            if repaired:
                parts.append("repaired " + ", ".join(repaired))
            self.emit("pit", "info", [c.idx], "Pit stop: " + ", ".join(parts), st=st, wall=wall, data=pit)

    # ------------------------------------------------------------- telemetry
    def _on_telemetry(self, c: Car, td: dict, st: float, wall: float) -> None:
        c.tel = td
        ld = c.lap
        lapd = ld.get("lapDistance", -1.0)
        speed = td["speed"]
        in_pit = bool(ld.get("pitStatus"))
        # ---- off-track excursions
        off = sum(1 for s in td["surfaceType"] if s in L.OFF_SURFACES)
        need = self.det["offtrack_min_wheels"]
        if off >= need and speed > 20 and not in_pit:
            if c._off is None:
                c._off = {"st": st, "wheels": off, "v0": speed, "vmin": speed, "d": lapd,
                          "surfaces": set(), "wall": wall}
            o = c._off
            o["wheels"] = max(o["wheels"], off)
            o["vmin"] = min(o["vmin"], speed)
            o["surfaces"].update(L.SURFACES.get(s, str(s)) for s in td["surfaceType"] if s in L.OFF_SURFACES)
        elif c._off is not None:
            o, c._off = c._off, None
            dur = st - o["st"]
            spinning = c._slide is not None or (st - c._last_spin_end < 5.0)
            if dur >= self.det["offtrack_min_s"] and not spinning:
                where = self.corner(o["d"])
                c.counts["offtrack_4w" if o["wheels"] >= 4 else "offtrack_3w"] += 1
                c.offtrack_where[where] += 1
                loss = o["v0"] - o["vmin"]
                major = o["wheels"] >= 4 and (dur >= self.det["offtrack_major_s"] or loss >= 60)
                if major:   # short excursions are counted as track limits below, not reported one by one
                    self.emit("offtrack", "major", [c.idx],
                              f"Off track at {where} ({'/'.join(sorted(o['surfaces']))}, {dur:.1f}s"
                              + (f", -{loss:.0f} km/h" if loss >= 15 else "") + ")",
                              st=o["st"], wall=o["wall"], lap_distance=o["d"],
                              data={"wheels": o["wheels"], "duration": round(dur, 2), "speed_loss": loss})
        # ---- excursions beyond the kerbs (grass, gravel, run-off). A wheel on the kerb is on the track.
        # One excursion = samples with 2+ wheels off, allowing 0.3 s gaps (at 10 Hz a car crossing the
        # inside of a hairpin shows front wheels off in one sample and rear wheels in the next).
        #   all four wheels went off during it  -> counts towards "four wheels off" (track limits)
        #   inside wheels left first            -> corner cut; outside wheels first -> ran wide
        off_w = [i for i, sf in enumerate(td["surfaceType"]) if sf not in (0, 1)]
        e = c._exc
        if e is not None and (len(off_w) < 2 or not (speed > 30 and not in_pit)) and st - e["last"] > 0.3:
            self._finish_excursion(c, e)
            c._exc = e = None
        if len(off_w) == 1:
            c._off1 = (st, set(off_w))     # the wheel that led the way off, if it went first on its own
        if len(off_w) >= 2 and speed > 30 and not in_pit:
            if e is None:
                first = c._off1[1] if (c._off1 and st - c._off1[0] <= 0.3) else set(off_w)
                c._exc = e = {"st": st, "last": st, "wall": wall, "lap": c.lap_num, "d": lapd, "first": first,
                              "union": set(), "max": 0, "steer": [], "surf": set()}
            e["last"] = st
            e["union"].update(off_w)
            e["max"] = max(e["max"], len(off_w))
            e["steer"].append(td["steer"])
            e["surf"].update(L.SURFACES.get(td["surfaceType"][i], "?") for i in off_w)
        # ---- braking points (exact distance where the pedal crosses 50%)
        brk_now = td["brake"]
        if brk_now > 0.5 >= c._prev_brk and not in_pit and lapd >= 0:
            c.cur_onsets.append((round(lapd, 1), speed))
        c._prev_brk = brk_now
        if c.initial_pressures is None and speed < 5 and c.lap_num <= 1:
            c.initial_pressures = [round(x, 1) for x in td["tyresPressure"]]
        # ---- style accumulators (above 40 km/h, on track)
        if speed > 40 and not in_pit:
            thr, brk, steer = td["throttle"], td["brake"], td["steer"]
            s = c.cur_style
            s["n"] += 1
            if thr > 0.98:
                s["full_thr"] += 1
            elif thr > 0.05:
                s["part_thr"] += 1
            if thr < 0.05 and brk < 0.05:
                s["coast"] += 1
            if brk > 0.05:
                s["brk"] += 1
                s["brk_sum_pct"] += int(brk * 100)
                if brk < 0.9:
                    s["brk_part"] += 1
                if abs(steer) > 0.25:
                    s["trail"] += 1
            if abs(steer) > 0.95:
                s["full_lock"] += 1
            if c._prev_steer is not None and abs(steer - c._prev_steer) > 0.5:
                s["snaps"] += 1
            c._prev_steer = steer
        # ---- 5 m trace for this lap
        if lapd >= 0:
            c.cur_trace[int(lapd // LP.BIN)] = (speed, round(td["throttle"], 2), round(td["brake"], 2),
                                                round(td["steer"], 2), td["gear"])

    # ---------------------------------------------------------------- status
    def _on_status(self, c: Car, cs: dict, key: str, st: float, wall: float) -> None:
        prev = c.status
        c.status = cs
        # ERS: deploying while "deployed this lap" is rising
        dep = float(cs.get("ersDeployedThisLap", 0) or 0)
        if c._ers and st > c._ers[0] and dep >= c._ers[1]:
            if (dep - c._ers[1]) / (st - c._ers[0]) > self.det["ers_deploy_w"]:
                c.deploy_until = st + 0.4
                d = c.lap.get("lapDistance", -1.0)
                if d >= 0:
                    c.cur_deploy_bins.add(int(d // LP.BIN))
        c._ers = (st, dep)
        c.ers_lap_deploy = max(c.ers_lap_deploy, dep)
        c.ers_lap_harvest = max(c.ers_lap_harvest, float(cs.get("ersHarvestedThisLapMGUK", 0) or 0)
                                + float(cs.get("ersHarvestedThisLapMGUH", 0) or 0))
        c.cur_mix[cs.get("fuelMix", 1)] += 1
        c.assists[(cs.get("tractionControl"), cs.get("antiLockBrakes"), cs.get("frontBrakeBias"))] += 1
        if c._fuel_lap_start is None:
            c._fuel_lap_start = float(cs.get("fuelInTank", 0) or 0)
        if not prev:
            self._stint(c, cs)
            return
        # damage
        for k, label in DAMAGE_PARTS:
            before, now = prev.get(k, 0) or 0, cs.get(k, 0) or 0
            step = now - before
            if label in ("engine", "gearbox") and step < 10:
                continue   # these parts wear gradually over a race; only sudden jumps are damage
            if step >= self.det["damage_min_pct"] and not c.lap.get("pitStatus"):
                other, dist = self._nearest(c.idx, key)
                recent = [o for o, t in c.recent_contact.items() if st - t < 2.5]
                if other is not None and recent and other not in recent:
                    other = recent[0]
                elif other is None and recent:
                    other = recent[0]
                collision = other is not None and (dist is not None and dist <= self.det["damage_collision_dist_m"] or other in recent)
                cars = [c.idx] + ([other] if collision else [])
                txt = f"{label.capitalize()} damage {before}% → {now}%"
                if collision:
                    txt = f"Collision with {self.display_name(other)}: {txt.lower()}"
                    if self._pair_recent(("coll",) + tuple(sorted((c.idx, other))), st, 3.0):
                        # already reported this collision; just add the damage note
                        self.emit("damage", "minor", [c.idx], txt, st=st, wall=wall,
                                  data={"part": label, "before": before, "after": now})
                        continue
                    c.counts["collisions"] += 1
                    self.emit("collision", "major", cars, txt, st=st, wall=wall, capture=True,
                              data={"part": label, "before": before, "after": now,
                                    "distance_m": round(dist, 1) if dist is not None else None})
                else:
                    major = now - before >= 20 or now >= 50 or label in ("engine", "gearbox")
                    self.emit("damage", "major" if major else "minor", [c.idx], txt, st=st, wall=wall,
                              capture=major, data={"part": label, "before": before, "after": now})
        # blue flags (humans only)
        flag = cs.get("vehicleFiaFlags", 0)
        if c.human:
            if flag == 2:
                if c._blue is None:
                    c._blue = {"st": st, "warned": False}
                elif not c._blue["warned"] and st - c._blue["st"] >= self.det["blue_flag_warn_s"]:
                    c._blue["warned"] = True
                    c.counts["blue_flag_warnings"] += 1
                    self.emit("blue_flag", "minor", [c.idx],
                              f"Under blue flags for {self.det['blue_flag_warn_s']:.0f}s+ without letting the car behind by",
                              st=st, wall=wall)
            elif c._blue is not None:
                dur = st - c._blue["st"]
                if c._blue["warned"]:
                    self.emit("blue_flag", "info", [c.idx], f"Blue flag cleared after {dur:.0f}s", st=st, wall=wall,
                              data={"duration": round(dur, 1)})
                c._blue = None
        self._stint(c, cs)

    def _stint(self, c: Car, cs: dict) -> None:
        label = L.tyre_label(cs.get("actualTyreCompound"), cs.get("visualTyreCompound"))
        age = cs.get("tyresAgeLaps", 0) or 0
        cur = c.stints[-1] if c.stints else None
        if cur is None or cur["tyre"] != label or age < cur["age_end"]:
            c.stints.append({"tyre": label, "from_lap": max(1, c.lap_num), "age_start": age, "age_end": age,
                             "to_lap": max(1, c.lap_num)})
        else:
            cur["age_end"] = age
            cur["to_lap"] = max(cur["to_lap"], c.lap_num)

    def _nearest(self, idx: int, key: str) -> tuple[int | None, float | None]:
        pos = self.last_positions.get(key)
        if not pos:
            return None, None
        pts = pos[1]
        x, z = pts[idx]
        best, bd = None, 1e9
        for j, (xo, zo) in enumerate(pts):
            if j == idx or not self.cars[j].active:
                continue
            d = math.hypot(xo - x, zo - z)
            if d < bd:
                best, bd = j, d
        return best, (bd if best is not None else None)

    # ---------------------------------------------------------------- motion
    def _on_motion(self, c: Car, m: dict, cm: list, key: str, st: float, wall: float) -> None:
        c.motion = m
        ld = c.lap
        lapd = ld.get("lapDistance", -1.0)
        in_pit = bool(ld.get("pitStatus"))
        vx, vz = m["worldVelocityX"], m["worldVelocityZ"]
        v_kmh = math.hypot(vx, vz) * 3.6
        if lapd >= 0:
            c.cur_line[int(lapd // 5)] = (round(m["worldPositionX"], 2), round(m["worldPositionZ"], 2))
        slip = 0.0
        if v_kmh > 5:
            ang = math.atan2(vx, vz) - m["yaw"]
            slip = math.degrees(math.atan2(math.sin(ang), math.cos(ang)))
        a = abs(slip)
        # ---- slides and spins
        if not in_pit and v_kmh > self.det["slide_min_kmh"] and a > self.det["slide_deg"]:
            if c._slide is None:
                c._slide = {"st": st, "max": a, "v": v_kmh, "d": lapd, "wall": wall}
            c._slide["max"] = max(c._slide["max"], a)
        elif c._slide is not None and (a < 12 or v_kmh < 15):
            s, c._slide = c._slide, None
            c._last_spin_end = st
            where = self.corner(s["d"])
            if s["max"] >= self.det["spin_deg"]:
                c.counts["spins"] += 1
                self.emit("spin", "major", [c.idx], f"Spun at {where} ({s['max']:.0f}° at {s['v']:.0f} km/h)",
                          st=s["st"], wall=s["wall"], capture=True, lap_distance=s["d"],
                          data={"max_deg": round(s["max"]), "speed": round(s["v"])})
            else:
                c.counts["slides"] += 1
                self.emit("slide", "minor", [c.idx], f"Big slide at {where} ({s['max']:.0f}°)",
                          st=s["st"], wall=s["wall"], lap_distance=s["d"],
                          data={"max_deg": round(s["max"]), "speed": round(v_kmh)})
        # ---- reversing (e.g. missed pit entry, recovering from a spin)
        if 5 < v_kmh < 60 and a > self.det["reverse_deg"] and not in_pit:
            if c._rev is None:
                c._rev = {"st": st, "d": lapd, "wall": wall, "reported": False}
            elif not c._rev["reported"] and st - c._rev["st"] >= self.det["reverse_min_s"]:
                c._rev["reported"] = True
                c.counts["reversing"] += 1
                self.emit("reversing", "minor", [c.idx], f"Reversing at {self.corner(c._rev['d'])}",
                          st=c._rev["st"], wall=c._rev["wall"], lap_distance=c._rev["d"])
        elif a < 90:
            c._rev = None
        # ---- contact: g-force jump with another car close by (same packet = same reference frame)
        glat, glon = m["gForceLateral"], m["gForceLongitudinal"]
        g_prev = c._g
        c._g = (key, glat, glon, st)
        if g_prev and g_prev[0] == key and 0 < st - g_prev[3] < 0.3 and not in_pit:
            dlat, dlon = abs(glat - g_prev[1]), glon - g_prev[2]
            if dlat > self.det["contact_dglat"] or dlon > self.det["contact_dglon"]:
                x, z = m["worldPositionX"], m["worldPositionZ"]
                best, bd = None, 1e9
                for j, o in enumerate(cm):
                    if j == c.idx or not self.cars[j].active:
                        continue
                    d = math.hypot(o["worldPositionX"] - x, o["worldPositionZ"] - z)
                    if d < bd:
                        best, bd = j, d
                if best is not None and bd <= self.det["contact_dist_m"]:
                    c.recent_contact[best] = st
                    self.cars[best].recent_contact[c.idx] = st
                    pair = tuple(sorted((c.idx, best)))
                    big = max(dlat, abs(dlon)) >= self.det["collision_dg"]
                    if big:
                        if not self._pair_recent(("coll",) + pair, st, 3.0):
                            c.counts["collisions"] += 1
                            self.emit("collision", "major", [c.idx, best],
                                      f"Collision with {self.display_name(best)} at {self.corner(lapd)} ({max(dlat, abs(dlon)):.0f} g spike)",
                                      st=st, wall=wall, capture=True, lap_distance=lapd,
                                      data={"distance_m": round(bd, 1), "dg_lat": round(dlat, 1), "dg_lon": round(dlon, 1)})
                    elif not self._pair_seen(("coll",) + pair, st, 3.0) and not self._pair_recent(("contact",) + pair, st, 3.0):
                        c.counts["contacts"] += 1
                        self.emit("contact", "minor", [c.idx, best],
                                  f"Contact with {self.display_name(best)} at {self.corner(lapd)}",
                                  st=st, wall=wall, lap_distance=lapd,
                                  data={"distance_m": round(bd, 1), "dg_lat": round(dlat, 1), "dg_lon": round(dlon, 1)})
        # ---- lockups / wheelspin (own car only: wheel slip is only sent for the player's car)
        if c.own and c.extra and not in_pit and v_kmh > 50:
            ws = c.extra.get("wheelSlip") or [0, 0, 0, 0]   # wheel order: RL, RR, FL, FR
            brk, thr = c.tel.get("brake", 0), c.tel.get("throttle", 0)
            lock = min(ws) < self.det["lockup_slip"] and brk > 0.2
            spin = max(ws[:2]) > self.det["wheelspin_slip"] and thr > 0.3
            if lock and not c._lock_on:
                c.counts["lockups"] += 1
                c.cur_style["lockups"] += 1
            if spin and not c._spin_on:
                c.counts["wheelspin"] += 1
                c.cur_style["wheelspin"] += 1
            c._lock_on, c._spin_on = lock, spin

    # -------------------------------------------------------------- queries
    def gaps(self) -> dict[int, dict]:
        """Gap to the leader and to the car ahead, from each car's distance trail."""
        order = sorted([c for c in self.cars if c.active and c.position], key=lambda c: c.position)
        out: dict[int, dict] = {}
        if not order:
            return out
        leader = order[0]
        for k, c in enumerate(order):
            td = c.lap.get("totalDistance", 0.0)
            st_now = c.trail_time_at(td)
            g = {"gap": None, "interval": None, "laps_down": 0}
            if k == 0:
                out[c.idx] = g
                continue
            lt = leader.trail_time_at(td)
            if self.track_length:
                g["laps_down"] = int((leader.lap.get("totalDistance", 0) - td) // self.track_length)
            if lt is not None and st_now is not None:
                g["gap"] = round(st_now - lt, 2)
            ahead = order[k - 1]
            at = ahead.trail_time_at(td)
            if at is not None and st_now is not None:
                g["interval"] = round(st_now - at, 2)
            out[c.idx] = g
        return out

    def style_metrics(self, c: Car, live: bool = False) -> dict:
        s = c.style + (c.cur_style if live else Counter())
        n = s["n"]
        if not n:
            return {}
        brk = s["brk"] or 1
        return {
            "samples": n,
            "full_throttle_pct": round(100 * s["full_thr"] / n, 1),
            "partial_throttle_pct": round(100 * s["part_thr"] / n, 1),
            "coasting_pct": round(100 * s["coast"] / n, 1),
            "braking_pct": round(100 * s["brk"] / n, 1),
            "partial_brake_pct": round(100 * s["brk_part"] / brk, 1),
            "mean_brake_pct": round(s["brk_sum_pct"] / brk, 1),
            "trail_brake_pct": round(100 * s["trail"] / brk, 1),
            "full_lock_pct": round(100 * s["full_lock"] / n, 1),
            "steer_snaps_pct": round(100 * s["snaps"] / n, 2),
            "lockups": s["lockups"], "wheelspin": s["wheelspin"],
        }

    def corner_profile(self, c: Car) -> list[dict]:
        return LP.profile(c.laps, self.zones)

    def corner_stats(self, c: Car) -> dict:
        """Per-corner averages keyed by corner name (used by incident prompts)."""
        return {r["corner"]: {"min_speed": r["min_v"], "brake_at": r["brake_at"], "brake_range": r["brake_range"],
                              "laps": r["laps"], "entry_v": r["entry_v"]} for r in self.corner_profile(c)}

    def elapsed_quiet(self, wall: float) -> float:
        return wall - self.last_wall

    def all_sources_ended(self) -> bool:
        return bool(self.sources) and all(s.ended for s in self.sources.values())


def now() -> float:
    return time.time()
