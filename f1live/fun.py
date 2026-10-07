"""The lobby's running jokes and season table, worked out from the recorded sessions.

- Points: F1 2020 scoring summed over every competitive race in the data folder.
- Tags (Smooth Operator, Lawnmower, Mr Saturday, Photo Finish): earned in a competitive race and shown for
  the next two races, i.e. while that race is one of the last two finished.
- Days since the last Maldonado: a human's solo crash (heavy wing damage with no other car involved).
- Simply lovely: a Red Bull human on pole, or winning the race.
- We are checking: once every 2 races a Ferrari human in the pit lane gets 10 s of "We are checking".
- Hammer Time: the serious switch. While it is on none of the above is shown (points stay).

A "competitive" race has a final classification, at least two human drivers who finished it (an abandoned
lobby has everyone "did not finish") and at least COMPETITIVE_KM of racing (the game's 25% distance is about 65 km at Monaco and 75-80 km elsewhere; 5-lap races are 20-30 km).
"""

from __future__ import annotations

import datetime
import json
import os
import time

from .model import Session
from .storage import write_json
from .summary import _best_valid

POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
COMPETITIVE_KM = 60
TAG_RACES = 2
CHECKING_EVERY = 2            # races
CHECKING_SECONDS = 10.0
LAWNMOWER_OFFS = 15           # all four wheels off more than this many times in a race
PHOTO_FINISH_S = 0.1
STATE_FILE = "_pitwall.json"

TAGS = {
    "smooth": "Smooth Operator",
    "lawnmower": "Lawnmower",
    "mr_saturday": "Mr Saturday",
    "photo": "Photo Finish",
}


def is_solo_crash(ev: dict) -> bool:
    """Heavy wing damage on a human's car with nobody else involved (collisions are reported as such)."""
    return (ev.get("kind") == "damage" and ev.get("severity") == "major" and bool(ev.get("humans"))
            and "wing damage" in ev.get("text", "") and not ev.get("text", "").startswith("Collision"))


def _first_human(ev: dict) -> str:
    """An event's "humans" are car indices; its "names" line up with its "cars"."""
    return next((n for i, n in zip(ev.get("cars") or [], ev.get("names") or []) if i in ev["humans"]),
                (ev.get("names") or ["?"])[0])


def _red_bull(team: str | None) -> bool:
    return str(team or "").startswith("Red Bull")


class Fun:
    def __init__(self, base: str):
        self.base = base
        self.path = os.path.join(base, STATE_FILE)
        self.state = {"hammer": False, "checking_races_left": 1, "counted": []}
        try:
            with open(self.path, encoding="utf-8") as f:
                self.state.update(json.load(f))
        except (OSError, ValueError):
            pass
        self.races: list[dict] = []        # competitive races, oldest first
        self.qualis: list[dict] = []
        self.maldonado: dict | None = None
        self.checking: dict | None = None  # {idx, name, key, until_st} while it shows
        self.refresh()
        self._scan_maldonado()

    # ------------------------------------------------------------------ history
    def refresh(self) -> None:
        """Re-read the finished sessions' summaries (at start-up and when a session finishes)."""
        races, qualis = [], []
        for name in sorted(os.listdir(self.base)):
            p = os.path.join(self.base, name, "summary.json")
            if not os.path.isfile(p):
                continue
            try:
                with open(p, encoding="utf-8") as f:
                    sm = json.load(f)
            except (OSError, ValueError):
                continue
            se = sm.get("session") or {}
            rows = sm.get("classification") or []
            if se.get("kind") == "qualifying" and rows:
                top = rows[0]
                qualis.append({"id": name, "started": se.get("started") or 0, "track": se.get("track"),
                               "pole": {"name": top.get("name"), "team": top.get("team"), "human": top.get("human"),
                                        "time": top.get("best_lap")}})
            if se.get("kind") != "race" or not se.get("final_classification_received"):
                continue
            km = (se.get("total_laps") or 0) * (se.get("track_length") or 0) / 1000
            if sum(1 for r in rows if r.get("human") and r.get("status") == "finished") < 2 or km < COMPETITIVE_KM:
                continue
            tl = {d["name"]: d.get("track_limits") for d in sm.get("drivers") or []}
            races.append({"id": name, "started": se.get("started") or 0, "track": se.get("track"),
                          "fastest": (se.get("fastest_lap") or {}).get("name"), "rows": rows, "track_limits": tl})
        races.sort(key=lambda r: r["started"])
        qualis.sort(key=lambda q: q["started"])
        self.races, self.qualis = races, qualis

    def _scan_maldonado(self) -> None:
        last = None
        for name in os.listdir(self.base):
            p = os.path.join(self.base, name, "events.jsonl")
            if not os.path.isfile(p):
                continue
            track = name.split("_")[2].replace("-", " ") if name.count("_") >= 3 else ""
            try:
                with open(p, encoding="utf-8") as f:
                    for line in f:
                        if '"damage"' not in line:
                            continue
                        try:
                            ev = json.loads(line)
                        except ValueError:
                            continue
                        if is_solo_crash(ev) and (last is None or ev["wall"] > last["wall"]):
                            last = {"wall": ev["wall"], "name": _first_human(ev), "track": track}
            except OSError:
                continue
        self.maldonado = last

    def _save(self) -> None:
        try:
            write_json(self.path, self.state)
        except OSError:
            pass

    # ------------------------------------------------------------------- season
    def standings(self) -> dict[str, dict]:
        """lower-case name -> {name, points, races}"""
        out: dict[str, dict] = {}
        for r in self.races:
            for row in r["rows"]:
                if not row.get("human"):
                    continue
                d = out.setdefault(row["name"].lower(), {"name": row["name"], "points": 0, "races": 0})
                d["name"] = row["name"]
                d["races"] += 1
                pos = row.get("pos") or 99
                if row.get("status") == "finished" and pos <= len(POINTS):
                    d["points"] += POINTS[pos - 1] + (1 if r["fastest"] == row["name"] else 0)
        return out

    def tags(self) -> dict[str, list[dict]]:
        """lower-case name -> tags earned in the last TAG_RACES competitive races (newest first)."""
        out: dict[str, list[dict]] = {}
        for r in reversed(self.races[-TAG_RACES:]):
            when = datetime.datetime.fromtimestamp(r["started"]).strftime("%d %b") if r["started"] else ""
            where = f"{r['track']}, {when}"
            for tag, name, why in race_tags(r):
                lst = out.setdefault(name.lower(), [])
                if not any(t["id"] == tag for t in lst):
                    lst.append({"id": tag, "label": TAGS[tag], "title": f"{why} · {where}"})
        return out

    # ----------------------------------------------------------------- live bits
    def on_event(self, ev: dict, track: str) -> None:
        if is_solo_crash(ev) and (self.maldonado is None or ev["wall"] >= self.maldonado["wall"]):
            self.maldonado = {"wall": ev["wall"], "name": _first_human(ev), "track": track}

    def set_hammer(self, on: bool) -> None:
        self.state["hammer"] = bool(on)
        if on:
            self.checking = None
        self._save()

    def tick(self, s: Session) -> None:
        """Counts races towards "We are checking" and sets it off when it's due."""
        if s.kind != "race" or s.race_start_st is None or len(s.humans()) < 2:
            return
        key = f"{s.uid:x}-{int(s.started_wall)}"
        if key not in self.state["counted"]:
            self.state["counted"] = (self.state["counted"] + [key])[-20:]
            self.state["checking_races_left"] = max(0, int(self.state["checking_races_left"]) - 1)
            self._save()
        if self.state["checking_races_left"] > 0 or self.state["hammer"] or self.checking or s.ended:
            return
        for c in s.humans():
            if c.team == "Ferrari" and c.lap.get("pitStatus") and c.lap_num >= 2:
                self.checking = {"idx": c.idx, "name": s.display_name(c.idx), "key": key,
                                 "until_st": s.st + CHECKING_SECONDS}
                self.state["checking_races_left"] = CHECKING_EVERY
                self._save()
                return

    def simply_lovely(self, s: Session) -> dict | None:
        if s.kind == "qualifying" and s.ended:
            best = [(_best_valid(c)[0], c) for c in s.cars if c.active and _best_valid(c)[0]]
            if best:
                t, c = min(best, key=lambda x: x[0])
                if c.human and _red_bull(c.team):
                    return {"name": s.display_name(c.idx), "why": "Pole position", "time": t}
        on_grid = s.race_start_st is None and all(
            (c.lap.get("totalDistance") or 0) < 50 for c in s.cars if c.active)   # not just "joined mid-race"
        if s.kind == "race" and on_grid:
            q = next((q for q in reversed(self.qualis) if q["track"] == s.track_name
                      and 0 < s.started_wall - q["started"] < 4 * 3600), None)
            if q and q["pole"]["human"] and _red_bull(q["pole"]["team"]):
                return {"name": q["pole"]["name"], "why": "Pole position", "time": q["pole"]["time"]}
        if s.kind == "race" and (s.chequered or s.ended):
            winner = next((c for c in s.cars if c.active and (c.result or {}).get("position", c.position) == 1), None)
            if winner and winner.human and _red_bull(winner.team):
                return {"name": s.display_name(winner.idx), "why": "Race win", "time": None}
        return None

    def snapshot(self, s: Session | None, now: float | None = None) -> dict:
        now = now or time.time()
        hammer = bool(self.state["hammer"])
        out: dict = {"hammer": hammer, "standings": self.standings()}
        if hammer:
            return out
        out["tags"] = self.tags()
        if self.maldonado:
            days = (datetime.date.fromtimestamp(now) - datetime.date.fromtimestamp(self.maldonado["wall"])).days
            out["maldonado"] = {"days": max(0, days), "name": self.maldonado["name"], "track": self.maldonado["track"]}
        if s is not None:
            ch = self.checking
            if ch and ch["key"] == f"{s.uid:x}-{int(s.started_wall)}" and s.st < ch["until_st"]:
                out["checking"] = {"idx": ch["idx"], "name": ch["name"]}
            elif ch and (s.st >= ch["until_st"] or ch["key"] != f"{s.uid:x}-{int(s.started_wall)}"):
                self.checking = None
            out["simply_lovely"] = self.simply_lovely(s)
        return out


def race_tags(r: dict) -> list[tuple[str, str, str]]:
    """(tag, driver, why) for one competitive race."""
    out = []
    rows = r["rows"]
    tl = r.get("track_limits") or {}
    for row in rows:
        if not row.get("human"):
            continue
        n, pos, grid = row["name"], row.get("pos"), row.get("grid")
        offs = tl.get(n)
        if pos == 1 and row.get("status") == "finished" and offs == 0:
            out.append(("smooth", n, "Won with zero wheels off"))
        if offs is not None and offs > LAWNMOWER_OFFS:
            out.append(("lawnmower", n, f"All four wheels off {offs} times"))
        if grid == 1 and (pos or 99) > 3:
            out.append(("mr_saturday", n, f"Pole, finished P{pos}"))
    # photo finish: crossed the line within PHOTO_FINISH_S of the car ahead or behind (same lap)
    fin = [x for x in rows if x.get("status") == "finished" and x.get("race_time")]
    fin.sort(key=lambda x: x["pos"] or 99)
    for a, b in zip(fin, fin[1:]):
        if a.get("laps") != b.get("laps"):
            continue
        ta = a["race_time"] - (a.get("penalties_s") or 0)
        tb = b["race_time"] - (b.get("penalties_s") or 0)
        if abs(tb - ta) < PHOTO_FINISH_S:
            for x, y in ((a, b), (b, a)):
                if x.get("human"):
                    out.append(("photo", x["name"], f"{abs(tb - ta):.3f} s from {y['name']} at the line"))
    return out
