"""Season points, tags, the Maldonado counter, "We are checking" and incident titles.

    python -m pytest tests/test_fun.py      (or)      python tests/test_fun.py
"""

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from f1live import fun as F  # noqa: E402
from f1live.incidents import incident_title  # noqa: E402
from f1live.model import Session  # noqa: E402


def row(pos, name, human=True, grid=None, status="finished", race_time=None, penalties=0, laps=18, team="Mercedes"):
    return {"pos": pos, "name": name, "human": human, "grid": grid or pos, "status": status, "team": team,
            "race_time": race_time, "penalties_s": penalties, "laps": laps}


def write_race(base, folder, started, rows, track_limits, fastest=None, laps=18, length=4300.0, final=True):
    os.makedirs(os.path.join(base, folder))
    sm = {"session": {"kind": "race", "track": folder.split("_")[2], "total_laps": laps, "track_length": length,
                      "started": started, "final_classification_received": final,
                      "fastest_lap": {"name": fastest, "time": 80.0} if fastest else None},
          "classification": rows,
          "drivers": [{"name": n, "track_limits": t} for n, t in track_limits.items()]}
    with open(os.path.join(base, folder, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(sm, f)


def test_points_tags_and_competitive_races():
    base = tempfile.mkdtemp()
    t = time.time() - 3 * 86400
    # 1: a competitive race; Marty wins clean, Aven from pole ends P5
    write_race(base, "2026-10-01_0100_Monza_Race_a", t, [
        row(1, "Marty Gua", race_time=1500.0), row(2, "BOTTAS", human=False, race_time=1501.0),
        row(3, "Franky Frank", race_time=1510.0), row(5, "Aven Luci", grid=1, race_time=1530.0)],
        {"Marty Gua": 0, "Franky Frank": 3, "Aven Luci": 16}, fastest="Franky Frank")
    # 2: a 5-lap sprint (too short) and an abandoned lobby (nobody finished): neither counts
    write_race(base, "2026-10-01_0200_Monza_Race_b", t + 100, [row(1, "Marty Gua"), row(2, "Aven Luci")], {}, laps=5)
    write_race(base, "2026-10-01_0300_Monza_Race_c", t + 200, [row(1, "Marty Gua", status="did not finish"),
                                                              row(2, "Aven Luci", status="did not finish")], {})
    # 3: photo finish between Franky and an AI car, penalties taken off the crossing time
    write_race(base, "2026-10-02_0100_Spa_Race_d", t + 86400, [
        row(1, "Aven Luci", race_time=1390.0), row(2, "Franky Frank", race_time=1405.05, penalties=5),
        row(3, "SAINZ", human=False, race_time=1400.09), row(4, "Marty Gua", race_time=1420.0)],
        {"Aven Luci": 2, "Franky Frank": 0, "Marty Gua": 1})
    fun = F.Fun(base)
    assert [r["id"][-1] for r in fun.races] == ["a", "d"]
    st = fun.standings()
    assert st["marty gua"]["points"] == 25 + 12
    assert st["franky frank"]["points"] == 15 + 1 + 18   # P3 with the fastest lap, then P2
    assert st["aven luci"]["points"] == 10 + 25
    tags = {k: sorted(t["id"] for t in v) for k, v in fun.tags().items()}
    assert tags["marty gua"] == ["smooth"]
    assert tags["aven luci"] == ["lawnmower", "mr_saturday"]
    assert tags["franky frank"] == ["photo"]

    # a third competitive race pushes the first one's tags out (they last two races)
    write_race(base, "2026-10-03_0100_Spa_Race_e", t + 2 * 86400, [row(1, "Franky Frank", race_time=1400.0),
                                                                  row(2, "Marty Gua", race_time=1460.0)],
               {"Franky Frank": 4, "Marty Gua": 0})
    fun.refresh()
    tags = {k: sorted(t["id"] for t in v) for k, v in fun.tags().items()}
    assert "marty gua" not in tags and "aven luci" not in tags
    assert tags["franky frank"] == ["photo"]


def test_hammer_time_hides_the_fun_and_persists():
    base = tempfile.mkdtemp()
    fun = F.Fun(base)
    fun.maldonado = {"wall": time.time(), "name": "Aven Luci", "track": "Silverstone"}
    snap = fun.snapshot(None)
    assert snap["maldonado"]["days"] == 0 and "tags" in snap
    fun.set_hammer(True)
    snap = fun.snapshot(None)
    assert snap["hammer"] and "tags" not in snap and "maldonado" not in snap and "standings" in snap
    assert F.Fun(base).state["hammer"] is True


def test_solo_crash_is_a_maldonado():
    solo = {"kind": "damage", "severity": "major", "cars": [3], "names": ["Aven Luci"], "humans": [3],
            "text": "Front-left wing damage 0% → 100%", "wall": 100.0}
    hit = dict(solo, kind="collision", text="Collision with Stroll: front-left wing damage 0% → 29%")
    minor = dict(solo, severity="minor")
    assert F.is_solo_crash(solo) and not F.is_solo_crash(hit) and not F.is_solo_crash(minor)
    fun = F.Fun(tempfile.mkdtemp())
    fun.on_event(solo, "Silverstone")
    assert fun.maldonado["name"] == "Aven Luci" and fun.maldonado["track"] == "Silverstone"


class _Car:
    def __init__(self, idx, human, team="", pit=0, lap_num=3):
        self.idx, self.human, self.team, self.lap_num = idx, human, team, lap_num
        self.lap = {"pitStatus": pit, "totalDistance": 900.0}
        self.active, self.result, self.position = True, None, idx + 1


class _Session:
    def __init__(self, cars):
        self.cars = cars
        self.kind, self.uid, self.started_wall, self.race_start_st, self.st, self.ended = "race", 7, 1000.0, 5.0, 100.0, False
        self.chequered, self.track_name = False, "Monza"

    def humans(self):
        return [c for c in self.cars if c.human]

    def display_name(self, i):
        return f"car{i}"


def test_we_are_checking_once_every_two_races():
    base = tempfile.mkdtemp()
    fun = F.Fun(base)
    s = _Session([_Car(0, True, "Ferrari"), _Car(1, True, "Red Bull")])
    fun.tick(s)                                   # counts this race; nobody in the pits yet
    assert fun.state["checking_races_left"] == 0 and fun.checking is None
    s.cars[0].lap["pitStatus"] = 1
    fun.tick(s)
    assert fun.checking and fun.checking["idx"] == 0
    assert fun.snapshot(s)["checking"]["name"] == "car0"
    s.st += F.CHECKING_SECONDS + 1                # ten seconds later it's gone
    assert "checking" not in fun.snapshot(s)
    assert fun.state["checking_races_left"] == F.CHECKING_EVERY
    fun.tick(s)                                   # same race: doesn't fire again or count again
    assert fun.checking is None and fun.state["checking_races_left"] == F.CHECKING_EVERY
    for n in range(1, F.CHECKING_EVERY):          # the races in between count down without firing
        s.uid += 1
        fun.tick(s)
        assert fun.checking is None, n
    s.uid += 1
    fun.tick(s)
    assert fun.checking is not None               # and the next one fires again


def test_simply_lovely_for_a_red_bull_pole_or_win():
    fun = F.Fun(tempfile.mkdtemp())
    fun.qualis = [{"id": "q", "started": 500.0, "track": "Monza",
                   "pole": {"name": "Marty Gua", "team": "Red Bull", "human": True, "time": 80.1}}]
    s = _Session([_Car(0, True, "Red Bull"), _Car(1, True, "Ferrari")])
    s.race_start_st = None
    for c in s.cars:
        c.lap["totalDistance"] = 0.0
    assert fun.simply_lovely(s)["why"] == "Pole position"        # on the grid after a Red Bull pole
    for c in s.cars:
        c.lap["totalDistance"] = 2500.0
    assert fun.simply_lovely(s) is None                           # joined mid-race: not "before the start"
    s.chequered = True
    assert fun.simply_lovely(s)["why"] == "Race win"              # car 0 (Red Bull) is P1
    s.cars[0].position, s.cars[1].position = 2, 1
    assert fun.simply_lovely(s) is None                           # a Ferrari win isn't lovely


def test_incident_titles_lead_with_the_humans():
    s = Session(1, 0.0, {})
    for i, (name, human) in enumerate([("Aven Luci", True), ("Franky Frank", True), ("BOTTAS", False), ("chappu mochi", True)]):
        s.cars[i].name, s.cars[i].human, s.cars[i].active = name, human, True
    s.display_name = lambda i: s.cars[i].name
    t = incident_title(s, {"kind": "collision", "cars": [0, 1], "names": ["Aven Luci", "Franky Frank"],
                           "text": "Collision with Franky Frank: front-left wing damage 0% → 8%"})
    assert t == "Aven Luci & Franky Frank · Collision: front-left wing damage 0% → 8%", t
    t = incident_title(s, {"kind": "collision", "cars": [2, 3], "names": ["BOTTAS", "chappu mochi"],
                           "text": "Collision with chappu mochi at S1 0.49 km (112 g spike)"})
    assert t == "chappu mochi · Collision at S1 0.49 km (112 g spike), with BOTTAS", t
    t = incident_title(s, {"kind": "penalty", "cars": [0, 1], "names": ["Aven Luci", "Franky Frank"],
                           "text": "Warning: Small collision (involving Franky Frank)"})
    assert t == "Aven Luci & Franky Frank · Warning: Small collision", t


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
