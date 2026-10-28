"""Episode evidence tests (P0.1).

Run directly:  python3 tests/test_evidence.py
"""

import copy
import json
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from f1live import engineer as E  # noqa: E402
from f1live import evidence as EV  # noqa: E402
from f1live import packets as P  # noqa: E402


def blank(pid: int, st: float = 0.0, player: int = 0) -> dict:
    b = bytearray(P.PACKET_SIZES[pid])
    struct.pack_into("<H", b, 0, 2020)
    b[5] = pid
    p = P.decode(bytes(b))
    p["header"].update(sessionTime=st, sessionUID=1, playerCarIndex=player)
    return p


def motion(slip, susp_v=(0.0, 0.0, 0.0, 0.0), glat=0.0, st=0.0):
    p = blank(P.MOTION, st)
    p["wheelSlip"], p["suspensionVelocity"] = list(slip), list(susp_v)
    p["carMotionData"][0]["gForceLateral"] = glat
    return p


def telemetry(speed=100, surfaces=(0, 0, 0, 0), steer=0.2, brake=0.0,
              brake_temp=(500, 500, 500, 500), st=0.0,
              tyre_temp=(95, 95, 95, 95)):
    p = blank(P.TELEMETRY, st)
    td = p["carTelemetryData"][0]
    td.update(speed=speed, surfaceType=list(surfaces), steer=steer, brake=brake,
              brakesTemperature=list(brake_temp),
              tyresInnerTemperature=list(tyre_temp))
    return p


def lap(status=4, pit=0, distance=100.0, num=1, last=0.0, invalid=0, st=0.0):
    p = blank(P.LAP, st)
    p["lapData"][0].update(currentLapNum=num, driverStatus=status, pitStatus=pit,
                           lapDistance=distance, lastLapTime=last,
                           currentLapInvalid=invalid)
    return p


def status(compound=16, age=5, wear=(0, 0, 0, 0), st=0.0):
    p = blank(P.STATUS, st)
    p["carStatusData"][0].update(visualTyreCompound=compound, tyresAgeLaps=age,
                                 tyresWear=list(wear), fuelInTank=20.0,
                                 fuelRemainingLaps=10.0)
    return p


def test_one_slide_is_one_episode_at_any_rate():
    for hz in (10, 20, 60):
        tr = EV.EpisodeTracker("oversteer", gap_s=0.2, cooldown_s=0.1)
        dt = 1.0 / hz
        closed = []
        t = 0.0
        # 0.1 s of slip, then 0.5 s of grip
        for _ in range(max(1, int(0.1 * hz))):
            ep = tr.update(t, True, {"slip": 0.4, "speed": 100, "lap": 1})
            assert ep is None
            t += dt
        for _ in range(max(2, int(0.5 * hz))):
            ep = tr.update(t, False, {})
            if ep:
                closed.append(ep)
            t += dt
        ep = tr.flush(t)
        if ep:
            closed.append(ep)
        assert len(closed) == 1, (hz, closed)
        assert closed[0].duration_s >= 0.09
        assert closed[0].max_slip == 0.4


def test_one_kerb_strike_is_one_episode():
    tr = EV.EpisodeTracker("kerb", gap_s=0.2, cooldown_s=0.1)
    closed = []
    t = 0.0
    for _ in range(6):  # six consecutive frames over 1000 mm/s
        ep = tr.update(t, True, {"wheels": {2}, "speed": 180, "lap": 1})
        if ep:
            closed.append(ep)
        t += 0.05
    for _ in range(10):
        ep = tr.update(t, False, {})
        if ep:
            closed.append(ep)
        t += 0.05
    ep = tr.flush(t)
    if ep:
        closed.append(ep)
    assert len(closed) == 1
    assert closed[0].wheels == {2}


def test_bounce_between_episodes_is_separate():
    tr = EV.EpisodeTracker("oversteer", gap_s=0.1, cooldown_s=0.05)
    closed = []
    t = 0.0
    for _ in range(4):
        ep = tr.update(t, True, {"slip": 0.3})
        if ep:
            closed.append(ep)
        t += 0.05
    for _ in range(4):
        ep = tr.update(t, False, {})
        if ep:
            closed.append(ep)
        t += 0.05
    # after the gap/cooldown, a new slide is a new episode
    for _ in range(4):
        ep = tr.update(t, True, {"slip": 0.3})
        if ep:
            closed.append(ep)
        t += 0.05
    ep = tr.flush(t)
    if ep:
        closed.append(ep)
    assert len(closed) == 2, closed


def test_summary_keeps_duration_and_peak():
    tr = EV.EpisodeTracker("oversteer", gap_s=0.1)
    t = 0.0
    for slip in (0.3, 0.5, 0.4):
        tr.update(t, True, {"slip": slip, "g": 2.0, "speed": 120, "lap": 2})
        t += 0.05
    ep = tr.flush(t)
    (row,) = [EV.summarize_episodes([ep])["oversteer"]]
    assert row["count"] == 1
    assert row["max_slip"] == 0.5
    assert row["max_g"] == 2.0
    assert row["seconds"] > 0


def test_gates_ignore_garage_pit_grass_and_low_speed():
    def oversteer_count(status=4, pit=0, surfaces=(0, 0, 0, 0), speed=100):
        f = E.EngineerFeed()
        f.feed(lap(status, pit), 0)
        f.feed(telemetry(speed, surfaces), 0)
        f.feed(motion((0.30, 0.10, 0.10, 0.05)), 0)
        return f.snapshot("x", {})["diagnostics"]["episodes"]["oversteer"]["count"]

    assert oversteer_count() == 1
    assert oversteer_count(status=0) == 0       # garage
    assert oversteer_count(pit=1) == 0          # pit lane
    assert oversteer_count(surfaces=(7, 7, 0, 0)) == 0  # grass
    assert oversteer_count(speed=10) == 0       # too slow


def test_gates_apply_to_kerbs_and_front_locks():
    def kerb_count(pit=0, surfaces=(0, 0, 0, 0), speed=100):
        f = E.EngineerFeed()
        f.feed(lap(4, pit), 0)
        f.feed(telemetry(speed, surfaces), 0)
        f.feed(motion((0, 0, 0, 0), susp_v=(0, 0, 1500.0, 0)), 0)
        return f.snapshot("x", {})["diagnostics"]["episodes"]["kerb"]["count"]

    assert kerb_count() == 1
    assert kerb_count(pit=1) == 0
    assert kerb_count(surfaces=(7, 7, 0, 0)) == 0

    def lock_count(pit=0, speed=100):
        f = E.EngineerFeed()
        f.feed(lap(4, pit), 0)
        f.feed(telemetry(speed, brake=0.8, brake_temp=(500, 500, 700, 700)), 0)
        return f.snapshot("x", {})["diagnostics"]["episodes"]["front_lock"]["count"]

    assert lock_count() == 1
    assert lock_count(pit=1) == 0
    assert lock_count(speed=10) == 0


def test_episode_window_uses_recent_push_laps_and_generation():
    def lap_row(n, push, count, generation=0):
        eps = [EV.Episode("oversteer", float(i), float(i) + 0.1, n) for i in range(count)]
        return {"lap": n, "push": push, "setup_generation": generation,
                "episodes": EV.summarize_episodes(eps)}

    laps = [
        lap_row(1, True, 2, 0),
        lap_row(2, False, 5, 0),   # out lap: must not count
        lap_row(3, True, 1, 0),
        lap_row(4, True, 3, 1),    # new setup generation
    ]
    w = EV.episode_window(laps, window=2)
    assert w["laps_used"] == 2
    assert w["oversteer"]["count"] == 4  # laps 3 + 4
    assert w["laps"] == [3, 4]

    w = EV.episode_window(laps, window=3, generation=1)
    assert w["oversteer"]["count"] == 3
    assert w["laps"] == [4]


def test_tyre_change_resets_episode_evidence():
    f = E.EngineerFeed()
    f.feed(status(compound=16, age=5), 0)
    f.feed(lap(4, 0), 0)
    f.feed(telemetry(100), 0)
    f.feed(motion((0.30, 0.10, 0.10, 0.05)), 0)
    assert f.snapshot("x", {})["diagnostics"]["episodes"]["oversteer"]["count"] == 1
    f.feed(status(compound=17, age=0), 0)
    snap = f.snapshot("x", {})
    assert snap["diagnostics"]["episodes"]["oversteer"]["count"] == 0
    assert snap["tyreGeneration"] == 1


def test_lap_labels_and_last_lap():
    f = E.EngineerFeed()
    f.feed(lap(4, 0, num=1, st=0.0), 0)
    f.feed(telemetry(100, st=0.0), 0)
    f.feed(lap(4, 0, num=2, last=90.0, st=1.0), 0)
    snap = f.snapshot("x", {})
    assert snap["lastLap"]["label"] == "Flying lap"
    assert snap["lastLap"]["push"] is True
    assert snap["completedLaps"][-1]["kind"] == "flying"

    # an in-lap
    f.feed(lap(2, 0, num=2, st=2.0), 0)
    f.feed(lap(4, 0, num=3, last=100.0, st=3.0), 0)
    assert f.snapshot("x", {})["lastLap"]["label"] == "In lap"
    assert f.snapshot("x", {})["completedLaps"][-1]["push"] is False

    # an out-lap
    f.feed(lap(3, 0, num=3, st=4.0), 0)
    f.feed(lap(4, 0, num=4, last=95.0, st=5.0), 0)
    assert f.snapshot("x", {})["lastLap"]["label"] == "Out lap"

    # a lap that entered the pit lane
    f.feed(lap(4, 1, num=4, st=6.0), 0)
    f.feed(lap(4, 0, num=5, last=95.0, st=7.0), 0)
    assert f.snapshot("x", {})["lastLap"]["label"] == "In lap"
    assert f.snapshot("x", {})["lastLap"]["pit"] is True

    # the payload carries the label to the Engineer page
    entry = {"idx": 0, "name": "A", "team": "T", "color": "#888", "own": True,
             "setup": False, "engine": "v2", "snapshot": f.snapshot("x", {})}
    out = E.payload(entry, "a", [])
    assert out["car"]["lastLap"]["label"] == "In lap"


def test_pit_temperature_does_not_pollute_lap_average():
    f = E.EngineerFeed()
    f.feed(lap(4, 0, num=1, st=0.0), 0)
    for i in range(4):
        f.feed(telemetry(100, st=0.1 + i * 0.1, tyre_temp=(98, 98, 98, 98)), 0)
    f.feed(lap(4, 1, num=1, st=0.6), 0)          # pit lane
    f.feed(telemetry(50, st=0.6, tyre_temp=(150, 150, 150, 150)), 0)
    f.feed(lap(4, 0, num=1, st=0.7), 0)
    f.feed(telemetry(100, st=0.7, tyre_temp=(98, 98, 98, 98)), 0)
    f.feed(lap(4, 0, num=2, last=90.0, st=1.0), 0)
    lap_row = f.snapshot("x", {})["completedLaps"][-1]
    assert lap_row["averageTyreTemps"]["fl"] < 105
    assert lap_row["maxTyreTemps"]["fl"] == 98
    assert lap_row["tempSamples"] >= 4
    # a normal flying lap after it is usable by the summary
    f.feed(telemetry(100, st=1.1, tyre_temp=(98, 98, 98, 98)), 0)
    f.feed(lap(4, 0, num=3, last=90.0, st=1.2), 0)
    summary = EV.temperature_summary(f.snapshot("x", {})["completedLaps"], window=3)
    assert summary["front_avg"] == 98.0


def test_wear_estimate_ignores_outliers_and_uses_push_laps():
    def lap_row(n, fl, fr, rl, rr, push=True, generation=0):
        return {"lap": n, "push": push, "valid": True, "setup_generation": generation,
                "tyreWearDelta": {"fl": fl, "fr": fr, "rl": rl, "rr": rr}}

    laps = [
        lap_row(1, 2.0, 2.0, 0.8, 0.8),
        lap_row(2, 2.2, 2.2, 0.9, 0.9),
        lap_row(3, 9.5, 9.5, 0.9, 0.9),   # puncture/damage outlier
        lap_row(4, 2.1, 2.1, 0.8, 0.8),
        lap_row(5, 8.0, 8.0, 8.0, 8.0, push=False),  # out lap
    ]
    w = EV.wear_estimate(laps)
    assert w["outliers"] == 1
    assert 2.0 <= w["front_rate"] <= 2.3
    assert 0.8 <= w["rear_rate"] <= 1.0
    assert w["laps"] == 3

    # 3+ laps use a regression, so a rising trend is captured
    trend = [lap_row(n, 2.0 + 0.2 * n, 2.0 + 0.2 * n, 1.0, 1.0) for n in range(1, 5)]
    w = EV.wear_estimate(trend)
    assert 2.4 <= w["front_rate"] <= 2.6
    assert w["provisional"] is False

    # a setup generation filters out the old setup's laps
    w = EV.wear_estimate(laps, generation=1)
    assert w["laps"] == 0


def _v2_snapshot():
    with open(os.path.join(os.path.dirname(__file__), "engineer_golden.json"), encoding="utf-8") as f:
        snap = copy.deepcopy(json.load(f)["cases"][0]["snapshot"])
    snap["session"].update(sessionType="Practice 1", sessionTypeId=1, totalLaps=99, trackLength=5000)
    snap["lapData"].update(currentLapNum=5)
    snap["damage"]["tyresWear"] = {"fl": 30.0, "fr": 30.0, "rl": 20.0, "rr": 20.0}
    snap["completedLaps"] = []
    for n in range(1, 4):
        over = [EV.Episode("oversteer", float(i), float(i) + 0.1, n,
                           max_speed_kmh=220 if n < 3 else 100) for i in range(2)]
        snap["completedLaps"].append({
            "lapNumber": n, "lap": n, "isValid": True, "valid": True,
            "kind": "flying", "push": True, "clean": True, "setup_generation": 0,
            "lapTime": 90.0, "maxSpeedKmh": 300 if n == 1 else 285,
            "tyreWearDelta": {"fl": 2.0, "fr": 2.0, "rl": 0.8, "rr": 0.8},
            "averageTyreTemps": {"fl": 100, "fr": 100, "rl": 98, "rr": 98},
            "episodes": EV.summarize_episodes(over),
        })
    snap["setupGeneration"] = 0
    snap["engineerConfig"] = {"engine": "v2"}
    return snap


def test_analyze_v2_uses_evidence_and_triggers_high_wear():
    result = E.analyze_v2(_v2_snapshot())
    assert result["engine"] == "v2"
    assert result["evidence"]["wear"]["laps"] == 3
    assert result["evidence"]["high_wear_trigger"] is True
    assert any(i["title"] == "High Tyre Wear / Degradation" for i in result["diagnosedIssues"])
    assert any(i["title"] == "Aerodynamic & High-Speed Instability" for i in result["diagnosedIssues"])


def test_analyze_v2_top_speed_hint():
    snap = _v2_snapshot()
    snap["completedLaps"][0]["maxSpeedKmh"] = 310
    snap["completedLaps"][1]["maxSpeedKmh"] = 290
    snap["completedLaps"][2]["maxSpeedKmh"] = 290
    result = E.analyze_v2(snap)
    assert result["evidence"]["top_speed_trigger"] is True
    assert any(i["title"] == "Excessive Aerodynamic Drag" for i in result["diagnosedIssues"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
