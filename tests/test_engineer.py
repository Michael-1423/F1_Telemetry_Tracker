"""Race engineer: the port must behave exactly like the original F1 2020 AI Race Engineer.

engineer_golden.json holds engine inputs and the outputs of the original TypeScript
(engineer_engine.ts + strategy.ts) for them; analyze() must reproduce every field. The detector
tests drive EngineerFeed and a Session with synthetic packets and check the relay's counting rules.

    python -m pytest tests/test_engineer.py      (or)      python tests/test_engineer.py
"""
import json
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from f1live import engineer as E        # noqa: E402
from f1live import packets as P         # noqa: E402
from f1live.model import Session        # noqa: E402

GOLDEN = os.path.join(os.path.dirname(__file__), "engineer_golden.json")


def diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                yield f"{path}.{k} missing"
            else:
                yield from diff(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            yield f"{path}: {len(a)} items vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            yield from diff(x, y, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if float(a) != float(b):
            yield f"{path}: {a!r} vs {b!r}"
    elif a != b:
        yield f"{path}: {a!r} vs {b!r}"


def test_matches_original_typescript():
    with open(GOLDEN, encoding="utf-8") as f:
        cases = json.load(f)["cases"]
    for i, c in enumerate(cases):
        got = json.loads(json.dumps(E.analyze(c["snapshot"], c["feedback"])))
        problems = list(diff(got, c["expected"]))
        assert not problems, f"case {i}: {problems[:5]}"


def test_js_number_semantics():
    assert E._to_fixed(1.005, 2) == "1.00"      # exact binary value is below the tie
    assert E._to_fixed(2.5, 0) == "3" and E._to_fixed(-2.5, 0) == "-3"
    assert E._to_fixed(-0.04, 1) == "-0.0"
    assert E._round(2.5) == 3 and E._round(-2.5) == -2
    assert E._num(1.0) == "1" and E._num(-0.4) == "-0.4"


# ------------------------------------------------------------------------------ detectors
def blank(pid: int, st: float = 0.0, player: int = 0) -> dict:
    b = bytearray(P.PACKET_SIZES[pid])
    struct.pack_into("<H", b, 0, 2020)
    b[5] = pid
    p = P.decode(bytes(b))
    p["header"].update(sessionTime=st, sessionUID=1, playerCarIndex=player)
    return p


def motion(slip, susp=(0.0, 0.0, 0.0, 0.0), glat=0.0):
    p = blank(P.MOTION)
    p["wheelSlip"], p["suspensionPosition"] = list(slip), list(susp)
    p["carMotionData"][0]["gForceLateral"] = glat
    return p


def test_detector_counts():
    f = E.EngineerFeed()
    f.feed(motion((0.30, 0.10, 0.10, 0.05)), 0)                  # rear slip, 1.5x the front: oversteer
    f.feed(motion((0.30, 0.10, 0.25, 0.05)), 0)                  # rear not 1.5x the front: nothing
    f.feed(motion((0.05, 0.05, 0.30, 0.10), glat=-2.0), 0)       # front slip with 2 g: understeer
    f.feed(motion((0.05, 0.05, 0.30, 0.10), glat=1.0), 0)        # ...but not at 1 g
    f.feed(motion((0, 0, 0, 0), susp=(0.05, -0.07, 0, 0)), 0)
    f.feed(motion((0, 0, 0, 0), susp=(0, 0, -0.09, 0)), 0)       # |0.09| > 0.08: kerb
    d = f.snapshot("x", {})["diagnostics"]
    assert (d["oversteerEvents"], d["understeerEvents"], d["kerbBottomingEvents"]) == (1, 1, 1), d

    t = blank(P.TELEMETRY)
    td = t["carTelemetryData"][0]
    td.update(brake=0.8, brakesTemperature=[500, 500, 700, 700])  # RL, RR, FL, FR: fronts 200 hotter
    f.feed(t, 0)
    td.update(brakesTemperature=[500, 500, 600, 600])             # only 100 hotter
    f.feed(t, 0)
    td.update(brake=0.6, brakesTemperature=[500, 500, 700, 700])  # not braking hard enough
    f.feed(t, 0)
    assert f.snapshot("x", {})["diagnostics"]["frontLockingEvents"] == 1


def test_lap_records_and_resets():
    f = E.EngineerFeed()
    s = blank(P.STATUS)
    s["carStatusData"][0].update(fuelInTank=20.0, tyresWear=[4, 5, 6, 7], visualTyreCompound=17)
    f.feed(s, 0)
    lap = blank(P.LAP)
    ld = lap["lapData"][0]
    ld.update(currentLapNum=1, currentLapInvalid=1)
    f.feed(lap, 0)
    f.feed(motion((0.30, 0.10, 0.10, 0.05)), 0)
    ld.update(currentLapNum=2, currentLapInvalid=0, lastLapTime=91.25)
    f.feed(lap, 0)
    snap = f.snapshot("x", {})
    (l1,) = snap["completedLaps"]
    assert l1["lapNumber"] == 1 and l1["lapTime"] == 91.25 and l1["lapTimeFormatted"] == "1:31.250"
    assert l1["isValid"] is False                       # the completed lap was invalid, not the new one
    assert l1["fuelUsedKg"] == 1.85                     # the relay's placeholder for the first lap
    assert l1["tyreWearDelta"] == {"fl": 6, "fr": 7, "rl": 4, "rr": 5}
    assert l1["tyreCompound"] == "Medium" and l1["oversteerEvents"] == 1
    assert snap["diagnostics"]["oversteerEvents"] == 0  # per-lap counters restart


def test_session_feeds_each_drivers_own_car():
    s = Session(1, 0.0, {})
    su = blank(P.SETUPS, player=3)
    su["carSetups"][3].update(frontWing=7, rearWing=8, brakeBias=56, frontLeftTyrePressure=23.0)
    s.handle(su, "10.0.0.2:1", 1.0)
    s.handle(motion((0.30, 0.10, 0.10, 0.05)), "10.0.0.1:1", 1.1)   # another game, player car 0
    assert set(s.engineers) == {0, 3}
    assert s.engineers[3].setup["frontWing"] == 7 and s.engineers[3].setup["brakeBias"] == 56
    assert s.engineers[0].oversteer_events == 1 and s.engineers[3].oversteer_events == 0
    zero = blank(P.SETUPS, player=3)                                # sent after retiring / finishing
    s.handle(zero, "10.0.0.2:1", 2.0)
    assert s.engineers[3].setup["frontWing"] == 7
    spectator = blank(P.MOTION, player=255)
    s.handle(spectator, "10.0.0.3:1", 2.1)
    assert set(s.engineers) == {0, 3}


def test_payload_filters_feedback_and_waits_for_setup():
    f = E.EngineerFeed()
    entry = {"idx": 0, "name": "A", "team": "T", "color": "#888", "own": True, "setup": False,
             "snapshot": f.snapshot("A", {"sessionType": "Practice 1", "sessionTypeId": 1, "trackName": "X"})}
    out = E.payload(entry, "a", ["oversteer_exit", "nonsense"])
    assert out["feedback"] == ["oversteer_exit"] and out["analysis"] is None and out["car"]
    assert E.payload(None, "b", [])["found"] is False


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
