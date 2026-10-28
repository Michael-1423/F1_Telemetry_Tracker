"""Episode evidence tests (P0.1).

Run directly:  python3 tests/test_evidence.py
"""

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


def motion(slip, susp_v=(0.0, 0.0, 0.0, 0.0), glat=0.0):
    p = blank(P.MOTION)
    p["wheelSlip"], p["suspensionVelocity"] = list(slip), list(susp_v)
    p["carMotionData"][0]["gForceLateral"] = glat
    return p


def telemetry(speed=100, surfaces=(0, 0, 0, 0), steer=0.2, brake=0.0,
              brake_temp=(500, 500, 500, 500)):
    p = blank(P.TELEMETRY)
    td = p["carTelemetryData"][0]
    td.update(speed=speed, surfaceType=list(surfaces), steer=steer, brake=brake,
              brakesTemperature=list(brake_temp))
    return p


def lap(status=4, pit=0, distance=100.0, num=1):
    p = blank(P.LAP)
    p["lapData"][0].update(currentLapNum=num, driverStatus=status, pitStatus=pit,
                           lapDistance=distance)
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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
