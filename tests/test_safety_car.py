"""Safety car race control messages.

F1 2020 sends safetyCarStatus 3 during a race's formation lap. That once crashed the tracker
(KeyError: 3) when the formation lap ended, and marked every lap as a safety car lap.

    python tests/test_safety_car.py
"""
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from f1live import packets as P          # noqa: E402
from f1live.model import Session         # noqa: E402


def session_pkt(st: float, sc: int) -> dict:
    b = bytearray(P.PACKET_SIZES[P.SESSION])
    struct.pack_into("<H", b, 0, 2020)
    b[5] = P.SESSION
    p = P.decode(bytes(b))
    p["header"]["sessionTime"] = st
    p["header"]["sessionUID"] = 1
    p.update(sessionType=10, trackId=17, trackLength=5848, totalLaps=5, safetyCarStatus=sc)
    return p


def race_control(s: Session) -> list[str]:
    return [e["text"] for e in s.events if e["kind"] == "race_control"]


def test_formation_lap_is_not_a_safety_car():
    s = Session(1, 0.0, {})
    for st, sc in ((0.0, 3), (1.0, 3), (2.0, 0), (3.0, 3)):
        s.handle(session_pkt(st, sc), "me", st)
        assert s.sc_status == 0
        assert not any("sc" in c.cur_lap_flags for c in s.cars)
    assert race_control(s) == []


def test_safety_car_deployed_and_ending():
    s = Session(1, 0.0, {})
    for st, sc in ((0.0, 3), (1.0, 0), (2.0, 1), (3.0, 0), (4.0, 2), (5.0, 0)):
        s.handle(session_pkt(st, sc), "me", st)
    assert race_control(s) == ["Safety Car deployed", "Safety Car ending, green flag",
                               "Virtual Safety Car deployed", "Virtual Safety Car ending, green flag"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
