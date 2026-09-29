"""Practice / qualifying lap bookkeeping.

The game only counts timed laps: an out lap, or a flying lap abandoned into the pits, never moves
currentLapNum. This drives a Session with synthetic packets through three qualifying runs and checks
every lap driven is recorded, with the right kind and timing, including after the game has marked
the car "retired" (terminal damage sends it back to the garage, and it keeps that status).

    python tests/test_practice_laps.py
"""
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from f1live import packets as P          # noqa: E402
from f1live.model import Session         # noqa: E402
from f1live.summary import timesheet     # noqa: E402

TRACK = 4650.0
HZ = 10
DT = 1 / HZ


def blank(pid: int, st: float) -> dict:
    b = bytearray(P.PACKET_SIZES[pid])
    struct.pack_into("<H", b, 0, 2020)
    b[5] = pid
    p = P.decode(bytes(b))
    p["header"]["sessionTime"] = st
    p["header"]["sessionUID"] = 1
    return p


class Sim:
    def __init__(self):
        self.s = Session(1, 0.0, {})
        self.st = 0.0
        self.lapn = 1
        self.t_lap = 0.0          # the game's currentLapTime
        self.last = 0.0
        self.invalid = 0
        self.d = 4550.0
        self.ds = 0
        self.pit = 1
        self.rs = 2               # resultStatus
        sp = blank(P.SESSION, 0.0)
        sp.update(sessionType=8, trackId=4, trackLength=int(TRACK), totalLaps=0)
        self.s.handle(sp, "me", 0.0)
        pp = blank(P.PARTICIPANTS, 0.0)
        pp["numActiveCars"] = 1
        pp["participants"][0].update(name="Tester", aiControlled=0, teamId=0)
        self.s.handle(pp, "me", 0.0)

    def step(self, v_kmh: float) -> None:
        self.st += DT
        if self.ds != 0:
            self.t_lap += DT
            self.d += v_kmh / 3.6 * DT
        wrapped = self.d >= TRACK
        if wrapped:
            self.d -= TRACK
        ld = blank(P.LAP, self.st)["lapData"][0]
        ld.update(currentLapNum=self.lapn, currentLapTime=self.t_lap, lastLapTime=self.last, lapDistance=self.d,
                  totalDistance=self.st * 50, driverStatus=self.ds, pitStatus=self.pit, resultStatus=self.rs,
                  currentLapInvalid=self.invalid, carPosition=1)
        pk = blank(P.LAP, self.st)
        pk["lapData"][0] = ld
        self.s.handle(pk, "me", self.st)
        tp = blank(P.TELEMETRY, self.st)
        tp["carTelemetryData"][0].update(speed=int(v_kmh), throttle=1.0 if v_kmh > 150 else 0.3, gear=5)
        self.s.handle(tp, "me", self.st)
        return wrapped

    def drive_to(self, d_target: float, v: float, *, cross: bool = False) -> None:
        """Drive at v km/h until reaching d_target (after crossing the line first, when cross=True)."""
        crossed = not cross
        for _ in range(100000):
            if self.step(v):
                crossed = True
            if crossed and self.d >= d_target:
                return
        raise RuntimeError("never got there")

    def line(self, timed: bool) -> None:
        """Cross the start/finish line on track."""
        if timed:
            self.last = self.t_lap
            self.lapn += 1
        self.t_lap = 0.0
        self.invalid = 0
        self.d = 0.0

    def garage(self) -> None:
        self.ds, self.pit, self.t_lap = 0, 1, 0.0
        for _ in range(20):
            self.step(0)

    def run(self, flying_v: float, abandon_at: float | None = None) -> None:
        self.ds, self.pit, self.t_lap = 3, 1, 0.0        # leave the garage; the pit lane crosses the line
        self.drive_to(300, 80, cross=True)
        self.pit = 0                                     # pit exit
        self.drive_to(TRACK - 10, 200)                   # out lap
        self.line(timed=False)                           # the game doesn't count the out lap
        self.ds = 1
        if abandon_at is not None:
            self.drive_to(abandon_at, flying_v)
            self.invalid = 1                             # cut a corner: lap invalidated, driver gives up
            self.ds = 2
        else:
            self.drive_to(TRACK - 10, flying_v)
            self.line(timed=True)
            self.ds = 2
        self.drive_to(4400, 150)                         # in lap to the pit entry
        self.pit = 1
        self.drive_to(4550, 80)
        self.garage()


def main() -> None:
    sim = Sim()
    sim.garage()
    sim.run(217)                    # ~77 s lap
    sim.rs = 7                      # terminal damage: the game says "retired" for the rest of the session,
                                    # but the driver goes back out from the garage
    sim.run(219, abandon_at=3000)   # invalidated at 3 km, back to the pits
    sim.run(220)                    # ~76 s lap
    c = sim.s.cars[0]
    got = [(l["lap"], l["kind"], l["timed"], l["valid"], round(l["time"], 1)) for l in c.laps]
    for g in got:
        print(g)
    kinds = [g[1] for g in got]
    assert kinds == ["out", "flying", "in", "out", "in", "out", "flying", "in"], kinds
    assert [g[0] for g in got] == list(range(1, 9)), got
    assert [g[2] for g in got] == [False, True, False, False, False, False, True, False], got
    assert [g[3] for g in got] == [True, True, True, True, False, True, True, True], "only the abandoned lap is invalid"
    assert 76.5 < got[1][4] < 78 and 75.5 < got[6][4] < 77, got
    assert not any(l["pit"] for l in c.laps if l["kind"] == "flying"), "flying laps must not carry the out lap's pit flag"
    assert c.best_lap() == c.laps[6]["time"]
    row = timesheet(sim.s)[0]
    assert row["best"] == c.laps[6]["time"] and row["best_lap"] == 7, row
    assert (row["valid_laps"], row["laps"], row["laps_all"], row["runs"]) == (2, 2, 8, 3), row
    print("ok")


if __name__ == "__main__":
    main()
