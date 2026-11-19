"""Tests for issue #17: per-corner throttle/braking mistake detection and coaching."""

import copy
import json
import math
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from f1live import engineer as E  # noqa: E402
from f1live import laps as LP  # noqa: E402
from f1live import packets as P  # noqa: E402
from f1live.model import Session  # noqa: E402

GOLDEN = os.path.join(os.path.dirname(__file__), "engineer_golden.json")


def _blank(pid: int, st: float = 0.0, player: int = 0) -> dict:
    b = bytearray(P.PACKET_SIZES[pid])
    struct.pack_into("<H", b, 0, 2020)
    b[5] = pid
    p = P.decode(bytes(b))
    p["header"].update(sessionTime=st, sessionUID=1, playerCarIndex=player)
    return p


def _base_setup() -> dict:
    return {
        "frontWing": 7, "rearWing": 7, "onThrottle": 65, "offThrottle": 55,
        "frontCamber": -3.0, "rearCamber": -1.5, "frontToe": 0.05, "rearToe": 0.30,
        "frontSuspension": 5, "rearSuspension": 5, "frontAntiRollBar": 5,
        "rearAntiRollBar": 6, "frontSuspensionHeight": 4, "rearSuspensionHeight": 5,
        "brakePressure": 95, "brakeBias": 58, "frontLeftTyrePressure": 23.0,
        "frontRightTyrePressure": 23.0, "rearLeftTyrePressure": 21.0,
        "rearRightTyrePressure": 21.0, "ballast": 5, "fuelLoad": 20.0,
    }


def _golden_snapshot() -> dict:
    with open(GOLDEN, encoding="utf-8") as f:
        snap = copy.deepcopy(json.load(f)["cases"][0]["snapshot"])
    snap["telemetry"]["tyresInnerTemperature"] = {"fl": 98, "fr": 98, "rl": 96, "rr": 96}
    snap["diagnostics"] = {
        "oversteerEvents": 0, "understeerEvents": 0,
        "frontLockingEvents": 0, "rearLockingEvents": 0,
        "kerbBottomingEvents": 0,
    }
    snap["completedLaps"] = []
    snap["damage"]["frontLeftWingDamage"] = 0
    snap["damage"]["frontRightWingDamage"] = 0
    snap["damage"]["rearWingDamage"] = 0
    snap["damage"]["tyresWear"] = {"fl": 10.0, "fr": 10.0, "rl": 10.0, "rr": 10.0}
    return snap


# ---------------------------------------------------------------------- laps.py metrics & profile
def test_corner_metrics_captures_coast_overlap_delay_and_lift():
    zones = [{"name": "T1", "start": 700.0, "end": 880.0}]
    # bins 140..176 (700..880 m). Apex at bin 152 (760 m, 80 km/h).
    # Heavy braking 142..145 (710..725 m), overlap at 142, coasting 146..151 (6 bins = 30 m),
    # then exit throttle rises to 0.80 at 154, dips to 0.40 at 156 (lift!), full 1.0 at 160 (800 m).
    trace = {
        140: (250, 1.0, 0.0, 0.0, 7),
        142: (220, 0.25, 0.90, 0.0, 6),  # overlap >= 0.2 & >= 0.2 (5 m)
        143: (180, 0.0, 0.95, 0.05, 5),
        144: (140, 0.0, 0.70, 0.15, 4),
        145: (110, 0.0, 0.30, 0.25, 3),  # last_brk = 145
        146: (95, 0.0, 0.0, 0.35, 3),    # coasting bins 146..151 = 6 * 5 = 30 m
        147: (92, 0.0, 0.0, 0.40, 3),
        148: (88, 0.0, 0.0, 0.45, 3),
        149: (85, 0.0, 0.0, 0.50, 2),
        150: (83, 0.0, 0.0, 0.55, 2),
        151: (81, 0.0, 0.0, 0.60, 2),
        152: (80, 0.10, 0.0, 0.55, 2),   # apex_d = 760 m, first_thr = 152
        154: (95, 0.80, 0.0, 0.35, 2),
        156: (102, 0.40, 0.0, 0.30, 3),  # lift: 0.80 - 0.40 = 0.40 >= 0.25
        160: (135, 1.00, 0.0, 0.10, 3),  # thr_on = 800 m -> delay = 40 m
        166: (170, 1.00, 0.0, 0.0, 4),
    }
    m = LP.corner_metrics(trace, [(710.0, 220)], zones)["T1"]
    assert m["brake_peak"] == 0.95
    assert m["coast_m"] == 30
    assert m["overlap_m"] == 5
    assert m["throttle_delay_m"] == 40.0
    assert m["throttle_lift"] is True


def test_profile_aggregates_new_metrics_and_reference_lap():
    zones = [{"name": "T1", "start": 700.0, "end": 880.0}]
    fast_lap = {
        "clean": True, "time": 80.0,
        "corners": {
            "T1": {
                "min_v": 95, "apex_d": 760, "brake_at": 730.0, "entry_v": 250,
                "brake_peak": 0.9, "coast_m": 5, "overlap_m": 0,
                "throttle_at": 780, "throttle_delay_m": 20.0, "throttle_lift": False,
                "exit_v": 185, "steer_peak": 0.5, "steer_at_apex": 0.4,
                "brake_at_apex": 0.1, "throttle_at_apex": 0.2, "trail_braking": True, "gear": 3,
            }
        },
    }
    slow_lap = {
        "clean": True, "time": 82.0,
        "corners": {
            "T1": {
                "min_v": 85, "apex_d": 760, "brake_at": 710.0, "entry_v": 245,
                "brake_peak": 0.95, "coast_m": 35, "overlap_m": 15,
                "throttle_at": 805, "throttle_delay_m": 45.0, "throttle_lift": True,
                "exit_v": 170, "steer_peak": 0.6, "steer_at_apex": 0.5,
                "brake_at_apex": 0.15, "throttle_at_apex": 0.1, "trail_braking": True, "gear": 3,
            }
        },
    }
    row = LP.profile([slow_lap, fast_lap], zones)[0]
    assert row["coast_m"] == 20.0
    assert row["overlap_m"] == 7.5
    assert row["throttle_delay_m"] == 32.5
    assert row["throttle_lift_pct"] == 50
    assert row["best_exit_v"] == 185
    assert row["ref_brake_at"] == 730.0
    assert row["ref_min_v"] == 95
    assert row["ref_exit_v"] == 185
    assert row["ref_throttle_delay_m"] == 20.0


# ---------------------------------------------------------------------- Session -> Engineer integration
def _init_session(track_id: int = 4, track_length: int = 4650) -> Session:
    s = Session(1, 0.0, {})
    sp = _blank(P.SESSION, 0.0)
    sp.update(sessionType=1, trackId=track_id, trackLength=track_length, totalLaps=20)
    s.handle(sp, "me", 0.0)
    pp = _blank(P.PARTICIPANTS, 0.0)
    pp["numActiveCars"] = 1
    pp["participants"][0].update(name="Tester", aiControlled=0, teamId=0)
    s.handle(pp, "me", 0.0)
    su = _blank(P.SETUPS, 0.0)
    su["carSetups"][0].update(_base_setup())
    s.handle(su, "me", 0.0)
    return s


def _feed_sample(
    s: Session,
    st: float,
    *,
    lap_num: int = 1,
    d: float = 0.0,
    speed: float = 150.0,
    thr: float = 0.0,
    brk: float = 0.0,
    steer: float = 0.0,
    surfaces: tuple = (0, 0, 0, 0),
    wheel_slip: tuple = (0.0, 0.0, 0.0, 0.0),
    yaw_deg: float = 0.0,
    glat: float = 0.0,
    brake_temp: tuple = (500, 500, 500, 500),
    lap_time: float = 10.0,
    last_lap: float = 0.0,
    invalid: int = 0,
) -> None:
    lp = _blank(P.LAP, st)
    lp["lapData"][0].update(
        currentLapNum=lap_num, currentLapTime=lap_time, lastLapTime=last_lap,
        lapDistance=d, totalDistance=max(0.0, (lap_num - 1) * (s.track_length or 4650) + d),
        driverStatus=1, pitStatus=0, resultStatus=2, currentLapInvalid=invalid, carPosition=1,
    )
    s.handle(lp, "me", st)

    tp = _blank(P.TELEMETRY, st)
    tp["carTelemetryData"][0].update(
        speed=int(speed), throttle=thr, brake=brk, steer=steer, gear=4,
        surfaceType=list(surfaces), brakesTemperature=list(brake_temp),
    )
    s.handle(tp, "me", st)

    mp = _blank(P.MOTION, st)
    mp["wheelSlip"] = list(wheel_slip)
    mp["carMotionData"][0].update(
        worldVelocityX=0.0, worldVelocityZ=speed / 3.6,
        yaw=math.radians(yaw_deg), gForceLateral=glat,
    )
    s.handle(mp, "me", st)


def test_session_records_corner_mistakes_and_feeds_engineer_catalunya():
    s = _init_session(track_id=4, track_length=4650)
    # 1. Front lockup under heavy braking at T1 (700-880 m)
    _feed_sample(s, 1.0, d=760.0, speed=200.0, brk=0.9, steer=0.1, wheel_slip=(0.0, 0.0, -0.35, -0.10))
    _feed_sample(s, 1.1, d=770.0, speed=180.0, brk=0.5, steer=0.2, wheel_slip=(0.0, 0.0, 0.0, 0.0))

    # 2. Wheelspin on throttle exit at T3 (1030-1500 m)
    _feed_sample(s, 2.0, d=1250.0, speed=140.0, thr=0.85, steer=0.4, wheel_slip=(0.35, 0.10, 0.0, 0.0))
    _feed_sample(s, 2.1, d=1260.0, speed=145.0, thr=0.85, steer=0.3, wheel_slip=(0.0, 0.0, 0.0, 0.0))

    # 3. Solo braking slide at T4 (1500-1880 m)
    _feed_sample(s, 3.0, d=1600.0, speed=160.0, brk=0.75, steer=0.3, yaw_deg=42.0)
    _feed_sample(s, 3.3, d=1620.0, speed=130.0, brk=0.20, steer=0.1, yaw_deg=5.0)

    # 4. Wide excursion at T10 (3350-3580 m): >5 s after the slide so it isn't merged as part of the spin
    _feed_sample(s, 10.0, d=3450.0, speed=120.0, brk=0.4, steer=-0.3, surfaces=(0, 7, 0, 7))
    _feed_sample(s, 10.4, d=3470.0, speed=120.0, brk=0.2, steer=-0.2, surfaces=(0, 0, 0, 0))

    mistakes = s.corner_mistakes(s.cars[0])
    by_kind = {(m["kind"], m["corner"]) for m in mistakes}
    assert ("front_lockup", "T1") in by_kind
    assert ("wheelspin", "T3") in by_kind
    assert ("slide", "T4") in by_kind
    assert ("wide", "T10") in by_kind

    _, by_name = E.inputs(s)
    res = E.payload(by_name["tester"], "tester", [])["analysis"]
    cs = res["cornerSuggestions"]
    by_corner = {(x["corner"], x["category"]): x for x in cs}
    assert ("T1", "Braking") in by_corner
    assert ("T3", "Throttle") in by_corner
    assert ("T4", "Braking") in by_corner
    assert ("T10", "Braking") in by_corner or ("T10", "Cornering") in by_corner
    # Setup recommendations still coexist cleanly
    assert "recommendations" in res


def test_auto_zones_retroactively_resolve_lap1_mistakes():
    # Singapore (trackId=12) has no built-in CORNERS table; zones come from _auto_zones()
    s = _init_session(track_id=12, track_length=3000)
    # Lap 1: front lockup at d=1000 m before auto_zones exist
    _feed_sample(s, 1.0, lap_num=1, d=1000.0, speed=180.0, brk=0.9, wheel_slip=(0.0, 0.0, -0.35, 0.0), lap_time=10.0)
    _feed_sample(s, 1.1, lap_num=1, d=1010.0, speed=170.0, brk=0.0, wheel_slip=(0.0, 0.0, 0.0, 0.0), lap_time=10.1)

    # Populate a clean lap trace of >200 bins on lap 1 with a clear speed minimum around d=1000 m
    c = s.cars[0]
    for b in range(0, 600):
        d_m = b * LP.BIN
        spd = 250 - max(0, 150 - abs(d_m - 1000))
        c.cur_trace[b] = (spd, 1.0 if spd > 180 else 0.0, 0.8 if 850 <= d_m < 1000 else 0.0, 0.2, 4)
    # Cross start/finish line at end of lap 1 -> triggers _auto_zones()
    _feed_sample(s, 90.0, lap_num=2, d=10.0, speed=250.0, thr=1.0, lap_time=0.1, last_lap=88.0)

    assert s.zones_auto is True and s.zones
    mistakes = s.corner_mistakes(c)
    assert any(m["kind"] == "front_lockup" and m["corner"] == "C1" for m in mistakes), mistakes


def test_attach_episode_corners_does_not_mutate_feed_laps():
    s = _init_session(track_id=4, track_length=4650)
    feed = s.engineers[0]
    # Start lap 1 from the line (currentLapTime <= 1.0) so EngineerFeed does not treat it as partial
    _feed_sample(s, 0.5, lap_num=1, d=10.0, speed=220.0, thr=1.0, lap_time=0.1)
    # Emit a front_lock episode at d=760 m on lap 1, then complete lap 1
    _feed_sample(s, 10.0, lap_num=1, d=760.0, speed=200.0, brk=0.9,
                 brake_temp=(500, 500, 720, 720), lap_time=10.0)
    _feed_sample(s, 10.5, lap_num=1, d=800.0, speed=140.0, brk=0.0,
                 brake_temp=(500, 500, 520, 520), lap_time=10.5)
    _feed_sample(s, 90.0, lap_num=2, d=10.0, speed=240.0, thr=1.0, lap_time=0.1, last_lap=89.0)

    assert feed.completed_laps[0]["episodes"]["front_lock"]["episodes"][0]["corner"] is None
    _, by_name = E.inputs(s)
    snap = by_name["tester"]["snapshot"]
    assert snap["completedLaps"][0]["episodes"]["front_lock"]["episodes"][0]["corner"] == "T1"
    # EngineerFeed internal completed_laps must remain untouched
    assert feed.completed_laps[0]["episodes"]["front_lock"]["episodes"][0]["corner"] is None


# ---------------------------------------------------------------------- coaching rules
def test_braking_rules_cover_all_mistake_types():
    snap = _golden_snapshot()
    snap["cornerMistakes"] = [
        {"corner": "T1", "kind": "front_lockup", "steer": 0.4, "d": 760},
        {"corner": "T10", "kind": "rear_lockup", "steer": 0.05, "d": 3400},
        {"corner": "T4", "kind": "wide", "d": 1550},
    ]
    snap["cornerProfile"] = [
        {"corner": "T1", "start": 700, "laps": 3, "brake_at_apex": 0.30, "brakes_pct": 100},
        {"corner": "T4", "start": 1500, "laps": 3, "brake_at": 1520.0, "brakes_pct": 100,
         "brake_spread": 32.0, "brake_range": [1505, 1537]},
        {"corner": "T5", "start": 1880, "laps": 3, "brakes_pct": 100, "brake_at": 1910.0,
         "overlap_m": 15.0, "coast_m": 45.0},
        {"corner": "T7", "start": 2440, "laps": 3, "brakes_pct": 100, "brake_at": 2460.0,
         "ref_brake_at": 2480.0, "min_v": 110.0, "ref_min_v": 120.0},
    ]
    res = E.analyze_v2(snap)
    cs = res["cornerSuggestions"]
    by_cm = {(c["corner"], c["mistake"]): c for c in cs}

    assert ("T1", "Front Lockup Under Braking") in by_cm
    assert "steering lock" in by_cm[("T1", "Front Lockup Under Braking")]["correction"]
    assert ("T10", "Rear Instability Under Braking") in by_cm
    assert ("T4", "Braking Too Late / Overshooting Entry") in by_cm
    assert ("T4", "Inconsistent Braking Point") in by_cm
    assert ("T5", "Brake & Throttle Overlap") in by_cm
    assert ("T5", "Excessive Coasting Mid-Corner") in by_cm
    assert ("T7", "Braking Too Early / Over-Slowing") in by_cm


def test_straight_line_front_lockup_wording_and_late_apex_braking():
    snap = _golden_snapshot()
    snap["cornerMistakes"] = [{"corner": "T1", "kind": "front_lockup", "steer": 0.05, "d": 740}]
    snap["cornerProfile"] = [
        {"corner": "T1", "start": 700, "laps": 2, "brake_at_apex": 0.05, "brakes_pct": 100},
        {"corner": "T12", "start": 3660, "laps": 3, "brake_at_apex": 0.50, "steer_at_apex": 0.45, "brakes_pct": 100},
    ]
    cs = E.build_corner_suggestions(snap)
    by_cm = {(c["corner"], c["mistake"]): c for c in cs}
    assert "bleed off peak brake pressure" in by_cm[("T1", "Front Lockup Under Braking")]["correction"]
    assert ("T12", "Braking Too Late / Overshooting Entry") in by_cm


def test_throttle_and_cornering_rules():
    snap = _golden_snapshot()
    snap["cornerMistakes"] = [
        {"corner": "T3", "kind": "wheelspin", "d": 1250},
        {"corner": "T9", "kind": "cut", "d": 2850},
        {"corner": "T16", "kind": "wide", "d": 4350},
    ]
    snap["cornerProfile"] = [
        {"corner": "T3", "start": 1030, "laps": 3, "throttle_at_apex": 0.45, "steer_at_apex": 0.55},
        {"corner": "T5", "start": 1880, "laps": 3, "throttle_lift_pct": 67},
        {"corner": "T10", "start": 3350, "laps": 3, "throttle_delay_m": 45.0,
         "ref_throttle_delay_m": 20.0, "exit_v": 160.0, "best_exit_v": 172.0},
        {"corner": "T16", "start": 4230, "laps": 3, "brakes_pct": 0},  # non-braking corner wide -> Cornering
    ]
    cs = E.build_corner_suggestions(snap)
    by_cm = {(c["corner"], c["mistake"]): c for c in cs}

    assert ("T3", "Wheelspin / Oversteer on Throttle Exit") in by_cm
    assert ("T5", "Abrupt Early Throttle (Mid-Exit Lift)") in by_cm
    assert ("T10", "Late Full-Throttle Pickup on Exit") in by_cm
    assert ("T16", "Mid-Corner Understeer / Running Wide") in by_cm
    assert ("T9", "Cutting Inside Kerb (Track Limits)") in by_cm


def test_straight_labels_ignored_and_caps_enforced():
    snap = _golden_snapshot()
    snap["cornerMistakes"] = [
        {"corner": "Main straight", "kind": "wheelspin", "d": 200},
        {"corner": "?", "kind": "front_lockup", "d": 300},
        {"corner": None, "kind": "cut", "d": 400},
        # T1 has 3 distinct mistakes -> cap at 2 per corner, preferring distinct categories
        {"corner": "T1", "kind": "front_lockup", "d": 740},
        {"corner": "T1", "kind": "wheelspin", "d": 820},
        {"corner": "T1", "kind": "cut", "d": 780},
    ]
    # Add 13 more corners with mistakes to verify global cap of 12
    for idx in range(2, 16):
        snap["cornerMistakes"].append({"corner": f"T{idx}", "kind": "cut", "d": 800 + idx * 100})

    cs = E.build_corner_suggestions(snap)
    assert all(c["corner"] not in ("Main straight", "?", None) for c in cs)
    t1_items = [c for c in cs if c["corner"] == "T1"]
    assert len(t1_items) == 2
    assert {c["category"] for c in t1_items} == {"Braking", "Throttle"}
    assert len(cs) == 12


def test_v2_coexists_with_setup_recommendations_and_v1_clean():
    snap = _golden_snapshot()
    snap["completedLaps"] = [
        {
            "lapNumber": 1, "isValid": True, "valid": True, "kind": "flying", "push": True,
            "episodes": {
                "oversteer": {
                    "count": 4,
                    "episodes": [
                        {"kind": "oversteer", "lap": 1, "start_st": float(i), "corner": "T3", "max_speed_kmh": 130.0}
                        for i in range(4)
                    ],
                }
            },
        }
    ]
    snap["cornerMistakes"] = [{"corner": "T3", "kind": "wheelspin", "d": 1250}]
    res = E.analyze_v2(snap)
    assert len(res["recommendations"]) > 0
    assert len(res["cornerSuggestions"]) == 1
    assert res["cornerSuggestions"][0]["corner"] == "T3"

    # When no corner data exists, v1 does not emit cornerSuggestions (golden compatibility)
    clean_snap = _golden_snapshot()
    v1 = E.analyze(clean_snap)
    assert "cornerSuggestions" not in v1
