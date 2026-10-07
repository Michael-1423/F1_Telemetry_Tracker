"""Race engineer: car-balance diagnostics and setup recommendations for each human driver.

Ported from the F1 2020 AI Race Engineer app:
  - EngineerFeed      <- relay/f1_relay.py  TelemetryState (detectors, lap records, snapshot)
  - analyze()         <- src/lib/engineer_engine.ts  analyzeTelemetryAndGenerateSetup
  - session_briefing  <- src/lib/strategy.ts  buildSessionBriefing
The logic, thresholds and wording are kept exactly as they were there, except the kerb detector (see
below). tests/test_engineer.py checks
analyze() against outputs of the original TypeScript. Snapshots and results use the original's
camelCase field names so the two stay easy to compare.

The original relay ran on the driver's own PC and read that game's player car. Here every game
sends to the host, so each driver gets an EngineerFeed fed only by their own game (wheel slip,
suspension and the car setup are only sent to a player's own game anyway).

Deliberate differences from the relay, all in what it read from the packets rather than the logic:
  - A completed lap's validity is the game's "lap invalid" flag for that lap. The relay's lap parser
    was laid out for F1 2021 and read the penalties byte of the new lap instead.
  - All-zero setup packets are ignored. The game sends them once a car retires or finishes, and the
    relay then worked from a setup of zeros. The relay also started from a made-up setup before the
    game sent one; here there is no analysis until the real setup arrives.
  - Kerb strikes are sharp suspension movements (velocity above 1000 mm/s), counted per lap. The relay
    counted suspension *position* above 0.08 over the whole session, but F1 2020 reports position in
    millimetres, so every sample counted and the engine always diagnosed kerb bottoming. Position isn't
    a usable signal (aero load at speed compresses the car more than kerbs do), and any whole-session
    count eventually passes the engine's "more than 3". The engine rule itself is unchanged.
  - When the driver changes the setup, the detector counts start again. The relay kept counting, so after
    a change the engine went on diagnosing from laps on the old setup and repeated its advice on top of
    the new values (e.g. +1 ride height, then +1 again straight away).
  - Detector thresholds can be changed under [engineer] in the config. The others are the relay's.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from . import lookups as L
from . import packets as P

# Detector thresholds. All but kerb_suspension_velocity are as hard-coded in relay/f1_relay.py.
DEFAULTS: dict[str, float] = {
    # rear wheel slip above this and slip_ratio x the front -> oversteer sample
    "oversteer_slip": 0.22,
    # front wheel slip above this and slip_ratio x the rear...
    "understeer_slip": 0.22,
    "understeer_lat_g": 1.8,  # ...with lateral g above this -> understeer sample
    "slip_ratio": 1.5,
    # any |suspension velocity| above this (mm/s) -> kerb strike sample. Singapore 2026-09-30, >3 per lap:
    # 7/13 laps on the stiffest setup (rear springs 11), 1/13 on the softest (rear springs 1).
    "kerb_suspension_velocity": 1000.0,
    "lock_brake": 0.7,  # brake pedal above this...
    # ...with front brakes this much hotter than the rears -> front locking sample
    "lock_brake_temp_gap": 120.0,
}

FEED_PACKETS = {P.MOTION, P.LAP, P.TELEMETRY, P.STATUS, P.SETUPS}

# Names as the relay's packet_parser.py produced them (they appear in the briefing text).
SESSION_TYPES = {
    0: "Unknown",
    1: "Practice 1",
    2: "Practice 2",
    3: "Practice 3",
    4: "Short Practice",
    5: "Qualifying 1",
    6: "Qualifying 2",
    7: "Qualifying 3",
    8: "Short Qualifying",
    9: "One-Shot Qualifying",
    10: "Race",
    11: "Race 2",
    12: "Time Trial",
}
WEATHER_NAMES = {
    0: "Clear",
    1: "Light Cloud",
    2: "Overcast",
    3: "Light Rain",
    4: "Heavy Rain",
    5: "Storm",
}
TYRE_COMPOUNDS = {
    16: "Soft",
    17: "Medium",
    18: "Hard",
    7: "Intermediate",
    8: "Wet",
}  # visual compound

HANDLING_FEEDBACK_OPTIONS: list[dict[str, str]] = [
    {
        "id": "oversteer_exit",
        "label": "Oversteer on Corner Exit",
        "description": "Rear kicks out when getting on the throttle out of slow/medium corners",
        "symptom": "Differential on-throttle too aggressive or rear anti-roll bar too stiff",
    },
    {
        "id": "understeer_entry",
        "label": "Understeer on Corner Entry",
        "description": "Car won't turn in when trail-braking or entering turns",
        "symptom": "Front wing insufficient, front brake bias too high, or off-throttle diff too locked",
    },
    {
        "id": "understeer_mid",
        "label": "Mid-Corner Push / Washout",
        "description": "Front end scrubs wide at apex while holding steady speed",
        "symptom": "Front anti-roll bar too stiff or lack of front camber / aerodynamic load",
    },
    {
        "id": "high_speed_snap",
        "label": "High-Speed Instability / Snap",
        "description": "Rear feels twitchy or snaps unpredictably in 6th/7th gear corners",
        "symptom": "Rear wing too low or rear ride height too high (excessive rake stall)",
    },
    {
        "id": "kerb_unstable",
        "label": "Kerbs Unsettle the Car",
        "description": "Hitting rumble strips launches or destabilizes the chassis",
        "symptom": "Suspension springs too stiff or ride height bottoming out",
    },
    {
        "id": "front_locking",
        "label": "Front Brakes Locking Up",
        "description": "Front tyres smoke and screech under initial heavy deceleration",
        "symptom": "Front brake bias too far forward or brake pressure too high",
    },
    {
        "id": "rear_locking",
        "label": "Rear Brakes Locking / Rotation on Entry",
        "description": "Car wants to spin around while braking hard in a straight line",
        "symptom": "Brake bias too far rearward",
    },
    {
        "id": "high_tyre_wear",
        "label": "High Tyre Wear / Degradation",
        "description": "Tyres are degrading too quickly over the stint, causing grip loss and a shorter competitive tyre life",
        "symptom": "Tyre pressures or geometry are generating excessive scrub and surface load",
    },
    {
        "id": "tyres_overheating",
        "label": "Tyres Overheating / Blistering",
        "description": "Tyre core temps exceed 108°C causing sudden grip loss after 3 laps",
        "symptom": "Tyre pressure too high or sliding across surface",
    },
    {
        "id": "lacking_top_speed",
        "label": "Lacking Straight-Line Speed",
        "description": "Getting overtaken on long straights / cannot reach speed traps",
        "symptom": "Wings set too high inducing unnecessary aerodynamic drag",
    },
]
FEEDBACK_IDS = {o["id"] for o in HANDLING_FEEDBACK_OPTIONS}


def _handling(setup: dict) -> dict:
    """The setup without fuel load, which isn't a handling change."""
    return {k: v for k, v in setup.items() if k != "fuelLoad"}


def _quad(v: list, nd: int | None = None) -> dict:
    """Game wheel order RL, RR, FL, FR -> {fl, fr, rl, rr}."""
    r = (lambda x: round(x, nd)) if nd is not None else (lambda x: x)
    return {"rl": r(v[0]), "rr": r(v[1]), "fl": r(v[2]), "fr": r(v[3])}


def format_lap_time(seconds: float) -> str:
    if seconds <= 0:
        return "--:--.---"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{mins}:{secs:02d}.{millis:03d}"


# --------------------------------------------------------------------------------------- detectors
class EngineerFeed:
    """One driver's view of their own car, accumulated the way relay/f1_relay.py TelemetryState did."""

    def __init__(self, thresholds: dict | None = None):
        self.th = {**DEFAULTS, **(thresholds or {})}
        self.last_st = 0.0
        self.telemetry: dict[str, Any] = {
            "speed": 0,
            "throttle": 0.0,
            "brake": 0.0,
            "steer": 0.0,
            "gear": 0,
            "engineRPM": 0,
            "drs": False,
            "tyresSurfaceTemperature": {"fl": 95, "fr": 95, "rl": 95, "rr": 95},
            "tyresInnerTemperature": {"fl": 100, "fr": 100, "rl": 100, "rr": 100},
            "brakesTemperature": {"fl": 450, "fr": 450, "rl": 450, "rr": 450},
            "tyresPressure": {"fl": 23.0, "fr": 23.0, "rl": 21.0, "rr": 21.0},
        }
        self.motion: dict[str, Any] = {
            "gForceLateral": 0.0,
            "gForceLongitudinal": 0.0,
            "wheelSlip": [0.0, 0.0, 0.0, 0.0],
            "suspensionPosition": [0.0, 0.0, 0.0, 0.0],
        }
        self.lap_data: dict[str, Any] = {
            "currentLapNum": 1,
            "currentLapTime": 0.0,
            "lastLapTime": 0.0,
            "bestLapTime": 0.0,
            "sector1TimeMs": 0,
            "sector2TimeMs": 0,
            "carPosition": 1,
            "isCurrentLapInvalid": False,
        }
        self.status: dict[str, Any] = {
            "tyreCompound": "Soft",
            "fuelMix": 1,
            "fuelInTank": 25.0,
            "fuelRemainingLaps": 1.2,
            "ersStoreEnergy": 4000000,
            "drsAllowed": False,
        }
        self.damage: dict[str, Any] = {
            "tyresWear": {"fl": 0.0, "fr": 0.0, "rl": 0.0, "rr": 0.0},
            "frontLeftWingDamage": 0,
            "frontRightWingDamage": 0,
            "rearWingDamage": 0,
        }
        self.setup: dict[str, Any] | None = (
            None  # the relay started from a made-up setup; we wait for the game's
        )
        self.completed_laps: list[dict] = []
        self.lap_start_fuel = 0.0
        self.lap_start_wear = {"fl": 0.0, "fr": 0.0, "rl": 0.0, "rr": 0.0}
        self.current_lap_max_speed = 0
        self._temp_sum = {
            "fl": 0,
            "fr": 0,
            "rl": 0,
            "rr": 0,
        }  # the relay kept every sample; sums give the same mean
        self._temp_n = 0
        self.oversteer_events = 0
        self.understeer_events = 0
        self.front_locking_events = 0
        self.rear_locking_events = 0
        self.kerb_bottoming_events = 0  # this lap
        self.kerb_last_lap = 0  # the last completed lap
        self.setup_changed_lap: int | None = (
            None  # lap of the last setup change (counters restart there)
        )
        self._lap_seen = False  # had a lap packet from this game yet
        self._partial_lap = False  # joined mid-lap: the lap in progress isn't logged

    def feed(self, pkt: dict, idx: int) -> None:
        """One decoded packet from this driver's own game; idx is their car (the header's playerCarIndex)."""
        pid = pkt["id"]
        self.last_st = pkt["header"]["sessionTime"]
        if pid == P.MOTION:
            self.on_motion(pkt["carMotionData"][idx], pkt)
        elif pid == P.TELEMETRY:
            self.on_telemetry(pkt["carTelemetryData"][idx])
        elif pid == P.LAP:
            self.on_lap_data(pkt["lapData"][idx])
        elif pid == P.STATUS:
            self.on_status(pkt["carStatusData"][idx])
        elif pid == P.SETUPS:
            self.on_setup(pkt["carSetups"][idx])

    def on_motion(self, cm: dict, extra: dict) -> None:
        m = {
            "gForceLateral": round(cm["gForceLateral"], 2),
            "gForceLongitudinal": round(cm["gForceLongitudinal"], 2),
            # [RL, RR, FL, FR]
            "wheelSlip": [round(x, 4) for x in extra["wheelSlip"]],
            "suspensionPosition": [round(x, 4) for x in extra["suspensionPosition"]],
            "suspensionVelocity": [round(x, 4) for x in extra["suspensionVelocity"]],
        }
        self.motion.update(m)
        th = self.th
        # Oversteer: significant rear wheel slip; understeer: front slip with high lateral g
        slips = m["wheelSlip"]
        if len(slips) >= 4:
            rear_slip = max(slips[0], slips[1])
            front_slip = max(slips[2], slips[3])
            lat_g = abs(m["gForceLateral"])
            if (
                rear_slip > th["oversteer_slip"]
                and rear_slip > front_slip * th["slip_ratio"]
            ):
                self.oversteer_events += 1
            elif (
                front_slip > th["understeer_slip"]
                and front_slip > rear_slip * th["slip_ratio"]
                and lat_g > th["understeer_lat_g"]
            ):
                self.understeer_events += 1
        # Kerb strikes: a wheel punched up or dropped sharply
        if any(
            abs(v) > th["kerb_suspension_velocity"] for v in m["suspensionVelocity"]
        ):
            self.kerb_bottoming_events += 1

    def on_telemetry(self, td: dict) -> None:
        t = {
            "speed": td["speed"],
            "throttle": round(td["throttle"], 3),
            "steer": round(td["steer"], 3),
            "brake": round(td["brake"], 3),
            "gear": td["gear"],
            "engineRPM": td["engineRPM"],
            "drs": bool(td["drs"]),
            "brakesTemperature": _quad(td["brakesTemperature"]),
            "tyresSurfaceTemperature": _quad(td["tyresSurfaceTemperature"]),
            "tyresInnerTemperature": _quad(td["tyresInnerTemperature"]),
            "engineTemperature": td["engineTemperature"],
            "tyresPressure": _quad(td["tyresPressure"], 1),
        }
        self.telemetry.update(t)
        if t["speed"] > self.current_lap_max_speed:
            self.current_lap_max_speed = t["speed"]
        # Thermal sample
        for k, v in t["tyresInnerTemperature"].items():
            self._temp_sum[k] += v
        self._temp_n += 1
        # Brake lockups: heavy braking with the fronts much hotter than the rears
        if t["brake"] > self.th["lock_brake"]:
            b = t["brakesTemperature"]
            b_front_avg = (b["fl"] + b["fr"]) / 2
            b_rear_avg = (b["rl"] + b["rr"]) / 2
            if b_front_avg > b_rear_avg + self.th["lock_brake_temp_gap"]:
                self.front_locking_events += 1

    def on_lap_data(self, ld: dict) -> None:
        lap = {
            "lastLapTime": round(ld["lastLapTime"], 3),
            "currentLapTime": round(ld["currentLapTime"], 3),
            "sector1TimeMs": ld["sector1TimeInMS"],
            "sector2TimeMs": ld["sector2TimeInMS"],
            "bestLapTime": round(ld["bestLapTime"], 3),
            "lapDistance": round(ld["lapDistance"], 1),
            "carPosition": ld["carPosition"],
            "currentLapNum": ld["currentLapNum"],
            "pitStatus": ld["pitStatus"],
            "isCurrentLapInvalid": bool(ld["currentLapInvalid"]),
            "penalties": ld["penalties"],
        }
        if not self._lap_seen:
            # The first lap packet from this game. If the lap is already under way the tracker (or the game)
            # joined mid-session: the relay logged a lap "1" here with the previous lap's time, then a first
            # full lap with half a lap of evidence and the whole stint's tyre wear. Wait for the next line.
            self._lap_seen = True
            self._partial_lap = lap["currentLapTime"] > 1.0
            self.lap_data.update(lap)
            return
        prev_lap_num = self.lap_data.get("currentLapNum", 1)
        # Did we just cross the line into a new lap?
        if lap["currentLapNum"] > prev_lap_num and lap["lastLapTime"] > 0:
            if self._partial_lap:
                self._start_first_full_lap()
            else:
                self._record_completed_lap(prev_lap_num, lap["lastLapTime"], lap)
        self.lap_data.update(lap)

    def _start_first_full_lap(self) -> None:
        """Crossing the line after joining mid-lap: drop what was counted on the partial lap and start the
        per-lap measurements (fuel, wear, temperatures, detectors) from here."""
        self._partial_lap = False
        self.lap_start_fuel = self.status.get("fuelInTank", 0.0)
        self.lap_start_wear = dict(self.damage.get("tyresWear", {"fl": 0, "fr": 0, "rl": 0, "rr": 0}))
        self._temp_sum = {"fl": 0, "fr": 0, "rl": 0, "rr": 0}
        self._temp_n = 0
        self.current_lap_max_speed = 0
        self.oversteer_events = self.understeer_events = 0
        self.kerb_last_lap = self.kerb_bottoming_events = 0

    def on_status(self, cs: dict) -> None:
        self.status.update(
            {
                "fuelMix": cs["fuelMix"],
                "fuelInTank": round(cs["fuelInTank"], 2),
                "fuelRemainingLaps": round(cs["fuelRemainingLaps"], 2),
                "drsAllowed": bool(cs["drsAllowed"]),
                "tyreCompound": TYRE_COMPOUNDS.get(cs["visualTyreCompound"], "Dry"),
                "tyresAgeLaps": cs["tyresAgeLaps"],
                "ersStoreEnergy": round(cs["ersStoreEnergy"], 0),
                "ersDeployMode": cs["ersDeployMode"],
            }
        )
        self.damage.update(
            {
                "tyresWear": _quad(cs["tyresWear"]),
                "frontLeftWingDamage": cs["frontLeftWingDamage"],
                "frontRightWingDamage": cs["frontRightWingDamage"],
                "rearWingDamage": cs["rearWingDamage"],
            }
        )

    def on_setup(self, su: dict) -> None:
        if not any(su.values()):  # other cars' setups arrive as zeros
            return
        old = self.setup
        self.setup = {
            "frontWing": su["frontWing"],
            "rearWing": su["rearWing"],
            "onThrottleDiff": su["onThrottle"],
            "offThrottleDiff": su["offThrottle"],
            "frontCamber": round(su["frontCamber"], 2),
            "rearCamber": round(su["rearCamber"], 2),
            "frontToe": round(su["frontToe"], 2),
            "rearToe": round(su["rearToe"], 2),
            "frontSuspension": su["frontSuspension"],
            "rearSuspension": su["rearSuspension"],
            "frontAntiRollBar": su["frontAntiRollBar"],
            "rearAntiRollBar": su["rearAntiRollBar"],
            "frontRideHeight": su["frontSuspensionHeight"],
            "rearRideHeight": su["rearSuspensionHeight"],
            "brakePressure": su["brakePressure"],
            "brakeBias": su["brakeBias"],
            "rearLeftTyrePressure": round(su["rearLeftTyrePressure"], 1),
            "rearRightTyrePressure": round(su["rearRightTyrePressure"], 1),
            "frontLeftTyrePressure": round(su["frontLeftTyrePressure"], 1),
            "frontRightTyrePressure": round(su["frontRightTyrePressure"], 1),
            "ballast": su["ballast"],
            "fuelLoad": round(su["fuelLoad"], 1),
        }
        if old and _handling(old) != _handling(self.setup):
            self._restart_evidence()

    def _restart_evidence(self) -> None:
        """The driver changed the setup: what the detectors counted belongs to the old one. Without this the
        engine keeps diagnosing (say) kerb bottoming from laps on the old setup and asks for the same change
        again on top of the new values."""
        self.oversteer_events = self.understeer_events = 0
        self.front_locking_events = self.rear_locking_events = 0
        self.kerb_bottoming_events = self.kerb_last_lap = 0
        self.setup_changed_lap = self.lap_data.get("currentLapNum")

    def _record_completed_lap(
        self, lap_num: int, lap_time: float, current_lap_packet: dict
    ) -> None:
        # Fuel used
        curr_fuel = self.status.get("fuelInTank", 0.0)
        fuel_used = (
            round(max(0.0, self.lap_start_fuel - curr_fuel), 2)
            if self.lap_start_fuel > 0
            else 1.85
        )
        self.lap_start_fuel = curr_fuel
        # Tyre wear delta
        curr_wear = self.damage.get("tyresWear", {"fl": 0, "fr": 0, "rl": 0, "rr": 0})
        wear_delta = {
            k: round(max(0.0, curr_wear.get(k, 0) - self.lap_start_wear.get(k, 0)), 1)
            for k in ["fl", "fr", "rl", "rr"]
        }
        self.lap_start_wear = dict(curr_wear)
        # Average inner tyre temps
        avg_temps = {"fl": 100, "fr": 100, "rl": 100, "rr": 100}
        if self._temp_n:
            for k in avg_temps:
                avg_temps[k] = round(self._temp_sum[k] / self._temp_n, 1)
        self._temp_sum = {"fl": 0, "fr": 0, "rl": 0, "rr": 0}
        self._temp_n = 0
        # Sector times (as the relay read them: from the first packet of the new lap)
        s1 = current_lap_packet.get("sector1TimeMs", 0) / 1000.0
        s2 = current_lap_packet.get("sector2TimeMs", 0) / 1000.0
        s3 = (
            round(lap_time - s1 - s2, 3)
            if (s1 > 0 and s2 > 0 and lap_time > (s1 + s2))
            else 0.0
        )
        self.completed_laps.append(
            {
                "lapNumber": lap_num,
                "lapTime": lap_time,
                "lapTimeFormatted": format_lap_time(lap_time),
                "sector1": round(s1, 3) if s1 > 0 else None,
                "sector2": round(s2, 3) if s2 > 0 else None,
                "sector3": round(s3, 3) if s3 > 0 else None,
                "maxSpeedKmh": self.current_lap_max_speed,
                "fuelUsedKg": fuel_used,
                "tyreWear": dict(curr_wear),
                "tyreWearDelta": wear_delta,
                "averageTyreTemps": avg_temps,
                "tyreCompound": self.status.get("tyreCompound", "Soft"),
                "isValid": not self.lap_data.get(
                    "isCurrentLapInvalid", False
                ),  # the lap just completed
                "oversteerEvents": self.oversteer_events,
                "understeerEvents": self.understeer_events,
                "kerbEvents": self.kerb_bottoming_events,
            }
        )
        self.current_lap_max_speed = 0
        self.oversteer_events = 0
        self.understeer_events = 0
        self.kerb_last_lap, self.kerb_bottoming_events = self.kerb_bottoming_events, 0

    def snapshot(self, driver_name: str, session: dict) -> dict:
        """The relay's TelemetrySnapshot for this driver (the engine's input)."""
        return {
            "driverName": driver_name,
            "session": dict(session),
            "telemetry": dict(self.telemetry),
            "motion": dict(self.motion),
            "lapData": dict(self.lap_data),
            "status": dict(self.status),
            "setup": dict(self.setup) if self.setup else None,
            "damage": dict(self.damage),
            "completedLaps": list(self.completed_laps),
            "participants": [],
            "setupChangedLap": self.setup_changed_lap,
            "diagnostics": {
                "oversteerEvents": self.oversteer_events,
                "understeerEvents": self.understeer_events,
                "frontLockingEvents": self.front_locking_events,
                "rearLockingEvents": self.rear_locking_events,
                # a whole lap's worth: the last completed lap, or this one once it has more
                "kerbBottomingEvents": max(
                    self.kerb_bottoming_events, self.kerb_last_lap
                ),
            },
            "timestamp": self.last_st,
        }


def session_info(s) -> dict:
    """The relay's session block, from a model.Session."""
    info = s.info or {}
    return {
        "trackName": (
            L.TRACKS.get(s.track_id, f"Track {s.track_id}")
            if s.track_id is not None
            else "Waiting for track..."
        ),
        "trackId": s.track_id,
        "weather": WEATHER_NAMES.get(s.weather, "Unknown"),
        "weatherId": s.weather,
        "sessionType": SESSION_TYPES.get(s.session_type, f"Type {s.session_type}"),
        "sessionTypeId": s.session_type,
        "airTemperature": info.get("airTemperature", 25),
        "trackTemperature": info.get("trackTemperature", 32),
        "totalLaps": s.total_laps,
        "trackLength": info.get("trackLength", 0),
    }


# ------------------------------------------------------------------------ JavaScript number semantics
def _to_fixed(x: float, digits: int) -> str:
    """Number.prototype.toFixed: rounds the exact binary value, ties away from zero."""
    q = abs(Decimal(x)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    return ("-" if x < 0 else "") + format(q, f".{digits}f")


def _fixed(x: float, digits: int) -> float:
    """Number(x.toFixed(digits))"""
    return float(_to_fixed(x, digits))


def _round(x: float) -> int:
    """Math.round: halves go up."""
    return math.floor(x + 0.5)


def _or0(x):
    """x ?? 0"""
    return 0 if x is None else x


def _num(x: float) -> str:
    """A number interpolated into a JavaScript template string."""
    if isinstance(x, bool):
        return "true" if x else "false"
    if float(x).is_integer():
        return str(int(x))
    return repr(float(x))


# ------------------------------------------------------------------------------------ setup engine
def analyze(snapshot: dict, active_feedback: list[str] | None = None) -> dict:
    """analyzeTelemetryAndGenerateSetup(snapshot, activeFeedback) from engineer_engine.ts."""
    fb = list(active_feedback or [])
    current_setup = dict(snapshot["setup"])
    target: dict = dict(current_setup)
    recommendations: list[dict] = []
    issues: list[dict] = []

    def rec(
        category, parameter, label, current, recommended, delta, unit, reason, priority
    ):
        recommendations.append(
            {
                "category": category,
                "parameter": parameter,
                "label": label,
                "currentValue": current,
                "recommendedValue": recommended,
                "delta": delta,
                "unit": unit,
                "reason": reason,
                "priority": priority,
            }
        )

    # Telemetry metrics
    inner = snapshot["telemetry"].get("tyresInnerTemperature") or {
        "fl": 100,
        "fr": 100,
        "rl": 100,
        "rr": 100,
    }
    front_inner_avg = (inner["fl"] + inner["fr"]) / 2
    rear_inner_avg = (inner["rl"] + inner["rr"]) / 2
    diag = snapshot.get("diagnostics") or {}
    oversteer_count = diag.get("oversteerEvents") or 0
    understeer_count = diag.get("understeerEvents") or 0
    front_lock_count = diag.get("frontLockingEvents") or 0
    kerb_hits = diag.get("kerbBottomingEvents") or 0
    # Behavioural evidence is useful even when the lap was invalid.
    # Invalid laps must not affect lap-time/pace analysis, but events such
    # as oversteer, understeer still describe the car's
    # behaviour and should influence setup recommendations.
    # Only laps on the current setup count: after a change, laps on the old one would
    # keep asking for the change that was just made.
    changed = snapshot.get("setupChangedLap")
    laps = [
        lap
        for lap in snapshot.get("completedLaps", [])
        if changed is None or lap.get("lapNumber", 0) >= changed
    ]
    for lap in laps[-3:]:
        oversteer_count += lap.get("oversteerEvents", 0) or 0
        understeer_count += lap.get("understeerEvents", 0) or 0

    balance = 0  # negative = oversteer, positive = understeer

    # 1. Oversteer vs understeer balance
    if oversteer_count > understeer_count + 2 or "oversteer_exit" in fb:
        balance -= 4
    if (
        understeer_count > oversteer_count + 2
        or "understeer_entry" in fb
        or "understeer_mid" in fb
    ):
        balance += 4
    if front_inner_avg > rear_inner_avg + 7:
        balance += 2  # front tyres overheating = scrubbing/understeering
    if rear_inner_avg > front_inner_avg + 7:
        balance -= 2  # rear tyres overheating = spinning/oversteering

    # DIAGNOSTIC 1: Corner entry understeer / turn-in bite
    if "understeer_entry" in fb or balance >= 3:
        issues.append(
            {
                "title": "Corner Entry Understeer Detected",
                "description": "Car lacks front-axle bite into turn-in; front tyres scrubbing and running high core temperature.",
                "severity": "warning",
            }
        )
        if target["frontWing"] < 11:
            old = target["frontWing"]
            target["frontWing"] = min(11, target["frontWing"] + 1)
            rec(
                "Aerodynamics",
                "frontWing",
                "Front Wing Aero",
                old,
                target["frontWing"],
                target["frontWing"] - old,
                "clicks",
                "Increases front downforce to pin front axle on entry into high & medium speed corners.",
                "High",
            )
        if target["frontAntiRollBar"] > 3:
            old = target["frontAntiRollBar"]
            target["frontAntiRollBar"] = max(2, target["frontAntiRollBar"] - 1)
            rec(
                "Suspension",
                "frontAntiRollBar",
                "Front Anti-Roll Bar",
                old,
                target["frontAntiRollBar"],
                target["frontAntiRollBar"] - old,
                "clicks",
                "Softening front ARB increases front mechanical grip and reduces understeer mid-corner.",
                "Medium",
            )
        if target["offThrottleDiff"] > 52:
            old = target["offThrottleDiff"]
            target["offThrottleDiff"] = max(50, target["offThrottleDiff"] - 3)
            rec(
                "Transmission",
                "offThrottleDiff",
                "Off-Throttle Differential",
                old,
                target["offThrottleDiff"],
                target["offThrottleDiff"] - old,
                "%",
                "Lower off-throttle diff allows wheels to rotate freely under braking, promoting turn-in rotation.",
                "Medium",
            )

    # DIAGNOSTIC 2: Corner exit oversteer / traction snap
    if "oversteer_exit" in fb or balance <= -3:
        issues.append(
            {
                "title": "Traction Loss & Exit Oversteer",
                "description": "Rear axle breakaways detected on throttle pickup. Rear tyres slipping under torque.",
                "severity": "danger",
            }
        )
        if target["onThrottleDiff"] > 55:
            old = target["onThrottleDiff"]
            target["onThrottleDiff"] = max(50, target["onThrottleDiff"] - 5)
            rec(
                "Transmission",
                "onThrottleDiff",
                "On-Throttle Differential",
                old,
                target["onThrottleDiff"],
                target["onThrottleDiff"] - old,
                "%",
                "Unlocking the on-throttle diff prevents the inside wheel from forcing the rear axle to break loose.",
                "High",
            )
        if target["rearAntiRollBar"] > 3:
            old = target["rearAntiRollBar"]
            target["rearAntiRollBar"] = max(2, target["rearAntiRollBar"] - 1)
            rec(
                "Suspension",
                "rearAntiRollBar",
                "Rear Anti-Roll Bar",
                old,
                target["rearAntiRollBar"],
                target["rearAntiRollBar"] - old,
                "clicks",
                "Allows rear tyres to squat and generate lateral traction on corner exit.",
                "High",
            )
        if target["rearLeftTyrePressure"] > 19.9:
            old = target["rearLeftTyrePressure"]
            target["rearLeftTyrePressure"] = max(
                19.5, _fixed(target["rearLeftTyrePressure"] - 0.4, 1)
            )
            target["rearRightTyrePressure"] = target["rearLeftTyrePressure"]
            rec(
                "Tyres",
                "rearLeftTyrePressure",
                "Rear Tyre Pressures",
                old,
                target["rearLeftTyrePressure"],
                _fixed(target["rearLeftTyrePressure"] - old, 1),
                "PSI",
                "Expands the rear tyre contact patch, maximizing traction out of slow corners.",
                "Medium",
            )

    # DIAGNOSTIC 3: High-speed instability
    if "high_speed_snap" in fb:
        issues.append(
            {
                "title": "Aerodynamic & High-Speed Instability",
                "description": "Rear aerodynamic load unstable at high speeds. Snap oversteer risk in rapid transitions.",
                "severity": "danger",
            }
        )
        if target["rearWing"] < 11:
            old = target["rearWing"]
            target["rearWing"] = min(11, target["rearWing"] + 1)
            rec(
                "Aerodynamics",
                "rearWing",
                "Rear Wing Aero",
                old,
                target["rearWing"],
                target["rearWing"] - old,
                "clicks",
                "Bolsters rear downforce to firmly plant the rear diffuser in high-speed sweeps.",
                "High",
            )
        if target["rearToe"] < 0.45:
            old = target["rearToe"]
            target["rearToe"] = min(0.50, _fixed(target["rearToe"] + 0.04, 2))
            rec(
                "Suspension Geometry",
                "rearToe",
                "Rear Toe-in",
                old,
                target["rearToe"],
                _fixed(target["rearToe"] - old, 2),
                "deg",
                "Adds straight-line and high-speed directional stability to rear axle.",
                "Low",
            )

    # DIAGNOSTIC 4: Kerb striking & suspension bottoming
    if "kerb_unstable" in fb or kerb_hits > 3:
        issues.append(
            {
                "title": "Chassis Bottoming on Kerbs",
                "description": "Suspension too stiff or ride height too low; bottoming out over curbs unsettled the car.",
                "severity": "warning",
            }
        )
        if target["frontSuspension"] > 3:
            old = target["frontSuspension"]
            target["frontSuspension"] = max(2, target["frontSuspension"] - 1)
            rec(
                "Suspension",
                "frontSuspension",
                "Front Suspension Springs",
                old,
                target["frontSuspension"],
                target["frontSuspension"] - old,
                "clicks",
                "Softens compliance so car absorbs rumble strips instead of skipping into the air.",
                "Medium",
            )
        if target["frontRideHeight"] < 5:
            old = target["frontRideHeight"]
            target["frontRideHeight"] = min(6, target["frontRideHeight"] + 1)
            target["rearRideHeight"] = min(7, target["rearRideHeight"] + 1)
            rec(
                "Suspension",
                "frontRideHeight",
                "Ride Height (Front & Rear)",
                old,
                target["frontRideHeight"],
                1,
                "clicks",
                "Provides underfloor clearance to prevent floor damage and aerodynamic stalling over kerbs.",
                "Medium",
            )

    # DIAGNOSTIC 5: Brake locking & thermal distribution
    if "front_locking" in fb or front_lock_count > 2:
        issues.append(
            {
                "title": "Front Brake Lockup Tendency",
                "description": "Excessive front braking torque causing front wheels to lock into heavy braking zones.",
                "severity": "warning",
            }
        )
        if target["brakeBias"] > 53:
            old = target["brakeBias"]
            target["brakeBias"] = max(50, target["brakeBias"] - 1)
            rec(
                "Brakes",
                "brakeBias",
                "Front Brake Bias",
                old,
                target["brakeBias"],
                target["brakeBias"] - old,
                "%",
                "Moves braking balance rearward, freeing front tyres from locking up and preventing flat spots.",
                "High",
            )

    if "rear_locking" in fb:
        issues.append(
            {
                "title": "Rear Axle Snapping Under Braking",
                "description": "Rear tyres locking up under deceleration, causing sudden yaw rotation.",
                "severity": "danger",
            }
        )
        if target["brakeBias"] < 60:
            old = target["brakeBias"]
            target["brakeBias"] = min(62, target["brakeBias"] + 2)
            rec(
                "Brakes",
                "brakeBias",
                "Front Brake Bias",
                old,
                target["brakeBias"],
                target["brakeBias"] - old,
                "%",
                "Shifts bias forward to prevent dangerous rear brake lockup spins.",
                "High",
            )
    # DIAGNOSTIC 6: High tyre wear / degradation
    #
    # Use the last three valid laps to estimate degradation rate.
    # This is intentionally separate from tyre temperature: tyres can wear quickly
    # because of scrub/load even while their core temperature remains acceptable.
    valid_laps = [lap for lap in snapshot["completedLaps"] if lap.get("isValid")]

    recent_wear_laps = valid_laps[-3:]

    front_wear_samples = []
    rear_wear_samples = []

    for lap in recent_wear_laps:
        delta = lap.get("tyreWearDelta", {})

        front_wear_samples.append(
            max(
                0.0,
                delta.get("fl", 0.0),
                delta.get("fr", 0.0),
            )
        )

        rear_wear_samples.append(
            max(
                0.0,
                delta.get("rl", 0.0),
                delta.get("rr", 0.0),
            )
        )

    front_wear_rate = (
        sum(front_wear_samples) / len(front_wear_samples) if front_wear_samples else 0.0
    )

    rear_wear_rate = (
        sum(rear_wear_samples) / len(rear_wear_samples) if rear_wear_samples else 0.0
    )

    current_wear = snapshot.get("damage", {}).get(
        "tyresWear",
        {"fl": 0.0, "fr": 0.0, "rl": 0.0, "rr": 0.0},
    )

    front_current_wear = max(
        current_wear.get("fl", 0.0),
        current_wear.get("fr", 0.0),
    )

    rear_current_wear = max(
        current_wear.get("rl", 0.0),
        current_wear.get("rr", 0.0),
    )

    peak_current_wear = max(
        front_current_wear,
        rear_current_wear,
    )

    if "high_tyre_wear" in fb:
        if recent_wear_laps:
            wear_description = (
                f"Recent tyre wear is approximately "
                f"{_to_fixed(front_wear_rate, 1)}%/lap front and "
                f"{_to_fixed(rear_wear_rate, 1)}%/lap rear. "
                f"Peak current wear is {_round(peak_current_wear)}%."
            )
        else:
            wear_description = (
                f"Current peak tyre wear is {_round(peak_current_wear)}%, "
                "but there are not enough valid laps to calculate a wear rate yet."
            )

        issues.append(
            {
                "title": "High Tyre Wear / Degradation",
                "description": wear_description,
                "severity": "warning",
            }
        )

        # Determine which axle is degrading faster.
        if not recent_wear_laps:
            dominant_axle = "front"
        elif front_wear_rate >= rear_wear_rate:
            dominant_axle = "front"
        else:
            dominant_axle = "rear"

        # ---------------------------------------------------------------
        # FRONT TYRE DEGRADATION
        # ---------------------------------------------------------------
        if dominant_axle == "front":
            # Reduce front tyre pressure.
            if target["frontLeftTyrePressure"] > 21.0:
                old = target["frontLeftTyrePressure"]

                target["frontLeftTyrePressure"] = max(
                    21.0,
                    _fixed(old - 0.4, 1),
                )

                target["frontRightTyrePressure"] = target["frontLeftTyrePressure"]

                rec(
                    "Tyres",
                    "frontLeftTyrePressure",
                    "Front Tyre Pressures",
                    old,
                    target["frontLeftTyrePressure"],
                    _fixed(
                        target["frontLeftTyrePressure"] - old,
                        1,
                    ),
                    "PSI",
                    "Reduces front tyre scrub and contact-patch stress to slow degradation.",
                    "High",
                )

            # Reduce front toe.
            if target["frontToe"] > 0.05:
                old = target["frontToe"]

                target["frontToe"] = max(
                    0.05,
                    _fixed(old - 0.02, 2),
                )

                rec(
                    "Suspension Geometry",
                    "frontToe",
                    "Front Toe",
                    old,
                    target["frontToe"],
                    _fixed(
                        target["frontToe"] - old,
                        2,
                    ),
                    "deg",
                    "Reducing front toe lowers rolling scrub and unnecessary tyre wear.",
                    "Medium",
                )

            # Reduce negative front camber slightly.
            if target["frontCamber"] < -2.5:
                old = target["frontCamber"]

                target["frontCamber"] = min(
                    -2.5,
                    _fixed(old + 0.2, 2),
                )

                rec(
                    "Suspension Geometry",
                    "frontCamber",
                    "Front Camber",
                    old,
                    target["frontCamber"],
                    _fixed(
                        target["frontCamber"] - old,
                        2,
                    ),
                    "deg",
                    "Reducing negative camber spreads load across the front tyre and limits inner-edge wear.",
                    "Low",
                )

        # ---------------------------------------------------------------
        # REAR TYRE DEGRADATION
        # ---------------------------------------------------------------
        else:
            # Reduce rear tyre pressure.
            if target["rearLeftTyrePressure"] > 19.5:
                old = target["rearLeftTyrePressure"]

                target["rearLeftTyrePressure"] = max(
                    19.5,
                    _fixed(old - 0.4, 1),
                )

                target["rearRightTyrePressure"] = target["rearLeftTyrePressure"]

                rec(
                    "Tyres",
                    "rearLeftTyrePressure",
                    "Rear Tyre Pressures",
                    old,
                    target["rearLeftTyrePressure"],
                    _fixed(
                        target["rearLeftTyrePressure"] - old,
                        1,
                    ),
                    "PSI",
                    "Reduces rear tyre scrub and contact-patch stress to slow degradation under traction.",
                    "High",
                )

            # Reduce rear toe.
            if target["rearToe"] > 0.20:
                old = target["rearToe"]

                target["rearToe"] = max(
                    0.20,
                    _fixed(old - 0.02, 2),
                )

                rec(
                    "Suspension Geometry",
                    "rearToe",
                    "Rear Toe",
                    old,
                    target["rearToe"],
                    _fixed(
                        target["rearToe"] - old,
                        2,
                    ),
                    "deg",
                    "Reducing rear toe lowers rolling scrub and unnecessary tyre temperature/wear.",
                    "Medium",
                )

            # Reduce negative rear camber slightly.
            if target["rearCamber"] < -1.0:
                old = target["rearCamber"]

                target["rearCamber"] = min(
                    -1.0,
                    _fixed(old + 0.2, 2),
                )

                rec(
                    "Suspension Geometry",
                    "rearCamber",
                    "Rear Camber",
                    old,
                    target["rearCamber"],
                    _fixed(
                        target["rearCamber"] - old,
                        2,
                    ),
                    "deg",
                    "Reducing negative camber spreads rear-axle load and limits inner-edge wear.",
                    "Low",
                )

            # Reduce on-throttle differential if rear degradation is dominant.
            if target["onThrottleDiff"] > 50:
                old = target["onThrottleDiff"]

                target["onThrottleDiff"] = max(
                    50,
                    target["onThrottleDiff"] - 3,
                )

                rec(
                    "Transmission",
                    "onThrottleDiff",
                    "On-Throttle Differential",
                    old,
                    target["onThrottleDiff"],
                    target["onThrottleDiff"] - old,
                    "%",
                    "A more open differential reduces rear wheel slip and torque-induced tyre degradation on corner exits.",
                    "Medium",
                )
    # DIAGNOSTIC 7: Tyre thermal overheating (>104°C)
    if "tyres_overheating" in fb or front_inner_avg > 105 or rear_inner_avg > 105:
        issues.append(
            {
                "title": "Tyre Thermal Degradation / Overheating",
                "description": f"Core temperatures reaching {_round(max(front_inner_avg, rear_inner_avg))}°C, outside optimal 95-102°C operating window.",
                "severity": "warning",
            }
        )
        if target["frontLeftTyrePressure"] > 21.8:
            old = target["frontLeftTyrePressure"]
            target["frontLeftTyrePressure"] = max(
                21.0, _fixed(target["frontLeftTyrePressure"] - 0.4, 1)
            )
            target["frontRightTyrePressure"] = target["frontLeftTyrePressure"]
            rec(
                "Tyres",
                "frontLeftTyrePressure",
                "Front Tyre Pressures",
                old,
                target["frontLeftTyrePressure"],
                _fixed(target["frontLeftTyrePressure"] - old, 1),
                "PSI",
                "Lowers internal thermal pressure buildup and widens footprint to dissipate heat.",
                "High",
            )

    # DIAGNOSTIC 8: Straight-line speed deficit
    if "lacking_top_speed" in fb:
        issues.append(
            {
                "title": "Excessive Aerodynamic Drag",
                "description": "Downforce trim is heavily penalizing top speed traps on primary straights.",
                "severity": "info",
            }
        )
        if target["rearWing"] > 2:
            old = target["rearWing"]
            target["rearWing"] = max(1, target["rearWing"] - 1)
            rec(
                "Aerodynamics",
                "rearWing",
                "Rear Wing Aero",
                old,
                target["rearWing"],
                target["rearWing"] - old,
                "clicks",
                "Trims drag on straights for higher terminal top speed and overtaking capability.",
                "Medium",
            )

    label = "Balanced / Neutral"
    if balance <= -5:
        label = "Heavy Oversteer"
    elif balance <= -2:
        label = "Mild Oversteer"
    elif balance >= 5:
        label = "Heavy Understeer"
    elif balance >= 2:
        label = "Mild Understeer"

    briefing = session_briefing(
        snapshot, label, recommendations[0] if recommendations else None
    )
    return {
        "radioMessage": briefing["radioMessage"],
        "briefing": briefing,
        "balanceScore": max(-10, min(10, balance)),
        "balanceLabel": label,
        "recommendations": recommendations,
        "diagnosedIssues": issues,
        "targetSetup": target,
    }


# ---------------------------------------------------------------------------------------- strategy
def _session_kind(snapshot: dict) -> str:
    sid = snapshot["session"].get("sessionTypeId")
    if sid is not None:
        if 1 <= sid <= 4:
            return "practice"
        if 5 <= sid <= 9:
            return "qualifying"
        if sid in (10, 11):
            return "race"
        if sid == 12:
            return "time_trial"
    name = snapshot["session"]["sessionType"].lower()
    if "practice" in name:
        return "practice"
    if "qualif" in name:
        return "qualifying"
    if "race" in name:
        return "race"
    if "time trial" in name:
        return "time_trial"
    return "unknown"


def _max_tyre_wear(snapshot: dict) -> float:
    return max(snapshot["damage"]["tyresWear"].values())


def _recent_wear_per_lap(snapshot: dict) -> float | None:
    samples = [
        max(lap["tyreWearDelta"].values())
        for lap in [l for l in snapshot["completedLaps"] if l["isValid"]][-3:]
    ]
    samples = [w for w in samples if w > 0]
    if not samples:
        return None
    total = 0.0
    for w in (
        samples
        # left to right like Array.reduce; Python's sum() compensates rounding and can differ
    ):
        total += w
    return total / len(samples)


def _next_race_compound(snapshot: dict, laps_remaining: int) -> str:
    weather = _or0(snapshot["session"].get("weatherId"))
    if weather >= 4:
        return "Wet"
    if weather == 3:
        return "Intermediate"
    current = snapshot["status"]["tyreCompound"]
    if current == "Soft":
        return "Medium"
    if current == "Medium":
        return "Soft" if laps_remaining <= 8 else "Hard"
    if current == "Hard":
        return "Soft" if laps_remaining <= 12 else "Medium"
    return (
        "Soft" if laps_remaining <= 7 else "Medium" if laps_remaining <= 18 else "Hard"
    )


def _race_fuel_call(fuel_delta: float, laps_remaining: int) -> str:
    if fuel_delta < -0.1:
        return "Lean"
    if fuel_delta > 1 and laps_remaining > 3:
        return "Rich"
    return "Standard"


def session_briefing(
    snapshot: dict, balance_label: str, top: dict | None = None
) -> dict:
    """buildSessionBriefing(snapshot, balanceLabel, topSetupRecommendation) from strategy.ts."""
    kind = _session_kind(snapshot)
    driver = snapshot.get("driverName") or "Driver"
    setup_action = (
        f"{top['label']} {'+' if top['delta'] > 0 else ''}{_num(top['delta'])} {
            top['unit']
        }"
        if top
        else "keep the current setup"
    )
    session_label = snapshot["session"]["sessionType"]

    if kind == "practice":
        clean_laps_needed = max(
            0, 2 - len([l for l in snapshot["completedLaps"] if l["isValid"]])
        )
        pit_call = (
            f"Stay out for {clean_laps_needed} clean {
                'lap' if clean_laps_needed == 1 else 'laps'
            }"
            if clean_laps_needed > 0
            else "Box when ready to compare setup"
        )
        return {
            "sessionKind": kind,
            "sessionLabel": session_label,
            "headline": "Build a clean setup baseline",
            "pitLabel": "Run plan",
            "pitCall": pit_call,
            "tyreCall": f"Stay on {snapshot['status']['tyreCompound']}",
            "fuelCall": "Standard",
            "rationale": f"{balance_label}. Next setup action: {setup_action}.",
            "radioMessage": f"{driver}, practice programme. {pit_call}. Fuel standard. {balance_label}; {setup_action}.",
        }

    if kind == "qualifying":
        invalid = snapshot["lapData"]["isCurrentLapInvalid"]
        pit_call = (
            "Abort lap, recharge, then box" if invalid else "Prepare tyres, then push"
        )
        return {
            "sessionKind": kind,
            "sessionLabel": session_label,
            "headline": (
                "Reset for the next qualifying run" if invalid else "One clean push lap"
            ),
            "pitLabel": "Run call",
            "pitCall": pit_call,
            "tyreCall": "Soft for the push lap",
            "fuelCall": "Rich",
            "rationale": (
                "This lap is invalid; protect the tyres and battery for the next attempt."
                if invalid
                else "Use the preparation lap to build tyre temperature before committing."
            ),
            "radioMessage": f"{driver}, qualifying mode. {pit_call}. Soft tyres, rich fuel for the push lap.",
        }

    if kind == "race":
        current_lap = max(1, snapshot["lapData"]["currentLapNum"])
        total_laps = max(current_lap, snapshot["session"]["totalLaps"])
        laps_remaining = max(0, total_laps - current_lap)
        wear = _max_tyre_wear(snapshot)
        wear_per_lap = _recent_wear_per_lap(snapshot)
        peak_temp = max(snapshot["telemetry"]["tyresInnerTemperature"].values())
        dmg = snapshot["damage"]
        wing_damage = max(
            _or0(dmg.get("frontLeftWingDamage")), _or0(dmg.get("frontRightWingDamage"))
        )

        pit_call = "Stay out — no stop needed yet"
        pit_reason = f"Peak wear {_round(wear)}%"
        if wear >= 70 or peak_temp >= 112 or wing_damage >= 35:
            pit_call = "Box this lap"
            pit_reason = (
                f"Peak tyre wear is {_round(wear)}%"
                if wear >= 70
                else (
                    f"Tyre temperature is {_round(peak_temp)}°C"
                    if peak_temp >= 112
                    else f"Front-wing damage is {_round(wing_damage)}%"
                )
            )
        elif wear_per_lap and laps_remaining > 2:
            safe_laps = max(0, math.floor((70 - wear) / wear_per_lap))
            if safe_laps <= 2:
                pit_call = "Box this lap"
            elif safe_laps < laps_remaining:
                window_start = min(total_laps - 1, current_lap + max(1, safe_laps - 2))
                window_end = min(total_laps - 1, window_start + 2)
                pit_call = (
                    f"Pit on lap {window_start}"
                    if window_start == window_end
                    else f"Pit window: laps {window_start}–{window_end}"
                )
            pit_reason = f"{_round(wear)}% wear, rising about {
                _to_fixed(wear_per_lap, 1)
            }% per lap"

        tyre_call = _next_race_compound(snapshot, laps_remaining)
        fuel_delta = snapshot["status"]["fuelRemainingLaps"] - laps_remaining
        fuel_call = _race_fuel_call(fuel_delta, laps_remaining)
        fuel_reason = (
            f"fuel delta {_to_fixed(fuel_delta, 1)} laps"
            if fuel_call == "Lean"
            else (
                f"fuel surplus {_to_fixed(fuel_delta, 1)} laps"
                if fuel_call == "Rich"
                else "fuel is on target"
            )
        )
        return {
            "sessionKind": kind,
            "sessionLabel": session_label,
            "headline": pit_call,
            "pitLabel": "Pit call",
            "pitCall": pit_call,
            "tyreCall": tyre_call,
            "fuelCall": fuel_call,
            "rationale": f"{pit_reason}; {fuel_reason}. Strategy updates each lap from live wear and fuel.",
            "radioMessage": f"{driver}, {pit_call.lower()}. Fit {tyre_call}. Fuel mix {fuel_call.lower()}. {pit_reason}.",
        }

    return {
        "sessionKind": kind,
        "sessionLabel": session_label,
        "headline": (
            "Focus on a clean benchmark lap"
            if kind == "time_trial"
            else "Waiting for session context"
        ),
        "pitLabel": "Run call",
        "pitCall": "Stay out",
        "tyreCall": f"Current {snapshot['status']['tyreCompound']}",
        "fuelCall": "Standard",
        "rationale": f"{balance_label}. Live setup analysis remains active.",
        "radioMessage": f"{driver}, telemetry received. {balance_label}. Keep the current run clean.",
    }


# ------------------------------------------------------------------------------------ dashboard glue
def inputs(s) -> tuple[list[dict], dict[str, dict]]:
    """Per human driver: a roster row for the dashboard and the engine input (keyed by lower-case name).
    Built by the processing thread; the result is not mutated afterwards, so web threads can read it.
    """
    roster, by_name = [], {}
    sess = session_info(s)
    for c in s.humans():
        name = s.display_name(c.idx)
        feed = s.engineers.get(c.idx)
        row = {
            "idx": c.idx,
            "name": name,
            "team": c.team,
            "color": L.TEAM_COLORS.get(c.team, "#888"),
            "own": c.own,
            "setup": bool(feed and feed.setup),
        }
        roster.append(row)
        by_name[name.lower()] = dict(
            row, snapshot=feed.snapshot(name, sess) if feed else None
        )
    return roster, by_name


def payload(entry: dict | None, name: str, feedback: list[str]) -> dict:
    """What the engineer page shows for one driver, with that viewer's handling feedback applied."""
    fb = [f for f in feedback if f in FEEDBACK_IDS]
    out: dict[str, Any] = {
        "driver": name,
        "found": entry is not None,
        "feedback": fb,
        "options": HANDLING_FEEDBACK_OPTIONS,
        "analysis": None,
        "car": None,
    }
    if entry is None:
        return out
    snap = entry["snapshot"]
    out.update({k: entry[k] for k in ("idx", "name", "team", "color", "own", "setup")})
    if snap is None:
        return out
    laps = snap["completedLaps"]
    out["car"] = {
        "session": snap["session"],
        "telemetry": snap["telemetry"],
        "motion": {"gForceLateral": snap["motion"]["gForceLateral"]},
        "status": snap["status"],
        "damage": snap["damage"],
        "lapData": snap["lapData"],
        "diagnostics": snap["diagnostics"],
        "setup": snap["setup"],
        "laps": len(laps),
        "validLaps": sum(1 for l in laps if l["isValid"]),
        "bestValidLap": min(
            (l["lapTime"] for l in laps if l["isValid"] and l["lapTime"] > 0), default=0
        ),
        "recentLaps": laps[-5:],
        "setupChangedLap": snap.get("setupChangedLap"),
    }
    if snap["setup"]:
        try:
            out["analysis"] = analyze(snap, fb)
        except Exception as e:  # never break a driver's live stream over one bad sample
            out["error"] = f"analysis failed: {e!r}"
    return out
