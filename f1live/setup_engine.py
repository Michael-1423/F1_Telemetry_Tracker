"""Setup engine v2: turns one driver's evidence into a small, coherent set of setup changes.

Replaces analyze() in engineer.py. Same signature (plus two optional arguments) and same output keys,
with a few extra ones (balanceByPhase, confidence, lapsOnSetup, notes, per-issue "evidence", per-rec "deferred").

Pipeline, one function per stage:

  1. _gather()         telemetry + driver feel  ->  symptoms, each with a severity 0..1 and its evidence
  2. _collect_votes()  each symptom votes for the levers that fix it (weighted, only if the lever has room)
  3. _resolve()        votes on the same parameter are netted: opposing symptoms cancel instead of both being applied
  4. stage + build     the strongest few changes are applied now, the rest are shown as "deferred"

Everything the engine "knows" is in three tables you can read top to bottom: TUNING (numbers), PARAMS (the
setup sliders and their windows) and LEVERS (which slider fixes which symptom, in which direction).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

PHASES = ("entry", "mid", "exit")  # braking / coasting / on power, see EngineerFeed.on_motion

# --------------------------------------------------------------------------------------------- tuning
TUNING: dict[str, float] = {
    # --- how much evidence is enough
    "evidence_laps": 3,  # most recent laps ON THE CURRENT SETUP that count
    "laps_full_confidence": 2,  # telemetry evidence is at full weight from this many laps
    "min_phase_samples": 60,  # ~3 s of cornering at 20 Hz in a phase before telemetry says anything
    # --- balance
    "prior_rate": 2.0,  # events/100 samples of "doubt": few events -> score stays near 0
    "telemetry_weight": 0.8,  # max pull of telemetry on a phase's balance score
    "feedback_weight": 0.7,  # pull of a ticked "how it feels" box
    "stale_factor": 0.5,  # a box ticked before the last setup change counts this much
    "mild": 0.20,  # |balance| below this = balanced. The SAME number decides whether advice is given
    "heavy": 0.50,
    "temp_deadband": 5.0,  # front-vs-rear core temp gap (C) ignored
    "temp_balance_weight": 0.25,
    # --- one-axis symptoms (average events per lap on this setup -> severity)
    "kerb_threshold": 3.0,  # as the relay: more than 3 per lap
    "kerb_span": 8.0,
    "lock_threshold": 6.0,  # slip-based lockup samples per lap (~0.3 s at 20 Hz). UNCALIBRATED, see replay
    "lock_span": 20.0,
    "heat_limit": 105.0,  # average inner core temp of an axle
    "heat_span": 10.0,
    # --- how many changes at once
    "max_changes": 3,  # change a few things at a time so the driver can feel each one
    "min_net_vote": 0.12,  # below this a parameter is left alone
}


# ------------------------------------------------------------------------------------------ setup sliders
@dataclass(frozen=True)
class Param:
    keys: tuple[str, ...]  # setup keys written together (a tyre-pressure pair, for instance)
    lo: float  # sensible window, NOT necessarily the game's hard limits
    hi: float
    step: float  # one click
    nd: int  # decimals
    unit: str
    label: str
    category: str
    max_clicks: int = 1  # how many clicks one run may move it when the evidence is strong


def _p(key, lo, hi, step, nd, unit, label, category, max_clicks=1, keys=None) -> Param:
    return Param(tuple(keys or (key,)), lo, hi, step, nd, unit, label, category, max_clicks)


# VERIFY these windows against the game's garage screen: they are the old engine's implied limits where it
# had them, and my recollection of the F1 2019/2020 ranges elsewhere (tyre pressures especially). They are
# the ONLY place limits live, so fixing one is a one-line change.
PARAMS: dict[str, Param] = {
    "frontWing": _p("frontWing", 1, 11, 1, 0, "clicks", "Front Wing Aero", "Aerodynamics", 2),
    "rearWing": _p("rearWing", 1, 11, 1, 0, "clicks", "Rear Wing Aero", "Aerodynamics", 2),
    "onThrottleDiff": _p("onThrottleDiff", 50, 90, 5, 0, "%", "On-Throttle Differential", "Transmission"),
    "offThrottleDiff": _p("offThrottleDiff", 50, 80, 3, 0, "%", "Off-Throttle Differential", "Transmission"),
    "frontCamber": _p("frontCamber", -3.5, -2.5, 0.1, 2, "deg", "Front Camber", "Suspension Geometry", 2),
    "rearCamber": _p("rearCamber", -2.0, -1.0, 0.1, 2, "deg", "Rear Camber", "Suspension Geometry", 2),
    "frontToe": _p("frontToe", 0.05, 0.50, 0.02, 2, "deg", "Front Toe", "Suspension Geometry"),
    "rearToe": _p("rearToe", 0.20, 0.50, 0.02, 2, "deg", "Rear Toe-in", "Suspension Geometry", 2),
    "frontSuspension": _p("frontSuspension", 3, 10, 1, 0, "clicks", "Front Suspension Springs", "Suspension"),
    "rearSuspension": _p("rearSuspension", 3, 10, 1, 0, "clicks", "Rear Suspension Springs", "Suspension"),
    "frontAntiRollBar": _p("frontAntiRollBar", 3, 10, 1, 0, "clicks", "Front Anti-Roll Bar", "Suspension", 2),
    "rearAntiRollBar": _p("rearAntiRollBar", 3, 10, 1, 0, "clicks", "Rear Anti-Roll Bar", "Suspension", 2),
    "frontRideHeight": _p("frontRideHeight", 1, 6, 1, 0, "clicks", "Front Ride Height", "Suspension"),
    "rearRideHeight": _p("rearRideHeight", 1, 7, 1, 0, "clicks", "Rear Ride Height", "Suspension"),
    "brakePressure": _p("brakePressure", 80, 100, 2, 0, "%", "Brake Pressure", "Brakes"),
    "brakeBias": _p("brakeBias", 50, 62, 1, 0, "%", "Front Brake Bias", "Brakes", 2),
    "frontTyrePressure": _p(
        "frontLeftTyrePressure", 22.5, 29.5, 0.4, 1, "PSI", "Front Tyre Pressures", "Tyres",
        keys=("frontLeftTyrePressure", "frontRightTyrePressure"),
    ),
    "rearTyrePressure": _p(
        "rearLeftTyrePressure", 20.5, 26.5, 0.4, 1, "PSI", "Rear Tyre Pressures", "Tyres",
        keys=("rearLeftTyrePressure", "rearRightTyrePressure"),
    ),
}


# ----------------------------------------------------------------------------------------- symptom -> levers
@dataclass(frozen=True)
class Lever:
    param: str
    direction: int  # +1 raise the value, -1 lower it
    weight: float  # 0..1: how good a fix this is. Orders the levers and scales the vote
    why: str


def _l(param, direction, weight, why) -> Lever:
    return Lever(param, direction, weight, why)


# Directions and wording of the rules that existed before are unchanged. Rules marked (new) are the mirror
# image of an existing rule, or cover a symptom the old engine could detect but never fix.
LEVERS: dict[str, list[Lever]] = {
    "entry_us": [
        _l("frontWing", +1, 0.9, "Increases front downforce to pin front axle on entry into high & medium speed corners."),
        _l("offThrottleDiff", -1, 0.7, "Lower off-throttle diff allows wheels to rotate freely under braking, promoting turn-in rotation."),
        _l("frontAntiRollBar", -1, 0.5, "Softening front ARB increases front mechanical grip and reduces understeer."),
        _l("brakeBias", -1, 0.4, "Less front brake bias unloads the front tyres so they can still turn the car while braking. (new)"),
    ],
    "mid_us": [
        _l("frontAntiRollBar", -1, 0.9, "Softening front ARB increases front mechanical grip and reduces understeer mid-corner."),
        _l("frontWing", +1, 0.7, "Increases front downforce to hold the front axle at the apex."),
        _l("rearAntiRollBar", +1, 0.5, "A stiffer rear ARB trades rear grip for rotation, moving the balance toward the front. (new)"),
        _l("frontCamber", -1, 0.4, "More negative front camber adds mid-corner front grip, at some cost in front tyre wear."),
    ],
    "exit_us": [  # (new) the old engine had no fix for understeer on power
        _l("onThrottleDiff", +1, 0.8, "A tighter on-throttle diff helps the car rotate under power."),
        _l("rearAntiRollBar", +1, 0.5, "A stiffer rear ARB trades rear grip for rotation on exit."),
        _l("frontWing", +1, 0.4, "More front downforce keeps the fronts biting as the throttle goes down."),
    ],
    "entry_os": [  # (new)
        _l("offThrottleDiff", +1, 0.8, "A tighter off-throttle diff keeps the rear stable while braking and turning in."),
        _l("brakeBias", +1, 0.5, "More front brake bias stabilises the rear under braking."),
        _l("rearWing", +1, 0.4, "More rear downforce settles the rear on entry."),
    ],
    "mid_os": [  # (new)
        _l("rearAntiRollBar", -1, 0.9, "Softer rear ARB gives the rear more mechanical grip mid-corner."),
        _l("rearWing", +1, 0.7, "More rear downforce settles the rear through the corner."),
        _l("frontAntiRollBar", +1, 0.4, "A stiffer front ARB moves the balance toward the front."),
    ],
    "exit_os": [
        _l("onThrottleDiff", -1, 1.0, "Unlocking the on-throttle diff prevents the inside wheel from forcing the rear axle to break loose."),
        _l("rearAntiRollBar", -1, 0.9, "Allows rear tyres to squat and generate lateral traction on corner exit."),
        _l("rearTyrePressure", -1, 0.5, "Expands the rear tyre contact patch, maximizing traction out of slow corners."),
        _l("rearWing", +1, 0.3, "More rear downforce adds traction in the faster exits. (new)"),
    ],
    "hs_instability": [
        _l("rearWing", +1, 1.0, "Bolsters rear downforce to firmly plant the rear diffuser in high-speed sweeps."),
        _l("rearToe", +1, 0.4, "Adds straight-line and high-speed directional stability to rear axle."),
        _l("rearRideHeight", -1, 0.4, "Less rake reduces the chance of the rear stalling at speed. (new)"),
    ],
    "kerb": [
        # (new) rear springs. The Singapore data in the detector comment (7/13 laps over the limit on rear
        # springs 11, 1/13 on 1) points at the rear, but the old engine only ever softened the front.
        _l("rearSuspension", -1, 0.9, "Softens the rear so it absorbs rumble strips instead of skipping into the air."),
        _l("frontSuspension", -1, 0.8, "Softens compliance so car absorbs rumble strips instead of skipping into the air."),
        _l("frontRideHeight", +1, 0.5, "Provides underfloor clearance to prevent floor damage and aerodynamic stalling over kerbs."),
        _l("rearRideHeight", +1, 0.5, "Provides underfloor clearance to prevent floor damage and aerodynamic stalling over kerbs."),
    ],
    "front_lock": [
        _l("brakeBias", -1, 1.0, "Moves braking balance rearward, freeing front tyres from locking up and preventing flat spots."),
        _l("brakePressure", -1, 0.3, "Less peak brake pressure makes the initial lockup easier to avoid, at some cost in stopping power. (new)"),
    ],
    "rear_lock": [
        _l("brakeBias", +1, 1.0, "Shifts bias forward to prevent dangerous rear brake lockup spins."),
    ],
    "wear_front": [
        _l("frontTyrePressure", -1, 1.0, "Reduces front tyre scrub and contact-patch stress to slow degradation."),
        _l("frontToe", -1, 0.7, "Reducing front toe lowers rolling scrub and unnecessary tyre wear."),
        _l("frontCamber", +1, 0.4, "Reducing negative camber spreads load across the front tyre and limits inner-edge wear."),
    ],
    "wear_rear": [
        _l("rearTyrePressure", -1, 1.0, "Reduces rear tyre scrub and contact-patch stress to slow degradation under traction."),
        _l("rearToe", -1, 0.7, "Reducing rear toe lowers rolling scrub and unnecessary tyre temperature/wear."),
        _l("rearCamber", +1, 0.4, "Reducing negative camber spreads rear-axle load and limits inner-edge wear."),
        _l("onThrottleDiff", -1, 0.5, "A more open differential reduces rear wheel slip and torque-induced tyre degradation on exits."),
    ],
    "heat_front": [
        _l("frontTyrePressure", -1, 1.0, "Lowers internal thermal pressure buildup and widens footprint to dissipate heat."),
        _l("frontCamber", +1, 0.3, "Less negative camber evens out the load across the tread and reduces edge heating."),
    ],
    "heat_rear": [  # (new) the old engine only ever cooled the fronts, even when the rears were the hot ones
        _l("rearTyrePressure", -1, 1.0, "Lowers internal thermal pressure buildup and widens footprint to dissipate heat."),
        _l("rearCamber", +1, 0.3, "Less negative camber evens out the load across the tread and reduces edge heating."),
    ],
    "top_speed": [
        _l("rearWing", -1, 1.0, "Trims drag on straights for higher terminal top speed and overtaking capability."),
        _l("frontWing", -1, 0.4, "Less front wing also trims drag, but moves the balance toward oversteer. (new)"),
    ],
}

# title, description, severity for the issue list
ISSUES: dict[str, tuple[str, str, str]] = {
    "entry_us": ("Corner Entry Understeer Detected", "Car lacks front-axle bite into turn-in while braking.", "warning"),
    "mid_us": ("Mid-Corner Understeer / Push", "Front end runs wide at the apex under steady load.", "warning"),
    "exit_us": ("Understeer on Power", "Front tyres push wide as the throttle goes down.", "warning"),
    "entry_os": ("Oversteer on Corner Entry", "Rear steps out while braking and turning in.", "danger"),
    "mid_os": ("Mid-Corner Oversteer", "Rear slides at the apex under steady load.", "danger"),
    "exit_os": ("Traction Loss & Exit Oversteer", "Rear axle breakaways detected on throttle pickup. Rear tyres slipping under torque.", "danger"),
    "hs_instability": ("Aerodynamic & High-Speed Instability", "Rear aerodynamic load unstable at high speeds. Snap oversteer risk in rapid transitions.", "danger"),
    "kerb": ("Chassis Bottoming on Kerbs", "Suspension too stiff or ride height too low; bottoming out over kerbs unsettles the car.", "warning"),
    "front_lock": ("Front Brake Lockup Tendency", "Excessive front braking torque causing front wheels to lock into heavy braking zones.", "warning"),
    "rear_lock": ("Rear Axle Snapping Under Braking", "Rear tyres locking up under deceleration, causing sudden yaw rotation.", "danger"),
    "wear_front": ("High Front Tyre Wear", "The front tyres are degrading faster than they should.", "warning"),
    "wear_rear": ("High Rear Tyre Wear", "The rear tyres are degrading faster than they should.", "warning"),
    "heat_front": ("Front Tyres Overheating", "Front core temperatures outside the optimal 95-102C window.", "warning"),
    "heat_rear": ("Rear Tyres Overheating", "Rear core temperatures outside the optimal 95-102C window.", "warning"),
    "top_speed": ("Excessive Aerodynamic Drag", "Downforce trim is heavily penalizing top speed on the primary straights.", "info"),
}

# which "how it feels" box pushes which phase's balance score (+ understeer, - oversteer)
FEEDBACK_BALANCE = {
    "entry": (("understeer_entry", +1),),
    "mid": (("understeer_mid", +1),),
    "exit": (("oversteer_exit", -1),),
}


@dataclass
class Sym:
    severity: float = 0.0
    evidence: list = field(default_factory=list)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


# ------------------------------------------------------------------------------------------ 1. evidence
def _laps_on_setup(snapshot: dict, T: dict) -> list[dict]:
    """The most recent completed laps driven on the current setup. The lap during which the setup changed is
    skipped: it is part old setup, and may be an out-lap."""
    changed = snapshot.get("setupChangedLap")
    laps = [
        l for l in snapshot.get("completedLaps", [])
        if changed is None or l.get("lapNumber", 0) > changed
    ]
    return laps[-int(T["evidence_laps"]):]


def _gather(snapshot: dict, fb: set, stale: set, T: dict) -> tuple[dict, dict, dict]:
    laps = _laps_on_setup(snapshot, T)
    n = len(laps)
    conf = min(1.0, n / T["laps_full_confidence"])
    syms: dict[str, Sym] = {}

    def driver(fid: str) -> float:
        """How hard the driver's tick-box pushes: full weight, halved if ticked before the last setup change."""
        if fid not in fb:
            return 0.0
        return T["feedback_weight"] * (T["stale_factor"] if fid in stale else 1.0)

    def add(key: str, sev: float, why: str) -> None:
        """Independent evidence for the same symptom reinforces ("noisy-or"): 0.5 + 0.5 -> 0.75, never above 1."""
        if sev <= 0:
            return
        s = syms.setdefault(key, Sym())
        s.severity = 1 - (1 - s.severity) * (1 - min(1.0, sev))
        s.evidence.append(why)

    # Axle temperatures from the lap averages, NOT the instantaneous sample the old engine used (that one
    # could be a pit-lane or straight-line reading and flipped the diagnosis from poll to poll).
    temps = [l["averageTyreTemps"] for l in laps if l.get("averageTyreTemps")]
    front_t = _mean((t["fl"] + t["fr"]) / 2 for t in temps) if temps else None
    rear_t = _mean((t["rl"] + t["rr"]) / 2 for t in temps) if temps else None

    # --- balance, one signed score per phase of the corner: + understeer, - oversteer
    bal: dict[str, float] = {}
    why: dict[str, list[str]] = {ph: [] for ph in PHASES}
    for ph in PHASES:
        os_n = sum((l.get("oversteerByPhase") or {}).get(ph, 0) for l in laps)
        us_n = sum((l.get("understeerByPhase") or {}).get(ph, 0) for l in laps)
        samples = sum((l.get("phaseSamples") or {}).get(ph, 0) for l in laps)
        tel = 0.0
        if samples >= T["min_phase_samples"]:
            # Rates per 100 cornering samples of this phase: comparable between Monaco and Monza, and between
            # a 1-lap and a 3-lap window (the old engine compared raw counts). The prior shrinks small
            # samples toward 0, so 2 stray events can't read as "heavy understeer".
            os_r, us_r = 100.0 * os_n / samples, 100.0 * us_n / samples
            tel = (us_r - os_r) / (us_r + os_r + T["prior_rate"])
            if os_n or us_n:
                why[ph].append(f"{us_n} understeer / {os_n} oversteer samples in {ph} phase over {n} lap(s)")
        tel *= conf * T["telemetry_weight"]
        drv = sum(sign * driver(fid) for fid, sign in FEEDBACK_BALANCE[ph])
        if drv:
            why[ph].append("driver reports it")
        bal[ph] = tel + drv  # feel and telemetry disagreeing cancel out: that IS the right answer

    if front_t is not None and rear_t is not None:
        d = front_t - rear_t  # hot fronts hint at scrubbing/understeer, hot rears at slide/oversteer
        if abs(d) > T["temp_deadband"]:
            bal["mid"] += math.copysign(
                min(1.0, (abs(d) - T["temp_deadband"]) / 15.0) * T["temp_balance_weight"] * conf, d
            )
            why["mid"].append(f"front tyres {abs(d):.0f}C {'hotter' if d > 0 else 'cooler'} than rear")

    for ph in PHASES:
        bal[ph] = _clamp(bal[ph], -1.0, 1.0)
        if abs(bal[ph]) >= T["mild"]:
            add(f"{ph}_{'us' if bal[ph] > 0 else 'os'}", abs(bal[ph]), "; ".join(why[ph]) or "balance")

    # --- one-axis symptoms: telemetry severity and driver feel combine with noisy-or
    kerb = _mean(l.get("kerbEvents", 0) or 0 for l in laps)
    if kerb > T["kerb_threshold"]:
        add("kerb", min(1.0, 0.4 + (kerb - T["kerb_threshold"]) / T["kerb_span"]),
            f"{kerb:.1f} kerb strikes per lap over {n} lap(s)")
    add("kerb", driver("kerb_unstable"), "driver reports it")

    for key, fid, fld in (("front_lock", "front_locking", "frontLockEvents"),
                          ("rear_lock", "rear_locking", "rearLockEvents")):
        v = _mean(l.get(fld, 0) or 0 for l in laps)
        if v > T["lock_threshold"]:
            add(key, min(1.0, 0.4 + (v - T["lock_threshold"]) / T["lock_span"]),
                f"{v:.0f} lockup samples per lap over {n} lap(s)")
        add(key, driver(fid), "driver reports it")

    add("hs_instability", driver("high_speed_snap"), "driver reports it")
    add("top_speed", driver("lacking_top_speed"), "driver reports it")

    for axle, t in (("front", front_t), ("rear", rear_t)):  # each axle on its own, the old engine only cooled the front
        if t is not None and t > T["heat_limit"]:
            add(f"heat_{axle}", min(1.0, 0.4 + (t - T["heat_limit"]) / T["heat_span"]),
                f"{axle} tyre core averaging {t:.0f}C over {n} lap(s)")
    if "tyres_overheating" in fb:
        axle = "rear" if (front_t is not None and rear_t is not None and rear_t > front_t) else "front"
        add(f"heat_{axle}", driver("tyres_overheating"), "driver reports it")

    if "high_tyre_wear" in fb:
        valid = [l for l in laps if l.get("isValid") and l.get("tyreWearDelta")]
        f_rate = _mean(max(l["tyreWearDelta"]["fl"], l["tyreWearDelta"]["fr"]) for l in valid)
        r_rate = _mean(max(l["tyreWearDelta"]["rl"], l["tyreWearDelta"]["rr"]) for l in valid)
        axle = "rear" if valid and r_rate > f_rate else "front"
        txt = (f"wear about {f_rate:.1f}%/lap front, {r_rate:.1f}%/lap rear" if valid
               else "no valid laps on this setup yet, assuming the fronts")
        add(f"wear_{axle}", driver("high_tyre_wear"), txt)

    return syms, bal, {"laps": n, "confidence": conf}


# ------------------------------------------------------------------------------------------------ 2. votes
def _headroom(p: Param, cur: float, direction: int) -> float:
    """1.0 with two or more clicks of room, 0.5 with one, 0 at the limit (the lever is then skipped, and the
    next lever for that symptom gets its turn)."""
    room = (p.hi - cur) if direction > 0 else (cur - p.lo)
    if room <= 1e-9:
        return 0.0
    return min(1.0, room / (2 * p.step))


def _collect_votes(syms: dict, setup: dict, T: dict) -> tuple[dict, list[str]]:
    votes: dict[str, dict[str, list]] = {}
    exhausted: list[str] = []
    for key, sym in sorted(syms.items(), key=lambda kv: -kv[1].severity):
        n_levers = 1 if sym.severity < 0.35 else 2 if sym.severity < 0.7 else 3  # worse -> more levers
        ranked = []
        for lv in LEVERS[key]:
            p = PARAMS[lv.param]
            if p.keys[0] not in setup:
                continue
            h = _headroom(p, setup[p.keys[0]], lv.direction)
            if h > 0:
                ranked.append((lv.weight * h, lv))
        ranked.sort(key=lambda x: -x[0])
        if not ranked:
            exhausted.append(key)
        for score, lv in ranked[:n_levers]:
            side = "up" if lv.direction > 0 else "down"
            votes.setdefault(lv.param, {"up": [], "down": []})[side].append((sym.severity * score, key, lv.why))
    return votes, exhausted


# ---------------------------------------------------------------------------------------------- 3. resolve
def _resolve(votes: dict, setup: dict, T: dict) -> tuple[list[dict], list[str], list[str]]:
    proposals: list[dict] = []
    conflicts: list[str] = []
    weak: list[str] = []
    for name, v in votes.items():
        p = PARAMS[name]
        up = sum(x[0] for x in v["up"])
        down = sum(x[0] for x in v["down"])
        net = up - down
        if abs(net) < T["min_net_vote"]:
            if up > T["min_net_vote"] and down > T["min_net_vote"]:
                a = ISSUES[max(v["up"])[1]][0]
                b = ISSUES[max(v["down"])[1]][0]
                conflicts.append(f"{p.label}: '{a}' wants it up, '{b}' wants it down. Left alone.")
            else:
                weak.append(ISSUES[(v["up"] or v["down"])[0][1]][0])
            continue
        direction = 1 if net > 0 else -1
        side = sorted(v["up"] if direction > 0 else v["down"], reverse=True)
        cur = setup[p.keys[0]]
        clicks = 2 if (abs(net) >= 0.75 and p.max_clicks >= 2) else 1
        new = round(_clamp(cur + direction * p.step * clicks, p.lo, p.hi), p.nd)
        if p.nd == 0:
            new = int(new)
        if abs(new - cur) < 10 ** -(p.nd + 1):
            continue
        reasons: list[str] = []
        for _, _, text in side:  # the two strongest distinct reasons
            if text not in reasons:
                reasons.append(text)
        reason = reasons[0] + (f" Also: {reasons[1][0].lower()}{reasons[1][1:]}" if len(reasons) > 1 else "")
        delta = round(new - cur, p.nd)
        proposals.append({
            "net": abs(net),
            "param": p,
            "rec": {
                "category": p.category, "parameter": p.keys[0], "label": p.label,
                "currentValue": cur, "recommendedValue": new,
                "delta": int(delta) if p.nd == 0 else delta,
                "unit": p.unit, "reason": reason,
                "priority": "High" if abs(net) >= 0.6 else "Medium" if abs(net) >= 0.3 else "Low",
            },
        })
    proposals.sort(key=lambda x: -x["net"])
    return proposals, conflicts, weak


# --------------------------------------------------------------------------------------------- 4. analyze
def _default_briefing() -> Callable:
    from .engineer import session_briefing  # imported late: engineer.py imports this module

    return session_briefing


def analyze(
    snapshot: dict,
    active_feedback: list[str] | None = None,
    stale_feedback: list[str] | None = None,
    tuning: dict | None = None,
    briefing_fn: Callable | None = None,
) -> dict:
    """Drop-in for engineer.analyze(). stale_feedback: boxes ticked before the last setup change."""
    T = {**TUNING, **(tuning or {})}
    fb, stale = set(active_feedback or []), set(stale_feedback or [])
    setup = dict(snapshot["setup"])

    syms, bal, meta = _gather(snapshot, fb, stale, T)
    votes, exhausted = _collect_votes(syms, setup, T)
    proposals, conflicts, weak = _resolve(votes, setup, T)

    applied, deferred = proposals[: int(T["max_changes"])], proposals[int(T["max_changes"]):]
    target = dict(setup)
    for pr in applied:
        for k in pr["param"].keys:
            target[k] = pr["rec"]["recommendedValue"]
    recommendations = [pr["rec"] | {"deferred": False} for pr in applied] + [
        pr["rec"] | {"deferred": True} for pr in deferred
    ]

    # balance label: the worst phase, with the SAME threshold that decides whether advice is given
    dom = max(PHASES, key=lambda ph: abs(bal[ph]))
    b = bal[dom]
    kind = "Understeer" if b > 0 else "Oversteer"
    label = ("Balanced / Neutral" if abs(b) < T["mild"]
             else f"Heavy {kind}" if abs(b) >= T["heavy"] else f"Mild {kind}")

    issues = [
        {"title": ISSUES[k][0], "description": ISSUES[k][1], "severity": ISSUES[k][2],
         "strength": round(s.severity, 2), "evidence": "; ".join(s.evidence)}
        for k, s in sorted(syms.items(), key=lambda kv: -kv[1].severity)
    ]

    notes: list[str] = []
    if meta["laps"] == 0:
        notes.append("No laps on this setup yet: advice is based on how the car feels only.")
    elif meta["confidence"] < 1:
        notes.append(f"Only {meta['laps']} lap on this setup so far: telemetry evidence is part-weighted.")
    notes += conflicts
    notes += [f"{ISSUES[k][0]}: every lever for this is already at its limit; the rest is in how the car is driven."
              for k in exhausted]
    notes += [f"{t}: evidence is too weak to justify a change yet." for t in dict.fromkeys(weak)]
    if deferred:
        notes.append(f"{len(deferred)} more change(s) held back: apply the first {len(applied)}, drive, then re-check.")
    if stale:
        notes.append("You changed the setup since ticking some boxes: tick them again if they still apply.")

    brief = briefing_fn or _default_briefing()
    briefing = brief(snapshot, label, applied[0]["rec"] if applied else None)
    if not applied and issues:  # don't say "keep the current setup" while diagnosing a problem
        for k in ("rationale", "radioMessage"):
            briefing[k] = briefing[k].replace("keep the current setup", "no setup change available yet (see notes)")

    return {
        "radioMessage": briefing["radioMessage"],
        "briefing": briefing,
        "balanceScore": int(max(-10, min(10, round(b * 10)))),
        "balanceLabel": label,
        "balanceByPhase": {ph: round(bal[ph], 2) for ph in PHASES},
        "confidence": round(meta["confidence"], 2),
        "lapsOnSetup": meta["laps"],
        "recommendations": recommendations,
        "diagnosedIssues": issues,
        "notes": notes,
        "targetSetup": target,
    }
