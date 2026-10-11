"""Tests for f1live/setup_engine.py. They pass their own briefing function, so they don't need engineer.py."""

import random

from f1live import setup_engine as S

BASE_SETUP = {
    "frontWing": 5, "rearWing": 6, "onThrottleDiff": 70, "offThrottleDiff": 60,
    "frontCamber": -3.0, "rearCamber": -1.5, "frontToe": 0.10, "rearToe": 0.30,
    "frontSuspension": 6, "rearSuspension": 6, "frontAntiRollBar": 6, "rearAntiRollBar": 6,
    "frontRideHeight": 3, "rearRideHeight": 4, "brakePressure": 98, "brakeBias": 56,
    "rearLeftTyrePressure": 22.0, "rearRightTyrePressure": 22.0,
    "frontLeftTyrePressure": 24.0, "frontRightTyrePressure": 24.0, "ballast": 6, "fuelLoad": 30.0,
}


def brief(snapshot, label, top):
    action = f"{top['label']} {top['delta']:+} {top['unit']}" if top else "keep the current setup"
    return {"rationale": f"{label}. Next setup action: {action}.", "radioMessage": f"{label}; {action}."}


def lap(n, os=None, us=None, samples=None, kerb=0, front_lock=0, rear_lock=0, temps=(100, 100), wear=(1, 1)):
    z = dict.fromkeys(S.PHASES, 0)
    return {
        "lapNumber": n, "isValid": True,
        "oversteerByPhase": {**z, **(os or {})}, "understeerByPhase": {**z, **(us or {})},
        "phaseSamples": {**dict.fromkeys(S.PHASES, 300), **(samples or {})},
        "kerbEvents": kerb, "frontLockEvents": front_lock, "rearLockEvents": rear_lock,
        "averageTyreTemps": {"fl": temps[0], "fr": temps[0], "rl": temps[1], "rr": temps[1]},
        "tyreWearDelta": {"fl": wear[0], "fr": wear[0], "rl": wear[1], "rr": wear[1]},
    }


def snap(laps=(), setup=None, changed=None):
    return {"setup": {**BASE_SETUP, **(setup or {})}, "completedLaps": list(laps), "setupChangedLap": changed}


def run(s, fb=None, **kw):
    return S.analyze(s, fb, briefing_fn=brief, **kw)


def params(res, applied_only=True):
    return {r["parameter"]: r for r in res["recommendations"] if not (applied_only and r["deferred"])}


def test_no_evidence_is_balanced_and_quiet():
    r = run(snap())
    assert r["balanceLabel"] == "Balanced / Neutral" and not r["recommendations"]


def test_feel_only_exit_oversteer_uses_original_levers():
    r = run(snap(), ["oversteer_exit"])
    p = params(r)
    assert p["onThrottleDiff"]["recommendedValue"] < 70
    assert p["rearAntiRollBar"]["recommendedValue"] < 6
    assert r["targetSetup"]["onThrottleDiff"] == p["onThrottleDiff"]["recommendedValue"]


def test_telemetry_understeer_on_entry_only_touches_entry_levers():
    laps = [lap(i, us={"entry": 25}) for i in (2, 3, 4)]
    r = run(snap(laps))
    assert r["balanceByPhase"]["entry"] > S.TUNING["mild"] > abs(r["balanceByPhase"]["exit"])
    assert "frontWing" in params(r)
    assert "onThrottleDiff" not in params(r)


def test_few_stray_events_do_not_trigger_advice():
    r = run(snap([lap(i, os={"exit": 1}) for i in (2, 3)]))
    assert r["balanceLabel"] == "Balanced / Neutral" and not r["recommendations"]


def test_feel_and_telemetry_disagreeing_cancel():
    laps = [lap(i, us={"exit": 20}) for i in (2, 3)]  # telemetry: understeer on exit
    r = run(snap(laps), ["oversteer_exit"])  # driver: oversteer on exit
    assert abs(r["balanceByPhase"]["exit"]) < 0.3


def test_opposing_symptoms_net_out_instead_of_both_applying():
    r = run(snap(), ["high_speed_snap", "lacking_top_speed"])  # one wants rear wing up, one down
    assert "rearWing" not in params(r, applied_only=False)
    assert any("Rear Wing" in n for n in r["notes"])


def test_each_parameter_appears_once():
    r = run(snap(), ["oversteer_exit", "high_tyre_wear", "tyres_overheating", "kerb_unstable", "high_speed_snap"])
    names = [x["parameter"] for x in r["recommendations"]]
    assert len(names) == len(set(names))


def test_changes_are_staged():
    r = run(snap(), ["oversteer_exit", "high_tyre_wear", "tyres_overheating", "kerb_unstable", "high_speed_snap"])
    assert sum(not x["deferred"] for x in r["recommendations"]) <= S.TUNING["max_changes"]


def test_exhausted_levers_say_so_instead_of_optimal():
    s = snap(setup={"frontWing": 11, "offThrottleDiff": 50, "frontAntiRollBar": 3, "brakeBias": 50})
    r = run(s, ["understeer_entry"])
    assert not r["recommendations"]
    assert any("already at its limit" in n for n in r["notes"])
    assert "keep the current setup" not in r["radioMessage"]


def test_laps_before_a_setup_change_are_ignored():
    laps = [lap(i, us={"entry": 25}) for i in (2, 3, 4)]
    r = run(snap(laps, changed=4))  # lap 4 is the change lap: skipped, nothing left
    assert r["lapsOnSetup"] == 0 and not r["recommendations"]


def test_values_stay_inside_their_windows():
    r = run(snap(setup={"rearTyrePressure": 0}) if False else snap(), ["oversteer_exit", "high_tyre_wear"])
    for rec in r["recommendations"]:
        p = S.PARAMS[next(k for k, v in S.PARAMS.items() if v.keys[0] == rec["parameter"])]
        assert p.lo <= rec["recommendedValue"] <= p.hi


def test_tyre_pressure_pairs_move_together():
    r = run(snap(), ["tyres_overheating"])
    if "frontLeftTyrePressure" in params(r):
        t = r["targetSetup"]
        assert t["frontLeftTyrePressure"] == t["frontRightTyrePressure"] < 24.0


def test_label_and_advice_never_disagree():
    """The reported bug: a Mild/Heavy label with the advice 'keep the current setup' and nothing explaining why."""
    rng = random.Random(7)
    ids = list(S.FEEDBACK_BALANCE["entry"] + S.FEEDBACK_BALANCE["mid"] + S.FEEDBACK_BALANCE["exit"])
    all_fb = [i for i, _ in ids] + ["high_speed_snap", "kerb_unstable", "front_locking", "rear_locking",
                                    "high_tyre_wear", "tyres_overheating", "lacking_top_speed"]
    for _ in range(3000):
        setup = {}
        for p in S.PARAMS.values():
            v = round(rng.uniform(p.lo, p.hi) / p.step) * p.step
            for k in p.keys:
                setup[k] = round(v, 2)
        laps = [lap(i + 2,
                    os={ph: rng.randint(0, 30) for ph in S.PHASES}, us={ph: rng.randint(0, 30) for ph in S.PHASES},
                    kerb=rng.randint(0, 12), front_lock=rng.randint(0, 40), rear_lock=rng.randint(0, 40),
                    temps=(rng.uniform(90, 118), rng.uniform(90, 118)))
                for i in range(rng.randint(0, 4))]
        r = run(snap(laps, setup), rng.sample(all_fb, rng.randint(0, 4)))
        if r["balanceLabel"] != "Balanced / Neutral" or r["diagnosedIssues"]:
            if not r["recommendations"]:
                assert any(("limit" in n) or ("Left alone" in n) or ("too weak" in n) for n in r["notes"]), r
                assert "keep the current setup" not in r["radioMessage"], r
