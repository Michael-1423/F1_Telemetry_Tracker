"""Live dashboard snapshot and end-of-session summary (JSON + Markdown)."""

from __future__ import annotations

import statistics

from . import lookups as L
from .model import Car, Session, fmt_clock, fmt_time


def _dmg(c: Car) -> dict:
    s = c.status
    return {"fl": s.get("frontLeftWingDamage", 0), "fr": s.get("frontRightWingDamage", 0),
            "rw": s.get("rearWingDamage", 0), "engine": s.get("engineDamage", 0),
            "gearbox": s.get("gearBoxDamage", 0), "drs_fault": s.get("drsFault", 0)}


def _state(s: Session, c: Car) -> str | None:
    rs = c.lap.get("resultStatus", 0)
    if rs in (3, 4, 5, 6, 7) and s.kind == "race":   # practice/quali: cars go back out after "retiring"
        return L.RESULT_STATUS[rs]
    ps = c.lap.get("pitStatus", 0)
    if ps:
        return "pit"
    return None


def car_row(s: Session, c: Car, gaps: dict) -> dict:
    g = gaps.get(c.idx, {})
    last = c.laps[-1] if c.laps else None
    return {
        "idx": c.idx, "name": s.display_name(c.idx), "team": c.team, "number": c.race_number,
        "human": c.human, "own": c.own, "pos": c.position, "grid": c.grid, "lap": c.lap_num,
        "gap": g.get("gap"), "interval": g.get("interval"), "laps_down": g.get("laps_down", 0),
        "last": last["time"] if last else None, "last_valid": last["valid"] if last else True,
        "best": c.best_lap(),
        "tyre": L.tyre_label(c.status.get("actualTyreCompound"), c.status.get("visualTyreCompound")),
        "tyre_age": c.status.get("tyresAgeLaps"), "pits": len(c.pit_stops),
        "penalties": c.lap.get("penalties", 0), "warnings": c.warnings,
        "state": _state(s, c), "flag": L.FIA_FLAGS.get(c.status.get("vehicleFiaFlags", 0)),
        "dmg": _dmg(c), "invalid": bool(c.lap.get("currentLapInvalid")),
        "sector": c.lap.get("sector", 0), "drs": c.tel.get("drs", 0), "speed": c.tel.get("speed"),
    }


def human_live(s: Session, c: Car, gaps: dict, fastest: float | None) -> dict:
    """What only the telemetry knows: tyre wear, fuel mix, ERS, track limits. Sent twice a second."""
    st, t = c.status, c.tel
    wear = st.get("tyresWear") or []
    last = c.laps[-1] if c.laps else None
    best = c.best_lap()
    best_lap = next((l["lap"] for l in c.laps if l["time"] == best), None) if best else None
    tyre = L.tyre_label(st.get("actualTyreCompound"), st.get("visualTyreCompound"))
    g = gaps.get(c.idx, {})
    return {
        "idx": c.idx, "name": s.display_name(c.idx), "team": c.team, "color": L.TEAM_COLORS.get(c.team, "#888"),
        "own": c.own, "pos": c.position, "grid": c.grid, "lap": c.lap_num, "state": _state(s, c),
        "gap": g.get("gap"), "interval": g.get("interval"), "laps_down": g.get("laps_down", 0),
        "tyre": tyre, "tyre_age": st.get("tyresAgeLaps"),
        # wheel order in the game is RL, RR, FL, FR; send FL, FR, RL, RR for display
        "wear": [wear[2], wear[3], wear[0], wear[1]] if len(wear) == 4 else [],
        "wear_avg": round(sum(wear) / 4, 1) if len(wear) == 4 else None,
        "mix": L.FUEL_MIX.get(st.get("fuelMix")), "fuel_kg": round(float(st.get("fuelInTank", 0) or 0), 2),
        "fuel_surplus": round(float(st.get("fuelRemainingLaps", 0) or 0), 2),
        "ers_pct": round(float(st.get("ersStoreEnergy", 0) or 0) / 4e6 * 100, 1),
        "ers_mode": L.ERS_MODE.get(st.get("ersDeployMode")),
        "deploying": s.st < c.deploy_until,
        "ers_lap_mj": round(float(st.get("ersDeployedThisLap", 0) or 0) / 1e6, 2),
        "last": last["time"] if last else None, "last_valid": last["valid"] if last else True,
        "best": best, "best_lap": best_lap, "fastest": bool(best and fastest and abs(best - fastest) < 1e-4),
        "track_limits": c.tl_count, "track_limits_lap": c.cur_tl,
        "penalties": c.lap.get("penalties", 0), "warnings": c.warnings, "dmg": _dmg(c),
        "invalid": bool(c.lap.get("currentLapInvalid")), "now": c.lap.get("currentLapTime"),
        "speed": t.get("speed"), "pit": c.lap.get("pitStatus", 0),
    }


def _deploy_pct(s: Session, lap: dict) -> int | None:
    zones = lap.get("deploy_zones") or []
    if not s.track_length:
        return None
    return round(100 * sum(b - a for a, b in zones) / s.track_length)


def _best_valid(c: Car) -> tuple[float | None, dict | None]:
    laps = [l for l in c.laps if l["valid"] and l.get("timed", True) and l["time"] > 0]
    if not laps:
        return None, None
    best = min(laps, key=lambda l: l["time"])
    return best["time"], best


def timesheet(s: Session) -> list[dict]:
    """Practice / qualifying: each human's fastest valid lap and what went into it."""
    rows = []
    for c in s.humans():
        best, bl = _best_valid(c)
        run = c.runs[bl["run"] - 1] if (bl and bl.get("run") and bl["run"] <= len(c.runs)) else (c.runs[-1] if c.runs else None)
        cc = s.cut_counts(c)
        cuts_on_best = sum(1 for i in (bl or {}).get("excursions", []) if i < len(c.excursions) and c.excursions[i]["kind"] == "cut")
        st, ld = c.status, c.lap
        delta = s.live_delta(c)
        rows.append({
            "idx": c.idx, "name": s.display_name(c.idx), "team": c.team, "color": L.TEAM_COLORS.get(c.team, "#888"),
            "own": c.own, "best": best, "best_lap": bl["lap"] if bl else None,
            "s1": bl["s1"] if bl else None, "s2": bl["s2"] if bl else None, "s3": bl["s3"] if bl else None,
            "tyre": bl["tyre"] if bl else L.tyre_label(st.get("actualTyreCompound"), st.get("visualTyreCompound")),
            "tyre_age": bl["tyre_age"] if bl else st.get("tyresAgeLaps"),
            "ers_mj": bl.get("ers_deployed_mj") if bl else None, "ers_pct": _deploy_pct(s, bl) if bl else None,
            "mix": bl.get("mix") if bl else None, "fuel_start": bl.get("fuel_start") if bl else None,
            "fuel_out": run["fuel_out"] if run else None, "run_mix": run["mix"] if run else None,
            "traffic_pct": bl.get("traffic_pct") if bl else None, "traffic_s": bl.get("traffic_s") if bl else None,
            "cuts": cc["cuts"], "cuts_best": cuts_on_best, "wides": cc["wides"], "four_off": c.tl_count,
            "game_cuts": c.game_cuts, "laps": sum(1 for l in c.laps if l.get("timed", True)), "laps_all": len(c.laps),
            "valid_laps": sum(1 for l in c.laps if l["valid"] and l.get("timed", True)),
            "runs": len(c.runs), "vmax": max((l.get("vmax") or 0) for l in c.laps) if c.laps else None,
            "speed_trap": c.speed_trap or None,
            # live
            "status": L.DRIVER_STATUS.get(ld.get("driverStatus")), "pit": ld.get("pitStatus", 0),
            "now": ld.get("currentLapTime"), "delta": delta, "invalid": bool(ld.get("currentLapInvalid")),
            "traffic_now": c.traffic_now["gap_s"] if c.traffic_now else None,
            "tyre_now": L.tyre_label(st.get("actualTyreCompound"), st.get("visualTyreCompound")),
            "mix_now": L.FUEL_MIX.get(st.get("fuelMix")), "fuel_now": round(float(st.get("fuelInTank", 0) or 0), 1),
            "ers_now": round(float(st.get("ersStoreEnergy", 0) or 0) / 4e6 * 100), "deploying": s.st < c.deploy_until,
        })
    rows.sort(key=lambda r: (r["best"] is None, r["best"] or 0))
    fastest = rows[0]["best"] if rows and rows[0]["best"] else None
    for i, r in enumerate(rows):
        r["pos"] = i + 1 if r["best"] else None
        r["gap"] = round(r["best"] - fastest, 3) if (r["best"] and fastest) else None
    return rows


SETUP_ORDER = [
    ("Aerodynamics", [("frontWing", "Front wing", ""), ("rearWing", "Rear wing", "")]),
    ("Transmission", [("onThrottle", "Diff on throttle", "%"), ("offThrottle", "Diff off throttle", "%")]),
    ("Suspension geometry", [("frontCamber", "Front camber", "°"), ("rearCamber", "Rear camber", "°"),
                             ("frontToe", "Front toe", "°"), ("rearToe", "Rear toe", "°")]),
    ("Suspension", [("frontSuspension", "Front suspension", ""), ("rearSuspension", "Rear suspension", ""),
                    ("frontAntiRollBar", "Front anti-roll bar", ""), ("rearAntiRollBar", "Rear anti-roll bar", ""),
                    ("frontSuspensionHeight", "Front ride height", ""), ("rearSuspensionHeight", "Rear ride height", "")]),
    ("Brakes", [("brakePressure", "Brake pressure", "%"), ("brakeBias", "Brake bias", "%")]),
    ("Tyres", [("frontLeftTyrePressure", "Front-left pressure", " psi"), ("frontRightTyrePressure", "Front-right pressure", " psi"),
               ("rearLeftTyrePressure", "Rear-left pressure", " psi"), ("rearRightTyrePressure", "Rear-right pressure", " psi")]),
    ("Weight", [("ballast", "Ballast", ""), ("fuelLoad", "Fuel load", " kg")]),
]


def _setup_groups(setup: dict) -> list[dict]:
    groups = []
    for title, fields in SETUP_ORDER:
        rows = []
        for key, label, unit in fields:
            v = setup.get(key)
            if v is None:
                continue
            rows.append({"label": label, "value": round(v, 2) if isinstance(v, float) else v, "unit": unit})
        groups.append({"title": title, "rows": rows})
    return groups


def driver_detail(s: Session) -> dict:
    """Setup, corner profile and lap-by-lap data for every human. Rebuilt when a lap completes."""
    drivers = []
    for c in sorted(s.humans(), key=lambda c: c.position or 99):
        st = c.status
        tc, abs_, bb = c.assists.most_common(1)[0][0] if c.assists else (st.get("tractionControl"), st.get("antiLockBrakes"), st.get("frontBrakeBias"))
        has_setup = bool(c.setup) and any(v for v in c.setup.values())
        cold = c.initial_pressures
        laps = []
        for l in c.laps:
            row = {k: l.get(k) for k in ("lap", "time", "s1", "s2", "s3", "valid", "pit", "sc", "clean", "incident",
                                          "position", "tyre", "tyre_age", "wear_avg", "fuel", "fuel_used", "mix",
                                          "ers_deployed_mj", "ers_harvested_mj", "track_limits", "avg_speed", "vmax",
                                          "corners", "kind", "fuel_start", "traffic_pct", "traffic_s", "run")}
            exc = [c.excursions[i] for i in l.get("excursions", []) if i < len(c.excursions)]
            row["cuts"] = sum(1 for e in exc if e["kind"] == "cut")
            row["cuts_where"] = [e["corner"] for e in exc if e["kind"] == "cut"]
            row["wides"] = sum(1 for e in exc if e["kind"] == "wide")
            metres: dict[str, float] = {}
            for a, b in l.get("deploy_zones") or []:
                for d in range(int(a), int(b), 5):
                    name = s.corner(d)
                    metres[name] = metres.get(name, 0) + 5
            total = sum(metres.values())
            row["deploy_pct"] = round(100 * total / s.track_length) if s.track_length else None
            row["deploy_where"] = [n for n, _ in sorted(metres.items(), key=lambda kv: -kv[1])[:3]]
            laps.append(row)
        drivers.append({
            "idx": c.idx, "name": s.display_name(c.idx), "team": c.team, "color": L.TEAM_COLORS.get(c.team, "#888"),
            "own": c.own, "pos": c.position,
            "setup": _setup_groups(c.setup) if has_setup else None,
            "known": {
                "Traction control": L.TC.get(tc), "ABS": "On" if abs_ else "Off",
                "Brake bias": f"{bb}% front" if bb is not None else None,
                "Cold tyre pressures (F / R)": f"{cold[2]:.1f} / {cold[0]:.1f} psi" if cold else None,
                "Fuel at start": f"{c.laps[0]['fuel'] + (c.laps[0].get('fuel_used') or 0):.1f} kg" if c.laps and c.laps[0].get("fuel") is not None else None,
                "Top speed": f"{max((l.get('vmax') or 0) for l in c.laps):.0f} km/h" if c.laps else None,
                "Speed trap": f"{c.speed_trap:.0f} km/h" if c.speed_trap else None,
            },
            "profile": s.corner_profile(c),
            "laps": laps,
            "cuts": s.cut_counts(c), "four_off": c.tl_count,
            "runs": [dict(r, laps=sum(1 for l in c.laps if l.get("run") == r["n"] and l.get("timed", True))) for r in c.runs],
            "track_limits_where": dict(c.tl_where.most_common()),
            "cuts_where": s.cut_counts(c)["cuts_where"],
            "style": s.style_metrics(c),
            "stints": c.stints,
        })
    return {"track": s.track_name, "kind": s.kind, "zones": [{"name": z["name"], "start": z["start"], "end": z["end"]} for z in s.zones],
            "zones_auto": s.zones_auto, "drivers": drivers}


def snapshot(s: Session | None, incidents: list[dict], extra: dict) -> dict:
    if s is None:
        return {"session": None, **extra}
    gaps = s.gaps()
    active = [c for c in s.cars if c.active]
    order = sorted(active, key=lambda c: (c.position or 99))
    events = s.events[-250:]
    return {
        "session": {
            "uid": str(s.uid), "id": s.path.replace("\\", "/").rsplit("/", 1)[-1] if s.path else None,
            "track": s.track_name, "track_id": s.track_id, "type": s.type_name, "is_race": s.is_race, "kind": s.kind,
            "lap": s.leader_lap(), "total_laps": s.total_laps, "sc": L.SAFETY_CAR.get(s.sc_status),
            "weather": L.WEATHER.get(s.weather), "track_temp": s.info.get("trackTemperature"),
            "air_temp": s.info.get("airTemperature"), "clock": fmt_clock(s.race_clock()) if s.race_start_st is not None else None,
            "time_left": s.info.get("sessionTimeLeft"), "paused": s.paused, "ended": s.ended,
            "chequered": s.chequered, "marshal_flags": s.info.get("marshalFlags"),
            "fastest_lap": {"name": s.display_name(s.fastest_lap[0]), "time": s.fastest_lap[1]} if s.fastest_lap else None,
        },
        "sources": [{"key": k, "car": src.player_car, "name": s.display_name(src.player_car) if src.player_car is not None and src.player_car < 22 else "spectator",
                     "rate": round(src.rate, 1), "age": round(s.last_wall - src.last_seen, 1), "packets": src.packets,
                     "primary": k == s.primary, "ended": src.ended} for k, src in s.sources.items()],
        "cars": [car_row(s, c, gaps) for c in order],
        "timesheet": timesheet(s) if not s.is_race else [],
        "live": [human_live(s, c, gaps, min((x.best_lap() or 1e9) for x in active) if active else None) for c in order if c.human],
        "events": [{k: e[k] for k in ("id", "st", "clock", "lap", "kind", "severity", "cars", "names", "humans",
                                      "corner", "text", "incident")} for e in events[-150:]],
        "incidents": incidents[-100:],
        **extra,
    }


# ----------------------------------------------------------------------------- summary
def build_summary(s: Session, incidents: list[dict]) -> dict:
    gaps = s.gaps()
    rows = []
    for c in sorted([c for c in s.cars if c.active], key=lambda c: c.position or 99):
        r = c.result or {}
        race_time = None
        if r.get("totalRaceTime"):
            race_time = r["totalRaceTime"] + (r.get("penaltiesTime") or 0)
        elif c.finish_st is not None and s.race_start_st is not None:
            race_time = c.finish_st - s.race_start_st + (c.lap.get("penalties") or 0)
        rows.append({
            "pos": r.get("position") or c.position, "idx": c.idx, "name": s.display_name(c.idx), "team": c.team,
            "human": c.human, "grid": r.get("gridPosition") or c.grid,
            "laps": r.get("numLaps") or len(c.laps), "status": L.RESULT_STATUS.get(r.get("resultStatus") or c.lap.get("resultStatus", 0)),
            "race_time": round(race_time, 3) if race_time else None,
            "penalties_s": r.get("penaltiesTime") if r else c.lap.get("penalties", 0),
            "best_lap": c.best_lap(), "pit_stops": len(c.pit_stops), "laps_down": gaps.get(c.idx, {}).get("laps_down", 0),
        })
    winner_time = next((r["race_time"] for r in rows if r["pos"] == 1 and r["race_time"]), None)
    winner_laps = next((r["laps"] for r in rows if r["pos"] == 1), None)
    for r in rows:
        down = (winner_laps - r["laps"]) if (winner_laps and r["laps"] < winner_laps) else 0
        r["laps_down"] = down
        r["gap"] = round(r["race_time"] - winner_time, 3) if (r["race_time"] and winner_time and not down) else None

    drivers = []
    for c in [c for c in s.cars if c.active and c.human]:
        laps = [l for l in c.laps if l["time"] > 0]
        clean = [l["time"] for l in laps if l.get("clean")]
        valid_sectors = [l for l in laps if not l.get("standing") and l.get("timed", True) and l["s1"] and l["s2"] and l["s3"]]
        th = None
        if valid_sectors:
            th = min(l["s1"] for l in valid_sectors) + min(l["s2"] for l in valid_sectors) + min(l["s3"] for l in valid_sectors)
        my_events = [e for e in s.events if c.idx in e["cars"] and e["severity"] in ("major", "minor")]
        drivers.append({
            "idx": c.idx, "name": s.display_name(c.idx), "team": c.team, "own_data": c.own,
            "grid": c.grid, "finish": c.position, "status": L.RESULT_STATUS.get(c.lap.get("resultStatus", 0)),
            "laps": laps, "best_lap": c.best_lap(), "theoretical_best": round(th, 3) if th else None,
            "clean_laps": len(clean), "clean_mean": round(statistics.mean(clean), 3) if clean else None,
            "clean_std": round(statistics.pstdev(clean), 3) if len(clean) > 1 else None,
            "positions": [c.grid] + c.lap_positions,
            "stints": c.stints, "pit_stops": [{k: v for k, v in p.items() if k != "damage_before"} for p in c.pit_stops],
            "penalties_s": c.lap.get("penalties", 0), "warnings": c.warnings,
            "counts": dict(c.counts), "offtrack_where": dict(c.offtrack_where.most_common()),
            "style": s.style_metrics(c), "corners": s.corner_stats(c),
            "assists": {"tc": L.TC.get(c.status.get("tractionControl")), "abs": bool(c.status.get("antiLockBrakes")),
                        "brake_bias": c.status.get("frontBrakeBias"), "fuel_mix": L.FUEL_MIX.get(c.status.get("fuelMix"))},
            "setup": c.setup or None, "speed_trap": c.speed_trap or None,
            "corner_profile": s.corner_profile(c), "track_limits": c.tl_count,
            "track_limits_where": dict(c.tl_where.most_common()),
            "mistakes": [{"clock": e["clock"], "lap": e["lap"], "severity": e["severity"], "kind": e["kind"],
                          "text": e["text"] if e["cars"][0] == c.idx else f"{e['names'][0]}: {e['text']}",
                          "with": [n for i, n in zip(e["cars"], e["names"]) if i != c.idx],
                          "incident": e.get("incident")} for e in my_events],
        })
    if not s.is_race:   # practice / qualifying: order by fastest valid lap
        best = {c.idx: _best_valid(c)[0] for c in s.cars if c.active}
        rows.sort(key=lambda r: (best.get(r["idx"]) is None, best.get(r["idx"]) or 0))
        top = next((best[r["idx"]] for r in rows if best.get(r["idx"])), None)
        for i, r in enumerate(rows):
            r["pos"] = i + 1
            r["best_lap"] = best.get(r["idx"])
            r["race_time"] = None
            r["laps_down"] = 0
            r["gap"] = round(r["best_lap"] - top, 3) if (r["best_lap"] and top and i) else None
    return {
        "session": {"uid": str(s.uid), "track": s.track_name, "type": s.type_name, "kind": s.kind, "total_laps": s.total_laps,
                    "track_length": s.track_length, "weather": L.WEATHER.get(s.weather),
                    "track_temp": s.info.get("trackTemperature"), "air_temp": s.info.get("airTemperature"),
                    "started": s.started_wall, "ended": s.last_wall,
                    "fastest_lap": {"name": s.display_name(s.fastest_lap[0]), "time": s.fastest_lap[1]} if s.fastest_lap else None,
                    "final_classification_received": s.final is not None},
        "classification": rows,
        "drivers": drivers,
        "race_control": [{"clock": e["clock"], "text": e["text"]} for e in s.events if e["kind"] == "race_control"],
        "incidents": incidents,
        "sources": [{"key": k, "car": v.player_car, "packets": v.packets} for k, v in s.sources.items()],
        "detail": driver_detail(s),
        "timesheet": timesheet(s) if not s.is_race else [],
    }


def summary_markdown(sm: dict) -> str:
    se = sm["session"]
    out = [f"# {se['track']} · {se['type']} · {se['total_laps']} laps\n"]
    w = out.append
    w(f"Weather {se['weather']}, track {se['track_temp']} °C, air {se['air_temp']} °C."
      + (f" Fastest lap: {se['fastest_lap']['name']} {fmt_time(se['fastest_lap']['time'])}." if se.get("fastest_lap") else ""))
    if se.get("kind") != "race" and sm.get("timesheet"):
        w("\n## Timesheet (human drivers)\n")
        w("| Pos | Driver | Fastest | Gap | S1 | S2 | S3 | Tyre | ERS MJ | Mix | Fuel out | Traffic | Cuts | 4 off | Laps |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for t in sm["timesheet"]:
            w(f"| {t['pos'] or '-'} | {t['name']} | {fmt_time(t['best'])} | {('+' + format(t['gap'], '.3f')) if t['gap'] else ''} "
              f"| {t['s1'] or '-'} | {t['s2'] or '-'} | {t['s3'] or '-'} | {t['tyre']} ({t['tyre_age']}) | {t['ers_mj'] if t['ers_mj'] is not None else '-'} "
              f"| {t['mix'] or '-'} | {t['fuel_out'] if t['fuel_out'] is not None else '-'} kg | {t['traffic_pct'] if t['traffic_pct'] is not None else '-'}% "
              f"| {t['cuts']} | {t['four_off']} | {t['valid_laps']}/{t['laps']} |")
    w("\n## Classification\n")
    w("| Pos | Driver | Team | Grid | Laps | Time / gap | Penalties | Best lap | Stops |")
    w("|---|---|---|---|---|---|---|---|---|")
    for r in sm["classification"]:
        if se.get("kind") != "race":
            tg = fmt_time(r.get("best_lap")) if r["pos"] == 1 else (f"+{r['gap']:.3f}" if r.get("gap") is not None else "-")
        else:
            tg = None
        tg = tg or (fmt_time(r["race_time"]) if r["pos"] == 1 and r["race_time"] else
              (f"+{r['gap']:.3f}" if r.get("gap") is not None else (f"+{r['laps_down']} lap" if r.get("laps_down") else r["status"])))
        name = f"**{r['name']}**" if r["human"] else r["name"]
        w(f"| {r['pos']} | {name} | {r['team']} | {r['grid']} | {r['laps']} | {tg} | {r['penalties_s'] or ''} | {fmt_time(r['best_lap'])} | {r['pit_stops']} |")
    for d in sm["drivers"]:
        w(f"\n## {d['name']} ({d['team']}) · P{d['grid']} → P{d['finish']}\n")
        w(f"- Best lap {fmt_time(d['best_lap'])}, theoretical {fmt_time(d['theoretical_best'])}, "
          f"clean-lap mean {fmt_time(d['clean_mean'])} (σ {d['clean_std'] if d['clean_std'] is not None else '-'} s over {d['clean_laps']} laps)")
        w("- Stints: " + " → ".join(f"{x['tyre']} (laps {x['from_lap']}-{x['to_lap']})" for x in d["stints"]))
        if d["pit_stops"]:
            w("- Pit stops: " + "; ".join(f"lap {p['lap']}: {p.get('lane_s')}s lane"
                                          + (f", repaired {', '.join(p['repaired'])}" if p.get("repaired") else "") for p in d["pit_stops"]))
        w(f"- Penalties {d['penalties_s']}s, warnings {d['warnings']}; counts: "
          + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in sorted(d["counts"].items())))
        prof = d.get("corner_profile") or []
        if prof:
            w(f"\n| Corner | Apex km/h (avg) | Best apex | Entry km/h | Brakes at (avg m) | Range | Gear |")
            w("|---|---|---|---|---|---|---|")
            for r in prof:
                rng = f"{r['brake_range'][0]}–{r['brake_range'][1]}" if r.get("brake_range") else "-"
                w(f"| {r['corner']} | {r['min_v']} | {r['best_min_v']} | {r['entry_v'] or '-'} | {r['brake_at'] or 'no braking'} | {rng} | {r['gear']} |")
            w("")
        w(f"- Track limits (all four wheels off): {d.get('track_limits', 0)}"
          + (" · " + ", ".join(f"{k} {v}" for k, v in (d.get('track_limits_where') or {}).items()) if d.get("track_limits_where") else ""))
        st = d["style"]
        if st:
            w(f"- Style (clean laps): full throttle {st['full_throttle_pct']}%, coasting {st['coasting_pct']}%, "
              f"partial brake {st['partial_brake_pct']}%, trail-braking {st['trail_brake_pct']}%, full lock {st['full_lock_pct']}%, "
              f"steer snaps {st['steer_snaps_pct']}%" + ("" if d["own_data"] else " (remote data, approximate)"))
        a = d["assists"]
        w(f"- Assists/setup: TC {a['tc']}, ABS {'on' if a['abs'] else 'off'}, brake bias {a['brake_bias']}%"
          + (f", wings {d['setup'].get('frontWing')}/{d['setup'].get('rearWing')}" if d.get("setup") else ""))
        if d["mistakes"]:
            w("\n| Clock | Lap | Severity | What happened |")
            w("|---|---|---|---|")
            for m in d["mistakes"]:
                w(f"| {m['clock'] or '-'} | {m['lap']} | {m['severity']} | {m['text']}{' · incident ' + m['incident'] if m.get('incident') else ''} |")
    if sm["incidents"]:
        w("\n## Saved incidents\n")
        for i in sm["incidents"]:
            w(f"- {i['id']} · lap {i.get('lap')} · {i['title']} ({i['status']})")
    return "\n".join(out) + "\n"
