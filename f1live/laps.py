"""Corner zones and per-lap corner metrics (apex speed, braking point, entry speed, throttle-on).

Every lap of every car is recorded as a 5 m-binned trace plus the exact distances where the
brake pedal crossed 50%. At the end of each lap those are reduced to one row per corner.
Corner zones come from the built-in map (COTA), the config file, or are detected automatically
from the fastest clean lap of the session for any other track.
"""

from __future__ import annotations

import statistics

from . import lookups as L

BIN = 5  # metres per trace bin


def builtin_zones(track_id: int | None, custom: dict) -> list[dict]:
    table = custom.get(track_id) or L.CORNERS.get(track_id if track_id is not None else -1)
    if not table:
        return []
    return [{"name": n, "start": float(lo), "end": float(hi)} for n, lo, hi in table if "straight" not in n.lower()]


def auto_zones(trace: dict[int, tuple], track_length: float) -> list[dict]:
    """Find corners as clear speed minima on a reference lap trace."""
    n = int(track_length // BIN) + 1 if track_length else (max(trace) + 1 if trace else 0)
    v: list[float | None] = [None] * n
    for b, row in trace.items():
        if 0 <= b < n:
            v[b] = float(row[0])
    known = [i for i, x in enumerate(v) if x is not None]
    if len(known) < 50:
        return []
    # fill gaps linearly, then smooth
    for a, b in zip(known, known[1:]):
        for i in range(a + 1, b):
            v[i] = v[a] + (v[b] - v[a]) * (i - a) / (b - a)
    first, last = known[0], known[-1]
    for i in range(0, first):
        v[i] = v[first]
    for i in range(last + 1, n):
        v[i] = v[last]
    s = [sum(v[max(0, i - 1):i + 2]) / len(v[max(0, i - 1):i + 2]) for i in range(n)]
    apexes: list[int] = []
    for i in range(n):
        lo, hi = max(0, i - 12), min(n, i + 13)
        if s[i] > min(s[lo:hi]):
            continue
        before = s[max(0, i - 80):i] or [s[i]]
        if max(before) - s[i] < 25:
            continue
        if apexes and i - apexes[-1] < 16:
            if s[i] < s[apexes[-1]]:
                apexes[-1] = i
            continue
        apexes.append(i)
    zones = []
    for k, a in enumerate(apexes):
        am = a * BIN
        prev_m = apexes[k - 1] * BIN if k else None
        next_m = apexes[k + 1] * BIN if k + 1 < len(apexes) else None
        start = max(am - 450, (prev_m + am) / 2 if prev_m is not None else 0.0)
        end = min(am + 200, (am + next_m) / 2 if next_m is not None else float(track_length or am + 200))
        zones.append({"name": f"C{k + 1}", "start": round(start), "end": round(end), "apex_ref": am,
                      "apex_speed_ref": round(s[a])})
    return zones


def corner_metrics(trace: dict[int, tuple], onsets: list[tuple[float, int]], zones: list[dict]) -> dict[str, dict]:
    """One row per corner for one lap. trace: bin -> (speed, thr, brk, steer, gear)."""
    out: dict[str, dict] = {}
    for z in zones:
        b0, b1 = int(z["start"] // BIN), int(z["end"] // BIN)
        bins = [b for b in range(b0, b1 + 1) if b in trace]
        if len(bins) < 3:
            continue
        apex = min(bins, key=lambda b: trace[b][0])
        apex_d = apex * BIN
        brake = next(((d, sp) for d, sp in onsets if z["start"] <= d <= apex_d + BIN), None)
        thr_on = next((b * BIN for b in bins if b > apex and trace[b][1] >= 0.95), None)
        after = [trace[b][0] for b in bins if b >= apex]
        apex_row = trace[apex]
        steering = [abs(trace[b][3]) for b in bins]
        brake_peak = round(max((trace[b][2] for b in bins if b <= apex), default=0.0), 2)
        brk_bins = [b for b in bins if b <= apex + 2 and trace[b][2] >= 0.05]
        if brk_bins and brake_peak >= 0.30:
            last_brk = brk_bins[-1]
            first_thr = next((b for b in bins if b > last_brk and trace[b][1] >= 0.05), None)
            coast_m = sum(
                BIN for b in bins
                if b > last_brk and (first_thr is None or b < first_thr)
                and trace[b][1] < 0.05 and trace[b][2] < 0.05
            )
        else:
            coast_m = 0
        overlap_m = sum(BIN for b in bins if trace[b][1] >= 0.2 and trace[b][2] >= 0.2)
        throttle_delay_m = round(thr_on - apex_d, 1) if thr_on is not None else None
        peak_exit_thr = 0.0
        throttle_lift = False
        for b in bins:
            if b > apex:
                tv = trace[b][1]
                peak_exit_thr = max(peak_exit_thr, tv)
                if peak_exit_thr >= 0.65 and peak_exit_thr - tv >= 0.25:
                    throttle_lift = True
                    break
        out[z["name"]] = {
            "min_v": trace[apex][0], "apex_d": apex_d, "gear": trace[apex][4],
            "brake_at": round(brake[0], 1) if brake else None, "entry_v": brake[1] if brake else None,
            "brake_peak": brake_peak, "coast_m": coast_m, "overlap_m": overlap_m,
            "throttle_at": thr_on, "throttle_delay_m": throttle_delay_m, "throttle_lift": throttle_lift,
            "exit_v": max(after) if after else None,
            "steer_peak": round(max(steering), 2), "steer_at_apex": round(abs(apex_row[3]), 2),
            "brake_at_apex": round(apex_row[2], 2), "throttle_at_apex": round(apex_row[1], 2),
            "trail_braking": bool(apex_row[2] >= 0.1 and abs(apex_row[3]) >= 0.2),
        }
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 1) if xs else None


def profile(laps: list[dict], zones: list[dict]) -> list[dict]:
    """Average corner behaviour over clean laps (falls back to all racing laps if none are clean)."""
    use = [l for l in laps if l.get("clean") and l.get("corners")]
    basis = "clean laps"
    if not use:
        use = [l for l in laps if l.get("corners") and not l.get("standing") and l.get("timed", True) and not l.get("pit")]
        basis = "all laps"
    timed_use = [l for l in use if (l.get("time") or 0) > 0]
    ref_lap = min(timed_use, key=lambda l: l["time"]) if timed_use else None
    rows = []
    for z in zones:
        vals = [l["corners"].get(z["name"]) for l in use]
        vals = [v for v in vals if v]
        if not vals:
            continue
        bps = [v["brake_at"] for v in vals if v.get("brake_at") is not None]
        mins = [v["min_v"] for v in vals]
        ref_corner = ref_lap["corners"].get(z["name"]) if ref_lap else None
        rows.append({
            "corner": z["name"], "start": z["start"], "end": z["end"], "laps": len(vals), "basis": basis,
            "min_v": _mean(mins), "best_min_v": max(mins), "apex_d": _mean([v.get("apex_d") for v in vals]),
            "entry_v": _mean([v.get("entry_v") for v in vals]),
            "brake_at": _mean(bps), "brake_range": [round(min(bps)), round(max(bps))] if bps else None,
            "brake_spread": round(max(bps) - min(bps), 1) if len(bps) > 1 else None,
            "brake_peak": _mean([v.get("brake_peak") for v in vals]),
            "coast_m": _mean([v.get("coast_m") for v in vals]),
            "overlap_m": _mean([v.get("overlap_m") for v in vals]),
            "brakes_pct": round(100 * len(bps) / len(vals)),
            "gear": statistics.mode([v["gear"] for v in vals]) if vals else None,
            "throttle_at": _mean([v.get("throttle_at") for v in vals]),
            "throttle_delay_m": _mean([v.get("throttle_delay_m") for v in vals]),
            "throttle_lift_pct": round(100 * sum(bool(v.get("throttle_lift")) for v in vals) / len(vals)),
            "exit_v": _mean([v.get("exit_v") for v in vals]),
            "best_exit_v": max((v["exit_v"] for v in vals if v.get("exit_v") is not None), default=None),
            "steer_peak": _mean([v.get("steer_peak") for v in vals]),
            "steer_at_apex": _mean([v.get("steer_at_apex") for v in vals]),
            "brake_at_apex": _mean([v.get("brake_at_apex") for v in vals]),
            "throttle_at_apex": _mean([v.get("throttle_at_apex") for v in vals]),
            "trail_braking_pct": round(100 * sum(bool(v.get("trail_braking")) for v in vals) / len(vals)),
            "ref_brake_at": ref_corner.get("brake_at") if ref_corner else None,
            "ref_min_v": ref_corner.get("min_v") if ref_corner else None,
            "ref_exit_v": ref_corner.get("exit_v") if ref_corner else None,
            "ref_throttle_delay_m": ref_corner.get("throttle_delay_m") if ref_corner else None,
        })
    return rows


def segments(bins: set[int], gap_bins: int = 6) -> list[list[int]]:
    """Merge 5 m bins into [start_m, end_m] runs (gaps up to gap_bins are bridged)."""
    out: list[list[int]] = []
    for b in sorted(bins):
        if out and b - out[-1][1] // BIN <= gap_bins:
            out[-1][1] = b * BIN + BIN
        else:
            out.append([b * BIN, b * BIN + BIN])
    return out
