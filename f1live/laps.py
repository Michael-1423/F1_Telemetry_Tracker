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
        out[z["name"]] = {
            "min_v": trace[apex][0], "apex_d": apex_d, "gear": trace[apex][4],
            "brake_at": round(brake[0], 1) if brake else None, "entry_v": brake[1] if brake else None,
            "throttle_at": thr_on, "exit_v": max(after) if after else None,
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
    rows = []
    for z in zones:
        vals = [l["corners"].get(z["name"]) for l in use]
        vals = [v for v in vals if v]
        if not vals:
            continue
        bps = [v["brake_at"] for v in vals if v.get("brake_at") is not None]
        mins = [v["min_v"] for v in vals]
        rows.append({
            "corner": z["name"], "start": z["start"], "end": z["end"], "laps": len(vals), "basis": basis,
            "min_v": _mean(mins), "best_min_v": max(mins), "entry_v": _mean([v.get("entry_v") for v in vals]),
            "brake_at": _mean(bps), "brake_range": [round(min(bps)), round(max(bps))] if bps else None,
            "brake_spread": round(max(bps) - min(bps), 1) if len(bps) > 1 else None,
            "brakes_pct": round(100 * len(bps) / len(vals)),
            "gear": statistics.mode([v["gear"] for v in vals]) if vals else None,
            "throttle_at": _mean([v.get("throttle_at") for v in vals]),
            "exit_v": _mean([v.get("exit_v") for v in vals]),
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
