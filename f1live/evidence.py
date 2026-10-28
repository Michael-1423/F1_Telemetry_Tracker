"""Evidence layer for the race engineer (P0).

The engineer's old detectors were plain per-packet counters: one slide at 20 Hz
counted twice as much as the same slide at 10 Hz, and a single kerb could count
several times.  This module turns those samples into episodes with a start, an
end, a duration and peak magnitudes, and provides the lap-quality / wear /
temperature helpers used by ``engineer.analyze_v2``.

Nothing here changes the legacy ``analyze()`` path; it is additive so the
original TypeScript-compatible golden tests keep passing.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable

EPISODE_KINDS = ("oversteer", "understeer", "kerb", "front_lock", "rear_lock")

# Same classification as model.lookups.OFF_SURFACES: tarmac/kerb/concrete and
# the other hard surfaces are on-track, grass/gravel/water/etc. are not.
OFF_SURFACES = {3, 4, 5, 6, 7, 8}

# Setup fields whose meaningful resolution is coarser than a float epsilon.
SETUP_EPS = {
    "frontCamber": 0.011,
    "rearCamber": 0.011,
    "frontToe": 0.011,
    "rearToe": 0.011,
    "frontLeftTyrePressure": 0.06,
    "frontRightTyrePressure": 0.06,
    "rearLeftTyrePressure": 0.06,
    "rearRightTyrePressure": 0.06,
    "fuelLoad": 0.0,  # ignored for evidence resets
}
DEFAULT_SETUP_EPS = 0.001


@dataclass
class Episode:
    """One contiguous detector event (a slide, a kerb strike, a lockup)."""

    kind: str
    start_st: float
    last_st: float
    lap: int
    start_distance: float = -1.0
    end_distance: float = -1.0
    corner: str | None = None
    max_slip: float = 0.0
    max_g: float = 0.0
    max_speed_kmh: float = 0.0
    min_speed_kmh: float = 999.0
    wheels: set[int] = field(default_factory=set)
    surfaces: set[int] = field(default_factory=set)
    in_pit: bool = False
    on_track: bool = True
    contact: bool = False

    @property
    def duration_s(self) -> float:
        return max(0.0, round(self.last_st - self.start_st, 3))

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "start_st": round(self.start_st, 3),
            "end_st": round(self.last_st, 3),
            "duration_s": self.duration_s,
            "lap": self.lap,
            "start_distance": round(self.start_distance, 1),
            "end_distance": round(self.end_distance, 1),
            "corner": self.corner,
            "max_slip": round(self.max_slip, 3),
            "max_g": round(self.max_g, 2),
            "max_speed_kmh": round(self.max_speed_kmh, 1),
            "min_speed_kmh": round(self.min_speed_kmh, 1) if self.min_speed_kmh < 999 else None,
            "wheels": sorted(self.wheels),
            "surfaces": sorted(self.surfaces),
            "in_pit": self.in_pit,
            "on_track": self.on_track,
            "contact": self.contact,
        }


class EpisodeTracker:
    """Turn a per-sample boolean into one episode per contiguous run.

    ``gap_s`` is the silence that closes an episode (tolerates one dropped
    packet); ``cooldown_s`` prevents a bounce from opening a second episode.
    """

    def __init__(self, kind: str, gap_s: float = 0.25, cooldown_s: float = 0.30):
        self.kind = kind
        self.gap_s = float(gap_s)
        self.cooldown_s = float(cooldown_s)
        self.active: Episode | None = None
        self.last_closed: float = -1e9
        self._last_now: float = -1e9

    def update(self, now: float, condition: bool, sample: dict[str, Any] | None = None) -> Episode | None:
        """Feed one sample. Returns a closed episode when the run ends."""
        if now < self._last_now:
            return None  # out-of-order packet
        self._last_now = now
        sample = sample or {}
        if condition:
            if self.active is None:
                if now - self.last_closed < self.cooldown_s:
                    return None
                self.active = Episode(
                    kind=self.kind,
                    start_st=now,
                    last_st=now,
                    lap=int(sample.get("lap", 0) or 0),
                    start_distance=float(sample.get("lap_distance", -1.0) or -1.0),
                )
            ep = self.active
            ep.last_st = now
            ep.end_distance = float(sample.get("lap_distance", ep.end_distance) or ep.end_distance)
            ep.corner = sample.get("corner", ep.corner)
            ep.max_slip = max(ep.max_slip, abs(float(sample.get("slip", 0.0) or 0.0)))
            ep.max_g = max(ep.max_g, abs(float(sample.get("g", 0.0) or 0.0)))
            speed = float(sample.get("speed", 0.0) or 0.0)
            ep.max_speed_kmh = max(ep.max_speed_kmh, speed)
            ep.min_speed_kmh = min(ep.min_speed_kmh, speed)
            wheels = sample.get("wheels")
            if wheels:
                ep.wheels.update(int(w) for w in wheels)
            surfaces = sample.get("surfaces")
            if surfaces:
                ep.surfaces.update(int(s) for s in surfaces)
            ep.in_pit = bool(sample.get("in_pit", ep.in_pit))
            ep.on_track = bool(sample.get("on_track", ep.on_track))
            ep.contact = bool(sample.get("contact", ep.contact))
            return None
        if self.active is not None and now - self.active.last_st >= self.gap_s:
            return self.flush(now)
        return None

    def flush(self, now: float | None = None) -> Episode | None:
        """Close the active episode immediately (lap line, pit entry, reset)."""
        if self.active is None:
            return None
        ep = self.active
        if now is not None and now >= ep.last_st:
            ep.last_st = now
        self.active = None
        self.last_closed = ep.last_st
        return ep

    def reset(self) -> None:
        self.active = None
        self.last_closed = -1e9
        self._last_now = -1e9

    @property
    def duration_s(self) -> float:
        return self.active.duration_s if self.active else 0.0


def summarize_episodes(episodes: Iterable[Episode | dict]) -> dict[str, dict]:
    """Group episodes by kind into a JSON-friendly summary."""
    out: dict[str, dict] = {}
    for item in episodes:
        ep = item.as_dict() if isinstance(item, Episode) else dict(item)
        kind = str(ep.get("kind", "?"))
        row = out.setdefault(kind, {
            "count": 0, "seconds": 0.0, "max_slip": 0.0, "max_g": 0.0,
            "max_speed_kmh": 0.0, "wheels": set(), "corners": set(), "episodes": [],
        })
        row["count"] += 1
        row["seconds"] = round(row["seconds"] + float(ep.get("duration_s") or 0.0), 3)
        row["max_slip"] = max(row["max_slip"], float(ep.get("max_slip") or 0.0))
        row["max_g"] = max(row["max_g"], float(ep.get("max_g") or 0.0))
        row["max_speed_kmh"] = max(row["max_speed_kmh"], float(ep.get("max_speed_kmh") or 0.0))
        row["wheels"].update(ep.get("wheels") or [])
        if ep.get("corner"):
            row["corners"].add(ep["corner"])
        row["episodes"].append(ep)
    for row in out.values():
        row["wheels"] = sorted(row["wheels"])
        row["corners"] = sorted(row["corners"])
        row["max_slip"] = round(row["max_slip"], 3)
        row["max_g"] = round(row["max_g"], 2)
        row["max_speed_kmh"] = round(row["max_speed_kmh"], 1)
    return out


def lap_is_push(lap: dict) -> bool:
    """A lap that can be used for setup evidence.

    Older records only have ``isValid``; treat those as flying laps so the
    function remains usable with captures made before lap labels existed.
    """
    if not lap:
        return False
    if "push" in lap:
        return bool(lap["push"])
    kind = str(lap.get("kind") or "flying")
    return bool(lap.get("isValid", lap.get("valid", True))) and kind == "flying" and not lap.get("pit")


def classify_lap(start_pit: bool, end_pit: bool, start_status: int | None,
                 end_status: int | None, invalid: bool, had_pit: bool = False) -> dict:
    """Label a completed practice/qualifying lap as out / flying / in.

    ``driverStatus``: 0 garage, 1 flying lap, 2 in lap, 3 out lap, 4 on track.
    The game only counts timed laps, but an in-lap can still be recorded if the
    game gives it a last lap time; this keeps it out of the evidence.
    """
    if bool(start_pit) or start_status == 3:
        kind = "out"
    elif bool(end_pit) or end_status == 2:
        kind = "in"
    else:
        kind = "flying"
    pit = bool(had_pit or start_pit or end_pit)
    push = bool(not invalid and kind == "flying" and not pit)
    label = {"out": "Out lap", "in": "In lap", "flying": "Flying lap"}[kind]
    if pit and kind == "flying":
        label = "Pit lap"
    if invalid:
        label += " (invalid)"
    return {"kind": kind, "pit": pit, "push": push, "valid": not invalid, "label": label}


def episode_window(laps: list[dict], window: int = 3, generation: int | None = None) -> dict[str, dict]:
    """Aggregate episodes over the last ``window`` push laps of one setup generation."""
    use = []
    for lap in laps:
        if generation is not None and lap.get("setup_generation") is not None \
                and int(lap["setup_generation"]) != int(generation):
            continue
        if lap_is_push(lap):
            use.append(lap)
    use = use[-max(1, int(window)):]
    out: dict[str, dict] = {}
    for kind in EPISODE_KINDS:
        rows = []
        for lap in use:
            for ep in (lap.get("episodes") or {}).get(kind, {}).get("episodes", []):
                rows.append(ep)
        out[kind] = summarize_episodes(rows).get(kind, {
            "count": 0, "seconds": 0.0, "max_slip": 0.0, "max_g": 0.0,
            "max_speed_kmh": 0.0, "wheels": [], "corners": [], "episodes": [],
        })
    out["laps_used"] = len(use)
    out["laps"] = [int(l.get("lap", 0) or 0) for l in use]
    return out


def _lap_wear(lap: dict) -> tuple[float | None, float | None]:
    delta = lap.get("tyreWearDelta") or {}
    try:
        front = [float(delta[k]) for k in ("fl", "fr") if delta.get(k) is not None]
        rear = [float(delta[k]) for k in ("rl", "rr") if delta.get(k) is not None]
    except (TypeError, ValueError):
        return None, None
    if not front or not rear:
        return None, None
    return statistics.fmean(front), statistics.fmean(rear)


def _slope(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 1e-9:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def wear_estimate(laps: list[dict], generation: int | None = None, min_laps: int = 2,
                  outlier_factor: float = 3.0) -> dict[str, Any]:
    """Front/rear wear rate from clean push laps on the current setup.

    Uses the mean of the two wheels on each axle, drops puncture/damage
    outliers, and switches to a linear regression once there are 3+ laps.
    """
    rows = []
    for lap in laps:
        if generation is not None and lap.get("setup_generation") is not None \
                and int(lap["setup_generation"]) != int(generation):
            continue
        if not lap_is_push(lap) or lap.get("valid", lap.get("isValid", True)) is False:
            continue
        front, rear = _lap_wear(lap)
        if front is None or rear is None:
            continue
        rows.append((int(lap.get("lap", 0) or len(rows) + 1), front, rear))
    if not rows:
        return {"front_rate": None, "rear_rate": None, "laps": 0, "samples": [],
                "outliers": 0, "provisional": True, "confidence": 0.0,
                "dominant_axle": None}
    # Drop values that are far above the median: a puncture or a damaged tyre
    # is not a degradation rate.
    med_f = statistics.median(r[1] for r in rows)
    med_r = statistics.median(r[2] for r in rows)
    kept, outliers = [], 0
    for row in rows:
        if (row[1] > max(med_f * outlier_factor, med_f + 2.0)
                or row[2] > max(med_r * outlier_factor, med_r + 2.0)):
            outliers += 1
            continue
        kept.append(row)
    rows = kept
    if len(rows) < min_laps:
        return {"front_rate": None, "rear_rate": None, "laps": len(rows),
                "samples": rows, "outliers": outliers, "provisional": True,
                "confidence": 0.0, "dominant_axle": None}
    xs = [float(r[0]) for r in rows]
    front_vals = [r[1] for r in rows]
    rear_vals = [r[2] for r in rows]
    f_rate = _slope(xs, front_vals) if len(rows) >= 3 else statistics.fmean(front_vals)
    r_rate = _slope(xs, rear_vals) if len(rows) >= 3 else statistics.fmean(rear_vals)
    f_rate = max(0.0, float(f_rate if f_rate is not None else statistics.fmean(front_vals)))
    r_rate = max(0.0, float(r_rate if r_rate is not None else statistics.fmean(rear_vals)))
    confidence = min(1.0, 0.35 + 0.15 * len(rows))
    return {
        "front_rate": round(f_rate, 2),
        "rear_rate": round(r_rate, 2),
        "laps": len(rows),
        "samples": [{"lap": r[0], "front": round(r[1], 2), "rear": round(r[2], 2)} for r in rows],
        "outliers": outliers,
        "provisional": len(rows) < 3,
        "confidence": round(confidence, 2),
        "dominant_axle": "front" if f_rate >= r_rate else "rear",
    }


def temperature_summary(laps: list[dict], window: int = 3, generation: int | None = None) -> dict[str, Any] | None:
    """Average per-lap inner tyre temperatures over recent push laps."""
    use = []
    for lap in laps:
        if generation is not None and lap.get("setup_generation") is not None \
                and int(lap["setup_generation"]) != int(generation):
            continue
        if lap_is_push(lap) and lap.get("averageTyreTemps"):
            use.append(lap)
    use = use[-max(1, int(window)):]
    if not use:
        return None
    keys = ("fl", "fr", "rl", "rr")
    vals = {k: [] for k in keys}
    for lap in use:
        temps = lap.get("averageTyreTemps") or {}
        for k in keys:
            if temps.get(k) is not None:
                vals[k].append(float(temps[k]))
    if not any(vals.values()):
        return None
    avg = {k: (round(statistics.fmean(v), 1) if v else None) for k, v in vals.items()}
    front = [x for k in ("fl", "fr") for x in vals[k]]
    rear = [x for k in ("rl", "rr") for x in vals[k]]
    return {
        "avg": avg,
        "front_avg": round(statistics.fmean(front), 1) if front else None,
        "rear_avg": round(statistics.fmean(rear), 1) if rear else None,
        "max": round(max(max(v) for v in vals.values() if v), 1) if any(vals.values()) else None,
        "laps": len(use),
    }


def setup_diff(old: dict | None, new: dict | None) -> dict[str, dict]:
    """Fields that really changed between two setup packets (fuel load ignored)."""
    if not old or not new:
        return {}
    changed = {}
    for key in set(old) | set(new):
        if key == "fuelLoad":
            continue
        a, b = old.get(key), new.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            eps = SETUP_EPS.get(key, DEFAULT_SETUP_EPS)
            if abs(float(a) - float(b)) <= eps:
                continue
        elif a == b:
            continue
        changed[key] = {"old": a, "new": b}
    return changed


def is_on_track(surfaces: Iterable[int] | None) -> bool:
    return not any(int(s) in OFF_SURFACES for s in (surfaces or []))


def off_wheels(surfaces: Iterable[int] | None) -> int:
    return sum(1 for s in (surfaces or []) if int(s) in OFF_SURFACES)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default
