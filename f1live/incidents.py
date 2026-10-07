"""Incidents: collisions, spins, heavy damage, penalties, retirements and manual flags that involve a
human driver, grouped when they are the same moment on track. They are listed in race control and in the
summary; the full race recording holds the data for a closer look afterwards.
"""

from __future__ import annotations

import re

from .model import Session


class IncidentLog:
    def __init__(self, cfg: dict):
        icfg = cfg.get("incidents", {})
        self.window = float(icfg.get("merge_seconds", 12))   # events this close in session time are one incident
        self.kinds = set(icfg.get("capture", ["collision", "spin", "damage", "penalty", "retirement", "manual"]))
        self.include_ai_only = bool(icfg.get("include_ai_only", False))
        self.done: list[dict] = []
        self._open: list[dict] = []

    def wants(self, s: Session, ev: dict) -> bool:
        if ev["kind"] == "manual":
            return True
        if not ev.get("capture") or ev["kind"] not in self.kinds:
            return False
        return self.include_ai_only or any(s.cars[i].human for i in ev["cars"] if 0 <= i < 22)

    def trigger(self, s: Session, ev: dict) -> dict | None:
        if not self.wants(s, ev):
            return None
        cars = [i for i in ev["cars"] if 0 <= i < 22] or [c.idx for c in s.humans()]
        self._open = [i for i in self._open if ev["st"] - i["st_last"] <= self.window]
        for inc in self._open:
            if set(cars) & set(inc["cars"]) or ev["kind"] == "manual":
                inc["events"].append(ev["id"])
                inc["st_last"] = max(inc["st_last"], ev["st"])
                inc["cars"] += [i for i in cars if i not in inc["cars"]]
                inc["humans"] = [s.display_name(i) for i in inc["cars"] if s.cars[i].human]
                ev["incident"] = inc["id"]
                return inc
        inc = {"id": f"{len(self.done) + 1:03d}", "title": incident_title(s, ev), "cars": list(cars),
               "humans": [s.display_name(i) for i in cars if s.cars[i].human], "events": [ev["id"]],
               "lap": ev.get("lap"), "clock": ev.get("clock"), "kind": ev["kind"], "st": ev["st"], "st_last": ev["st"]}
        ev["incident"] = inc["id"]
        self.done.append(inc)
        self._open.append(inc)
        return inc


def incident_title(s: Session, ev: dict) -> str:
    """The human driver(s) first, then what happened: "Aven Luci & Franky Frank · Collision at S1 0.43 km,
    front-left wing damage 0% → 8%", "chappu mochi · Collision at S1 0.49 km (112 g spike), with Bottas"."""
    pairs = list(zip(ev.get("cars") or [], ev.get("names") or []))
    humans = [n for i, n in pairs if 0 <= i < 22 and s.cars[i].human]
    others = [n for i, n in pairs if not (0 <= i < 22 and s.cars[i].human)]
    text = ev["text"]
    if ev["kind"] == "manual":
        return text
    for n in humans:   # the names lead the title, so drop them from the description
        q = re.escape(n)
        text = re.sub(rf"\s*\((?:involving|with) {q}\)", "", text)
        text = re.sub(rf"\s+with {q}\b", "", text)
    text = re.sub(r"\s+:", ":", text).strip()
    named = [n for n in others if n in text]
    rest = [n for n in others if n not in named]
    if rest:
        text += ", with " + " & ".join(rest)
    if not humans:
        return text
    return " & ".join(humans) + " · " + (text[:1].upper() + text[1:])
