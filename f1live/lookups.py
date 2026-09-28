"""Human-readable names for F1 2020 telemetry codes, plus per-track corner maps."""

from __future__ import annotations

TRACKS = {
    0: "Melbourne", 1: "Paul Ricard", 2: "Shanghai", 3: "Sakhir", 4: "Catalunya", 5: "Monaco",
    6: "Montreal", 7: "Silverstone", 8: "Hockenheim", 9: "Hungaroring", 10: "Spa", 11: "Monza",
    12: "Singapore", 13: "Suzuka", 14: "Abu Dhabi", 15: "Texas (COTA)", 16: "Brazil", 17: "Austria",
    18: "Sochi", 19: "Mexico", 20: "Baku", 21: "Sakhir Short", 22: "Silverstone Short",
    23: "Texas Short", 24: "Suzuka Short", 25: "Hanoi", 26: "Zandvoort",
}

TEAMS = {
    0: "Mercedes", 1: "Ferrari", 2: "Red Bull", 3: "Williams", 4: "Racing Point", 5: "Renault",
    6: "AlphaTauri", 7: "Haas", 8: "McLaren", 9: "Alfa Romeo", 41: "Generic", 255: "My Team",
}

SESSION_TYPES = {
    0: "Unknown", 1: "Practice 1", 2: "Practice 2", 3: "Practice 3", 4: "Short Practice",
    5: "Qualifying 1", 6: "Qualifying 2", 7: "Qualifying 3", 8: "Short Qualifying",
    9: "One-Shot Qualifying", 10: "Race", 11: "Race 2", 12: "Time Trial",
}
RACE_SESSION_TYPES = {10, 11}
SESSION_KIND = {1: "practice", 2: "practice", 3: "practice", 4: "practice", 5: "qualifying", 6: "qualifying",
                7: "qualifying", 8: "qualifying", 9: "qualifying", 10: "race", 11: "race", 12: "time_trial"}

# Official 2020 livery colours used in F1 broadcast graphics.
TEAM_COLORS = {
    "Mercedes": "#00D2BE", "Ferrari": "#DC0000", "Red Bull": "#1E41FF", "Williams": "#0082FA",
    "Racing Point": "#F596C8", "Renault": "#FFF500", "AlphaTauri": "#469BFF", "Haas": "#787878",
    "McLaren": "#FF8700", "Alfa Romeo": "#9B0000",
}

WEATHER = {0: "Clear", 1: "Light cloud", 2: "Overcast", 3: "Light rain", 4: "Heavy rain", 5: "Storm"}
SAFETY_CAR = {0: None, 1: "Safety Car", 2: "Virtual Safety Car"}
FIA_FLAGS = {-1: "unknown", 0: None, 1: "green", 2: "blue", 3: "yellow", 4: "red"}
PIT_STATUS = {0: None, 1: "pitting", 2: "in pit area"}
# F1 2020 sends 7 for retired cars (verified against real race data).
RESULT_STATUS = {0: "invalid", 1: "inactive", 2: "active", 3: "finished", 4: "did not finish",
                 5: "disqualified", 6: "not classified", 7: "retired"}
OUT_OF_RACE = {4, 5, 6, 7}
DRIVER_STATUS = {0: "garage", 1: "flying lap", 2: "in lap", 3: "out lap", 4: "on track"}
FUEL_MIX = {0: "Lean", 1: "Standard", 2: "Rich", 3: "Max"}
ERS_MODE = {0: "None", 1: "Medium", 2: "Overtake", 3: "Hotlap"}
TC = {0: "Off", 1: "Medium", 2: "Full"}

TYRE_ACTUAL = {16: "C5", 17: "C4", 18: "C3", 19: "C2", 20: "C1", 7: "Inter", 8: "Wet",
               9: "Dry (classic)", 10: "Wet (classic)", 11: "Super soft", 12: "Soft", 13: "Medium",
               14: "Hard", 15: "Wet"}
TYRE_VISUAL = {16: "Soft", 17: "Medium", 18: "Hard", 7: "Inter", 8: "Wet", 15: "Wet", 19: "Super soft",
               20: "Soft", 21: "Medium", 22: "Hard"}
TYRE_LETTER = {"Soft": "S", "Medium": "M", "Hard": "H", "Inter": "I", "Wet": "W", "Super soft": "SS"}

SURFACES = {0: "tarmac", 1: "rumble strip", 2: "concrete", 3: "rock", 4: "gravel", 5: "mud", 6: "sand",
            7: "grass", 8: "water", 9: "cobblestone", 10: "metal", 11: "ridged"}
# Surfaces that count as "off the track" for track-limit / excursion detection.
OFF_SURFACES = {3, 4, 5, 6, 7, 8}

PENALTY_TYPES = {
    0: "Drive through", 1: "Stop Go", 2: "Grid penalty", 3: "Penalty reminder", 4: "Time penalty",
    5: "Warning", 6: "Disqualified", 7: "Removed from formation lap", 8: "Parked too long timer",
    9: "Tyre regulations", 10: "This lap invalidated", 11: "This and next lap invalidated",
    12: "This lap invalidated without reason", 13: "This and next lap invalidated without reason",
    14: "This and previous lap invalidated", 15: "This and previous lap invalidated without reason",
    16: "Retired", 17: "Black flag timer",
}
# Penalty types that actually cost the driver something (vs. warnings / lap invalidations).
SERIOUS_PENALTIES = {0, 1, 2, 4, 6, 17}

INFRINGEMENTS = {
    0: "Blocking by slow driving", 1: "Blocking by wrong way driving", 2: "Reversing off the start line",
    3: "Big collision", 4: "Small collision", 5: "Collision, failed to hand back position (single)",
    6: "Collision, failed to hand back position (multiple)", 7: "Corner cutting gained time",
    8: "Corner cutting overtake (single)", 9: "Corner cutting overtake (multiple)",
    10: "Crossed pit exit lane", 11: "Ignoring blue flags", 12: "Ignoring yellow flags",
    13: "Ignoring drive through", 14: "Too many drive throughs",
    15: "Drive through reminder: serve within n laps", 16: "Drive through reminder: serve this lap",
    17: "Pit lane speeding", 18: "Parked for too long", 19: "Ignoring tyre regulations",
    20: "Too many penalties", 21: "Multiple warnings", 22: "Approaching disqualification",
    23: "Tyre regulations select single", 24: "Tyre regulations select multiple",
    25: "Lap invalidated: corner cutting", 26: "Lap invalidated: running wide",
    27: "Corner cutting, ran wide, gained time (minor)",
    28: "Corner cutting, ran wide, gained time (significant)",
    29: "Corner cutting, ran wide, gained time (extreme)", 30: "Lap invalidated: wall riding",
    31: "Lap invalidated: flashback used", 32: "Lap invalidated: reset to track",
    33: "Blocking the pit lane", 34: "Jump start", 35: "Safety car to car collision",
    36: "Safety car illegal overtake", 37: "Safety car exceeding allowed pace",
    38: "Virtual safety car exceeding allowed pace", 39: "Formation lap below allowed speed",
    40: "Retired: mechanical failure", 41: "Retired: terminally damaged",
    42: "Safety car falling too far back", 43: "Black flag timer", 44: "Unserved stop go penalty",
    45: "Unserved drive through penalty", 46: "Engine component change", 47: "Gearbox change",
    48: "League grid penalty", 49: "Retry penalty", 50: "Illegal time gain", 51: "Mandatory pitstop",
}
COLLISION_INFRINGEMENTS = {3, 4, 5, 6, 35}
# The game's own corner-cutting calls (warnings, time penalties, lap invalidations for cutting).
CUT_INFRINGEMENTS = {7, 8, 9, 25, 27, 28, 29}

# Corner maps: (name, start m, end m) in lap distance. Tracks without a map fall back to
# "S<sector> <km>" labels. Add your own in config.toml under [corners."<trackId>"].
CORNERS: dict[int, list[tuple[str, float, float]]] = {
    15: [  # Circuit of the Americas, measured from 2020 telemetry
        ("T1", 540, 760), ("T2", 900, 1150), ("Esses T3-T6", 1150, 1950), ("T7-T9", 1950, 2350),
        ("T10", 2350, 2450), ("T11", 2450, 2720), ("Back straight", 2720, 3550), ("T12", 3550, 3900),
        ("T13-T14", 3900, 4200), ("T15", 4200, 4420), ("T16-T18", 4420, 4950), ("T19", 4950, 5200),
        ("T20", 5200, 5514), ("Main straight", 0, 540),
    ],
}


def corner_name(track_id: int | None, lap_distance: float | None, track_length: float | None = None,
                custom: dict | None = None) -> str:
    if lap_distance is None:
        return "?"
    table = (custom or {}).get(track_id) or CORNERS.get(track_id or -1)
    if table:
        for name, lo, hi in table:
            if lo <= lap_distance < hi:
                return name
    if track_length:
        sector = 1 + min(2, int(3 * max(lap_distance, 0) / track_length))
        return f"S{sector} {lap_distance / 1000:.2f} km"
    return f"{lap_distance / 1000:.2f} km"


def tyre_label(actual: int | None, visual: int | None) -> str:
    v = TYRE_VISUAL.get(visual or -1)
    a = TYRE_ACTUAL.get(actual or -1)
    if v and a and a != v:
        return f"{v} ({a})"
    return v or a or "?"
