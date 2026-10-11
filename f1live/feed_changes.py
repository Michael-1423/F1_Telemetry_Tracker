"""Changes to EngineerFeed in engineer.py. Not a runnable module: each block says what it replaces.

What changes and why:
  * Oversteer/understeer samples are only counted while cornering (|steer| above a threshold), so wheelspin on
    a straight or the launch no longer reads as "oversteer".
  * Each cornering sample is tagged entry / mid / exit from the pedals, and counted per phase together with
    how many samples each phase had. The engine needs both: events alone depend on how long you spent there.
    This tagging is also the foundation for the driver-style step (trail braking, throttle pickup).
  * Lockups are detected from wheel slip under braking (front vs rear, the same pattern as under/oversteer).
    The old test, "front brakes 120C hotter than the rears", is normally true for a braking F1 car, and
    front_locking_events was never reset per lap, so after enough laps the engine ALWAYS diagnosed front
    lockups. rear_locking_events was never incremented at all.
  * Every counter is per lap and saved in the lap record, so the engine can look at "the last 3 laps on this
    setup" instead of a session-long total.
  * setup_version goes up on every handling change, so the page can tell that tick-boxes are stale.
"""

# ---------------------------------------------------------------- 1. DEFAULTS: add these keys
DEFAULTS_ADD = {
    "cornering_steer": 0.08,  # |steer| above this = cornering sample
    "entry_brake": 0.05,  # brake above this while cornering = entry
    "exit_throttle": 0.5,  # throttle above this while cornering (no brake) = exit; otherwise mid
    "lock_slip": 0.25,  # braking with one axle's slip above this and slip_ratio x the other = lockup sample
    # lock_brake_temp_gap is no longer used
}

PHASES = ("entry", "mid", "exit")


# ---------------------------------------------------------------- 2. __init__: REPLACE the counter lines
#   self.oversteer_events = 0
#   self.understeer_events = 0
#   self.front_locking_events = 0
#   self.rear_locking_events = 0
# with:
def _init_counters(self):
    self.os_by_phase = dict.fromkeys(PHASES, 0)
    self.us_by_phase = dict.fromkeys(PHASES, 0)
    self.phase_samples = dict.fromkeys(PHASES, 0)
    self.front_lock_lap = 0
    self.rear_lock_lap = 0
    self.setup_version = 0


# and ADD these two properties to the class (the totals the old code and the UI still read):
#
#   @property
#   def oversteer_events(self): return sum(self.os_by_phase.values())
#   @property
#   def understeer_events(self): return sum(self.us_by_phase.values())
#
# Because they are read-only, every place that did `self.oversteer_events = 0` must call the helper below
# instead: _start_first_full_lap, _restart_evidence and _record_completed_lap (shown in section 5).


def _reset_phase_counters(self):
    for d in (self.os_by_phase, self.us_by_phase, self.phase_samples):
        for k in d:
            d[k] = 0
    self.front_lock_lap = self.rear_lock_lap = 0


# ---------------------------------------------------------------- 3. on_motion: REPLACE the whole method
def on_motion(self, cm: dict, extra: dict) -> None:
    m = {
        "gForceLateral": round(cm["gForceLateral"], 2),
        "gForceLongitudinal": round(cm["gForceLongitudinal"], 2),
        "wheelSlip": [round(x, 4) for x in extra["wheelSlip"]],  # [RL, RR, FL, FR]
        "suspensionPosition": [round(x, 4) for x in extra["suspensionPosition"]],
        "suspensionVelocity": [round(x, 4) for x in extra["suspensionVelocity"]],
    }
    self.motion.update(m)
    th = self.th
    slips = m["wheelSlip"]
    if len(slips) >= 4:
        rear_slip = max(slips[0], slips[1])
        front_slip = max(slips[2], slips[3])
        lat_g = abs(m["gForceLateral"])
        t = self.telemetry  # latest car telemetry packet: pedals and steering at this instant
        brake, throttle, steer = t["brake"], t["throttle"], abs(t["steer"])

        if steer > th["cornering_steer"]:
            # Which part of the corner are we in? Braking = entry, on the power = exit, else mid.
            phase = ("entry" if brake > th["entry_brake"]
                     else "exit" if throttle > th["exit_throttle"] else "mid")
            self.phase_samples[phase] += 1
            if rear_slip > th["oversteer_slip"] and rear_slip > front_slip * th["slip_ratio"]:
                self.os_by_phase[phase] += 1
            elif (front_slip > th["understeer_slip"] and front_slip > rear_slip * th["slip_ratio"]
                  and lat_g > th["understeer_lat_g"]):
                self.us_by_phase[phase] += 1

        if brake > th["lock_brake"]:  # hard braking: whichever axle is sliding much more than the other
            if front_slip > th["lock_slip"] and front_slip > rear_slip * th["slip_ratio"]:
                self.front_lock_lap += 1
            elif rear_slip > th["lock_slip"] and rear_slip > front_slip * th["slip_ratio"]:
                self.rear_lock_lap += 1

    if any(abs(v) > th["kerb_suspension_velocity"] for v in m["suspensionVelocity"]):
        self.kerb_bottoming_events += 1  # unchanged


# ---------------------------------------------------------------- 4. on_telemetry: DELETE this block
#   # Brake lockups: heavy braking with the fronts much hotter than the rears
#   if t["brake"] > self.th["lock_brake"]:
#       ...front_locking_events += 1
# (lockups are now detected in on_motion, where the wheel slip is)


# ---------------------------------------------------------------- 5. the three reset sites
# _start_first_full_lap: replace
#       self.oversteer_events = self.understeer_events = 0
#       self.kerb_last_lap = self.kerb_bottoming_events = 0
# with
#       _reset_phase_counters(self)
#       self.kerb_last_lap = self.kerb_bottoming_events = 0
#
# _restart_evidence: replace the whole body with
#       _reset_phase_counters(self)
#       self.kerb_bottoming_events = self.kerb_last_lap = 0
#       self.setup_changed_lap = self.lap_data.get("currentLapNum")
#       self.setup_version += 1
#
# _record_completed_lap: in the dict appended to completed_laps, REPLACE the three event fields with
#       "oversteerEvents": self.oversteer_events,
#       "understeerEvents": self.understeer_events,
#       "oversteerByPhase": dict(self.os_by_phase),
#       "understeerByPhase": dict(self.us_by_phase),
#       "phaseSamples": dict(self.phase_samples),
#       "frontLockEvents": self.front_lock_lap,
#       "rearLockEvents": self.rear_lock_lap,
#       "kerbEvents": self.kerb_bottoming_events,
# and at the end of the method, replace
#       self.oversteer_events = 0
#       self.understeer_events = 0
# with
#       _reset_phase_counters(self)


# ---------------------------------------------------------------- 6. snapshot(): two small edits
#   "diagnostics": {... "frontLockingEvents": self.front_lock_lap, "rearLockingEvents": self.rear_lock_lap, ...}
#   and add   "setupVersion": self.setup_version,   next to "setupChangedLap".
