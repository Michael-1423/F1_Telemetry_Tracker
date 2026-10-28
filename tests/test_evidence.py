"""Episode evidence tests (P0.1).

Run directly:  python3 tests/test_evidence.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from f1live import evidence as EV  # noqa: E402


def test_one_slide_is_one_episode_at_any_rate():
    for hz in (10, 20, 60):
        tr = EV.EpisodeTracker("oversteer", gap_s=0.2, cooldown_s=0.1)
        dt = 1.0 / hz
        closed = []
        t = 0.0
        # 0.1 s of slip, then 0.5 s of grip
        for _ in range(max(1, int(0.1 * hz))):
            ep = tr.update(t, True, {"slip": 0.4, "speed": 100, "lap": 1})
            assert ep is None
            t += dt
        for _ in range(max(2, int(0.5 * hz))):
            ep = tr.update(t, False, {})
            if ep:
                closed.append(ep)
            t += dt
        ep = tr.flush(t)
        if ep:
            closed.append(ep)
        assert len(closed) == 1, (hz, closed)
        assert closed[0].duration_s >= 0.09
        assert closed[0].max_slip == 0.4


def test_one_kerb_strike_is_one_episode():
    tr = EV.EpisodeTracker("kerb", gap_s=0.2, cooldown_s=0.1)
    closed = []
    t = 0.0
    for _ in range(6):  # six consecutive frames over 1000 mm/s
        ep = tr.update(t, True, {"wheels": {2}, "speed": 180, "lap": 1})
        if ep:
            closed.append(ep)
        t += 0.05
    for _ in range(10):
        ep = tr.update(t, False, {})
        if ep:
            closed.append(ep)
        t += 0.05
    ep = tr.flush(t)
    if ep:
        closed.append(ep)
    assert len(closed) == 1
    assert closed[0].wheels == {2}


def test_bounce_between_episodes_is_separate():
    tr = EV.EpisodeTracker("oversteer", gap_s=0.1, cooldown_s=0.05)
    closed = []
    t = 0.0
    for _ in range(4):
        ep = tr.update(t, True, {"slip": 0.3})
        if ep:
            closed.append(ep)
        t += 0.05
    for _ in range(4):
        ep = tr.update(t, False, {})
        if ep:
            closed.append(ep)
        t += 0.05
    # after the gap/cooldown, a new slide is a new episode
    for _ in range(4):
        ep = tr.update(t, True, {"slip": 0.3})
        if ep:
            closed.append(ep)
        t += 0.05
    ep = tr.flush(t)
    if ep:
        closed.append(ep)
    assert len(closed) == 2, closed


def test_summary_keeps_duration_and_peak():
    tr = EV.EpisodeTracker("oversteer", gap_s=0.1)
    t = 0.0
    for slip in (0.3, 0.5, 0.4):
        tr.update(t, True, {"slip": slip, "g": 2.0, "speed": 120, "lap": 2})
        t += 0.05
    ep = tr.flush(t)
    (row,) = EV.summarize_episodes([ep]).values()
    assert row["count"] == 1
    assert row["max_slip"] == 0.5
    assert row["max_g"] == 2.0
    assert row["seconds"] > 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
