# F1 2020 Telemetry Tracker (Pit Wall)

A live race-control dashboard for F1 2020 online lobbies. Every driver's game sends UDP telemetry to one
computer. The tracker merges the feeds, shows each human driver's car condition, lap times and mistakes as
the race happens, saves an incident package around every collision, and writes a summary at the end.

- **Live dashboard** at `http://<recording computer>:8020`, styled like F1 broadcast graphics:
  - **Drivers**: one live strip per human driver with the current tyre and its average wear (all four ÷ 4,
    plus each corner), fuel mix, ERS store % with a live *Deploying* indicator, last lap, fastest lap
    (purple = fastest overall), and how many times all four wheels have gone off the white lines.
  - **Driver profile**: the car setup (wings, diff, geometry, suspension, brakes, tyre pressures, ballast,
    fuel) and, for every corner, the average apex, entry and exit speed, braking point and its spread, gear,
    throttle-on point and track-limit count.
  - **Lap by lap**: every lap with sectors, position, tyre, average wear, fuel used, fuel mix, ERS deployed
    and where, track limits, average and top speed, and per-corner apex speed, braking point, entry speed,
    throttle-on or exit speed (switchable).
  - Race control feed, auto-saved incidents, and a Sessions page that shows the same profiles for past races.
- **Practice and qualifying** get a timesheet instead of the race view: each human's fastest valid lap with
  sectors, gap, tyre, ERS used on that lap, fuel mix, fuel at the start of the lap and when leaving the pits,
  traffic (share of the lap within 1 s behind another car), corners cut, four-wheels-off, valid/total laps
  and runs, plus what each driver is doing right now (out lap / flying lap with a live delta to their best).
  The driver profile and lap-by-lap sections are the same as in races, with lap types (out / flying / in).
  Every lap driven is listed, including out laps and laps abandoned into the pits. The game doesn't count or
  time these, so the tracker numbers the laps itself and measures their times (shown with ~).
  Only the summary and event log are saved; no raw data and no incident packages.
- **Race engineer** (the *Engineer* tab, one page per driver): setup advice from the driver's own car. It
  detects oversteer, understeer, front locking and kerb strikes from their telemetry, takes how the car
  feels to them (tick boxes such as *Oversteer on corner exit*), and suggests setup clicks with the
  reasons, plus a full setup sheet to copy and a run / pit / tyre / fuel call for the session. This is the
  F1 2020 AI Race Engineer's logic, ported unchanged (see below).
- **Corners cut vs running wide**: every excursion with 2+ wheels beyond the kerbs is classed by which side
  of the car left the track first compared with the way the driver is steering: inside wheels = a cut,
  outside wheels = running wide. Useful when the game's corner cutting is set to lenient.
- **Real-time detection**: off-tracks, spins, slides, reversing, contacts and collisions, wing, engine and
  gearbox damage, lap invalidations, penalties and warnings (with the game's reason), blue-flag episodes,
  pit stops (lane time, stationary time, tyres, repairs), slow laps with the likely reason, and retirements.
- **Incident packages**: 20 s before and 10 s after each incident involving a human, saved to disk. Each
  package has the raw packets, per-car traces, relative positions and an LLM-ready `prompt.md` with 2020
  rules excerpts.
- **End-of-session summary**: `summary.json` and `summary.md` with classification, stints, pit stops,
  penalties, mistakes, driving-style metrics and typical braking points.
- **Automatic cleanup**: raw packets of the previous race are deleted when the next race starts, unless you
  keep them. Their incident packages are deleted at the same time. Summaries and event logs are always kept.

No dependencies beyond Python 3.11+.

## Race night

1. On the recording computer:

   ```
   python -m f1live
   ```

   Open `http://localhost:8020`. Other drivers can watch at `http://<your IP>:8020` (your Tailscale IP works).

2. Every driver, in **Game Options → Settings → Telemetry Settings**:
   - UDP Telemetry: **On**
   - UDP IP Address: the recording computer's IP (Tailscale IP for remote drivers)
   - UDP Port: **20777**
   - UDP Send Rate: **20 Hz** (10 Hz also works)
   - UDP Format: **2020**
   - Your Telemetry: **Public**. If it's Restricted, the other games can't see your fuel, tyre wear and damage.

   Each driver who streams gets full-resolution data for their own car (inputs, wheel slip, setup). Drivers
   who don't stream are still tracked through the other players' games, at lower resolution.

3. **Practice:** each driver opens `http://<recording computer>:8020/#engineer`, picks their name and keeps
   that page open (the link becomes `.../#engineer/<name>`, so it can be bookmarked). Work through the
   setup changes, tick what the car feels like, and use **Copy setup** for the garage. **Qualifying and
   race:** everyone watches the *Pit wall* tab.

4. Race. Press **F** (or the Flag incident button) to save the last 20 s and next 10 s by hand.

5. Stop with **Ctrl+C**. The session is finalised automatically 90 s after the last packet in any case.

### Keeping raw data

Raw packets are only recorded for races (about 170 MB per 14-lap race with five drivers streaming); practice
and qualifying keep just the summary. Race raw data and incident packages are deleted when the next race
starts (set `delete_incidents = false` to keep incidents). You'll see a banner with a 2-minute countdown and a
**Keep it** button, which keeps both. Other ways to keep them:

```
python -m f1live --keep-raw               # keep the race you're about to record
python -m f1live --keep-previous-raw      # keep the last recorded session
python -m f1live keep <session-id>        # keep any session; --off to undo
python -m f1live sessions                 # list sessions, sizes and what's kept
```

The Sessions page on the dashboard can do the same, and can delete a session's raw data and incidents straight away.

## Race engineer

The engineer comes from the F1 2020 AI Race Engineer app. `f1live/engineer.py` is a line-by-line port
of its relay's detectors (`relay/f1_relay.py`) and its setup engine (`src/lib/engineer_engine.ts` and
`strategy.ts`), with the same thresholds, rules and wording, except the kerb detector (below).
`tests/test_engineer.py` checks the port against outputs of the original TypeScript.

- It only works for drivers whose game sends telemetry to the recording computer: wheel slip,
  suspension and the car setup only go to a driver's own game. The page says so when that's missing,
  and waits for the setup to arrive before advising.
- The handling boxes a driver ticks stay in that browser, per driver and track, like in the original
  app. Two people can look at the same driver with different boxes ticked.
- The counters are per telemetry packet, as in the original. Oversteer and understeer restart each lap;
  kerb strikes count over the current or last lap (whichever is higher); front locking adds up over the
  session.
- Kerb strikes are sharp suspension movements, faster than 1000 mm/s. The original counted suspension
  *position* above 0.08 over the whole session; F1 2020 reports position in millimetres, so every sample
  counted and every car was diagnosed as bottoming on kerbs. Position can't tell kerbs apart anyway (aero
  load at speed compresses the car more than kerbs do). On the Singapore race of 2026-09-30, more than 3
  strikes in a lap flagged 7 of 13 laps on the stiffest setup (rear springs 11) and 1 of 13 on the softest.
- Other differences from the original relay, both in what it read from the game: a lap's validity is the
  game's flag for that lap (the relay's parser was laid out for F1 2021 and read the wrong byte), and the
  all-zero setup the game sends after a car retires or finishes is ignored.
- The detector thresholds can be changed under `[engineer]` in the config.

`GET /api/engineer?driver=<name>&fb=oversteer_exit,front_locking` returns one driver's analysis as JSON.

## Reviewing incidents with an LLM

Each incident folder in `data/races/<session>/incidents/` contains:

| File | What it is |
|---|---|
| `prompt.md` | A brief for an LLM steward: the session, the cars, what triggered it, gap and blue-flag history, driver inputs every 0.2 s, braking point against the driver's usual one, relative position of the cars, data caveats, and 2020 regulation excerpts. |
| `incident.json.gz` | Everything the prompt was built from, at full resolution. |
| `packets.f1raw.gz` | The raw packets, replayable with `python -m f1live replay`. |
| `meta.json` | Small index entry. |

Open an incident on the dashboard and use **Copy prompt**, then paste it into Claude. If you change the
rules text in `f1live/rules.py`, rebuild a prompt with `python -m f1live prompt <incident folder>`.

## Replays and testing

```
python -m f1live replay data/races/<session>/raw.f1raw.gz --speed 10     # watch a race again
python -m f1live replay f1_telemetry_capture.jsonl --speed max --no-server --record-raw
                                           # re-analyse an old f1_capture.py recording (and convert it)
python -m f1live send <recording> --to <tracker IP>:20777 --speed 1
                                           # test race-night setup: replays over UDP, one socket per driver
python tests/test_packets.py               # decoder check against the f1_2020_telemetry library, if installed
python tests/test_engineer.py              # race engineer port against the original's outputs
```

## Files

```
data/races/2026-09-28_0055_Texas-COTA_Race_a57668/
  session.json     session metadata
  raw.f1raw.gz     every packet (deleted at the next race unless kept)
  events.jsonl     live event log
  summary.json     end-of-session summary (also summary.md)
  incidents/001_lap1_collision_chappu-mochi_Franky-Frank/ ...
  KEEP_RAW         present when raw data is kept
```

## Configuration

Copy `f1live.example.toml` to `f1live.toml`. You can change ports, the data folder, retention, incident
windows, detector thresholds, driver display names, and corner names for tracks other than COTA.

## How it works

- `f1live/packets.py` is a dependency-free F1 2020 decoder, checked field by field against the reference library.
- `f1live/model.py` merges the feeds into one session state and runs the detectors. Each car is read from its own driver's game when available, otherwise from one stable primary game.
- `f1live/engineer.py` is the race engineer: one detector per driver, fed by their own game, and the setup engine.
- `f1live/incidents.py` keeps a rolling buffer, merges overlapping triggers, and writes the packages.
- `f1live/summary.py` builds the dashboard snapshot and the end-of-session summary.
- `f1live/storage.py` handles session folders, the raw recorder and reader, and retention.
- `f1live/server.py` serves the dashboard, a live Server-Sent Events stream and the JSON API.

## Limits

- F1 2020 has no collision event. Contacts are inferred from g-force spikes with another car within about
  4.5 m, wing damage appearing with a car nearby, or the game's own collision penalties. Expect the
  occasional false "contact" when cars run side by side over kerbs.
- Wheel slip, suspension and car setup are only sent for a player's own car.
- "Four wheels off" counts excursions in which every wheel went beyond the kerbs (grass, gravel, painted
  run-off), allowing for the 10–20 Hz sample rate, so a car crossing the inside of a hairpin or crossing a
  painted strip onto tarmac run-off counts. A car straddling the kerb with its outside wheels on the grass
  doesn't: a wheel on the kerb is on the track, as in the game. Bounces within a second count once.
- Corners come from a built-in map for COTA. On other tracks they are detected automatically from the
  fastest clean lap (named C1, C2, ...); give them real names under `[corners]` in the config.
- A car's setup is only sent by that driver's own game, so setups appear for drivers who stream telemetry.

`f1_capture.py` is the original capture script, kept for reference. `f1live` replaces it and can replay its
recordings.
