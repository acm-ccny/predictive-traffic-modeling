# team-1

## Google multi-route comparison app

This app compares:
- Google Directions API route durations (defaults to static `duration`; optional `duration_in_traffic` via env)
- Our model ETA on overview polylines by default, with **partial** per-route length calibration vs Google's `distance_m`

### Setup

1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Configure Google Maps API key (Directions + Geocoding enabled):
   - Option A: put `GOOGLE_MAPS_API_KEY=...` in the project root `.env` (loaded automatically via `python-dotenv` when the app imports `google_places`)
   - Option B: set env var `GOOGLE_MAPS_API_KEY` in your shell before `streamlit run`
   - Option C: create `.streamlit/secrets.toml` with:
     - `GOOGLE_MAPS_API_KEY = "your-key"`

### Run

- `streamlit run streamlit_UI.py`

### Route benchmark tests (pytest)

- Install deps: `pip install -r requirements.txt`
- Run benchmark routes against live Google APIs:
  - `pytest -s -m integration tests/test_route_benchmark.py`
- Optional pass/fail guardrail by delta magnitude (minutes):
  - `ROUTE_BENCH_MAX_ABS_DELTA_MIN=4.0 pytest -s -m integration tests/test_route_benchmark.py`
- The test prints a compact table with departure context, Google ETA, model ETA, delta, and route summary.
- Benchmarks pass `day_of_week` + `hour` into Directions (used when traffic requests are enabled).

### Notes

- Users can enter any start/end place (resolved with Google geocoding).
- The model uses hour + weekend/rush-hour + borough context; day-of-week is mapped to weekend flag (`Saturday/Sunday => is_weekend=1`).

### Routing tuning (optional env vars)

- `ROUTING_HEAL_PAIR_DEG` — KDTree pair distance in degrees for local healing (default `0.02`).
- `ROUTING_MAX_HEAL_M` — skip healed links longer than this many meters (default `850`).
- `ROUTING_MAX_BRIDGE_M` — cap effective length used for travel time on component-bridge edges (default `1200`).
- `ROUTING_RF_CONGESTION_GAIN` — increases/decreases RF congestion sensitivity around baseline `1.0` (default `2.1`).
- `ROUTING_RF_MULT_MIN`, `ROUTING_RF_MULT_MAX` — lower/upper clip bounds for congestion multiplier stability (defaults `0.40`, `3.85`).
- `ROUTING_URBAN_DELAY_BASE_SEC` — base additive per-edge local-road friction delay before borough/rush/length scaling (default `6.8`).
- `ROUTING_OFFPEAK_MULT_BONUS` — additive multiplier applied only outside rush-hour to reduce off-peak optimism (default `0.22`).
- `ROUTING_WEEKEND_RUSH_MULT_BONUS` — additive multiplier on weekend rush-hour trips only (default `0.10`).
- `ROUTING_DELAY_HWY_DAMP_M` — meters scale for damping intersection delay on long segments (bridges/expressway chords; default `420`).
- `GOOGLE_DIRECTIONS_USE_TRAFFIC` — when `1`/`true`, send `departure_time` + `traffic_model` (default `0` for stable deltas vs static duration).
- `GOOGLE_DIRECTIONS_USE_STEP_POLYLINES` — when `1`/`true`, merge leg step polylines for scoring/map (default `0`).
- `GOOGLE_DIRECTIONS_TRAFFIC_MODEL` — `best_guess`, `pessimistic`, or `optimistic` (default `best_guess`).
- `GOOGLE_DIRECTIONS_TZ` — IANA zone used to interpret the next upcoming weekday+hour for `departure_time` (default `America/New_York`).
- `ROUTE_POLYLINE_CHORD_FACTOR` — uniform chord inflation when Google distance is **not** used for scaling (default `1.12`).
- `ROUTE_POLYLINE_SCALE_MIN`, `ROUTE_POLYLINE_SCALE_MAX` — clamp on `google_distance_m / sum(chords)` before blending (defaults `0.93`, `1.28`).
- `ROUTE_POLYLINE_DISTANCE_BLEND` — `length_scale = 1 + blend * (clamped_ratio - 1)` (default `0.45`).
- `ROUTE_ETA_CALIB_A`, `ROUTE_ETA_CALIB_B` — affine calibration on predicted seconds: `A * seconds + B` (defaults `1.0`, `0.0`). Applied in `predict_route` and `google_route_scoring.score_polyline_eta_seconds`.
  - Practical tuning: increase `A` if ETAs are low proportionally; increase `B` if ETAs are low by a mostly fixed offset.
- `GOOGLE_DIRECTIONS_AVOID` — passed to Google Directions `avoid` (default `highways`, matching local-road training). Use pipe-separated values per Google docs (e.g. `highways|tolls`), or set empty to disable.