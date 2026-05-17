# Predictive Traffic Routing (NYC)

Compare **Google Directions** route durations with **our congestion model** on the same alternative routes. The Streamlit app geocodes any start/end place, fetches up to three Google driving routes, scores each with the local model, and shows them on one map.

## Repository layout

```
team-1/
├── app/
│   └── streamlit_app.py      # Streamlit UI (run this)
├── src/traffic_routing/      # Python package (routing model + Google APIs)
├── scripts/
│   └── ensure_ml_datasets.py # Unzip ML CSVs if only the .zip is present
├── tests/                    # Pytest (integration tests call live Google APIs)
├── notebooks/                # Data cleaning, merging, EDA, modeling (offline)
├── data/                     # Raw/processed data and routing graph CSVs
├── docs/                     # Design notes
├── requirements.txt
├── pyproject.toml            # Editable install: pip install -e .
└── .env.example              # Copy to .env for your API key
```

## Quick start

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .

python scripts/ensure_ml_datasets.py   # only needed if routing CSVs are missing

cp .env.example .env
# Edit .env and set GOOGLE_MAPS_API_KEY=...

streamlit run app/streamlit_app.py
```

Or use the Makefile (same steps):

```bash
make install
make setup-data
make run
```

### Google Maps API key

Enable **Directions** and **Geocoding** on your key. Configure it using one of:

- **`.env`** in the project root (recommended): `GOOGLE_MAPS_API_KEY=...` — loaded automatically via `python-dotenv`
- **Shell**: `export GOOGLE_MAPS_API_KEY=...` before `streamlit run`
- **Streamlit secrets**: `.streamlit/secrets.toml` with `GOOGLE_MAPS_API_KEY = "..."`

### Routing data

The app loads graph and training CSVs from `data/ml_datasets/`:

- `routing_nodes.csv`, `routing_edges.csv`, `congestion_ml.csv`

If those files are missing but `data/ml_datasets/ml_datasets.zip` is present, run:

```bash
python scripts/ensure_ml_datasets.py
```

## Route benchmark tests (pytest)

Integration tests call live Google APIs (key required):

```bash
pytest -s -m integration tests/test_route_benchmark.py
```

Optional pass/fail guardrail by delta magnitude (minutes):

```bash
ROUTE_BENCH_MAX_ABS_DELTA_MIN=4.0 pytest -s -m integration tests/test_route_benchmark.py
```

## Notebooks

Jupyter notebooks under `notebooks/` cover data cleaning, merging, EDA, and offline modeling. They are not required to run the Streamlit app.

## Routing tuning (optional env vars)

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
- `ROUTE_ETA_CALIB_A`, `ROUTE_ETA_CALIB_B` — affine calibration on predicted seconds: `A * seconds + B` (defaults `1.0`, `0.0`). Applied in `predict_route` and `score_polyline_eta_seconds`.
  - Practical tuning: increase `A` if ETAs are low proportionally; increase `B` if ETAs are low by a mostly fixed offset.
- `GOOGLE_DIRECTIONS_AVOID` — passed to Google Directions `avoid` (default `highways`, matching local-road training). Use pipe-separated values per Google docs (e.g. `highways|tolls`), or set empty to disable.

Further design detail: [docs/google-multi-route-model-scoring.md](docs/google-multi-route-model-scoring.md).
