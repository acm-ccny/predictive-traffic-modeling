"""
ML Preprocessing Pipeline
==========================
Loads the outputs of merge_aggregated.py and merge_detailed.py and transforms
them into two ML-ready datasets:

  1. congestion_ml.csv
     Tabular dataset for training a model that predicts traffic congestion
     (volume level) at any road segment given a location and time.
     One row = one segment × one volume reading.

  2. routing_edges.csv + routing_nodes.csv
     Graph-structured dataset for a routing model that recommends the fastest
     path between a start and end point given a date and time of week.
     routing_edges: directed edge list where each edge is a road segment with
                    an estimated travel time for each hour of the day.
     routing_nodes: unique intersections with lat/lon coordinates.

Key design note — time range mismatch:
  Dataset2 (speed/travel time) covers Apr 2026; Dataset6 (volume) covers
  2023–2025. There is no temporal overlap, so time-joined columns in the
  merged CSVs are all null. Speed and travel time features are instead built
  from static per-link averages across all available readings in datasets 1/2.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
# Resolved relative to this script file, not the working directory.
# Merged CSVs are written to the same folder as merge_detailed/aggregated.py.
# Dataset1/2 paths mirror the BASE convention used in those merge scripts.
HERE     = Path(__file__).parent
BASE     = HERE.parent.parent / "data" / "processed"

DETAIL_CSV   = HERE / "merged_dataset_detailed.csv"
AGG_CSV      = HERE / "merged_dataset_aggregated.csv"
DATASET1_CSV = BASE / "datasets_cleaned" / "dataset1_cleaned.csv"
DATASET2_CSV = BASE / "datasets_cleaned" / "dataset2_cleaned.csv"
DATASET6_CSV = BASE / "dataset6_cleaned.csv"

# =============================================================================
# SHARED HELPERS
# =============================================================================

def load_and_clean(path):
    """Load a merged CSV and fix known issues common to both merged files."""
    df = pd.read_csv(path)

    # Resolve borough naming collision from the merge (borough_x = dataset6 source)
    if "borough_x" in df.columns:
        df = df.rename(columns={"borough_x": "borough"})
    if "borough_y" in df.columns:
        df = df.drop(columns=["borough_y"])

    # Vol is stored as a string in dataset6
    df["Vol"] = pd.to_numeric(df["Vol"], errors="coerce")

    # Reconstruct a proper datetime from the separate time columns
    df["datetime"] = pd.to_datetime(
        dict(year=df["Yr"], month=df["M"], day=df["D"], hour=df["HH"], minute=df["MM"]),
        errors="coerce"
    )
    return df


def build_static_link_features(dataset1_path, dataset2_path):
    """
    Build per-link static feature averages from datasets 1 and 2.
    These are joined on link_name without any time dimension, working
    around the temporal mismatch between the datasets.
    """
    ds1 = pd.read_csv(dataset1_path)
    ds2 = pd.read_csv(dataset2_path)

    ds1_static = ds1.groupby("link_name", as_index=False).agg(
        link_length_ft       = ("link_length_ft",    "mean"),
        avg_median_tt_sec    = ("median_tt_sec",      "mean"),
        avg_median_speed_fps = ("median_speed_fps",   "mean"),
        avg_n_samples        = ("n_samples",          "mean"),
    )

    ds2_static = ds2.groupby("link_name", as_index=False).agg(
        avg_speed       = ("speed",        "mean"),
        avg_travel_time = ("travel_time",  "mean"),
    )

    return ds1_static.merge(ds2_static, on="link_name", how="outer")


def build_static_segment_features(dataset6_path):
    """
    Build per-segment historical volume statistics from dataset6.
    These are joined on SegmentID and capture the baseline traffic
    behaviour of each physical road segment across all recorded dates.
    Useful features:
      avg_vol_hist   — typical volume for this segment across all readings
      peak_vol_hist  — highest volume ever recorded (capacity signal)
      std_vol_hist   — variability (high std = unpredictable segment)
      peak_hour_hist — hour of day that typically sees the most traffic
    """
    ds6 = pd.read_csv(dataset6_path)
    ds6 = ds6.rename(columns={"Boro": "borough"})
    ds6["Vol"] = pd.to_numeric(ds6["Vol"], errors="coerce")
    ds6 = ds6.dropna(subset=["Vol", "SegmentID"])

    seg_stats = ds6.groupby("SegmentID", as_index=False).agg(
        avg_vol_hist  = ("Vol", "mean"),
        peak_vol_hist = ("Vol", "max"),
        std_vol_hist  = ("Vol", "std"),
    )

    # Peak hour: the hour that has the highest average volume for each segment
    peak_hour = (
        ds6.groupby(["SegmentID", "HH"])["Vol"].mean()
        .reset_index()
        .sort_values("Vol", ascending=False)
        .drop_duplicates("SegmentID")
        .rename(columns={"HH": "peak_hour_hist"})[["SegmentID", "peak_hour_hist"]]
    )

    return seg_stats.merge(peak_hour, on="SegmentID", how="left")


def engineer_time_features(df):
    """Add all time-based features needed for congestion and routing models."""
    # Raw features — useful for tree-based models and interpretability
    df["hour"]        = df["HH"]
    df["day_of_week"] = df["datetime"].dt.dayofweek      # 0=Mon … 6=Sun
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    # Morning rush 7–10 AM, evening rush 4–7 PM
    df["is_rush_hour"] = (
        df["HH"].between(7, 10) | df["HH"].between(16, 19)
    ).astype(int)

    # Cyclical encoding: makes hour 23 and hour 0 appear close to the model.
    # Kept alongside raw hour — neural nets and SVMs benefit from cyclical,
    # tree-based models (Random Forest, XGBoost) benefit from raw integers.
    df["hour_sin"]  = np.sin(2 * np.pi * df["HH"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["HH"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["M"]  / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["M"]  / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def engineer_lag_rolling_features(df):
    """
    Compute lag and rolling volume features per segment.

    Captures short-term temporal patterns — e.g. if the last 3 readings
    were high, congestion is likely still building. Requires df to already
    have SegmentID and datetime columns present.

    Lag features:
      vol_lag_1 / vol_lag_2 / vol_lag_3 — Vol from 1, 2, 3 time steps back
    Rolling feature:
      vol_rolling_avg_3 — mean of the previous 3 Vol readings per segment
    """
    df = df.sort_values(["SegmentID", "datetime"]).copy()

    grp = df.groupby("SegmentID")["Vol"]
    df["vol_lag_1"] = grp.shift(1)
    df["vol_lag_2"] = grp.shift(2)
    df["vol_lag_3"] = grp.shift(3)

    # Rolling mean of the 3 readings immediately before the current one
    df["vol_rolling_avg_3"] = (
        grp.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    # Fill NaN (first rows of each segment) with that segment's median Vol
    lag_cols = ["vol_lag_1", "vol_lag_2", "vol_lag_3", "vol_rolling_avg_3"]
    for col in lag_cols:
        df[col] = df.groupby("SegmentID")[col].transform(
            lambda x: x.fillna(x.median())
        )
    return df


# =============================================================================
# DATASET 1 — CONGESTION PREDICTION
# Uses the detailed merge (more individual readings = better training data).
# Target: predict Vol (regression) or is_congested (binary classification).
# =============================================================================

print("=" * 60)
print("Building congestion prediction dataset …")
print("=" * 60)

detail = load_and_clean(DETAIL_CSV)
static = build_static_link_features(DATASET1_CSV, DATASET2_CSV)
seg_features = build_static_segment_features(DATASET6_CSV)

# Drop null columns from the merged CSV that collide with static feature names.
# These columns exist in the merged CSV but are all null due to the time range
# mismatch between datasets — the static versions have real values.
_static_overlap = ["link_length_ft", "aggregation_period_sec", "n_samples",
                   "median_tt_sec", "median_speed_fps"]
detail = detail.drop(columns=[c for c in _static_overlap if c in detail.columns])

# Attach static link features (from datasets 1+2, joined on link_name)
detail = detail.merge(static, on="link_name", how="left")

# Attach historical segment volume stats (from dataset6, joined on SegmentID)
# SegmentID is dropped later — join must happen before the drop_cols step
detail["SegmentID"] = pd.to_numeric(detail["SegmentID"], errors="coerce")
detail = detail.merge(seg_features, on="SegmentID", how="left")

# Engineer time features
detail = engineer_time_features(detail)

# Lag and rolling features — must be called after time engineering so
# datetime exists, and before drop_cols so SegmentID is still present
detail = engineer_lag_rolling_features(detail)

# One-hot encode low-cardinality categoricals
detail = pd.get_dummies(detail, columns=["borough", "Direction"], drop_first=False)

# Drop columns not useful for ML.
# SegmentID is kept as a numeric location identifier per the suggestion.
drop_cols = [
    "WktGeom", "polyline", "link_points", "encoded_poly_line",
    "encoded_poly_line_lvls", "sid", "id", "link_id",
    "transcom_id", "street", "fromSt", "toSt", "link_name",
    "datetime", "median_calculation_timestamp", "data_as_of",
    "aggregation_period_sec", "owner", "match_distance_deg",
    "Yr", "M", "D", "HH", "MM",
    # speed has only 5 unique values (borough-level code, not real speed)
    "speed",
    # status is of unknown/negative meaning
    "status",
    # time-joined columns are all null due to dataset date range mismatch
    "median_tt_sec", "median_speed_fps", "n_samples", "travel_time",
]
detail = detail.drop(columns=[c for c in drop_cols if c in detail.columns])

# Drop rows where Vol is missing (can't train without the target)
detail = detail.dropna(subset=["Vol"])

# Fill remaining numeric NaN with column medians
num_cols = detail.select_dtypes(include="number").columns
detail[num_cols] = detail[num_cols].fillna(detail[num_cols].median())

# Build target columns
vol_median = detail["Vol"].median()
detail["is_congested"] = (detail["Vol"] > vol_median).astype(int)
print(f"  Congestion threshold (median Vol): {vol_median:.1f}")
print(f"  Class balance:\n{detail['is_congested'].value_counts().to_string()}")

# Separate features and targets
congestion_targets = detail[["Vol", "is_congested"]].copy()
congestion_features = detail.drop(columns=["Vol", "is_congested"])

# Scale continuous features (leave boolean/dummy columns unscaled)
bool_cols  = [c for c in congestion_features.columns
              if set(congestion_features[c].dropna().unique()).issubset({0, 1})]
scale_cols = [c for c in congestion_features.select_dtypes(include="number").columns
              if c not in bool_cols]

scaler = StandardScaler()
congestion_features[scale_cols] = scaler.fit_transform(congestion_features[scale_cols])

congestion_ml = pd.concat([congestion_features, congestion_targets], axis=1)
congestion_ml.to_csv("congestion_ml.csv", index=False)

print(f"  Shape: {congestion_ml.shape}")
print(f"  Saved → congestion_ml.csv")
print(f"  Features: {congestion_features.columns.tolist()}\n")


# =============================================================================
# DATASET 2 — ROUTING GRAPH
# Uses the aggregated merge (summarised per-segment per-hour conditions).
# Produces a directed edge list and node list for a graph-based routing model.
#
# Graph structure:
#   Node  = unique road intersection, identified as "STREET @ CROSS_STREET"
#   Edge  = directed road segment from from_node → to_node
#   Weight= estimated travel time in seconds, varying by hour of day
#
# Travel time estimation (no direct measurement available due to time mismatch):
#   base_travel_time = avg_travel_time from dataset2 static averages (seconds)
#   congestion_ratio = hourly avg Vol for this segment / median Vol for this segment
#   est_travel_time  = base_travel_time × max(1.0, congestion_ratio)
#   This makes travel time longer during high-volume hours.
# =============================================================================

print("=" * 60)
print("Building routing graph dataset …")
print("=" * 60)

agg = load_and_clean(AGG_CSV)
agg = agg.drop(columns=[c for c in _static_overlap if c in agg.columns])
agg = agg.merge(static, on="link_name", how="left")

# Attach historical segment volume stats from dataset6
agg["SegmentID"] = pd.to_numeric(agg["SegmentID"], errors="coerce")
agg = agg.merge(seg_features, on="SegmentID", how="left")

agg = engineer_time_features(agg)

# Drop rows where Vol or location is missing
agg = agg.dropna(subset=["Vol", "lat", "lon", "street", "fromSt", "toSt"])

# ── Build node identifiers from intersection names ────────────────────────────
# Each segment runs from (street ∩ fromSt) to (street ∩ toSt).
# Normalise to upper-case to avoid duplicates from mixed capitalisation.
def make_node_id(street, cross):
    s = str(street).strip().upper()
    c = str(cross).strip().upper()
    # Always sort alphabetically so "A @ B" and "B @ A" map to the same node
    parts = sorted([s, c])
    return f"{parts[0]} @ {parts[1]}"

agg["from_node"] = agg.apply(lambda r: make_node_id(r["street"], r["fromSt"]), axis=1)
agg["to_node"]   = agg.apply(lambda r: make_node_id(r["street"], r["toSt"]),   axis=1)

# ── Per-segment per-hour congestion profile ───────────────────────────────────
# Compute average and median Vol grouped by segment + hour across all dates.
seg_hour = agg.groupby(["link_name", "HH"], as_index=False).agg(
    avg_vol         = ("Vol",  "mean"),
    from_node       = ("from_node", "first"),
    to_node         = ("to_node",   "first"),
    lat             = ("lat",  "mean"),
    lon             = ("lon",  "mean"),
    borough         = ("borough", "first"),
    link_length_ft  = ("link_length_ft", "mean"),
    avg_travel_time = ("avg_travel_time", "mean"),  # static from ds2
    avg_speed       = ("avg_speed",       "mean"),  # static from ds2
    day_of_week     = ("day_of_week", lambda x: x.mode()[0] if len(x) else np.nan),
    is_weekend      = ("is_weekend",  "mean"),
    is_rush_hour    = ("is_rush_hour","mean"),
    hour_sin        = ("hour_sin",    "first"),
    hour_cos        = ("hour_cos",    "first"),
    dow_sin         = ("dow_sin",     "mean"),
    dow_cos         = ("dow_cos",     "mean"),
    # Historical volume stats from dataset6
    avg_vol_hist    = ("avg_vol_hist",   "mean"),
    peak_vol_hist   = ("peak_vol_hist",  "max"),
    std_vol_hist    = ("std_vol_hist",   "mean"),
    peak_hour_hist  = ("peak_hour_hist", "first"),
)

# Per-segment median Vol (used as the congestion baseline)
seg_median_vol = agg.groupby("link_name")["Vol"].median().rename("median_vol")
seg_hour = seg_hour.merge(seg_median_vol, on="link_name", how="left")

# Estimated travel time: stretch base time by congestion ratio
# Clamp to at least 1× so travel time never shrinks below the base
seg_hour["congestion_ratio"] = (
    seg_hour["avg_vol"] / seg_hour["median_vol"].replace(0, np.nan)
).fillna(1.0).clip(lower=1.0)

seg_hour["est_travel_time_sec"] = (
    seg_hour["avg_travel_time"].fillna(60) * seg_hour["congestion_ratio"]
)

# ── Routing edge list ─────────────────────────────────────────────────────────
routing_edges = seg_hour[[
    "link_name", "from_node", "to_node", "HH",
    "lat", "lon", "borough", "link_length_ft",
    "avg_vol", "median_vol", "congestion_ratio",
    "avg_speed", "avg_travel_time", "est_travel_time_sec",
    "avg_vol_hist", "peak_vol_hist", "std_vol_hist", "peak_hour_hist",
    "is_rush_hour", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]].copy()

routing_edges.to_csv("routing_edges.csv", index=False)
print(f"  Routing edges shape: {routing_edges.shape}")
print(f"  Saved → routing_edges.csv")

# ── Routing node list ─────────────────────────────────────────────────────────
# Collect all unique intersections with an approximate lat/lon.
from_nodes = agg[["from_node", "lat", "lon"]].rename(columns={"from_node": "node_id"})
to_nodes   = agg[["to_node",   "lat", "lon"]].rename(columns={"to_node":   "node_id"})
routing_nodes = (
    pd.concat([from_nodes, to_nodes])
    .dropna(subset=["node_id"])
    .groupby("node_id", as_index=False)
    .agg(lat=("lat", "mean"), lon=("lon", "mean"))
)

routing_nodes.to_csv("routing_nodes.csv", index=False)
print(f"  Routing nodes shape: {routing_nodes.shape}")
print(f"  Saved → routing_nodes.csv")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("Output files")
print("=" * 60)
print("  congestion_ml.csv   — congestion prediction training data")
print("  routing_edges.csv   — directed graph edges with travel times")
print("  routing_nodes.csv   — intersection nodes with coordinates")
print()
print("How to use routing_edges.csv with a routing algorithm:")
print("  1. Load routing_edges.csv and filter to the desired hour (HH).")
print("  2. Build a directed graph: edge = (from_node → to_node),")
print("     weight = est_travel_time_sec.")
print("  3. Run Dijkstra (e.g. networkx.shortest_path) from the")
print("     from_node nearest to the origin to the to_node nearest")
print("     to the destination.")
