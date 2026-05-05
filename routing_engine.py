import heapq
import math
import os

import folium
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.ensemble import RandomForestRegressor

# =========================
# 1. LOAD DATA
# =========================
nodes = pd.read_csv("data/ml_datasets/routing_nodes.csv")
edges = pd.read_csv("data/ml_datasets/routing_edges.csv")
ml_data = pd.read_csv("data/ml_datasets/congestion_ml.csv")

# Optional affine calibration vs reference (e.g. Google) — defaults are no-op.
ROUTE_ETA_CALIB_A = float(os.environ.get("ROUTE_ETA_CALIB_A", "1.0"))
ROUTE_ETA_CALIB_B = float(os.environ.get("ROUTE_ETA_CALIB_B", "0.0"))

# Tighter healing than before: degree threshold + hard cap in meters.
KDTREE_PAIR_DEG = float(os.environ.get("ROUTING_HEAL_PAIR_DEG", "0.02"))
MAX_HEAL_STRAIGHTLINE_M = float(os.environ.get("ROUTING_MAX_HEAL_M", "850"))
MAX_BRIDGE_GAP_M = float(os.environ.get("ROUTING_MAX_BRIDGE_M", "1200"))

# Google Directions `avoid` flags (pipe-separated), aligned with local-road training.
# Default excludes motorways / limited-access highways. Set to "" to allow highways.
GOOGLE_DIRECTIONS_AVOID = os.environ.get("GOOGLE_DIRECTIONS_AVOID", "highways").strip()

# =========================
# 2. TRAIN MODEL
# =========================
ml_features = [
    "hour",
    "day_of_week",
    "is_weekend",
    "is_rush_hour",
    "dow_sin",
    "dow_cos",
    "borough_Bronx",
    "borough_Brooklyn",
    "borough_Manhattan",
    "borough_Queens",
    "borough_Staten Island",
]

# Keep compatibility if older training snapshots are missing newer time features.
ml_features = [f for f in ml_features if f in ml_data.columns]

ml_data_clean = ml_data.dropna(subset=["avg_travel_time"])
X_train = ml_data_clean[ml_features]
y_train = ml_data_clean["avg_travel_time"]

rf_model = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

_RF_REFERENCE_SECONDS = float(max(np.median(y_train), 30.0))
_RF_CONGESTION_GAIN = float(os.environ.get("ROUTING_RF_CONGESTION_GAIN", "2.1"))
_RF_MULT_MIN = float(os.environ.get("ROUTING_RF_MULT_MIN", "0.40"))
_RF_MULT_MAX = float(os.environ.get("ROUTING_RF_MULT_MAX", "3.85"))
_URBAN_DELAY_BASE_SEC = float(os.environ.get("ROUTING_URBAN_DELAY_BASE_SEC", "6.8"))
_OFFPEAK_MULT_BONUS = float(os.environ.get("ROUTING_OFFPEAK_MULT_BONUS", "0.22"))
_DELAY_HWY_DAMP_M = float(os.environ.get("ROUTING_DELAY_HWY_DAMP_M", "420"))


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def _borough_flags(borough: str) -> dict[str, int]:
    names = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
    return {f"borough_{name}": (1 if borough == name else 0) for name in names}


def _is_rush_hour(hour: int) -> int:
    return 1 if (7 <= hour <= 9 or 16 <= hour <= 19) else 0


def _day_index(day_of_week: int | str | None, is_weekend: int = 0) -> int:
    if isinstance(day_of_week, str):
        days = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }
        idx = days.get(day_of_week.strip().lower())
        if idx is not None:
            return idx
    if isinstance(day_of_week, (int, np.integer)):
        return int(day_of_week) % 7
    # Fallback keeps backward compatibility for callers that only pass weekend/weekday.
    return 6 if int(is_weekend) else 2


def _urban_free_flow_mps(borough: str, hour: int) -> float:
    """Typical free-flow-ish urban driving speed (m/s), lower during rush."""
    rush = _is_rush_hour(hour)
    # (off_peak_mps, rush_mps): conservative local-road speeds by borough.
    table = {
        "Manhattan": (5.85, 4.15),
        "Brooklyn": (6.85, 4.95),
        "Queens": (6.25, 5.25),
        "Bronx": (7.2, 5.15),
        "Staten Island": (10.8, 7.6),
    }
    off, r = table.get(borough, (7.0, 5.1))
    return r if rush else off


def _virtual_speed_mps(borough: str, hour: int) -> float:
    """Synthetic / healed edges: deliberately conservative for realism."""
    base = _urban_free_flow_mps(borough, hour)
    return max(3.6, base * 0.50)


def rf_travel_multiplier(
    hour: int, is_weekend: int, borough: str, day_of_week: int | str | None = None
) -> float:
    """
    Map RF output (trained on avg_travel_time scale) to a congestion multiplier
    instead of using raw seconds as a per-edge travel time.
    """
    b_flags = _borough_flags(borough)
    day_idx = _day_index(day_of_week, is_weekend=is_weekend)
    dow_rad = 2.0 * math.pi * (float(day_idx) / 7.0)
    inp = pd.DataFrame(
        [
            {
                "hour": hour,
                "day_of_week": day_idx,
                "is_weekend": int(is_weekend),
                "is_rush_hour": _is_rush_hour(hour),
                "dow_sin": math.sin(dow_rad),
                "dow_cos": math.cos(dow_rad),
                **b_flags,
            }
        ]
    )
    raw = float(rf_model.predict(inp[ml_features])[0])
    if not np.isfinite(raw):
        return 1.0
    ratio = raw / _RF_REFERENCE_SECONDS
    mult = 1.0 + _RF_CONGESTION_GAIN * (ratio - 1.0)
    # Off-peak routes were still a bit optimistic; apply a small additive slowdown.
    if not _is_rush_hour(hour):
        mult += _OFFPEAK_MULT_BONUS
    # Weekend peak still trended fast vs Google on long cross-borough polylines.
    if int(is_weekend) and _is_rush_hour(hour):
        mult += float(os.environ.get("ROUTING_WEEKEND_RUSH_MULT_BONUS", "0.10"))
    return float(np.clip(mult, _RF_MULT_MIN, _RF_MULT_MAX))


def _urban_intersection_delay_sec(
    borough: str, hour: int, length_m: float, is_virtual: bool = False
) -> float:
    """
    Approximate urban friction from signals/turns.
    Length scaling avoids over-penalizing tiny polyline segments.
    """
    rush = _is_rush_hour(hour)
    borough_mult = {
        "Manhattan": 1.38,
        "Brooklyn": 1.18,
        "Queens": 1.08,
        "Bronx": 1.14,
        "Staten Island": 0.78,
    }.get(borough, 1.0)
    rush_mult = 1.38 if rush else 1.0
    virtual_mult = 0.62 if is_virtual else 1.0
    length_scale = min(1.0, max(0.16, float(length_m) / 100.0))
    # Long chords (bridges / expressway segments) should not get full per-segment signal friction.
    hwy_damp = 1.0 / (1.0 + max(0.0, float(length_m)) / max(1.0, _DELAY_HWY_DAMP_M))
    return _URBAN_DELAY_BASE_SEC * borough_mult * rush_mult * virtual_mult * length_scale * hwy_damp


def _interpolated_historical_seconds(times_dict: dict, hour: int) -> float | None:
    """
    Linearly interpolate stored HH travel times when an exact hour key is missing.
    times_dict keys are integers 0..23 from training.
    """
    if not times_dict:
        return None
    hour = int(hour) % 24
    if hour in times_dict:
        return float(times_dict[hour])

    hours_sorted = sorted(int(float(h)) for h in times_dict.keys())
    if len(hours_sorted) == 1:
        return float(times_dict[hours_sorted[0]])

    if hour <= hours_sorted[0]:
        return float(times_dict[hours_sorted[0]])
    if hour >= hours_sorted[-1]:
        return float(times_dict[hours_sorted[-1]])

    for lo, hi in zip(hours_sorted, hours_sorted[1:]):
        if lo <= hour <= hi:
            if lo == hi:
                return float(times_dict[lo])
            w = (hour - lo) / (hi - lo)
            return (1.0 - w) * float(times_dict[lo]) + w * float(times_dict[hi])
    return float(times_dict[hours_sorted[-1]])


def _edge_length_m(graph: nx.DiGraph, u, v, stored_m: float | None) -> float:
    if stored_m is not None and stored_m > 1.0:
        return float(stored_m)
    nu, nv = graph.nodes[u], graph.nodes[v]
    return max(
        1.0,
        _haversine_m(float(nu["lat"]), float(nu["lon"]), float(nv["lat"]), float(nv["lon"])),
    )


# =========================
# 3. BUILD GRAPH
# =========================
G = nx.DiGraph()
edge_lookup = {}

for _, n in nodes.iterrows():
    G.add_node(n["node_id"], lat=n["lat"], lon=n["lon"])

for _, row in edges.iterrows():
    u, v = row["from_node"], row["to_node"]
    if u not in G or v not in G:
        continue

    if (u, v) not in edge_lookup:
        edge_lookup[(u, v)] = {"times": {}, "borough": row["borough"], "length_m": None}

    t_val = max(row["est_travel_time_sec"], 10)
    edge_lookup[(u, v)]["times"][int(row["HH"])] = float(t_val)

    if "link_length_ft" in row and pd.notna(row["link_length_ft"]):
        try:
            lf_m = float(row["link_length_ft"]) * 0.3048
            if lf_m > 1.0:
                edge_lookup[(u, v)]["length_m"] = lf_m
        except (TypeError, ValueError):
            pass

    if edge_lookup[(u, v)]["length_m"] is None:
        edge_lookup[(u, v)]["length_m"] = _edge_length_m(G, u, v, None)

    G.add_edge(u, v)

# =========================
# 4. HEAL GRAPH (constrained)
# =========================
coords = nodes[["lat", "lon"]].values
tree = KDTree(coords)
pairs = tree.query_pairs(KDTREE_PAIR_DEG)

for i, j in pairs:
    u, v = nodes.iloc[i]["node_id"], nodes.iloc[j]["node_id"]
    if G.has_edge(u, v):
        continue
    dist_m = _haversine_m(
        float(nodes.iloc[i]["lat"]),
        float(nodes.iloc[i]["lon"]),
        float(nodes.iloc[j]["lat"]),
        float(nodes.iloc[j]["lon"]),
    )
    if dist_m > MAX_HEAL_STRAIGHTLINE_M:
        continue

    borough = "Manhattan"

    edge_info = {
        "times": {},
        "borough": borough,
        "is_virtual": True,
        "length_m": dist_m,
    }
    edge_lookup[(u, v)] = edge_info
    edge_lookup[(v, u)] = edge_info

    G.add_edge(u, v)
    G.add_edge(v, u)


def _bridge_components(graph, lookup, components_fn):
    while True:
        comps = list(components_fn(graph))
        if len(comps) <= 1:
            break

        best_d, best_u, best_v = float("inf"), None, None
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                for u in comps[i]:
                    nu = graph.nodes[u]
                    for v in comps[j]:
                        nv = graph.nodes[v]
                        d = math.sqrt(
                            (nu["lat"] - nv["lat"]) ** 2 + (nu["lon"] - nv["lon"]) ** 2
                        )
                        if d < best_d:
                            best_d, best_u, best_v = d, u, v

        gap_m = _haversine_m(
            float(graph.nodes[best_u]["lat"]),
            float(graph.nodes[best_u]["lon"]),
            float(graph.nodes[best_v]["lat"]),
            float(graph.nodes[best_v]["lon"]),
        )
        eff_m = min(gap_m, MAX_BRIDGE_GAP_M)
        borough = "Manhattan"
        edge_info = {
            "times": {},
            "borough": borough,
            "is_virtual": True,
            "length_m": eff_m,
        }
        lookup[(best_u, best_v)] = edge_info
        lookup[(best_v, best_u)] = edge_info
        graph.add_edge(best_u, best_v)
        graph.add_edge(best_v, best_u)


_bridge_components(G, edge_lookup, nx.weakly_connected_components)
_bridge_components(G, edge_lookup, nx.strongly_connected_components)

# =========================
# 5. COST FUNCTION
# =========================
def get_dynamic_cost(u, v, current_time_sec, is_weekend=0):
    edge = edge_lookup[(u, v)]
    hour = (int(current_time_sec) // 3600) % 24
    borough = str(edge.get("borough", "Manhattan"))
    length_m = _edge_length_m(G, u, v, edge.get("length_m"))

    hist = _interpolated_historical_seconds(edge.get("times") or {}, hour)
    if hist is not None and "is_virtual" not in edge:
        return float(hist)

    if "is_virtual" in edge:
        spd = _virtual_speed_mps(borough, hour)
        base = length_m / max(1.0, spd)
        delay = _urban_intersection_delay_sec(borough, hour, length_m, is_virtual=True)
        return max(10.0, base + delay)

    ff = length_m / max(1.0, _urban_free_flow_mps(borough, hour))
    mult = rf_travel_multiplier(hour, is_weekend, borough)
    delay = _urban_intersection_delay_sec(borough, hour, length_m)
    return max(6.5, ff * mult + delay)


# =========================
# 6. ROUTING
# =========================
def predict_route(source, target, start_hour, is_weekend=0):
    if source == target:
        return [source], 0

    start_sec = start_hour * 3600
    pq = [(start_sec, source)]
    best = {source: start_sec}
    parent = {}

    while pq:
        t, u = heapq.heappop(pq)
        if u == target:
            break
        if t > best.get(u, float("inf")):
            continue

        for v in G.successors(u):
            cost = get_dynamic_cost(u, v, t, is_weekend=is_weekend)
            new_t = t + max(6, cost)

            if new_t < best.get(v, float("inf")):
                best[v] = new_t
                parent[v] = u
                heapq.heappush(pq, (new_t, v))

    if target not in parent:
        return None, 0
    path = []
    curr = target
    while curr != source:
        path.append(curr)
        curr = parent[curr]
    path.append(source)
    raw_sec = best[target] - start_sec
    # Affine ETA calibration for practical fit to reference ETAs.
    # Tune by adjusting A for slope and B for fixed offset.
    calibrated = raw_sec * ROUTE_ETA_CALIB_A + ROUTE_ETA_CALIB_B
    return path[::-1], max(0.0, calibrated)


# =========================
# 7. MAP RENDERING
# =========================
def render_map(G, path):
    if not path:
        return None
    m = folium.Map(location=[G.nodes[path[0]]["lat"], G.nodes[path[0]]["lon"]], zoom_start=13)
    coords = [(G.nodes[n]["lat"], G.nodes[n]["lon"]) for n in path]
    folium.PolyLine(coords, color="blue", weight=5, opacity=0.8).add_to(m)
    folium.Marker(coords[0], popup="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(coords[-1], popup="End", icon=folium.Icon(color="red")).add_to(m)
    return m


def render_map_with_congestion(G, path, start_hour=None):
    """Alias for Streamlit; congestion styling does not vary by hour in this map."""
    return render_map(G, path)


def render_multi_route_map(routes_meta):
    """
    Render multiple route alternatives on one folium map.
    routes_meta items require:
      - points: [(lat, lon), ...]
      - name: route label
      - label: recommendation label text
      - color: polyline color
    """
    if not routes_meta:
        return None

    first_points = routes_meta[0]["points"]
    m = folium.Map(location=[first_points[0][0], first_points[0][1]], zoom_start=12)

    for route in routes_meta:
        points = route["points"]
        folium.PolyLine(
            points,
            color=route.get("color", "blue"),
            weight=5,
            opacity=0.85,
            tooltip=f"{route.get('name', 'Route')} - {route.get('label', '')}",
        ).add_to(m)

    start_pt = first_points[0]
    end_pt = first_points[-1]
    folium.Marker(start_pt, popup="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(end_pt, popup="End", icon=folium.Icon(color="red")).add_to(m)
    return m


if __name__ == "__main__":
    start = "1 AVENUE @ EAST 116 STREET"
    end = "1 AVENUE @ EAST 34 STREET"
    path, time_sec = predict_route(start, end, 8, is_weekend=0)
    if path:
        print(f"Success! Route found with {len(path)} nodes.")
        print(f"Total travel time: {time_sec / 60:.2f} minutes")
