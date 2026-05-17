"""
Score Google-provided route polylines using the same baseline + congestion
multiplier logic as routing_engine.get_dynamic_cost (not raw RF seconds per segment).
"""
from __future__ import annotations

import math
import os
from typing import Iterable

from scipy.spatial import KDTree

from traffic_routing.encoded_polyline import decode_polyline
from traffic_routing.routing_engine import (
    _urban_free_flow_mps,
    _urban_intersection_delay_sec,
    nodes,
    rf_travel_multiplier,
)

ROUTE_ETA_CALIB_A = float(os.environ.get("ROUTE_ETA_CALIB_A", "1.0"))
ROUTE_ETA_CALIB_B = float(os.environ.get("ROUTE_ETA_CALIB_B", "0.0"))
# When Google route distance is unavailable, inflate chords slightly vs true driven length.
ROUTE_POLYLINE_CHORD_FACTOR = float(os.environ.get("ROUTE_POLYLINE_CHORD_FACTOR", "1.12"))
# Partial calibration toward Google's distance (full match was unstable with step polylines + traffic).
ROUTE_POLYLINE_SCALE_MIN = float(os.environ.get("ROUTE_POLYLINE_SCALE_MIN", "0.93"))
ROUTE_POLYLINE_SCALE_MAX = float(os.environ.get("ROUTE_POLYLINE_SCALE_MAX", "1.28"))
ROUTE_POLYLINE_DISTANCE_BLEND = float(os.environ.get("ROUTE_POLYLINE_DISTANCE_BLEND", "0.45"))

_node_coords = nodes[["lat", "lon"]].values
_node_tree = KDTree(_node_coords)


def _haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = a
    lat2, lon2 = b
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def _nearest_borough(lat: float, lon: float) -> str:
    _, idx = _node_tree.query([lat, lon])
    return str(nodes.iloc[idx].get("borough", "Manhattan"))


def score_polyline_eta_seconds(
    points: Iterable[tuple[float, float]],
    departure_hour: int,
    is_weekend: int,
    *,
    day_of_week: int | str | None = None,
    google_distance_m: float | None = None,
) -> float:
    """
    Sum segment travel times using baseline + congestion multiplier + urban delay
    semantics aligned with routing_engine.get_dynamic_cost model-derived path.

    If ``google_distance_m`` is set, chord lengths are scaled **partially** toward Google's
    reported distance: ``1 + blend * (clamp(google/raw) - 1)``. Otherwise
    ``ROUTE_POLYLINE_CHORD_FACTOR`` is applied uniformly.
    """
    pts = list(points)
    if len(pts) < 2:
        return 0.0

    chords = [max(_haversine_m(pts[i], pts[i + 1]), 1.0) for i in range(len(pts) - 1)]
    raw_sum = float(sum(chords))
    if google_distance_m is not None and float(google_distance_m) > 1.0 and raw_sum > 1.0:
        ratio_raw = float(google_distance_m) / raw_sum
        ratio_clamped = max(
            ROUTE_POLYLINE_SCALE_MIN, min(ROUTE_POLYLINE_SCALE_MAX, ratio_raw)
        )
        b = max(0.0, min(1.0, ROUTE_POLYLINE_DISTANCE_BLEND))
        length_scale = 1.0 + b * (ratio_clamped - 1.0)
    else:
        length_scale = ROUTE_POLYLINE_CHORD_FACTOR

    current_time_sec = float(departure_hour * 3600)
    total_sec = 0.0

    for i in range(len(pts) - 1):
        a = pts[i]
        b = pts[i + 1]
        length_m = chords[i] * length_scale
        hour = (int(current_time_sec) // 3600) % 24
        borough = _nearest_borough(a[0], a[1])
        ff = length_m / max(1e-6, _urban_free_flow_mps(borough, hour))
        mult = rf_travel_multiplier(hour, is_weekend, borough, day_of_week=day_of_week)
        delay = _urban_intersection_delay_sec(borough, hour, length_m)
        segment_seconds = max(0.85, ff * mult + delay)
        total_sec += segment_seconds
        current_time_sec += segment_seconds

    return max(0.0, total_sec * ROUTE_ETA_CALIB_A + ROUTE_ETA_CALIB_B)
