"""
Google Directions API client for alternative routes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import requests

from encoded_polyline import decode_polyline
from google_places import get_maps_api_key

# Keep default behavior aligned with routing_engine without importing heavy model code.
GOOGLE_DIRECTIONS_AVOID = os.environ.get("GOOGLE_DIRECTIONS_AVOID", "highways").strip()
# Default on: use `duration_in_traffic` when departure_time is sent (closer to in-app Maps live).
# Set GOOGLE_DIRECTIONS_USE_TRAFFIC=0 to force static `duration`.
GOOGLE_DIRECTIONS_USE_TRAFFIC = os.environ.get("GOOGLE_DIRECTIONS_USE_TRAFFIC", "1").strip().lower() in (
    "1",
    "true",
    "yes",
)
GOOGLE_DIRECTIONS_TRAFFIC_MODEL = os.environ.get("GOOGLE_DIRECTIONS_TRAFFIC_MODEL", "best_guess").strip()
GOOGLE_DIRECTIONS_TZ = os.environ.get("GOOGLE_DIRECTIONS_TZ", "America/New_York")
# Step polylines add many short segments (extra delay stacking). Default overview for scoring.
GOOGLE_DIRECTIONS_USE_STEP_POLYLINES = os.environ.get(
    "GOOGLE_DIRECTIONS_USE_STEP_POLYLINES", "0"
).strip().lower() in ("1", "true", "yes")

DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"
_NY_TZ = ZoneInfo(GOOGLE_DIRECTIONS_TZ)


def _next_departure_unix(day_of_week: str, hour: int) -> int:
    """Next future occurrence of weekday + clock hour in GOOGLE_DIRECTIONS_TZ (for traffic requests)."""
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    target_wd = order.index(day_of_week)
    now = datetime.now(_NY_TZ)
    for delta in range(0, 15):
        day = now.date() + timedelta(days=delta)
        cand = datetime.combine(day, time(hour, 0, 0), tzinfo=_NY_TZ)
        if cand.weekday() != target_wd:
            continue
        if cand <= now:
            continue
        return int(cand.timestamp())
    # Rare fallback: next hour (still satisfies departure_time for traffic fields).
    return int((now + timedelta(hours=1)).timestamp())


def _points_from_route(route: dict) -> list[tuple[float, float]] | None:
    """Merge leg step polylines into one path (finer than overview_polyline)."""
    pts: list[tuple[float, float]] = []
    for leg in route.get("legs", []) or []:
        for step in leg.get("steps", []) or []:
            enc = (step.get("polyline") or {}).get("points")
            if not enc:
                continue
            part = decode_polyline(enc)
            if not part:
                continue
            if pts:
                la, lo = pts[-1]
                if abs(la - part[0][0]) < 1e-8 and abs(lo - part[0][1]) < 1e-8:
                    part = part[1:]
            pts.extend(part)
    return pts if len(pts) >= 2 else None


def _leg_duration_pair(leg: dict) -> tuple[float, float, bool]:
    """Returns (preferred_seconds, static_seconds, used_in_traffic)."""
    static = float((leg.get("duration") or {}).get("value", 0))
    dit = leg.get("duration_in_traffic")
    if isinstance(dit, dict) and dit.get("value") is not None:
        return float(dit["value"]), static, True
    return static, static, False


@dataclass
class GoogleRoute:
    polyline: str
    duration_sec: float
    distance_m: float
    summary: str
    path_points: list[tuple[float, float]] | None = None
    duration_static_sec: float = 0.0
    uses_duration_in_traffic: bool = False


def fetch_route_alternatives(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    *,
    api_key: str | None = None,
    max_routes: int = 3,
    day_of_week: str | None = None,
    departure_hour: int | None = None,
) -> tuple[list[GoogleRoute], str | None]:
    """Fetch up to max_routes alternatives from Directions API."""
    key = api_key or get_maps_api_key()
    if not key:
        return [], "Missing GOOGLE_MAPS_API_KEY (set env var or Streamlit secrets)."

    params: dict[str, str] = {
        "origin": f"{origin_lat},{origin_lon}",
        "destination": f"{dest_lat},{dest_lon}",
        "mode": "driving",
        "alternatives": "true",
        "key": key,
    }
    if GOOGLE_DIRECTIONS_AVOID:
        params["avoid"] = GOOGLE_DIRECTIONS_AVOID

    use_traffic = GOOGLE_DIRECTIONS_USE_TRAFFIC
    if use_traffic and day_of_week is not None and departure_hour is not None:
        params["departure_time"] = str(_next_departure_unix(day_of_week, int(departure_hour)))
        if GOOGLE_DIRECTIONS_TRAFFIC_MODEL:
            params["traffic_model"] = GOOGLE_DIRECTIONS_TRAFFIC_MODEL

    url = f"{DIRECTIONS_URL}?{urlencode(params)}"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        return [], f"Directions request failed: {exc}"

    status = payload.get("status")
    if status != "OK":
        if status == "ZERO_RESULTS":
            return [], "No driving route found between these points."
        return [], f"Directions API error: {payload.get('error_message') or status}"

    routes: list[GoogleRoute] = []
    for route in payload.get("routes", [])[:max_routes]:
        polyline = route.get("overview_polyline", {}).get("points")
        if not polyline:
            continue
        legs = route.get("legs", []) or []
        duration_static_total = 0.0
        duration_pref_total = 0.0
        any_traffic = False
        for leg in legs:
            pref, static, used_traffic = _leg_duration_pair(leg)
            duration_pref_total += pref
            duration_static_total += static
            any_traffic = any_traffic or used_traffic

        distance_m = sum(float((leg.get("distance") or {}).get("value", 0)) for leg in legs)
        path_points = _points_from_route(route) if GOOGLE_DIRECTIONS_USE_STEP_POLYLINES else None

        routes.append(
            GoogleRoute(
                polyline=polyline,
                duration_sec=duration_pref_total,
                distance_m=distance_m,
                summary=route.get("summary", ""),
                path_points=path_points,
                duration_static_sec=duration_static_total,
                uses_duration_in_traffic=any_traffic,
            )
        )

    return routes, None
