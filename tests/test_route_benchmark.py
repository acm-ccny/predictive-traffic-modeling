from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from google_directions import fetch_route_alternatives
from google_places import get_maps_api_key, resolve_place
from google_route_scoring import decode_polyline, score_polyline_eta_seconds


@dataclass(frozen=True)
class RouteCase:
    name: str
    origin: str
    destination: str
    origin_borough: str
    destination_borough: str
    scenario: str
    day_of_week: str
    hour: int


ROUTE_CASES = [
    RouteCase(
        name="Flushing Main St -> Bayside LIRR",
        origin="Flushing–Main St Station, Queens, NY",
        destination="Bayside LIRR Station, Queens, NY",
        origin_borough="Queens",
        destination_borough="Queens",
        scenario="weekday_offpeak",
        day_of_week="Monday",
        hour=11,
    ),
    RouteCase(
        name="Times Square -> Wall Street",
        origin="Times Square, New York, NY",
        destination="Wall Street, New York, NY",
        origin_borough="Manhattan",
        destination_borough="Manhattan",
        scenario="weekday_peak",
        day_of_week="Monday",
        hour=8,
    ),
    RouteCase(
        name="Astoria -> JFK Terminal 4",
        origin="Astoria, Queens, NY",
        destination="JFK Terminal 4, Queens, NY",
        origin_borough="Queens",
        destination_borough="Queens",
        scenario="weekday_offpeak",
        day_of_week="Wednesday",
        hour=14,
    ),
    RouteCase(
        name="Yankee Stadium -> Coney Island",
        origin="Yankee Stadium, Bronx, NY",
        destination="Coney Island, Brooklyn, NY",
        origin_borough="Bronx",
        destination_borough="Brooklyn",
        scenario="weekend_offpeak",
        day_of_week="Saturday",
        hour=13,
    ),
    RouteCase(
        name="St George Terminal -> SI Mall",
        origin="St George Ferry Terminal, Staten Island, NY",
        destination="Staten Island Mall, Staten Island, NY",
        origin_borough="Staten Island",
        destination_borough="Staten Island",
        scenario="weekday_peak",
        day_of_week="Thursday",
        hour=17,
    ),
    RouteCase(
        name="Pelham Bay -> Fordham",
        origin="Pelham Bay Park, Bronx, NY",
        destination="Fordham University, Bronx, NY",
        origin_borough="Bronx",
        destination_borough="Bronx",
        scenario="weekday_offpeak",
        day_of_week="Tuesday",
        hour=11,
    ),
    RouteCase(
        name="Downtown Brooklyn -> Williamsburg",
        origin="Downtown Brooklyn, Brooklyn, NY",
        destination="Williamsburg, Brooklyn, NY",
        origin_borough="Brooklyn",
        destination_borough="Brooklyn",
        scenario="weekend_peak",
        day_of_week="Saturday",
        hour=8,
    ),
    RouteCase(
        name="Long Island City -> Flushing",
        origin="Long Island City, Queens, NY",
        destination="Flushing, Queens, NY",
        origin_borough="Queens",
        destination_borough="Queens",
        scenario="weekend_offpeak",
        day_of_week="Sunday",
        hour=14,
    ),
    RouteCase(
        name="Harlem -> Astoria",
        origin="Harlem, New York, NY",
        destination="Astoria, Queens, NY",
        origin_borough="Manhattan",
        destination_borough="Queens",
        scenario="weekday_peak",
        day_of_week="Tuesday",
        hour=8,
    ),
    RouteCase(
        name="Bayside -> Midtown",
        origin="Bayside, Queens, NY",
        destination="Midtown Manhattan, New York, NY",
        origin_borough="Queens",
        destination_borough="Manhattan",
        scenario="weekday_offpeak",
        day_of_week="Wednesday",
        hour=13,
    ),
    RouteCase(
        name="Park Slope -> SoHo",
        origin="Park Slope, Brooklyn, NY",
        destination="SoHo, New York, NY",
        origin_borough="Brooklyn",
        destination_borough="Manhattan",
        scenario="weekday_peak",
        day_of_week="Friday",
        hour=17,
    ),
    RouteCase(
        name="Chelsea -> Bushwick",
        origin="Chelsea, New York, NY",
        destination="Bushwick, Brooklyn, NY",
        origin_borough="Manhattan",
        destination_borough="Brooklyn",
        scenario="weekend_offpeak",
        day_of_week="Sunday",
        hour=12,
    ),
    RouteCase(
        name="Bronx Zoo -> LaGuardia",
        origin="Bronx Zoo, Bronx, NY",
        destination="LaGuardia Airport, Queens, NY",
        origin_borough="Bronx",
        destination_borough="Queens",
        scenario="weekday_peak",
        day_of_week="Thursday",
        hour=8,
    ),
    RouteCase(
        name="Jamaica -> Yankee Stadium",
        origin="Jamaica, Queens, NY",
        destination="Yankee Stadium, Bronx, NY",
        origin_borough="Queens",
        destination_borough="Bronx",
        scenario="weekday_offpeak",
        day_of_week="Monday",
        hour=11,
    ),
    RouteCase(
        name="Bay Ridge -> Bronx Terminal",
        origin="Bay Ridge, Brooklyn, NY",
        destination="Bronx Terminal Market, Bronx, NY",
        origin_borough="Brooklyn",
        destination_borough="Bronx",
        scenario="weekend_peak",
        day_of_week="Saturday",
        hour=9,
    ),
    RouteCase(
        name="Morris Park -> Downtown Brooklyn",
        origin="Morris Park, Bronx, NY",
        destination="Downtown Brooklyn, Brooklyn, NY",
        origin_borough="Bronx",
        destination_borough="Brooklyn",
        scenario="weekend_offpeak",
        day_of_week="Sunday",
        hour=15,
    ),
    RouteCase(
        name="St George -> Financial District",
        origin="St George Ferry Terminal, Staten Island, NY",
        destination="Financial District, New York, NY",
        origin_borough="Staten Island",
        destination_borough="Manhattan",
        scenario="weekday_peak",
        day_of_week="Wednesday",
        hour=8,
    ),
    RouteCase(
        name="Lower Manhattan -> Staten Island Mall",
        origin="Battery Park, New York, NY",
        destination="Staten Island Mall, Staten Island, NY",
        origin_borough="Manhattan",
        destination_borough="Staten Island",
        scenario="weekday_offpeak",
        day_of_week="Tuesday",
        hour=14,
    ),
    RouteCase(
        name="Sunset Park -> SI Mall",
        origin="Sunset Park, Brooklyn, NY",
        destination="Staten Island Mall, Staten Island, NY",
        origin_borough="Brooklyn",
        destination_borough="Staten Island",
        scenario="weekend_peak",
        day_of_week="Saturday",
        hour=8,
    ),
    RouteCase(
        name="Staten Island Mall -> Bay Ridge",
        origin="Staten Island Mall, Staten Island, NY",
        destination="Bay Ridge, Brooklyn, NY",
        origin_borough="Staten Island",
        destination_borough="Brooklyn",
        scenario="weekend_offpeak",
        day_of_week="Sunday",
        hour=13,
    ),
    RouteCase(
        name="SI Mall -> Flushing",
        origin="Staten Island Mall, Staten Island, NY",
        destination="Flushing, Queens, NY",
        origin_borough="Staten Island",
        destination_borough="Queens",
        scenario="weekday_peak",
        day_of_week="Friday",
        hour=17,
    ),
    RouteCase(
        name="Flushing -> St George Terminal",
        origin="Flushing, Queens, NY",
        destination="St George Ferry Terminal, Staten Island, NY",
        origin_borough="Queens",
        destination_borough="Staten Island",
        scenario="weekday_offpeak",
        day_of_week="Thursday",
        hour=12,
    ),
    RouteCase(
        name="Arthur Ave -> Staten Island Mall",
        origin="Arthur Avenue, Bronx, NY",
        destination="Staten Island Mall, Staten Island, NY",
        origin_borough="Bronx",
        destination_borough="Staten Island",
        scenario="weekend_peak",
        day_of_week="Saturday",
        hour=9,
    ),
    RouteCase(
        name="St George -> Fordham",
        origin="St George Ferry Terminal, Staten Island, NY",
        destination="Fordham University, Bronx, NY",
        origin_borough="Staten Island",
        destination_borough="Bronx",
        scenario="weekend_offpeak",
        day_of_week="Sunday",
        hour=14,
    ),
]


def _is_weekend(day_of_week: str) -> int:
    return 1 if day_of_week in {"Saturday", "Sunday"} else 0


def _fmt_minutes(seconds: float) -> str:
    return f"{seconds / 60.0:6.2f}"


def _build_output_table(rows: list[dict[str, str]]) -> str:
    headers = ["Case", "Dep", "Google(min)", "Model(min)", "Delta(min)", "Dist(km)", "Google Route"]
    cols = [
        [r["case"] for r in rows],
        [r["dep"] for r in rows],
        [r["google_min"] for r in rows],
        [r["model_min"] for r in rows],
        [r["delta_min"] for r in rows],
        [r["distance_km"] for r in rows],
        [r["summary"] for r in rows],
    ]
    widths = [max(len(headers[i]), max((len(v) for v in col), default=0)) for i, col in enumerate(cols)]

    line = " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    sep = "-+-".join("-" * widths[i] for i in range(len(headers)))
    body = [
        " | ".join(
            [
                r["case"].ljust(widths[0]),
                r["dep"].ljust(widths[1]),
                r["google_min"].rjust(widths[2]),
                r["model_min"].rjust(widths[3]),
                r["delta_min"].rjust(widths[4]),
                r["distance_km"].rjust(widths[5]),
                r["summary"].ljust(widths[6]),
            ]
        )
        for r in rows
    ]
    return "\n".join([line, sep, *body])


def test_route_case_coverage_matrix() -> None:
    """
    Guardrail to keep benchmark breadth:
    - each borough appears multiple times as origin and destination
    - includes both intra- and inter-borough routes
    - includes all requested day/time scenarios
    """
    boroughs = {"Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"}
    required_scenarios = {
        "weekend_peak",
        "weekend_offpeak",
        "weekday_peak",
        "weekday_offpeak",
    }

    origin_counts = {b: 0 for b in boroughs}
    destination_counts = {b: 0 for b in boroughs}
    seen_scenarios: set[str] = set()
    has_intra = False
    has_inter = False

    for case in ROUTE_CASES:
        origin_counts[case.origin_borough] += 1
        destination_counts[case.destination_borough] += 1
        seen_scenarios.add(case.scenario)
        has_intra = has_intra or (case.origin_borough == case.destination_borough)
        has_inter = has_inter or (case.origin_borough != case.destination_borough)

    assert seen_scenarios == required_scenarios
    assert has_intra, "Expected at least one within-borough route."
    assert has_inter, "Expected at least one cross-borough route."
    assert all(origin_counts[b] >= 2 for b in boroughs), f"Origin counts: {origin_counts}"
    assert all(destination_counts[b] >= 2 for b in boroughs), f"Destination counts: {destination_counts}"


@pytest.mark.integration
def test_route_benchmark_against_google() -> None:
    """
    Integration benchmark:
    - Resolves each place pair via Google Geocoding
    - Fetches Google alternatives
    - Scores Google route polyline with local model ETA
    - Prints compact comparison table

    Optional guardrail:
    - Set ROUTE_BENCH_MAX_ABS_DELTA_MIN to fail if abs(delta) exceeds this value.
      Example: ROUTE_BENCH_MAX_ABS_DELTA_MIN=4.0 pytest -s -m integration tests/test_route_benchmark.py
    """
    api_key = get_maps_api_key()
    if not api_key:
        pytest.skip("GOOGLE_MAPS_API_KEY missing; skipping integration route benchmark.")

    max_abs_delta_env = os.environ.get("ROUTE_BENCH_MAX_ABS_DELTA_MIN")
    max_abs_delta_min = float(max_abs_delta_env) if max_abs_delta_env else None

    rows: list[dict[str, str]] = []
    violations: list[str] = []

    for case in ROUTE_CASES:
        origin, origin_err = resolve_place(case.origin, api_key=api_key)
        assert not origin_err and origin is not None, f"{case.name}: origin resolution failed: {origin_err}"

        destination, dest_err = resolve_place(case.destination, api_key=api_key)
        assert not dest_err and destination is not None, (
            f"{case.name}: destination resolution failed: {dest_err}"
        )

        routes, route_err = fetch_route_alternatives(
            origin.lat,
            origin.lon,
            destination.lat,
            destination.lon,
            api_key=api_key,
            max_routes=3,
            day_of_week=case.day_of_week,
            departure_hour=case.hour,
        )
        assert not route_err and routes, f"{case.name}: directions fetch failed: {route_err}"

        route = min(routes, key=lambda r: r.duration_sec)
        points = route.path_points if route.path_points else decode_polyline(route.polyline)
        model_sec = score_polyline_eta_seconds(
            points=points,
            departure_hour=case.hour,
            is_weekend=_is_weekend(case.day_of_week),
            google_distance_m=route.distance_m,
        )
        google_sec = float(route.duration_sec)
        delta_min = (model_sec - google_sec) / 60.0

        if max_abs_delta_min is not None and abs(delta_min) > max_abs_delta_min:
            violations.append(
                f"{case.name}: abs(delta)={abs(delta_min):.2f} > {max_abs_delta_min:.2f} minutes"
            )

        rows.append(
            {
                "case": case.name,
                "dep": f"{case.day_of_week[:3]} {case.hour:02d}:00",
                "google_min": _fmt_minutes(google_sec),
                "model_min": _fmt_minutes(model_sec),
                "delta_min": f"{delta_min:6.2f}",
                "distance_km": f"{route.distance_m / 1000.0:6.2f}",
                "summary": route.summary or "-",
            }
        )

    print("\nRoute benchmark summary (best Google alternative per case):")
    print(_build_output_table(rows))

    if violations:
        pytest.fail("Route delta guardrail failed:\n- " + "\n- ".join(violations))
