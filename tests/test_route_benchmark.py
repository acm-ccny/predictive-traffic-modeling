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


@dataclass(frozen=True)
class RouteGroup:
    name: str
    origin: str
    destination: str
    origin_borough: str
    destination_borough: str


ROUTE_GROUPS = [
    RouteGroup(
        name="Times Square -> Wall Street",
        origin="Times Square, New York, NY",
        destination="Wall Street, New York, NY",
        origin_borough="Manhattan",
        destination_borough="Manhattan",
    ),
    RouteGroup(
        name="Downtown Brooklyn -> Williamsburg",
        origin="Downtown Brooklyn, Brooklyn, NY",
        destination="Williamsburg, Brooklyn, NY",
        origin_borough="Brooklyn",
        destination_borough="Brooklyn",
    ),
    RouteGroup(
        name="Long Island City -> Flushing",
        origin="Long Island City, Queens, NY",
        destination="Flushing, Queens, NY",
        origin_borough="Queens",
        destination_borough="Queens",
    ),
    RouteGroup(
        name="Pelham Bay -> Fordham",
        origin="Pelham Bay Park, Bronx, NY",
        destination="Fordham University, Bronx, NY",
        origin_borough="Bronx",
        destination_borough="Bronx",
    ),
    RouteGroup(
        name="St George Terminal -> SI Mall",
        origin="St George Ferry Terminal, Staten Island, NY",
        destination="Staten Island Mall, Staten Island, NY",
        origin_borough="Staten Island",
        destination_borough="Staten Island",
    ),
    RouteGroup(
        name="Harlem -> Astoria",
        origin="Harlem, New York, NY",
        destination="Astoria, Queens, NY",
        origin_borough="Manhattan",
        destination_borough="Queens",
    ),
    RouteGroup(
        name="Bay Ridge -> Bronx Terminal",
        origin="Bay Ridge, Brooklyn, NY",
        destination="Bronx Terminal Market, Bronx, NY",
        origin_borough="Brooklyn",
        destination_borough="Bronx",
    ),
    RouteGroup(
        name="St George -> Financial District",
        origin="St George Ferry Terminal, Staten Island, NY",
        destination="Financial District, New York, NY",
        origin_borough="Staten Island",
        destination_borough="Manhattan",
    ),
    RouteGroup(
        name="Lower Manhattan -> Staten Island Mall",
        origin="Battery Park, New York, NY",
        destination="Staten Island Mall, Staten Island, NY",
        origin_borough="Manhattan",
        destination_borough="Staten Island",
    ),
    RouteGroup(
        name="Bronx Zoo -> LaGuardia",
        origin="Bronx Zoo, Bronx, NY",
        destination="LaGuardia Airport, Queens, NY",
        origin_borough="Bronx",
        destination_borough="Queens",
    ),
]

SCENARIOS = [
    ("weekday_peak", "Monday", 8),
    ("weekday_offpeak", "Wednesday", 14),
    ("weekend_peak", "Saturday", 9),
    ("weekend_offpeak", "Sunday", 13),
]

ROUTE_CASES = [
    RouteCase(
        name=group.name,
        origin=group.origin,
        destination=group.destination,
        origin_borough=group.origin_borough,
        destination_borough=group.destination_borough,
        scenario=scenario,
        day_of_week=day,
        hour=hour,
    )
    for group in ROUTE_GROUPS
    for scenario, day, hour in SCENARIOS
]


def _is_weekend(day_of_week: str) -> int:
    return 1 if day_of_week in {"Saturday", "Sunday"} else 0


def _fmt_minutes(seconds: float) -> str:
    return f"{seconds / 60.0:6.2f}"


def _build_output_table(rows: list[dict[str, str]]) -> str:
    headers = [
        "Case",
        "Scenario",
        "Dep",
        "Google(min)",
        "Model(min)",
        "Delta(min)",
        "Dist(km)",
        "GoogleTraffic",
        "Google Route",
    ]
    cols = [
        [r["case"] for r in rows],
        [r["scenario"] for r in rows],
        [r["dep"] for r in rows],
        [r["google_min"] for r in rows],
        [r["model_min"] for r in rows],
        [r["delta_min"] for r in rows],
        [r["distance_km"] for r in rows],
        [r["google_traffic"] for r in rows],
        [r["summary"] for r in rows],
    ]
    widths = [max(len(headers[i]), max((len(v) for v in col), default=0)) for i, col in enumerate(cols)]

    line = " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    sep = "-+-".join("-" * widths[i] for i in range(len(headers)))
    body = [
        " | ".join(
            [
                r["case"].ljust(widths[0]),
                r["scenario"].ljust(widths[1]),
                r["dep"].ljust(widths[2]),
                r["google_min"].rjust(widths[3]),
                r["model_min"].rjust(widths[4]),
                r["delta_min"].rjust(widths[5]),
                r["distance_km"].rjust(widths[6]),
                r["google_traffic"].ljust(widths[7]),
                r["summary"].ljust(widths[8]),
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
            day_of_week=case.day_of_week,
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
                "scenario": case.scenario,
                "dep": f"{case.day_of_week[:3]} {case.hour:02d}:00",
                "google_min": _fmt_minutes(google_sec),
                "model_min": _fmt_minutes(model_sec),
                "delta_min": f"{delta_min:6.2f}",
                "distance_km": f"{route.distance_m / 1000.0:6.2f}",
                "google_traffic": "yes" if route.uses_duration_in_traffic else "no",
                "summary": route.summary or "-",
            }
        )

    print("\nRoute benchmark summary (best Google alternative per case):")
    print(_build_output_table(rows))

    if violations:
        pytest.fail("Route delta guardrail failed:\n- " + "\n- ".join(violations))
