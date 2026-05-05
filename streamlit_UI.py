# This STREAMLIT APP should actually be a separate .py file, but for simplicity we include it here.
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from google_directions import fetch_route_alternatives
from google_places import resolve_place
from google_route_scoring import decode_polyline, score_polyline_eta_seconds
from routing_engine import render_multi_route_map

st.title("Predictive Traffic Routing: Google vs Our Model")

if "compare_result" not in st.session_state:
    st.session_state.compare_result = None
if "last_request_params" not in st.session_state:
    st.session_state.last_request_params = None

start_query = st.text_input("Start place", value="Times Square, New York, NY")
end_query = st.text_input("End place", value="Wall Street, New York, NY")

day_options = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
day_of_week = st.selectbox("Departure day", day_options, index=0)
hour = st.slider("Departure hour", 0, 23, 8)
is_weekend = 1 if day_of_week in {"Saturday", "Sunday"} else 0


def _recompute_routes() -> None:
    start_place, start_err = resolve_place(start_query)
    end_place, end_err = resolve_place(end_query)
    if start_err:
        st.session_state.compare_result = {"ok": False, "error": f"Start place error: {start_err}"}
    elif end_err:
        st.session_state.compare_result = {"ok": False, "error": f"End place error: {end_err}"}
    else:
        routes, route_err = fetch_route_alternatives(
            start_place.lat,
            start_place.lon,
            end_place.lat,
            end_place.lon,
            max_routes=3,
            day_of_week=day_of_week,
            departure_hour=int(hour),
        )
        if route_err:
            st.session_state.compare_result = {"ok": False, "error": route_err}
        elif not routes:
            st.session_state.compare_result = {"ok": False, "error": "No routes returned by Google."}
        else:
            rows = []
            routes_meta = []
            colors = ["blue", "orange", "purple"]
            for idx, route in enumerate(routes, start=1):
                points = route.path_points if route.path_points else decode_polyline(route.polyline)
                model_eta_sec = score_polyline_eta_seconds(
                    points,
                    hour,
                    is_weekend=is_weekend,
                    day_of_week=day_of_week,
                    google_distance_m=route.distance_m,
                )
                rows.append(
                    {
                        "route_idx": idx,
                        "google_duration_min": route.duration_sec / 60.0,
                        "model_eta_min": model_eta_sec / 60.0,
                        "delta_min_model_minus_google": (model_eta_sec - route.duration_sec) / 60.0,
                        "distance_km": route.distance_m / 1000.0,
                        "summary": route.summary,
                    }
                )
                routes_meta.append(
                    {
                        "name": f"Route {idx}",
                        "points": points,
                        "color": colors[(idx - 1) % len(colors)],
                        "google_duration_sec": route.duration_sec,
                        "model_eta_sec": model_eta_sec,
                        "label": "",
                    }
                )

            df = pd.DataFrame(rows)
            google_best = int(df["google_duration_min"].idxmin())
            model_best = int(df["model_eta_min"].idxmin())
            routes_meta[google_best]["label"] = "Google recommendation"
            if model_best == google_best:
                routes_meta[model_best]["label"] = "Google + Our model recommendation"
            else:
                routes_meta[model_best]["label"] = "Our model recommendation"

            st.session_state.compare_result = {
                "ok": True,
                "start_name": start_place.formatted_address,
                "end_name": end_place.formatted_address,
                "routes_meta": routes_meta,
                "df": df,
                "google_best": google_best,
                "model_best": model_best,
                "day_of_week": day_of_week,
                "hour": hour,
            }

current_params = {
    "start_query": start_query.strip(),
    "end_query": end_query.strip(),
    "day_of_week": day_of_week,
    "hour": int(hour),
}

# Re-run full comparison whenever any user input changes,
# including a fresh Directions API call.
if current_params != st.session_state.last_request_params:
    st.session_state.last_request_params = current_params
    with st.spinner("Refreshing routes and ETAs..."):
        _recompute_routes()

result = st.session_state.compare_result
if result is not None:
    if not result["ok"]:
        st.error(result["error"])
    else:
        st.caption(
            f"Trip: {result['start_name']} -> {result['end_name']} | "
            f"{result['day_of_week']} @ {result['hour']:02d}:00"
        )
        map_obj = render_multi_route_map(result["routes_meta"])
        st_folium(map_obj, width=900, height=520, key="compare_route_map")

        df = result["df"].copy()
        df["google_duration_min"] = df["google_duration_min"].round(2)
        df["model_eta_min"] = df["model_eta_min"].round(2)
        df["delta_min_model_minus_google"] = df["delta_min_model_minus_google"].round(2)
        df["distance_km"] = df["distance_km"].round(2)
        st.subheader("Route-by-route comparison")
        st.dataframe(df, hide_index=True, use_container_width=True)

        g_idx = result["google_best"] + 1
        m_idx = result["model_best"] + 1
        g_min = float(df.loc[result["google_best"], "google_duration_min"])
        m_min = float(df.loc[result["model_best"], "model_eta_min"])
        st.subheader("Summary and conclusion")
        if g_idx == m_idx:
            st.success(
                f"Both systems recommend Route {g_idx}. "
                f"Google ETA: {g_min:.2f} min, Our model ETA: {m_min:.2f} min."
            )
        else:
            st.warning(
                f"Recommendations differ: Google selects Route {g_idx}, "
                f"our model selects Route {m_idx}."
            )
        st.info(
            "Comparison notes: Our model uses hour, day-of-week/weekend signals, rush-hour flag, and "
            "borough context per segment, with **partial** length calibration vs Google's "
            "reported distance. By default Google uses static duration and overview "
            "polylines; enable traffic or step polylines via env vars (see README / "
            "google_directions)."
        )
