"""
Google place resolution helpers for free-form origin/destination inputs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode

import requests


def _load_dotenv() -> None:
    """Load project root `.env` so `os.getenv` sees GOOGLE_MAPS_API_KEY for Streamlit."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.is_file():
        load_dotenv(env_path)


_load_dotenv()

GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"


def get_maps_api_key() -> str | None:
    """Read API key from Streamlit secrets first, then environment (including `.env`)."""
    try:
        import streamlit as st

        if hasattr(st, "secrets"):
            value = st.secrets.get("GOOGLE_MAPS_API_KEY")
            if value:
                return str(value).strip()
    except Exception:
        pass

    env_value = os.getenv("GOOGLE_MAPS_API_KEY")
    return env_value.strip() if env_value else None


@dataclass
class PlaceResolution:
    query: str
    formatted_address: str
    lat: float
    lon: float
    place_id: str


def resolve_place(query: str, api_key: str | None = None) -> tuple[PlaceResolution | None, str | None]:
    """
    Resolve a user-entered place string to a single lat/lon via Geocoding API.
    Returns (resolution, error_message).
    """
    key = api_key or get_maps_api_key()
    if not key:
        return None, "Missing GOOGLE_MAPS_API_KEY (set env var or Streamlit secrets)."

    params = {"address": query, "key": key}
    url = f"{GEOCODE_URL}?{urlencode(params)}"
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        return None, f"Geocoding request failed: {exc}"

    status = payload.get("status")
    if status != "OK":
        if status == "ZERO_RESULTS":
            return None, f"No location found for '{query}'."
        return None, f"Geocoding API error: {payload.get('error_message') or status}"

    result = payload["results"][0]
    loc = result["geometry"]["location"]
    place = PlaceResolution(
        query=query,
        formatted_address=result.get("formatted_address", query),
        lat=float(loc["lat"]),
        lon=float(loc["lng"]),
        place_id=result.get("place_id", ""),
    )
    return place, None
