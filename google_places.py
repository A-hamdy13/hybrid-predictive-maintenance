"""Google Places helper for nearby mechanics."""
from __future__ import annotations

from typing import Optional

import requests

from config import GOOGLE_PLACES_API_KEY


def _nearby_search(lat: float, lon: float, keyword: str) -> Optional[dict]:
    """Return first nearby automotive place for a keyword using Google Places Nearby Search."""
    if not GOOGLE_PLACES_API_KEY:
        return None

    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lon}",
        "rankby": "distance",
        # Use a more specific automotive-focused keyword to avoid electronics / non-car shops.
        "keyword": keyword,
        "key": GOOGLE_PLACES_API_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    results = (data or {}).get("results") or []
    if not results:
        return None

    best = results[0]
    name = best.get("name")
    vicinity = best.get("vicinity") or ""
    if not name:
        return None

    # Use a query (name + vicinity) instead of query_place_id, since this
    # more reliably opens the specific place for many users.
    query = f"{name} {vicinity}".strip()
    maps_url = (
        "https://www.google.com/maps/search/?api=1&query="
        + requests.utils.quote(query)
    )

    return {"name": name, "maps_url": maps_url}


def nearest_general_mechanic(lat: float, lon: float) -> Optional[dict]:
    # Bias toward actual car workshops / mechanics.
    return _nearby_search(lat, lon, "auto mechanic car repair workshop")


def nearest_dealer_service(lat: float, lon: float, make: str) -> Optional[dict]:
    if not make:
        return None
    # Prefer authorized/official service for the given make.
    return _nearby_search(lat, lon, f"{make} authorized car service center")


def nearest_brake_tire(lat: float, lon: float) -> Optional[dict]:
    # Focus on automotive brake / tire shops.
    return _nearby_search(lat, lon, "car brake service tire shop")

