"""Affiliate repair centers selected by admin."""
import json
import math

from config import DATA_DIR

AFFILIATES_FILE = DATA_DIR / "affiliate_centers.json"


def _ensure() -> None:
    AFFILIATES_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not AFFILIATES_FILE.exists():
        AFFILIATES_FILE.write_text("[]", encoding="utf-8")


def _load() -> list[dict]:
    _ensure()
    with open(AFFILIATES_FILE, encoding="utf-8") as f:
        data = json.load(f) or []
    if not isinstance(data, list):
        return []
    return data


def _save(rows: list[dict]) -> None:
    _ensure()
    with open(AFFILIATES_FILE, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def list_affiliates() -> list[dict]:
    return _load()


def add_affiliate(center: dict) -> dict:
    rows = _load()
    place_id = (center.get("place_id") or "").strip()
    name = (center.get("name") or "").strip()
    if not name:
        raise ValueError("name is required")
    if place_id and any((r.get("place_id") or "") == place_id for r in rows):
        return next(r for r in rows if (r.get("place_id") or "") == place_id)
    row = {
        "id": place_id or str(len(rows) + 1),
        "place_id": place_id or None,
        "name": name,
        "address": (center.get("address") or "").strip(),
        "lat": float(center.get("lat")),
        "lon": float(center.get("lon")),
        "maps_url": (center.get("maps_url") or "").strip(),
    }
    rows.append(row)
    _save(rows)
    return row


def remove_affiliate(center_id: str) -> None:
    key = (center_id or "").strip()
    rows = [r for r in _load() if (str(r.get("id")) != key and str(r.get("place_id") or "") != key)]
    _save(rows)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    d1 = math.radians(lat2 - lat1)
    d2 = math.radians(lon2 - lon1)
    a = math.sin(d1 / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(d2 / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def nearest_affiliates(lat: float, lon: float, limit: int = 3) -> list[dict]:
    out = []
    for c in _load():
        try:
            d = _haversine_km(lat, lon, float(c.get("lat")), float(c.get("lon")))
        except Exception:
            continue
        row = dict(c)
        row["distance_km"] = d
        out.append(row)
    out.sort(key=lambda x: x.get("distance_km") or 1e9)
    return out[:limit]
