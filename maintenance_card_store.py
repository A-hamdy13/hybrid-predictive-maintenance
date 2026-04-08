"""Dynamic maintenance cards backed by extracted rules and service events."""
import json
from datetime import datetime, timezone
from pathlib import Path

from config import DATA_DIR
from rule_store import load_rules
from vehicle_logs import get_log as get_vehicle_log
from vehicle_registry import get_vehicle

CARDS_DIR = DATA_DIR / "maintenance_cards"

MAINTENANCE_CHECK_LIST: list[tuple[str, str]] = []


def _path(license_plate: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (license_plate or "").strip())
    if not safe:
        safe = "unknown"
    CARDS_DIR.mkdir(parents=True, exist_ok=True)
    return CARDS_DIR / f"{safe}.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _humanize(s: str) -> str:
    return (s or "").replace("_", " ").strip().title()


def _vehicle_parts(license_plate: str) -> list[dict]:
    rec = get_vehicle(license_plate)
    if not rec:
        return []
    rules = load_rules(rec.to_identity())
    if not rules:
        return []
    by_component: dict[str, dict] = {}
    for item in (rules.service_schedule.normal_service or []):
        if not item.found or not item.component:
            continue
        comp = item.component
        row = by_component.get(comp) or {
            "part_id": comp,
            "part_label": _humanize(comp),
            "recommended_service_interval_time": None,
            "recommended_service_interval_mileage": None,
        }
        if item.interval_time_value is not None and not row["recommended_service_interval_time"]:
            unit = item.interval_time_unit or ""
            row["recommended_service_interval_time"] = f"{item.interval_time_value:g} {unit}".strip()
        if item.interval_distance_value is not None and not row["recommended_service_interval_mileage"]:
            unit = item.interval_distance_unit or ""
            row["recommended_service_interval_mileage"] = f"{item.interval_distance_value:g} {unit}".strip()
        by_component[comp] = row
    return sorted(by_component.values(), key=lambda x: x["part_label"])


def _load_raw(license_plate: str) -> dict:
    p = _path(license_plate)
    if not p.exists():
        return {"license_plate": license_plate, "events": []}
    with open(p, encoding="utf-8") as f:
        data = json.load(f) or {}
    # migrate old format
    if "events" not in data:
        data = {"license_plate": license_plate, "events": []}
    if not isinstance(data.get("events"), list):
        data["events"] = []
    if not data.get("license_plate"):
        data["license_plate"] = license_plate
    return data


def _save_raw(license_plate: str, data: dict) -> None:
    p = _path(license_plate)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _latest_odometer_km(license_plate: str) -> float | None:
    log = get_vehicle_log(license_plate)
    for snap in reversed(log.operational_inputs or []):
        km = snap.get("current_mileage_km")
        if km is None:
            km = snap.get("odometer_reading")
        if km is not None:
            return float(km)
    return None


def _last_service_for_part(events: list[dict], part_id: str) -> tuple[str | None, float | None]:
    for ev in sorted(events, key=lambda e: e.get("produced_at") or "", reverse=True):
        status = (ev.get("items") or {}).get(part_id)
        if status in ("routine", "failure"):
            return (ev.get("produced_at") or "")[:10], ev.get("mileage_km")
    return None, None


def get_card(license_plate: str) -> dict:
    raw = _load_raw(license_plate)
    parts = _vehicle_parts(license_plate)
    events = raw.get("events") or []
    out_items = []
    for p in parts:
        last_date, last_mileage = _last_service_for_part(events, p["part_id"])
        out_items.append({
            **p,
            "last_serviced_date": last_date,
            "last_serviced_mileage": last_mileage,
            "routine_maintenance": False,
            "failure_maintenance": False,
        })
    return {
        "license_plate": license_plate,
        "current_mileage_km": _latest_odometer_km(license_plate),
        "items": out_items,
    }


def _event_card_id(license_plate: str, produced_at: str, existing: list[dict]) -> str:
    base = f"{license_plate}-{(produced_at or '')[:10]}"
    card_id = base
    i = 2
    existing_ids = {e.get("card_id") for e in existing}
    while card_id in existing_ids:
        card_id = f"{base}-{i}"
        i += 1
    return card_id


def save_card(license_plate: str, items: list[dict], mileage_km: float | None = None) -> dict:
    raw = _load_raw(license_plate)
    produced_at = _now_iso()
    parts = _vehicle_parts(license_plate)
    valid = {p["part_id"] for p in parts}
    statuses: dict[str, str] = {pid: "none" for pid in valid}
    for item in items or []:
        pid = item.get("part_id") or item.get("check_id")
        if pid not in valid:
            continue
        status = item.get("status")
        if status not in ("routine", "failure", "none"):
            routine = bool(item.get("routine_maintenance"))
            failure = bool(item.get("failure_maintenance"))
            status = "failure" if failure else ("routine" if routine else "none")
        statuses[pid] = status
    event = {
        "card_id": _event_card_id(license_plate, produced_at, raw.get("events") or []),
        "produced_at": produced_at,
        "mileage_km": mileage_km if mileage_km is not None else _latest_odometer_km(license_plate),
        "items": statuses,
    }
    raw["events"] = (raw.get("events") or [])[-499:] + [event]
    _save_raw(license_plate, raw)
    return get_card(license_plate)


def list_cards(license_plate: str) -> list[dict]:
    raw = _load_raw(license_plate)
    events = sorted(raw.get("events") or [], key=lambda e: e.get("produced_at") or "", reverse=True)
    return [{"card_id": e.get("card_id"), "produced_at": e.get("produced_at"), "mileage_km": e.get("mileage_km")} for e in events]


def get_card_snapshot(license_plate: str, card_id: str) -> dict:
    raw = _load_raw(license_plate)
    parts = _vehicle_parts(license_plate)
    event = next((e for e in (raw.get("events") or []) if e.get("card_id") == card_id), None)
    if not event:
        raise KeyError("card not found")
    part_by_id = {p["part_id"]: p for p in parts}
    items = []
    for pid, p in part_by_id.items():
        items.append({
            **p,
            "status": (event.get("items") or {}).get(pid, "none"),
        })
    return {
        "license_plate": license_plate,
        "card_id": event.get("card_id"),
        "produced_at": event.get("produced_at"),
        "mileage_km": event.get("mileage_km"),
        "items": items,
    }


def get_history_table(license_plate: str) -> dict:
    parts = _vehicle_parts(license_plate)
    events = list_cards(license_plate)
    raw = _load_raw(license_plate)
    event_by_id = {e.get("card_id"): e for e in (raw.get("events") or [])}
    columns = []
    for e in reversed(events):
        columns.append({
            "card_id": e.get("card_id"),
            "column_label": f"{(e.get('produced_at') or '')[:10]}-{int(e.get('mileage_km') or 0)}km",
        })
    rows = []
    for p in parts:
        values = {}
        for c in columns:
            ev = event_by_id.get(c["card_id"]) or {}
            values[c["card_id"]] = (ev.get("items") or {}).get(p["part_id"], "none")
        rows.append({
            **p,
            "part_id": p["part_id"],
            "values": values,
        })
    return {"license_plate": license_plate, "columns": columns, "rows": rows}


def get_simplified_table(license_plate: str, interval_km: int = 500) -> dict:
    parts = _vehicle_parts(license_plate)
    raw = _load_raw(license_plate)
    events = raw.get("events") or []
    latest_km = _latest_odometer_km(license_plate) or 0
    max_km = int(max([latest_km] + [float(e.get("mileage_km") or 0) for e in events]))
    columns = [{"bucket_km": km, "column_label": f"{km} km"} for km in range(0, max_km + interval_km, interval_km)]
    rows = []
    for p in parts:
        by_bucket: dict[int, str] = {c["bucket_km"]: "none" for c in columns}
        for e in events:
            status = (e.get("items") or {}).get(p["part_id"], "none")
            km = int(float(e.get("mileage_km") or 0))
            bucket = (km // interval_km) * interval_km
            by_bucket[bucket] = status
        rows.append({
            **p,
            "part_id": p["part_id"],
            "values": by_bucket,
        })
    return {"license_plate": license_plate, "interval_km": interval_km, "columns": columns, "rows": rows}


def save_simplified_cell(license_plate: str, part_id: str, bucket_km: int, status: str) -> dict:
    if status not in ("routine", "failure", "none"):
        raise ValueError("status must be one of routine|failure|none")
    return save_card(
        license_plate,
        items=[{"part_id": part_id, "status": status}],
        mileage_km=float(bucket_km),
    )


def _interval_mileage_km(text: str | None) -> float | None:
    if not text:
        return None
    parts = str(text).lower().split()
    if not parts:
        return None
    try:
        val = float(parts[0])
    except Exception:
        return None
    unit = parts[1] if len(parts) > 1 else "km"
    if unit.startswith("mile"):
        return val * 1.60934
    return val


def _interval_days(text: str | None) -> float | None:
    if not text:
        return None
    parts = str(text).lower().split()
    if not parts:
        return None
    try:
        val = float(parts[0])
    except Exception:
        return None
    unit = parts[1] if len(parts) > 1 else ""
    if unit.startswith("day"):
        return val
    if unit.startswith("week"):
        return val * 7
    if unit.startswith("month"):
        return val * 30.4375
    if unit.startswith("year"):
        return val * 365.25
    if unit.startswith("hour"):
        return val / 24
    return None


def due_parts_summary(license_plate: str) -> dict:
    card = get_card(license_plate)
    now_km = card.get("current_mileage_km")
    due_parts: list[dict] = []
    for item in card.get("items") or []:
        rec_km = _interval_mileage_km(item.get("recommended_service_interval_mileage"))
        rec_days = _interval_days(item.get("recommended_service_interval_time"))
        last_km = item.get("last_serviced_mileage")
        last_date = item.get("last_serviced_date")

        by_mileage = False
        by_time = False
        if rec_km is not None and last_km is not None and now_km is not None:
            by_mileage = float(now_km) - float(last_km) >= rec_km
        if rec_days is not None and last_date:
            try:
                d = datetime.fromisoformat(f"{last_date}T00:00:00+00:00")
                by_time = (datetime.now(timezone.utc) - d).total_seconds() / 86400 >= rec_days
            except Exception:
                by_time = False

        if by_mileage or by_time:
            due_parts.append({
                "part_id": item.get("part_id"),
                "part_label": item.get("part_label"),
                "due_by_mileage": by_mileage,
                "due_by_time": by_time,
            })
    return {"license_plate": license_plate, "due": len(due_parts) > 0, "due_parts": due_parts}


def maintenance_template(license_plate: str, interval_km: int = 500, n_cols: int = 16) -> dict:
    card = get_card(license_plate)
    current_km = int(float(card.get("current_mileage_km") or 0))
    # Start from next 5,000 block so printed sheets are cleaner and stable.
    start = ((current_km // 5000) + 1) * 5000
    columns = [start + i * interval_km for i in range(max(1, n_cols))]
    return {
        "license_plate": license_plate,
        "current_mileage_km": current_km,
        "interval_km": interval_km,
        "columns": columns,
        "parts": card.get("items") or [],
    }
