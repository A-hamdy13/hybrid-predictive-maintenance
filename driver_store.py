"""Driver registry and many-to-many vehicle assignments."""
import json

from config import DATA_DIR

DRIVERS_FILE = DATA_DIR / "drivers.json"
DEMO_DRIVER_CREDENTIALS = {
    "driver.alex": {"password": "driver123", "driver_id": "D001", "name": "Alex Driver", "phone": "+1-555-0101"},
    "driver.samira": {"password": "driver123", "driver_id": "D002", "name": "Samira Driver", "phone": "+1-555-0102"},
}


def _ensure() -> None:
    DRIVERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not DRIVERS_FILE.exists():
        DRIVERS_FILE.write_text(json.dumps({"drivers": [], "assignments": {}}, indent=2), encoding="utf-8")


def _load() -> dict:
    _ensure()
    with open(DRIVERS_FILE, encoding="utf-8") as f:
        data = json.load(f) or {}
    if not isinstance(data.get("drivers"), list):
        data["drivers"] = []
    if not isinstance(data.get("assignments"), dict):
        data["assignments"] = {}
    return data


def _save(data: dict) -> None:
    _ensure()
    with open(DRIVERS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _normalize_driver_id(driver_id: str) -> str:
    return (driver_id or "").strip()


def _normalize_plate(plate: str) -> str:
    return (plate or "").strip().upper().replace(" ", "")


def list_drivers() -> list[dict]:
    data = _load()
    return data["drivers"]


def add_driver(driver: dict) -> dict:
    data = _load()
    driver_id = _normalize_driver_id(driver.get("driver_id") or "")
    if not driver_id:
        raise ValueError("driver_id is required")
    if any((_normalize_driver_id(d.get("driver_id") or "") == driver_id) for d in data["drivers"]):
        raise ValueError("driver_id already exists")
    row = {"driver_id": driver_id, "name": (driver.get("name") or "").strip(), "phone": (driver.get("phone") or "").strip()}
    if not row["name"]:
        raise ValueError("name is required")
    data["drivers"].append(row)
    _save(data)
    return row


def remove_driver(driver_id: str) -> None:
    data = _load()
    key = _normalize_driver_id(driver_id)
    data["drivers"] = [d for d in data["drivers"] if _normalize_driver_id(d.get("driver_id") or "") != key]
    for plate, ids in list(data["assignments"].items()):
        data["assignments"][plate] = [i for i in (ids or []) if _normalize_driver_id(i) != key]
    _save(data)


def import_drivers(rows: list[dict]) -> dict:
    added = 0
    skipped = 0
    for r in rows or []:
        try:
            add_driver(r)
            added += 1
        except Exception:
            skipped += 1
    return {"added": added, "skipped": skipped}


def assign_vehicle(license_plate: str, driver_ids: list[str]) -> dict:
    data = _load()
    plate = _normalize_plate(license_plate)
    existing_ids = {_normalize_driver_id(d.get("driver_id") or "") for d in data["drivers"]}
    keep = []
    for d in driver_ids or []:
        k = _normalize_driver_id(d)
        if k and k in existing_ids and k not in keep:
            keep.append(k)
    data["assignments"][plate] = keep
    _save(data)
    return {"license_plate": plate, "driver_ids": keep}


def drivers_for_vehicle(license_plate: str) -> list[str]:
    data = _load()
    return data["assignments"].get(_normalize_plate(license_plate), []) or []


def vehicles_for_driver(driver_id: str) -> list[str]:
    key = _normalize_driver_id(driver_id)
    data = _load()
    out = []
    for plate, ids in (data["assignments"] or {}).items():
        if key in (ids or []):
            out.append(plate)
    return sorted(out)


def ensure_demo_drivers() -> None:
    data = _load()
    existing = {_normalize_driver_id(d.get("driver_id") or "") for d in data["drivers"]}
    changed = False
    for item in DEMO_DRIVER_CREDENTIALS.values():
        did = item["driver_id"]
        if did not in existing:
            data["drivers"].append({"driver_id": did, "name": item["name"], "phone": item["phone"]})
            changed = True
    if changed:
        _save(data)


def verify_driver_login(username: str, password: str) -> dict | None:
    u = (username or "").strip().lower()
    p = password or ""
    row = DEMO_DRIVER_CREDENTIALS.get(u)
    if not row or row["password"] != p:
        return None
    ensure_demo_drivers()
    return {"driver_id": row["driver_id"], "name": row["name"]}
