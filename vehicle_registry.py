"""Registry of vehicles by license plate."""
import json
from pathlib import Path

from config import DATA_DIR
from schemas import VehicleRecord

VEHICLES_FILE = DATA_DIR / "vehicles.json"


def _ensure_file():
    VEHICLES_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not VEHICLES_FILE.exists():
        VEHICLES_FILE.write_text("[]", encoding="utf-8")


def _normalize_plate(plate: str) -> str:
    return (plate or "").strip().upper().replace(" ", "")


def list_vehicles() -> list[VehicleRecord]:
    _ensure_file()
    with open(VEHICLES_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return [VehicleRecord.model_validate(item) for item in data]


def get_vehicle(license_plate: str) -> VehicleRecord | None:
    key = _normalize_plate(license_plate)
    for v in list_vehicles():
        if _normalize_plate(v.license_plate) == key:
            return v
    return None


def create_vehicle(record: VehicleRecord) -> VehicleRecord:
    key = _normalize_plate(record.license_plate)
    vehicles = list_vehicles()
    for v in vehicles:
        if _normalize_plate(v.license_plate) == key:
            raise ValueError(f"Vehicle with license plate {record.license_plate} already exists")
    vehicles.append(record)
    _ensure_file()
    with open(VEHICLES_FILE, "w", encoding="utf-8") as f:
        json.dump([v.model_dump() for v in vehicles], f, indent=2)
    return record


def update_vehicle(license_plate: str, updates: dict) -> VehicleRecord:
    key = _normalize_plate(license_plate)
    vehicles = list_vehicles()
    for i, v in enumerate(vehicles):
        if _normalize_plate(v.license_plate) == key:
            updated = v.model_copy(update=updates)
            vehicles[i] = updated
            with open(VEHICLES_FILE, "w", encoding="utf-8") as f:
                json.dump([x.model_dump() for x in vehicles], f, indent=2)
            return updated
    raise ValueError(f"Vehicle with license plate {license_plate} not found")


def delete_vehicle(license_plate: str) -> None:
    key = _normalize_plate(license_plate)
    vehicles = [v for v in list_vehicles() if _normalize_plate(v.license_plate) != key]
    with open(VEHICLES_FILE, "w", encoding="utf-8") as f:
        json.dump([v.model_dump() for v in vehicles], f, indent=2)
