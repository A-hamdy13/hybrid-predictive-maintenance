"""Store and load validated rules as JSON."""
import json
from pathlib import Path

from config import RULES_DIR
from schemas import ExtractedManualRules, VehicleIdentity


def _vehicle_key(v: VehicleIdentity) -> str:
    parts = [v.make.strip().lower(), v.model.strip().lower(), str(v.year)]
    if v.trim_or_engine:
        parts.append(v.trim_or_engine.strip().lower().replace(" ", "_"))
    return "_".join(parts).replace("/", "-")


def path_for_vehicle(vehicle: VehicleIdentity) -> Path:
    return RULES_DIR / f"{_vehicle_key(vehicle)}.json"


def save_rules(rules: ExtractedManualRules) -> Path:
    path = path_for_vehicle(rules.vehicle)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rules.model_dump(), f, indent=2)
    return path


def load_rules(vehicle: VehicleIdentity) -> ExtractedManualRules | None:
    path = path_for_vehicle(vehicle)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return ExtractedManualRules.model_validate(data)


def list_stored_vehicles() -> list[dict]:
    """Return list of {make, model, year, trim_or_engine} for each stored rule file."""
    out = []
    for p in RULES_DIR.glob("*.json"):
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            v = data.get("vehicle", {})
            out.append({
                "make": v.get("make", ""),
                "model": v.get("model", ""),
                "year": v.get("year", 0),
                "trim_or_engine": v.get("trim_or_engine"),
            })
        except Exception:
            continue
    return out
