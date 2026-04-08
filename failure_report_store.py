"""Per-vehicle failure reports: driver-submitted failure type with timestamp."""

import json
from datetime import datetime, timezone
from pathlib import Path

from config import DATA_DIR

FAILURES_DIR = DATA_DIR / "failure_reports"


def _path(license_plate: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (license_plate or "").strip())
    if not safe:
        safe = "unknown"
    FAILURES_DIR.mkdir(parents=True, exist_ok=True)
    return FAILURES_DIR / f"{safe}.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_reports(license_plate: str) -> list[dict]:
    p = _path(license_plate)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        data = json.load(f) or {}
    reports = data.get("reports") or []
    reports = [r for r in reports if isinstance(r, dict) and r.get("submitted_at") and r.get("failure_type")]
    reports.sort(key=lambda r: r.get("submitted_at") or "", reverse=True)
    return reports


def add_report(license_plate: str, failure_type: str) -> dict:
    failure_type = (failure_type or "").strip()
    if not failure_type:
        raise ValueError("failure_type is required")
    p = _path(license_plate)
    data = {"license_plate": license_plate, "reports": []}
    if p.exists():
        with open(p, encoding="utf-8") as f:
            data = json.load(f) or data
    reports = data.get("reports") or []
    reports.append({"submitted_at": _now_iso(), "failure_type": failure_type})
    data["license_plate"] = license_plate
    data["reports"] = reports[-2000:]
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return {"license_plate": license_plate, "reports": list_reports(license_plate)}

