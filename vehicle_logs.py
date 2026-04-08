"""Per-vehicle logs: recommendations, operational inputs, alerts."""
import json
from datetime import datetime, timezone
from pathlib import Path

from config import DATA_DIR
from schemas import VehicleLog, LogRecommendationEntry, LogAlertEntry

LOGS_DIR = DATA_DIR / "vehicle_logs"


def _path(license_plate: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (license_plate or "").strip())
    if not safe:
        safe = "unknown"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return LOGS_DIR / f"{safe}.json"


def _load_log(license_plate: str) -> VehicleLog:
    p = _path(license_plate)
    if not p.exists():
        return VehicleLog(license_plate=license_plate)
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return VehicleLog.model_validate(data)


def _save_log(log: VehicleLog) -> None:
    p = _path(log.license_plate)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(log.model_dump(), f, indent=2)


def get_log(license_plate: str) -> VehicleLog:
    return _load_log(license_plate)


def append_recommendation(
    license_plate: str,
    decision: dict,
    explanation: str,
    inputs_snapshot: dict,
    use_severe_service: bool = False,
) -> VehicleLog:
    log = _load_log(license_plate)
    log.recommendations.append(
        LogRecommendationEntry(
            at=datetime.now(timezone.utc).isoformat(),
            decision=decision,
            explanation=explanation,
            inputs_snapshot=inputs_snapshot,
            use_severe_service=use_severe_service,
        )
    )
    log.operational_inputs.append(inputs_snapshot)
    _save_log(log)
    return log


def append_alert(license_plate: str, kind: str, message: str, component: str | None = None) -> VehicleLog:
    log = _load_log(license_plate)
    log.alerts.append(
        LogAlertEntry(
            at=datetime.now(timezone.utc).isoformat(),
            kind=kind,
            message=message,
            component=component,
        )
    )
    _save_log(log)
    return log
