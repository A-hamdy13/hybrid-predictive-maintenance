"""FastAPI app: manual retrieval, extraction, rules, decision engine, explanation."""
import json
import logging
import sys
import csv
import io
import re
import secrets
from urllib.parse import unquote
from pathlib import Path
from datetime import datetime, timezone
import requests

from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Response

log = logging.getLogger(__name__)

# Ensure manual-finder and agent logs are visible (uvicorn often sets root handler to WARNING)
_manual_log_names = (
    "main",
    "manual_agent",
    "manual_finder",
    "openai_web_search",
    "manual_downloader",
    "manual_extractor",
)
_handler = logging.StreamHandler(sys.stderr)
_handler.setLevel(logging.INFO)
_handler.setFormatter(logging.Formatter("%(message)s"))
for _name in _manual_log_names:
    _logger = logging.getLogger(_name)
    _logger.setLevel(logging.INFO)
    _logger.addHandler(_handler)
    _logger.propagate = True  # still propagate so uvicorn can show them too if it wants

from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from fastapi.responses import HTMLResponse
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from config import OPENAI_API_KEY, DATA_DIR, GOOGLE_PLACES_API_KEY, SCRAPED_MANUALS_DIR
from decision_engine import run_decision_engine
from explanation_generator import generate_explanation
from manual_downloader import (
    get_manual_text_from_url,
    fetch_manual_text_and_type,
    extract_pdf_links_from_page,
)
from manual_extractor import extract_rules
from manual_finder import find_manual_candidates, resolve_fetchable_urls
from manual_agent import run_manual_finder_agent, is_agent_available
from google_places import (
    nearest_brake_tire,
    nearest_dealer_service,
    nearest_general_mechanic,
)
from ml_predictor import get_ml_prediction
from rule_store import load_rules, save_rules, list_stored_vehicles, path_for_vehicle
from vehicle_registry import list_vehicles, get_vehicle, create_vehicle, update_vehicle
from vehicle_logs import get_log as get_vehicle_log, append_recommendation as append_vehicle_recommendation
from maintenance_card_store import (
    get_card as get_maintenance_card,
    save_card as save_maintenance_card,
    list_cards as list_maintenance_cards,
    get_card_snapshot as get_maintenance_card_snapshot,
    get_history_table as get_maintenance_history_table,
    get_simplified_table as get_simplified_maintenance_table,
    save_simplified_cell as save_simplified_maintenance_cell,
    due_parts_summary as get_due_parts_summary,
    get_simplified_table as get_simplified_table_for_csv,
    maintenance_template as get_maintenance_template,
)
from failure_report_store import add_report as add_failure_report, list_reports as list_failure_reports
from driver_store import (
    list_drivers,
    add_driver,
    remove_driver,
    import_drivers,
    assign_vehicle,
    drivers_for_vehicle,
    vehicles_for_driver,
    verify_driver_login,
    ensure_demo_drivers,
)
from affiliate_store import (
    list_affiliates,
    add_affiliate,
    remove_affiliate,
    nearest_affiliates,
)
from rule_validator import validate_extraction, verifier_pass
from schemas import (
    VehicleIdentity,
    VehicleRecord,
    VehicleLog,
    OperationalInputs,
    RecommendationRequest,
    RecommendationResponse,
    DecisionOutput,
    MLOutput,
    ExtractedManualRules,
    ServiceSchedule,
    NearbyMechanic,
)

app = FastAPI(title="Hybrid Predictive Maintenance (Manual + ML)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
DRIVER_SESSION_COOKIE = "driver_session"
_DRIVER_SESSIONS: dict[str, str] = {}


# --- Demo vehicles ---
@app.get("/api/vehicles/demo")
def api_vehicles_demo():
    path = DATA_DIR / "demo_vehicles.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --- Stored rules ---
@app.get("/api/vehicles/rules")
def api_vehicles_rules():
    return list_stored_vehicles()


# --- Agent availability (browser + LLM manual finder) ---
@app.get("/api/agent/available")
def api_agent_available():
    return {"available": is_agent_available()}


# --- Find manual URLs ---
@app.post("/api/manuals/find")
def api_manuals_find(vehicle: VehicleIdentity):
    candidates = find_manual_candidates(vehicle, api_key=OPENAI_API_KEY or None)
    return {"candidates": [c.model_dump() for c in candidates]}


class ExtractRequest(BaseModel):
    vehicle: VehicleIdentity
    manual_url: str | None = None
    manual_text: str | None = None


def _do_extract_and_validate(
    text: str,
    vehicle: VehicleIdentity,
    manual_url: str | None,
) -> tuple[ExtractedManualRules, dict]:
    """Extract rules, validate; do not save. Returns (validated_rules, response_dict)."""
    raw = extract_rules(text, vehicle)
    if manual_url:
        raw.source_url = manual_url
    validated, errors = validate_extraction(raw)
    warnings = verifier_pass(validated)
    if manual_url:
        validated.source_url = manual_url
        if not getattr(validated, "source_urls", None):
            validated.source_urls = [manual_url]
        # Set per-item source_url so each rule shows where it came from
        new_normal = [item.model_copy(update={"source_url": manual_url}) for item in validated.service_schedule.normal_service]
        new_severe = [item.model_copy(update={"source_url": manual_url}) for item in validated.service_schedule.severe_service]
        validated = validated.model_copy(update={"service_schedule": ServiceSchedule(normal_service=new_normal, severe_service=new_severe)})
    # Count only items that were actually found in the text
    n = sum(1 for item in validated.service_schedule.normal_service if item.found)
    s = sum(1 for item in validated.service_schedule.severe_service if item.found)
    return validated, {
        "saved": False,
        "vehicle": validated.vehicle.model_dump(),
        "normal_service_count": n,
        "severe_service_count": s,
        "validation_errors": errors,
        "verifier_warnings": warnings,
    }


def _do_extract_and_store(
    text: str,
    vehicle: VehicleIdentity,
    manual_url: str | None,
) -> dict:
    """Extract rules, validate, store. Returns response dict."""
    validated, resp = _do_extract_and_validate(text, vehicle, manual_url)
    save_rules(validated)
    resp["saved"] = True
    return resp


def _vehicle_matches(requested: VehicleIdentity, extracted: VehicleIdentity) -> bool:
    """True if the extracted manual is for the same make/model/year (ignore trim)."""
    return (
        (requested.make or "").strip().lower() == (extracted.make or "").strip().lower()
        and (requested.model or "").strip().lower() == (extracted.model or "").strip().lower()
        and requested.year == extracted.year
    )


def _merge_rules(
    base: ExtractedManualRules | None,
    new: ExtractedManualRules,
) -> ExtractedManualRules:
    """Merge rules from multiple sources.

    - Keeps the first occurrence of a given (component, conditions, action) tuple.
    - Later sources only fill in components/conditions that are missing.
    - source_urls collects every URL that contributed at least one rule.
    """
    if base is None:
        urls = [new.source_url] if new.source_url else []
        return ExtractedManualRules(
            vehicle=new.vehicle,
            service_schedule=new.service_schedule.model_copy(deep=True),
            source_url=new.source_url,
            source_label=new.source_label,
            source_urls=urls,
        )

    def _key(item):
        return (item.component, item.conditions or "", item.action)

    base_normal = list(base.service_schedule.normal_service)
    base_severe = list(base.service_schedule.severe_service)

    seen_normal = {_key(it) for it in base_normal if it.found}
    seen_severe = {_key(it) for it in base_severe if it.found}

    added_any = False
    for item in new.service_schedule.normal_service:
        if not item.found:
            continue
        k = _key(item)
        if k not in seen_normal:
            base_normal.append(item)
            seen_normal.add(k)
            added_any = True

    for item in new.service_schedule.severe_service:
        if not item.found:
            continue
        k = _key(item)
        if k not in seen_severe:
            base_severe.append(item)
            seen_severe.add(k)
            added_any = True

    merged_schedule = ServiceSchedule(
        normal_service=base_normal,
        severe_service=base_severe,
    )

    base_urls = getattr(base, "source_urls", None) or ([base.source_url] if base.source_url else [])
    new_url = new.source_url if added_any and new.source_url else None
    all_urls = list(dict.fromkeys(base_urls + ([new_url] if new_url else [])))

    return ExtractedManualRules(
        vehicle=base.vehicle,
        service_schedule=merged_schedule,
        source_url=base.source_url or new.source_url,
        source_label=base.source_label or new.source_label,
        source_urls=all_urls,
    )


# --- Extract from URL and store ---
@app.post("/api/manuals/extract")
def api_manuals_extract(body: ExtractRequest):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not set")
    vehicle = body.vehicle
    manual_url = body.manual_url
    manual_text = body.manual_text
    if manual_text:
        text = manual_text
    elif manual_url:
        try:
            text = get_manual_text_from_url(manual_url)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")
    else:
        raise HTTPException(status_code=400, detail="Provide manual_url or manual_text")

    if len(text.strip()) < 100:
        raise HTTPException(status_code=400, detail="Extracted text too short; try a direct PDF link.")

    validated, resp = _do_extract_and_validate(text, vehicle, manual_url)
    if resp["normal_service_count"] + resp["severe_service_count"] == 0:
        raise HTTPException(
            status_code=400,
            detail="No maintenance intervals found in this page. Try a direct PDF link to the maintenance schedule or owner manual.",
        )
    save_rules(validated)
    resp["saved"] = True
    return resp


class FindAndExtractRequest(BaseModel):
    vehicle: VehicleIdentity
    manual_url: str | None = None


# --- Find manual (agent) then extract, validate, store. Default: agent finds; optional manual_url overrides. ---
@app.post("/api/manuals/find-and-extract")
def api_manuals_find_and_extract(body: FindAndExtractRequest):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not set")
    vehicle = body.vehicle
    manual_url = body.manual_url

    if manual_url and manual_url.strip():
        # User provided URL: fetch and extract
        try:
            text = get_manual_text_from_url(manual_url.strip())
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")
        if len(text.strip()) < 100:
            raise HTTPException(status_code=400, detail="Extracted text too short; try a direct PDF link.")
        return _do_extract_and_store(text, vehicle, manual_url.strip())

    # OpenAI Responses API + web_search tool only (no Bing, DuckDuckGo, agent)
    log.info(
        "[find-and-extract] vehicle=%s %s %s, manual_url=%s (OpenAI web search)",
        vehicle.make,
        vehicle.model,
        vehicle.year,
        bool(manual_url),
    )
    try:
        urls_to_try = list(resolve_fetchable_urls(vehicle, api_key=OPENAI_API_KEY))
    except RuntimeError as e:
        log.warning("[find-and-extract] Web search failed: %s", e)
        raise HTTPException(status_code=503, detail=str(e)) from e
    log.info("[find-and-extract] resolve_fetchable_urls returned %s URLs", len(urls_to_try))

    if not urls_to_try:
        log.warning("[find-and-extract] No manual candidates found")
        raise HTTPException(
            status_code=503,
            detail=(
                "Web search returned no manual URLs. Your model may not support the web_search tool, or the response format changed. "
                "Check the server terminal for [openai_web_search] logs. You can paste a manual PDF or page URL above to skip search."
            ),
        )

    seen_urls: set[str] = set()
    last_error: str | None = None
    merged: ExtractedManualRules | None = None
    save_scraped_path = SCRAPED_MANUALS_DIR / (path_for_vehicle(vehicle).name.replace(".json", ".txt"))
    i = 0

    while i < len(urls_to_try):
        url = urls_to_try[i]
        i += 1
        if "google.com/search" in url or url in seen_urls:
            log.debug("[find-and-extract] Skip url (search or seen): %s", url[:60])
            continue
        seen_urls.add(url)
        log.info("[find-and-extract] Trying URL [%s]: %s", i - 1, url[:85])
        try:
            text, content_type = fetch_manual_text_and_type(url)
            log.info("[find-and-extract] Fetched len=%s content_type=%s", len(text.strip()), content_type)
            if len(text.strip()) < 100:
                last_error = "Extracted text too short"
                log.warning("[find-and-extract] Text too short")
                continue
            validated, resp = _do_extract_and_validate(text, vehicle, url)
            n = resp["normal_service_count"] + resp["severe_service_count"]
            log.info(
                "[find-and-extract] Extracted rules: normal=%s severe=%s (manual is for %s %s %s)",
                resp["normal_service_count"],
                resp["severe_service_count"],
                validated.vehicle.make,
                validated.vehicle.model,
                validated.vehicle.year,
            )
            if n > 0:
                if not _vehicle_matches(vehicle, validated.vehicle):
                    log.warning(
                        "[find-and-extract] Skipping URL: manual is for %s %s %s, not %s %s %s",
                        validated.vehicle.make,
                        validated.vehicle.model,
                        validated.vehicle.year,
                        vehicle.make,
                        vehicle.model,
                        vehicle.year,
                    )
                    continue
                merged = _merge_rules(merged, validated)
                if merged and save_scraped_path and not save_scraped_path.exists():
                    save_scraped_path.parent.mkdir(parents=True, exist_ok=True)
                    save_scraped_path.write_text(text, encoding="utf-8")
                    log.info("[find-and-extract] Saved scraped text for NotebookLM: %s", save_scraped_path)
            if n == 0 and content_type == "html":
                pdf_links = extract_pdf_links_from_page(url)
                log.info("[find-and-extract] Hub page: %s PDF links found", len(pdf_links))
                for link in reversed(pdf_links):
                    if link not in seen_urls:
                        urls_to_try.insert(i, link)
        except Exception as e:
            last_error = str(e)
            log.warning("[find-and-extract] Failed for %s: %s", url[:60], e)
            continue

    if not merged:
        raise HTTPException(
            status_code=400,
            detail=(
                "No maintenance intervals were found in any fetched source. "
                "Many manufacturer pages are hub pages without schedule text. "
                "Try pasting a direct PDF link to the maintenance schedule or owner manual."
            ),
        )

    # Save under the requested vehicle (not the first extracted manual's vehicle)
    merged = merged.model_copy(update={"vehicle": vehicle})

    # Save merged rules and return a combined summary
    save_rules(merged)
    total_normal = sum(1 for item in merged.service_schedule.normal_service if item.found)
    total_severe = sum(1 for item in merged.service_schedule.severe_service if item.found)
    warnings = verifier_pass(merged)
    return {
        "saved": True,
        "vehicle": merged.vehicle.model_dump(),
        "normal_service_count": total_normal,
        "severe_service_count": total_severe,
        "validation_errors": [],
        "verifier_warnings": warnings,
    }


# --- Get rules for vehicle ---
@app.get("/api/rules")
def api_rules(make: str, model: str, year: int, trim_or_engine: str | None = None):
    v = VehicleIdentity(make=make, model=model, year=year, trim_or_engine=trim_or_engine)
    rules = load_rules(v)
    if not rules:
        raise HTTPException(status_code=404, detail="No rules found for this vehicle")
    return rules.model_dump()


# --- Save rules (create or update from UI edit / add manually) ---
@app.post("/api/rules/save")
def api_rules_save(body: dict):
    """Accept full rules JSON (vehicle + service_schedule + optional source_url/source_urls). Validate and save."""
    try:
        rules = ExtractedManualRules.model_validate(body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid rules format: {e}") from e
    validated, errors = validate_extraction(rules)
    warnings = verifier_pass(validated)
    save_rules(validated)
    n = sum(1 for item in validated.service_schedule.normal_service if item.found)
    s = sum(1 for item in validated.service_schedule.severe_service if item.found)
    return {
        "saved": True,
        "vehicle": validated.vehicle.model_dump(),
        "normal_service_count": n,
        "severe_service_count": s,
        "validation_errors": errors,
        "verifier_warnings": warnings,
    }


# --- Import rules from NotebookLM (same schema as extraction: vehicle + service_schedule) ---
@app.post("/api/rules/import-from-notebooklm")
def api_rules_import_from_notebooklm(body: dict):
    """
    Accept JSON in the same shape as ExtractedManualRules (vehicle + service_schedule with
    normal_service and severe_service). Use this after uploading scraped manual text to
    NotebookLM and having it produce structured maintenance intervals.
    """
    try:
        rules = ExtractedManualRules.model_validate(body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid rules format: {e}") from e
    validated, errors = validate_extraction(rules)
    warnings = verifier_pass(validated)
    save_rules(validated)
    n = sum(1 for item in validated.service_schedule.normal_service if item.found)
    s = sum(1 for item in validated.service_schedule.severe_service if item.found)
    return {
        "saved": True,
        "vehicle": validated.vehicle.model_dump(),
        "normal_service_count": n,
        "severe_service_count": s,
        "validation_errors": errors,
        "verifier_warnings": warnings,
    }


# --- Vehicles (by license plate) ---
@app.get("/api/vehicles")
def api_vehicles_list():
    out = []
    for v in list_vehicles():
        item = v.model_dump()
        try:
            due = get_due_parts_summary(v.license_plate)
            item["has_due"] = bool(due.get("due"))
        except Exception:
            item["has_due"] = False
        out.append(item)
    return out


@app.post("/api/vehicles")
def api_vehicles_create(body: dict):
    try:
        record = VehicleRecord.model_validate(body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    try:
        create_vehicle(record)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return record.model_dump()


@app.get("/api/vehicles/{license_plate}")
def api_vehicle_get(license_plate: str):
    v = get_vehicle(license_plate)
    if not v:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return v.model_dump()


@app.put("/api/vehicles/{license_plate}")
def api_vehicle_update(license_plate: str, body: dict):
    v = get_vehicle(license_plate)
    if not v:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    allowed = {k: v for k, v in body.items() if k in ("make", "model", "year", "trim_or_engine", "notes", "severe_service_flags")}
    try:
        updated = update_vehicle(license_plate, allowed)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return updated.model_dump()


@app.get("/api/vehicles/{license_plate}/logs")
def api_vehicle_logs(license_plate: str):
    v = get_vehicle(license_plate)
    if not v:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    log = get_vehicle_log(license_plate)
    return log.model_dump()


@app.get("/api/vehicles/{license_plate}/maintenance-card")
def api_maintenance_card_get(license_plate: str):
    v = get_vehicle(license_plate)
    if not v:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return get_maintenance_card(license_plate)


@app.put("/api/vehicles/{license_plate}/maintenance-card")
def api_maintenance_card_put(license_plate: str, body: dict):
    v = get_vehicle(license_plate)
    if not v:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    items = body.get("items")
    if not isinstance(items, list):
        raise HTTPException(status_code=400, detail="body must include 'items' array")
    mileage_km = body.get("mileage_km")
    return save_maintenance_card(license_plate, items, mileage_km=mileage_km)


@app.get("/api/vehicles/{license_plate}/maintenance-cards")
def api_maintenance_cards_list(license_plate: str):
    v = get_vehicle(license_plate)
    if not v:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return {"license_plate": license_plate, "cards": list_maintenance_cards(license_plate)}


@app.get("/api/vehicles/{license_plate}/maintenance-history-table")
def api_maintenance_history_table(license_plate: str):
    v = get_vehicle(license_plate)
    if not v:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return get_maintenance_history_table(license_plate)


@app.get("/api/vehicles/{license_plate}/maintenance-simplified-table")
def api_maintenance_simplified_table(license_plate: str, interval_km: int = 500):
    v = get_vehicle(license_plate)
    if not v:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return get_simplified_maintenance_table(license_plate, interval_km=interval_km)


@app.get("/api/vehicles/{license_plate}/due-parts")
def api_due_parts(license_plate: str):
    v = get_vehicle(license_plate)
    if not v:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return get_due_parts_summary(license_plate)


@app.post("/api/vehicles/{license_plate}/maintenance-simplified-cell")
def api_maintenance_simplified_cell_put(license_plate: str, body: dict):
    v = get_vehicle(license_plate)
    if not v:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    part_id = (body.get("part_id") or "").strip()
    status = (body.get("status") or "").strip().lower()
    try:
        bucket_km = int(body.get("bucket_km"))
    except Exception:
        raise HTTPException(status_code=400, detail="bucket_km must be an integer")
    if not part_id:
        raise HTTPException(status_code=400, detail="part_id is required")
    try:
        return save_simplified_maintenance_cell(license_plate, part_id, bucket_km, status)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/drivers")
def api_drivers_list():
    ensure_demo_drivers()
    return {"drivers": list_drivers()}


@app.post("/api/auth/login")
def api_auth_login(body: dict, response: Response):
    username = (body.get("username") or "").strip()
    password = body.get("password") or ""
    if username.lower() == "admin" and password == "admin123":
        return {"ok": True, "role": "admin", "redirect_to": "/admin"}
    driver = verify_driver_login(username, password)
    if driver:
        token = secrets.token_urlsafe(32)
        _DRIVER_SESSIONS[token] = driver["driver_id"]
        response.set_cookie(
            key=DRIVER_SESSION_COOKIE,
            value=token,
            httponly=True,
            samesite="lax",
            secure=True,
            max_age=60 * 60 * 12,
        )
        return {
            "ok": True,
            "role": "driver",
            "driver_id": driver["driver_id"],
            "redirect_to": "/driver",
        }
    raise HTTPException(status_code=401, detail="Invalid credentials")


@app.post("/api/auth/logout")
def api_auth_logout(request: Request, response: Response):
    token = request.cookies.get(DRIVER_SESSION_COOKIE)
    if token:
        _DRIVER_SESSIONS.pop(token, None)
    response.delete_cookie(DRIVER_SESSION_COOKIE)
    return {"ok": True}


@app.get("/api/auth/me")
def api_auth_me(request: Request):
    token = request.cookies.get(DRIVER_SESSION_COOKIE)
    driver_id = _DRIVER_SESSIONS.get(token or "")
    if not driver_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {"ok": True, "role": "driver", "driver_id": driver_id}


@app.post("/api/drivers")
def api_drivers_add(body: dict):
    try:
        return add_driver(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/drivers/{driver_id}")
def api_drivers_remove(driver_id: str):
    remove_driver(driver_id)
    return {"removed": True}


@app.post("/api/drivers/import")
def api_drivers_import(body: dict):
    rows = body.get("drivers")
    if not isinstance(rows, list):
        raise HTTPException(status_code=400, detail="drivers must be an array")
    return import_drivers(rows)


@app.get("/api/vehicles/{license_plate}/drivers")
def api_vehicle_drivers(license_plate: str):
    v = get_vehicle(license_plate)
    if not v:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return {"license_plate": license_plate, "driver_ids": drivers_for_vehicle(license_plate)}


@app.post("/api/vehicles/{license_plate}/drivers")
def api_vehicle_drivers_assign(license_plate: str, body: dict):
    v = get_vehicle(license_plate)
    if not v:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    ids = body.get("driver_ids")
    if not isinstance(ids, list):
        raise HTTPException(status_code=400, detail="driver_ids must be an array")
    return assign_vehicle(license_plate, ids)


@app.get("/api/drivers/{driver_id}/vehicles")
def api_driver_vehicles(driver_id: str, request: Request):
    token = request.cookies.get(DRIVER_SESSION_COOKIE)
    session_driver_id = _DRIVER_SESSIONS.get(token or "")
    if session_driver_id and session_driver_id != driver_id:
        raise HTTPException(status_code=403, detail="Forbidden")
    return {"driver_id": driver_id, "license_plates": vehicles_for_driver(driver_id)}


@app.get("/api/affiliates")
def api_affiliates_list():
    return {"centers": list_affiliates()}


@app.post("/api/affiliates")
def api_affiliates_add(body: dict):
    try:
        return add_affiliate(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/affiliates/{center_id}")
def api_affiliates_remove(center_id: str):
    remove_affiliate(center_id)
    return {"removed": True}


@app.get("/api/affiliates/search")
def api_affiliates_search(query: str):
    if not GOOGLE_PLACES_API_KEY:
        raise HTTPException(status_code=503, detail="GOOGLE_PLACES_API_KEY not set")
    # Absolute search by query text only; no user/location proximity bias.
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": query, "key": GOOGLE_PLACES_API_KEY}
    log.info("[affiliates.search] query=%r", query)
    try:
        resp = requests.get(url, params=params, timeout=8)
        log.info("[affiliates.search] http_status=%s", resp.status_code)
        resp.raise_for_status()
        data = resp.json()
        log.info(
            "[affiliates.search] google_status=%s results=%s error_message=%r",
            data.get("status"),
            len(data.get("results") or []),
            data.get("error_message"),
        )
        if data.get("status") not in ("OK", "ZERO_RESULTS"):
            log.warning(
                "[affiliates.search] non_ok_status=%s payload=%r",
                data.get("status"),
                str(data)[:1200],
            )
    except Exception as e:
        body = ""
        if "resp" in locals():
            try:
                body = resp.text[:1200]
            except Exception:
                body = ""
        log.exception("[affiliates.search] request_failed error=%r body=%r", e, body)
        raise HTTPException(status_code=500, detail=f"Google Places search failed: {e}")
    out = []
    for r in (data.get("results") or [])[:10]:
        loc = ((r.get("geometry") or {}).get("location") or {})
        name = r.get("name")
        if not name or loc.get("lat") is None or loc.get("lng") is None:
            continue
        address = r.get("formatted_address") or r.get("vicinity") or ""
        q = requests.utils.quote(f"{name} {address}".strip())
        out.append({
            "place_id": r.get("place_id"),
            "name": name,
            "address": address,
            "lat": loc.get("lat"),
            "lon": loc.get("lng"),
            "maps_url": f"https://www.google.com/maps/search/?api=1&query={q}",
        })
    return {"results": out}


def _expand_maps_url(url: str) -> str:
    text = str(url or "").strip()
    if not text:
        return text
    try:
        # Expand short Google Maps links (maps.app.goo.gl) to final maps URL.
        resp = requests.get(text, allow_redirects=True, timeout=10)
        if resp.url:
            return resp.url
    except Exception:
        pass
    return text


def _parse_lat_lon_from_maps_url(url: str) -> tuple[float | None, float | None]:
    text = _expand_maps_url(url)
    patterns = [
        r"@(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)",
        r"[?&]q=(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)",
        r"[?&]query=(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)",
        r"!3d(-?\d+(?:\.\d+)?)!4d(-?\d+(?:\.\d+)?)",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return float(m.group(1)), float(m.group(2))
    return None, None


def _infer_name_from_maps_url(url: str) -> str | None:
    text = str(url or "")
    m = re.search(r"/maps/place/([^/]+)/", text)
    if not m:
        return None
    name = unquote(m.group(1)).replace("+", " ").strip()
    return name or None


def _infer_address_from_lat_lon(lat: float, lon: float) -> str:
    if not GOOGLE_PLACES_API_KEY:
        return ""
    try:
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"latlng": f"{lat},{lon}", "key": GOOGLE_PLACES_API_KEY},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results") or []
        if not results:
            return ""
        return (results[0].get("formatted_address") or "").strip()
    except Exception:
        return ""


@app.post("/api/affiliates/from-link")
def api_affiliates_add_from_link(body: dict):
    maps_url = (body.get("maps_url") or "").strip()
    name = (body.get("name") or "").strip()
    if not maps_url:
        raise HTTPException(status_code=400, detail="maps_url is required")
    expanded = _expand_maps_url(maps_url)
    if not name:
        name = _infer_name_from_maps_url(expanded) or "Affiliate Center"
    lat, lon = _parse_lat_lon_from_maps_url(expanded)
    if lat is None or lon is None:
        raise HTTPException(status_code=400, detail="Could not read coordinates from this Google Maps link. Use a full Google Maps place/share URL.")
    address = _infer_address_from_lat_lon(lat, lon)
    try:
        return add_affiliate({
            "place_id": None,
            "name": name,
            "address": address,
            "lat": lat,
            "lon": lon,
            "maps_url": expanded or maps_url,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/vehicles/{license_plate}/maintenance-cards/{card_id}")
def api_maintenance_card_snapshot_get(license_plate: str, card_id: str):
    v = get_vehicle(license_plate)
    if not v:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    try:
        return get_maintenance_card_snapshot(license_plate, card_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Card not found")


@app.get("/api/vehicles/{license_plate}/maintenance-cards/{card_id}/csv")
def api_maintenance_card_snapshot_csv_v2(license_plate: str, card_id: str):
    """CSV download for a snapshot. Uses a separate path segment to avoid route conflicts."""
    v = get_vehicle(license_plate)
    if not v:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    try:
        card = get_maintenance_card_snapshot(license_plate, card_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Card not found")
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["license_plate", "produced_at", "mileage_km", "part_id", "part_label", "status", "recommended_interval_time", "recommended_interval_mileage"])
    for it in card.get("items") or []:
        w.writerow([
            license_plate,
            card.get("produced_at"),
            it.get("mileage_km"),
            it.get("part_id"),
            it.get("part_label"),
            it.get("status"),
            it.get("recommended_service_interval_time"),
            it.get("recommended_service_interval_mileage"),
        ])
    data = buf.getvalue().encode("utf-8")
    filename = f"{license_plate}_maintenance_card_{card_id.replace(':','-')}.csv"
    return StreamingResponse(iter([data]), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/vehicles/{license_plate}/maintenance-cards/{card_id}.csv")
def api_maintenance_card_snapshot_csv(license_plate: str, card_id: str):
    v = get_vehicle(license_plate)
    if not v:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    try:
        card = get_maintenance_card_snapshot(license_plate, card_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Card not found")
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["license_plate", "produced_at", "mileage_km", "part_id", "part_label", "status", "recommended_interval_time", "recommended_interval_mileage"])
    for it in card.get("items") or []:
        w.writerow([
            license_plate,
            card.get("produced_at"),
            it.get("mileage_km"),
            it.get("part_id"),
            it.get("part_label"),
            it.get("status"),
            it.get("recommended_service_interval_time"),
            it.get("recommended_service_interval_mileage"),
        ])
    data = buf.getvalue().encode("utf-8")
    filename = f"{license_plate}_maintenance_card_{card_id.replace(':','-')}.csv"
    return StreamingResponse(iter([data]), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/vehicles/{license_plate}/maintenance-history.csv")
def api_maintenance_history_csv(license_plate: str):
    v = get_vehicle(license_plate)
    if not v:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    cards = list_maintenance_cards(license_plate)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["license_plate", "produced_at", "mileage_km", "part_id", "part_label", "status", "recommended_interval_time", "recommended_interval_mileage"])
    for c in reversed(cards):  # oldest to newest
        card = get_maintenance_card_snapshot(license_plate, c["card_id"])
        for it in card.get("items") or []:
            w.writerow([
                license_plate,
                card.get("produced_at"),
                it.get("mileage_km"),
                it.get("part_id"),
                it.get("part_label"),
                it.get("status"),
                it.get("recommended_service_interval_time"),
                it.get("recommended_service_interval_mileage"),
            ])
    data = buf.getvalue().encode("utf-8")
    filename = f"{license_plate}_maintenance_history.csv"
    return StreamingResponse(iter([data]), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/vehicles/{license_plate}/maintenance-simplified.csv")
def api_maintenance_simplified_csv(license_plate: str, interval_km: int = 500):
    v = get_vehicle(license_plate)
    if not v:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    table = get_simplified_table_for_csv(license_plate, interval_km=interval_km)
    cols = table.get("columns") or []
    rows = table.get("rows") or []
    buf = io.StringIO()
    w = csv.writer(buf)
    header = [
        "license_plate",
        "part_id",
        "part_label",
        "recommended_service_interval_time",
        "recommended_service_interval_mileage",
    ] + [f"{c.get('bucket_km')}km" for c in cols]
    w.writerow(header)
    for r in rows:
        base = [
            license_plate,
            r.get("part_id"),
            r.get("part_label"),
            r.get("recommended_service_interval_time"),
            r.get("recommended_service_interval_mileage"),
        ]
        vals = [((r.get("values") or {}).get(c.get("bucket_km"), "none")) for c in cols]
        w.writerow(base + vals)
    data = buf.getvalue().encode("utf-8")
    filename = f"{license_plate}_maintenance_simplified.csv"
    return StreamingResponse(iter([data]), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/vehicles/{license_plate}/maintenance-template")
def api_maintenance_template(license_plate: str, interval_km: int = 500, n_cols: int = 8):
    v = get_vehicle(license_plate)
    if not v:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return get_maintenance_template(license_plate, interval_km=interval_km, n_cols=n_cols)


@app.get("/api/vehicles/{license_plate}/maintenance-template/print", response_class=HTMLResponse)
def api_maintenance_template_print(license_plate: str, interval_km: int = 500, n_cols: int = 8):
    v = get_vehicle(license_plate)
    if not v:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    t = get_maintenance_template(license_plate, interval_km=interval_km, n_cols=n_cols)
    cols = t.get("columns") or []
    parts = t.get("parts") or []
    hdr = "".join([f"<th>{c} km</th>" for c in cols])
    rows = ""
    for p in parts:
        pid = p.get("part_id") or ""
        label = p.get("part_label") or pid
        rec_time = p.get("recommended_service_interval_time") or "—"
        rec_mileage = p.get("recommended_service_interval_mileage") or "—"
        cells = "".join(['<td><div style="display:flex;gap:6px;justify-content:center;"><span>○R</span><span>○F</span></div></td>' for _ in cols])
        rows += f"<tr><td><strong>{label}</strong><div style='font-size:10px;color:#555'>{pid}</div></td><td>{rec_time}</td><td>{rec_mileage}</td>{cells}</tr>"
    filename = f"{license_plate}_maintenance_card_template_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>{filename}</title>
<style>body{{font-family:Arial,sans-serif;padding:16px}} table{{border-collapse:collapse;width:100%;font-size:11px}} th,td{{border:1px solid #444;padding:4px;text-align:center}} th:first-child,td:first-child{{text-align:left;position:sticky;left:0;background:#fff}} .meta{{margin-bottom:8px}}</style></head>
<body><div class='meta'><h3>Maintenance Card Template</h3><div>Vehicle: {license_plate} | Current mileage: {t.get("current_mileage_km")} km</div><div>Mark one bubble per part per mileage column (R=routine, F=failure), then upload photo in driver app.</div></div>
<table><thead><tr><th>Part</th><th>Recommended service interval: time</th><th>Recommended service interval: mileage</th>{hdr}</tr></thead><tbody>{rows}</tbody></table><script>setTimeout(()=>window.print(),250)</script></body></html>"""
    return HTMLResponse(content=html, headers={"Content-Disposition": f'inline; filename="{filename}.html"'})


@app.get("/api/vehicles/{license_plate}/maintenance-template.pdf")
def api_maintenance_template_pdf(license_plate: str, interval_km: int = 500, n_cols: int = 8):
    v = get_vehicle(license_plate)
    if not v:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    t = get_maintenance_template(license_plate, interval_km=interval_km, n_cols=n_cols)
    cols = t.get("columns") or []
    parts = t.get("parts") or []
    filename = f"{license_plate}_maintenance_card_template_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.pdf"
    try:
        from reportlab.lib.pagesizes import landscape, A4  # type: ignore
        from reportlab.pdfgen import canvas  # type: ignore
    except Exception:
        raise HTTPException(status_code=503, detail="PDF dependency not installed (reportlab)")

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=landscape(A4))
    w, h = landscape(A4)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(20, h - 24, "Maintenance Card Template")
    c.setFont("Helvetica", 9)
    c.drawString(20, h - 38, f"Vehicle: {license_plate} | Current mileage: {t.get('current_mileage_km')} km")
    c.drawString(20, h - 48, "Bubble key: left box = Routine (R), right box = Failure (F)")
    c.rect(8, h - 14, 8, 8, stroke=0, fill=1)
    c.rect(w - 16, h - 14, 8, 8, stroke=0, fill=1)
    c.rect(8, 6, 8, 8, stroke=0, fill=1)
    c.rect(w - 16, 6, 8, 8, stroke=0, fill=1)

    x0 = 20
    y = h - 64
    row_h = 24
    c.setFont("Helvetica-Bold", 7)
    c.drawString(x0, y, "Part")
    c.drawString(x0 + 170, y, "Rec. time")
    c.drawString(x0 + 250, y, "Rec. mileage")
    x = x0 + 350
    right_margin = 16
    n_cols_safe = max(1, len(cols))
    col_w = max(34, (w - right_margin - x) / n_cols_safe)
    for col in cols:
        label = f"{int(col):,}"
        c.drawCentredString(x + (col_w / 2), y, label)
        x += col_w
    y -= 11
    c.setFont("Helvetica", 6)
    for p in parts:
        if y < 24:
            c.showPage()
            y = h - 24
            c.setFont("Helvetica", 6)
        c.drawString(x0, y, str((p.get("part_label") or "")[:36]))
        c.drawString(x0 + 170, y, str((p.get("recommended_service_interval_time") or "—")[:14]))
        c.drawString(x0 + 250, y, str((p.get("recommended_service_interval_mileage") or "—")[:16]))
        x = x0 + 350
        for _ in cols:
            # Draw two blank bubble boxes centered in this column cell.
            box = 10
            gap = 6
            pair_w = box * 2 + gap
            left = x + max(0, (col_w - pair_w) / 2)
            box_y = y - 5
            c.rect(left, box_y, box, box, stroke=1, fill=0)             # routine
            c.rect(left + box + gap, box_y, box, box, stroke=1, fill=0) # failure
            x += col_w
        y -= row_h
    c.save()
    pdf = buf.getvalue()
    return StreamingResponse(iter([pdf]), media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.post("/api/vehicles/{license_plate}/maintenance-card-ocr")
async def api_maintenance_card_ocr(license_plate: str, image: UploadFile = File(...)):
    v = get_vehicle(license_plate)
    if not v:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    tmpl = get_maintenance_template(license_plate, interval_km=500, n_cols=8)
    cols = tmpl.get("columns") or []
    parts = tmpl.get("parts") or []
    data = await image.read()
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        raise HTTPException(status_code=503, detail="OMR dependencies not installed (opencv-python-headless, numpy)")

    arr = np.frombuffer(data, dtype=np.uint8)
    gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    def _order_points(pts):
        pts = np.array(pts, dtype="float32")
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1)
        return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype="float32")

    def _warp_to_template(g):
        blur = cv2.GaussianBlur(g, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        doc = None
        img_area = float(g.shape[0] * g.shape[1])
        for c in cnts[:20]:
            area = float(cv2.contourArea(c))
            if area < (img_area * 0.20):
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                doc = approx.reshape(4, 2)
                break
        out_w, out_h = 1684, 1190  # 2x landscape A4
        dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype="float32")
        if doc is not None:
            M = cv2.getPerspectiveTransform(_order_points(doc), dst)
            return cv2.warpPerspective(g, M, (out_w, out_h))
        # Fallback: no large page contour found (already close to frontal, such as screenshots).
        return cv2.resize(g, (out_w, out_h))

    warped = _warp_to_template(gray)
    bin_img = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 12)

    # Geometry mirrors pdf template endpoint.
    tw_pts, th_pts = 842.0, 595.0
    out_w, out_h = warped.shape[1], warped.shape[0]
    sx = out_w / tw_pts
    sy = out_h / th_pts
    x_cols = 370.0
    y_first_row = 75.0  # y0 + 11
    row_h = 24.0
    n_cols = max(1, len(cols))
    col_w = max(34.0, (tw_pts - 16.0 - x_cols) / n_cols)
    box = 10.0
    gap = 6.0
    pair_w = box * 2 + gap
    min_fill = 0.60
    min_delta = 0.12

    scores: list[list[tuple[float, float]]] = []
    for ci in range(n_cols):
        cell_x = x_cols + ci * col_w
        left = cell_x + max(0.0, (col_w - pair_w) / 2.0)
        col_scores: list[tuple[float, float]] = []
        for ri in range(len(parts)):
            y = y_first_row + ri * row_h
            box_y = y - 5.0
            rx1, ry1 = int(left * sx), int(box_y * sy)
            bw, bh = max(1, int(box * sx)), max(1, int(box * sy))
            rx2 = int((left + box + gap) * sx)
            # Use inner region to avoid counting the printed border as "filled".
            pad = max(1, int(min(bw, bh) * 0.22))
            roi_r = bin_img[ry1 + pad:ry1 + bh - pad, rx1 + pad:rx1 + bw - pad]
            roi_f = bin_img[ry1 + pad:ry1 + bh - pad, rx2 + pad:rx2 + bw - pad]
            rr = float(np.count_nonzero(roi_r)) / float(max(1, roi_r.size))
            rf = float(np.count_nonzero(roi_f)) / float(max(1, roi_f.size))
            col_scores.append((rr, rf))
        scores.append(col_scores)

    marked_per_col: list[int] = []
    for ci in range(n_cols):
        cnt = 0
        for rr, rf in scores[ci]:
            if max(rr, rf) >= min_fill and abs(rr - rf) >= min_delta:
                cnt += 1
        marked_per_col.append(cnt)

    active_cols = [i for i, cnt in enumerate(marked_per_col) if cnt > 0]
    if not active_cols:
        raise HTTPException(status_code=422, detail="Could not confidently detect marked bubbles. Retake image straight and closer.")

    saved_cards = []
    detected_total = 0
    for ci in active_cols:
        mileage = cols[ci]
        items = []
        for pi, p in enumerate(parts):
            rr, rf = scores[ci][pi]
            st = "none"
            if max(rr, rf) >= min_fill and abs(rr - rf) >= min_delta:
                st = "routine" if rr > rf else "failure"
                detected_total += 1
            items.append({"part_id": (p.get("part_id") or ""), "status": st})
        saved_cards.append(save_maintenance_card(license_plate, items, mileage_km=mileage))

    if detected_total == 0:
        raise HTTPException(status_code=422, detail="No marked bubbles detected.")
    return {
        "saved": True,
        "saved_events": len(saved_cards),
        "active_mileage_columns_km": [cols[i] for i in active_cols],
        "detected_marks": detected_total,
        "card": saved_cards[-1] if saved_cards else None,
    }


@app.post("/api/vehicles/{license_plate}/failures")
def api_failure_report(license_plate: str, body: dict):
    v = get_vehicle(license_plate)
    if not v:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    failure_type = (body.get("failure_type") or "").strip()
    try:
        return add_failure_report(license_plate, failure_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/vehicles/{license_plate}/failures")
def api_failure_list(license_plate: str):
    v = get_vehicle(license_plate)
    if not v:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return {"license_plate": license_plate, "reports": list_failure_reports(license_plate)}


@app.get("/api/vehicles/{license_plate}/failure-history.csv")
def api_failure_history_csv(license_plate: str):
    v = get_vehicle(license_plate)
    if not v:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    reps = list_failure_reports(license_plate)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["license_plate", "submitted_at", "failure_type"])
    for r in reversed(reps):  # oldest to newest
        w.writerow([license_plate, r.get("submitted_at"), r.get("failure_type")])
    data = buf.getvalue().encode("utf-8")
    filename = f"{license_plate}_failure_history.csv"
    return StreamingResponse(iter([data]), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{filename}"'})


def _nearest_by_time(target_iso: str, candidates: list[dict], time_key: str) -> tuple[dict | None, float | None]:
    """Return (nearest_candidate, abs_delta_seconds). candidates must have ISO timestamp at time_key."""
    if not target_iso or not candidates:
        return None, None
    try:
        from datetime import datetime
        t = datetime.fromisoformat(target_iso.replace("Z", "+00:00"))
    except Exception:
        return None, None
    best = None
    best_dt = None
    for c in candidates:
        ts = (c or {}).get(time_key)
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            continue
        d = abs((dt - t).total_seconds())
        if best is None or d < (best_dt or 1e18):
            best = c
            best_dt = d
    return best, best_dt


@app.get("/api/vehicles/{license_plate}/combined-report.csv")
def api_combined_report_csv(license_plate: str):
    v = get_vehicle(license_plate)
    if not v:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    log = get_vehicle_log(license_plate)
    recs = [r.model_dump() for r in (log.recommendations or [])]

    # Build event list = maintenance cards + failure reports, aligned to nearest recommendation timestamp (telemetry snapshot).
    card_meta = list_maintenance_cards(license_plate)
    cards = []
    for c in card_meta:
        try:
            cards.append(get_maintenance_card_snapshot(license_plate, c["card_id"]))
        except Exception:
            continue
    failures = list_failure_reports(license_plate)

    events = []
    for card in cards:
        events.append({"event_kind": "maintenance_card", "event_at": card.get("produced_at"), "card": card})
    for f in failures:
        events.append({"event_kind": "failure_report", "event_at": f.get("submitted_at"), "failure": f})
    events = [e for e in events if e.get("event_at")]
    events.sort(key=lambda e: e.get("event_at") or "")

    # Determine all input keys to output (operational + telemetry used in requests)
    input_keys: set[str] = set()
    for r in recs:
        snap = (r.get("inputs_snapshot") or {})
        for k in snap.keys():
            input_keys.add(k)
    input_keys = set(sorted(input_keys))

    # Maintenance checklist columns (fixed)
    from maintenance_card_store import MAINTENANCE_CHECK_LIST
    maint_cols = []
    for check_id, _label in MAINTENANCE_CHECK_LIST:
        maint_cols.extend([f"maint_{check_id}_last_completed", f"maint_{check_id}_mileage_km", f"maint_{check_id}_notes"])

    header = (
        ["license_plate", "event_kind", "event_at", "telemetry_at", "telemetry_delta_seconds"]
        + [f"input_{k}" for k in input_keys]
        + maint_cols
        + ["failure_type", "failure_submitted_at"]
    )

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)

    for e in events:
        nearest, delta = _nearest_by_time(e["event_at"], recs, "at")
        snap = (nearest or {}).get("inputs_snapshot") or {}
        row = {
            "license_plate": license_plate,
            "event_kind": e["event_kind"],
            "event_at": e["event_at"],
            "telemetry_at": (nearest or {}).get("at"),
            "telemetry_delta_seconds": delta,
            "failure_type": "",
            "failure_submitted_at": "",
        }
        if e["event_kind"] == "failure_report":
            row["failure_type"] = (e.get("failure") or {}).get("failure_type") or ""
            row["failure_submitted_at"] = (e.get("failure") or {}).get("submitted_at") or ""

        # flatten nearest inputs
        for k in input_keys:
            row[f"input_{k}"] = snap.get(k)

        # flatten maintenance card snapshot (aligned event’s own card, not nearest)
        maint_map = {}
        if e["event_kind"] == "maintenance_card":
            for it in (e.get("card") or {}).get("items") or []:
                cid = it.get("check_id")
                if not cid:
                    continue
                maint_map[cid] = it
        for check_id, _label in MAINTENANCE_CHECK_LIST:
            it = maint_map.get(check_id) or {}
            row[f"maint_{check_id}_last_completed"] = it.get("last_completed")
            row[f"maint_{check_id}_mileage_km"] = it.get("mileage_km")
            row[f"maint_{check_id}_notes"] = it.get("notes")

        w.writerow([row.get(col) for col in header])

    data = buf.getvalue().encode("utf-8")
    filename = f"{license_plate}_combined_report.csv"
    return StreamingResponse(iter([data]), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/all-data.csv")
def api_all_data_csv():
    """Combined report across all registered vehicles."""
    vehicles = list_vehicles()
    # Build one big CSV by concatenating per-vehicle combined report rows under a shared header.
    # We recompute header using the union of all input keys across vehicles.
    all_input_keys: set[str] = set()
    for v in vehicles:
        log = get_vehicle_log(v.license_plate)
        for r in log.recommendations or []:
            for k in (r.inputs_snapshot or {}).keys():
                all_input_keys.add(k)
    all_input_keys = set(sorted(all_input_keys))
    from maintenance_card_store import MAINTENANCE_CHECK_LIST
    maint_cols = []
    for check_id, _label in MAINTENANCE_CHECK_LIST:
        maint_cols.extend([f"maint_{check_id}_last_completed", f"maint_{check_id}_mileage_km", f"maint_{check_id}_notes"])
    header = (
        ["license_plate", "event_kind", "event_at", "telemetry_at", "telemetry_delta_seconds"]
        + [f"input_{k}" for k in all_input_keys]
        + maint_cols
        + ["failure_type", "failure_submitted_at"]
    )
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    for v in vehicles:
        plate = v.license_plate
        log = get_vehicle_log(plate)
        recs = [r.model_dump() for r in (log.recommendations or [])]
        card_meta = list_maintenance_cards(plate)
        cards = []
        for c in card_meta:
            try:
                cards.append(get_maintenance_card_snapshot(plate, c["card_id"]))
            except Exception:
                continue
        failures = list_failure_reports(plate)
        events = []
        for card in cards:
            events.append({"event_kind": "maintenance_card", "event_at": card.get("produced_at"), "card": card})
        for f in failures:
            events.append({"event_kind": "failure_report", "event_at": f.get("submitted_at"), "failure": f})
        events = [e for e in events if e.get("event_at")]
        events.sort(key=lambda e: e.get("event_at") or "")
        for e in events:
            nearest, delta = _nearest_by_time(e["event_at"], recs, "at")
            snap = (nearest or {}).get("inputs_snapshot") or {}
            row = {
                "license_plate": plate,
                "event_kind": e["event_kind"],
                "event_at": e["event_at"],
                "telemetry_at": (nearest or {}).get("at"),
                "telemetry_delta_seconds": delta,
                "failure_type": "",
                "failure_submitted_at": "",
            }
            if e["event_kind"] == "failure_report":
                row["failure_type"] = (e.get("failure") or {}).get("failure_type") or ""
                row["failure_submitted_at"] = (e.get("failure") or {}).get("submitted_at") or ""
            for k in all_input_keys:
                row[f"input_{k}"] = snap.get(k)
            maint_map = {}
            if e["event_kind"] == "maintenance_card":
                for it in (e.get("card") or {}).get("items") or []:
                    cid = it.get("check_id")
                    if cid:
                        maint_map[cid] = it
            for check_id, _label in MAINTENANCE_CHECK_LIST:
                it = maint_map.get(check_id) or {}
                row[f"maint_{check_id}_last_completed"] = it.get("last_completed")
                row[f"maint_{check_id}_mileage_km"] = it.get("mileage_km")
                row[f"maint_{check_id}_notes"] = it.get("notes")
            w.writerow([row.get(col) for col in header])
    data = buf.getvalue().encode("utf-8")
    return StreamingResponse(
        iter([data]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="all_vehicles_combined_report.csv"'},
    )


@app.get("/api/data/ml-training.csv")
def api_ml_training_csv():
    """Per-event, per-part training rows: telemetry snapshot + maintenance label."""
    vehicles = list_vehicles()
    all_rows: list[dict] = []
    telemetry_keys: set[str] = set()
    for v in vehicles:
        plate = v.license_plate
        log = get_vehicle_log(plate)
        recs = [r.model_dump() for r in (log.recommendations or [])]
        for r in recs:
            for k in ((r.get("inputs_snapshot") or {}).keys()):
                telemetry_keys.add(k)
        meta = list_maintenance_cards(plate)
        for m in meta:
            try:
                card = get_maintenance_card_snapshot(plate, m["card_id"])
            except Exception:
                continue
            nearest, _delta = _nearest_by_time(card.get("produced_at"), recs, "at")
            snap = (nearest or {}).get("inputs_snapshot") or {}
            for it in (card.get("items") or []):
                row = {
                    "license_plate": plate,
                    "event": card.get("produced_at"),
                    "event_mileage_km": card.get("mileage_km"),
                    "maintenance_part": it.get("part_id"),
                    "label": it.get("status"),
                }
                for k, vval in snap.items():
                    row[f"telemetry_{k}"] = vval
                all_rows.append(row)
    telemetry_cols = [f"telemetry_{k}" for k in sorted(telemetry_keys)]
    header = ["license_plate", "event", "event_mileage_km"] + telemetry_cols + ["maintenance_part", "label"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    for r in all_rows:
        w.writerow([r.get(c) for c in header])
    data = buf.getvalue().encode("utf-8")
    return StreamingResponse(
        iter([data]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="ml_training_all_vehicles.csv"'},
    )


# --- Recommendation (rules ± ML + explanation) ---
@app.post("/api/recommend", response_model=RecommendationResponse)
def api_recommend(req: RecommendationRequest):
    vehicle = req.vehicle
    rec = None
    if req.license_plate:
        rec = get_vehicle(req.license_plate)
        if not rec:
            raise HTTPException(status_code=404, detail=f"Vehicle with license plate {req.license_plate} not found")
        vehicle = rec.to_identity()
    rules = load_rules(vehicle)
    ml_out: MLOutput | None = None
    if req.use_ml:
        ml_out = get_ml_prediction(req.inputs, enable=True)
    else:
        ml_out = MLOutput(enabled=False)

    decision = run_decision_engine(
        rules, req.inputs, ml_out, use_severe_service=req.use_severe_service
    )

    explanation = decision.summary
    if OPENAI_API_KEY:
        try:
            explanation = generate_explanation(
                vehicle,
                decision,
                user_location=req.user_location,
                language=req.ui_language,
            )
        except Exception:
            pass

    nearby_mechanics: list[NearbyMechanic] = []
    has_due = any(c.priority in ("red", "yellow") for c in decision.components)
    if (
        has_due
        and req.user_location is not None
        and GOOGLE_PLACES_API_KEY
    ):
        lat = req.user_location.latitude
        lon = req.user_location.longitude

        for a in nearest_affiliates(lat, lon, limit=3):
            nearby_mechanics.append(
                NearbyMechanic(
                    kind="affiliate",
                    label="Affiliated repair center near you",
                    name=a.get("name") or "Affiliate Center",
                    maps_url=a.get("maps_url") or "",
                )
            )

        gen = nearest_general_mechanic(lat, lon)
        if gen:
            nearby_mechanics.append(
                NearbyMechanic(
                    kind="general",
                    label="General auto repair near you",
                    name=gen["name"],
                    maps_url=gen["maps_url"],
                )
            )

        make = (vehicle.make or "").strip()
        dealer = nearest_dealer_service(lat, lon, make) if make else None
        if dealer:
            nearby_mechanics.append(
                NearbyMechanic(
                    kind="dealer",
                    label=f"Authorized {make} service near you",
                    name=dealer["name"],
                    maps_url=dealer["maps_url"],
                )
            )

        brake = nearest_brake_tire(lat, lon)
        if brake:
            nearby_mechanics.append(
                NearbyMechanic(
                    kind="brake_tire",
                    label="Brake & tire shop near you",
                    name=brake["name"],
                    maps_url=brake["maps_url"],
                )
            )

    if req.license_plate:
        append_vehicle_recommendation(
            req.license_plate,
            decision.model_dump(),
            explanation,
            req.inputs.model_dump(),
            use_severe_service=req.use_severe_service,
        )

    return RecommendationResponse(
        decision=decision,
        explanation=explanation,
        vehicle=vehicle,
        nearby_mechanics=nearby_mechanics or None,
    )


# --- Static frontend ---
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
DOCS_DIR = Path(__file__).resolve().parent / "docs"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/api/walkthrough.pdf")
    def walkthrough_pdf():
        path = DOCS_DIR / "user_walkthrough_v1.pdf"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Walkthrough PDF not found")
        return FileResponse(path, media_type="application/pdf", filename="user_walkthrough_v1.pdf")

    @app.get("/")
    def index():
        return FileResponse(FRONTEND_DIR / "login.html")

    @app.get("/admin")
    def admin():
        return FileResponse(FRONTEND_DIR / "admin.html")

    @app.get("/driver")
    def driver(request: Request):
        token = request.cookies.get(DRIVER_SESSION_COOKIE)
        if not _DRIVER_SESSIONS.get(token or ""):
            return RedirectResponse(url="/")
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/simplified")
    def simplified():
        return FileResponse(FRONTEND_DIR / "simplified.html")
