# Hybrid Predictive Maintenance — Project Documentation

This document describes the **Hybrid Predictive Maintenance** application as implemented in this repository. It was produced by reading the source files (`main.py`, `frontend/index.html`, `decision_engine.py`, `schemas.py`, `rule_validator.py`, `manual_extractor.py`, `explanation_generator.py`, `ml_predictor.py`, stores, and related modules). If something is not listed here, it was not verified in code for this write-up.

---

## 1. Project description

**Hybrid Predictive Maintenance** is a web application that helps drivers reason about vehicle maintenance by combining:

1. **Rule-based scheduling** — Maintenance intervals extracted from (or manually aligned with) manufacturer documentation, stored per vehicle make/model/year (and optional trim), then compared to the driver’s operational inputs (mileage, time since last service, etc.).
2. **Optional ML / telemetry risk** — When enabled, telemetry fields feed a trained model if `data/model.pkl` (and feature list) exists, or a **heuristic placeholder** if the model is missing or inputs are incomplete.
3. **Natural-language explanation** — When `OPENAI_API_KEY` is set, an LLM turns the **already computed** structured decision into short prose. The LLM is instructed **not** to change the recommendation.
4. **Per–license-plate workflow** — Vehicles can be registered by license plate; recommendations are logged, maintenance cards can be filled and exported, and failures can be reported for record-keeping and CSV export.

The backend is **FastAPI** (`main.py`). The UI is a single-page **static HTML/JS** app served at `/` and under `/static`. Data lives under `data/` (rules JSON, vehicle registry, logs, maintenance cards, failure reports, optional model files).

---

## 2. What was built (high-level)

| Area | Implementation |
|------|------------------|
| **API** | REST endpoints for vehicles, rules, manual find/extract, recommendations, maintenance cards, failures, CSV exports |
| **Frontend** | `frontend/index.html` — vehicle selection, rules source, operational inputs, toggles, recommendation display, rules editor modal, maintenance card table |
| **Decision logic** | `decision_engine.py` — deterministic merge of schedule evaluation + optional ML component risks |
| **Extraction** | `manual_extractor.py` — LLM JSON extraction from manual text with a fixed schema |
| **Validation** | `rule_validator.py` — closed components/units, drop non-actionable items, optional quote verifier warnings |
| **Manual fetch** | `manual_downloader.py` — fetch PDF/HTML, cache, text extraction; `manual_finder.py` — OpenAI web search URL resolution |
| **ML** | `ml_predictor.py` — joblib model + fallback heuristics |
| **Explanation** | `explanation_generator.py` — LLM paraphrase of structured output only |
| **Persistence** | `rule_store.py`, `vehicle_registry.py`, `vehicle_logs.py`, `maintenance_card_store.py`, `failure_report_store.py` |

---

## 3. User interface — controls and actions

### 3.1 License plate and registration

| Control | Type | Description |
|---------|------|-------------|
| **Your vehicle (license plate)** | `<select id="licensePlateSelect">` | Lists registered vehicles from `GET /api/vehicles`; includes “Add vehicle…”. |
| **Add vehicle** | Button `btnAddVehicle` | Opens the add-vehicle flow (same as choosing “Add vehicle…” in the dropdown). |
| **Save vehicle** | Button `btnSubmitVehicle` | `POST /api/vehicles` with plate, make, model, year, trim; then reloads list and selects the new plate. |
| **Cancel** | Button `btnCancelAddVehicle` | Hides the form and clears selection. |

When a plate is selected, the UI loads `GET /api/vehicles/{plate}`, logs (`GET .../logs`), maintenance card (`GET .../maintenance-card`), syncs **Vehicle (rules source)** if make/model/year match a rules file, and may disable that dropdown when matched.

### 3.2 Vehicle log and maintenance card

| Control | Description |
|---------|-------------|
| **Vehicle log** sections | Summary + scrollable text of recent recommendations and alerts (from log JSON). |
| **Download CSV** (`btnDownloadCardCsv`) | Downloads CSV for the **current card snapshot** (requires a saved card / `produced_at`). Uses `produced_at` as `card_id` in the URL. |
| **Download maintenance history** (`btnDownloadMaintenanceHistory`) | `GET .../maintenance-history.csv`. |
| **Save card** (`btnSaveCard`) | `PUT .../maintenance-card` with table row data (last completed, mileage, notes). |
| **Download PDF** (`btnDownloadCardPdf`) | Opens a print dialog with HTML built in the browser (not a server-generated PDF file). |
| **Saved maintenance cards** list | For each snapshot: dynamically created **PDF** and **CSV** buttons (fetch snapshot + print or navigate to CSV URL). |
| **Failure type** | `<select id="failureType">` — predefined types (engine, transmission, brake, etc.). |
| **Report failure** (`btnReportFailure`) | `POST .../failures` with `failure_type`. |
| **Download failure history** | `GET .../failure-history.csv`. |
| **Download combined report** | `GET .../combined-report.csv` (events aligned to recommendation telemetry snapshots). |
| **Download all data** | `GET /api/all-data.csv` (all vehicles). |

### 3.3 Vehicle (rules source)

| Control | Description |
|---------|-------------|
| **Vehicle (rules source)** | `<select id="vehicleSelect">` — vehicles that have rules in `data/rules/` (`GET /api/vehicles/rules`), plus **Other (add new vehicle)**. |
| **Edit rules** (`btnEditRules`) | Visible when a stored-rules vehicle is selected. Loads `GET /api/rules?...` and opens the modal. |
| **Add manually** (`btnAddManually`) | Opens modal to enter make/model/year/trim and normal/severe schedule rows; saves via `POST /api/rules/save`. |
| **Other** panel | Make, model, year, trim, optional manual URL. |
| **Find manual & extract rules** (`btnFindAndExtract`) | `POST /api/manuals/find-and-extract` with optional `manual_url`; shows counts and verifier warnings. |

### 3.4 Operational inputs

Fields are shown **conditionally** when the selected rules include intervals for relevant components (and ML telemetry block when **Use ML** is checked): current mileage; oil change time/km; brake/tire km; battery age; extended telemetry fields (RPM, temps, pressures, wheel speeds, etc.). See `OperationalInputs` in `schemas.py` for the full API field set.

### 3.5 Toggles and recommendation

| Control | Description |
|---------|-------------|
| **Use ML prediction (telemetry risk)** (`useMl`) | Sends `use_ml: true` to `/api/recommend`; shows telemetry block in operational inputs when relevant. |
| **Use severe service schedule** (`useSevereService`) | Sends `use_severe_service: true`; decision engine uses `severe_service` instead of `normal_service`. |
| **Load defaults** (`btnLoadDefaults`) | Fills operational/telemetry fields with built-in demo numbers in the page script. |
| **Load failure scenario** (`btnLoadFailureScenario`) | Sets defaults then overwrites key fields with stressed values (high temp, low oil pressure, etc.). |
| **Get recommendation** (`btnRecommend`) | `POST /api/recommend` with vehicle, inputs, flags, optional `license_plate`, and browser geolocation when available. Renders overall priority, explanation, ML block, per-component lines, and optional Google Maps / Places links when there is a **red** priority and location + API key. |

### 3.6 Rules editor modal

| Control | Description |
|---------|-------------|
| **Save** / **Cancel** | Save posts full rules JSON to `POST /api/rules/save`; Cancel closes overlay. |
| **+ Add normal service item** / **+ Add severe service item** | Appends a rule card. |
| **Remove** (per card) | Removes that rule row. |
| Per-row fields | Component, action, distance/time intervals and units, conditions, source quote, source URL. |

---

## 4. Frontend JavaScript (main functions)

These are defined in `frontend/index.html` (inline `<script>`).

| Function | Role |
|----------|------|
| `byId(id)` | Shorthand for `document.getElementById`. |
| `vehicle()` | Builds `VehicleIdentity` from the “Other” form fields. |
| `setVehicleFromStored(v)` | Copies stored make/model/year/trim into those fields. |
| `inputs()` | Collects all operational/telemetry inputs into the payload shape expected by the API. |
| `loadLicensePlateList()` | Fetches `/api/vehicles` and fills the license plate dropdown. |
| `buildVehicleLogSummary(logData)` | Renders short summary of last recommendation into the log summary box. |
| `renderMaintenanceCardTable(items)` | Builds table rows for the maintenance card. |
| `escapeHtml(s)` | Escapes text for safe HTML insertion. |
| `getMaintenanceCardDataFromTable()` | Reads edited cells back into an `items` array for save. |
| `buildMaintenanceCardPdfHtml(vehicleRecord, cardData)` | HTML string for print/PDF view. |
| `onLicensePlateChange()` | Main handler when plate changes: fetch vehicle, logs, card, sync rules dropdown. |
| `loadVehicleList()` | Fetches `/api/vehicles/rules` for the rules-source dropdown. |
| `fmtDateTime(iso)` | Localized display of ISO timestamps. |
| `renderMaintenanceHistoryList(cards)` | Lists past cards with PDF/CSV actions. |
| `refreshMaintenanceCardPanels(licensePlate)` | Reloads current card and history list. |
| `emptyRuleItem()` | Default template for a new rule row. |
| `renderRuleItem(item, index, section, onRemove)` | DOM for one rule card in the modal. |
| `collectRuleItems(container, section)` | Reads modal cards back into arrays. |
| `openRulesModal(isAddManually, vehicleJson)` | Opens overlay; wires add-row and save handlers. |
| `getComponentsWithIntervals(rules)` | Which components have actionable intervals (for showing/hiding inputs). |
| `updateOperationalInputsVisibility()` | Shows/hides `data-show-when` blocks based on rules + ML toggle. |
| `loadOperationalDefaults()` / `loadFailureScenario()` | Preset input values. |
| `getUserLocation()` | `navigator.geolocation.getCurrentPosition` (with timeout). |

---

## 5. Backend API endpoints (`main.py`)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/vehicles/demo` | Load `data/demo_vehicles.json` if present. |
| GET | `/api/vehicles/rules` | List vehicles with stored rule files. |
| GET | `/api/agent/available` | `{ "available": ... }` for Playwright-based agent (see `manual_agent.is_agent_available`). |
| POST | `/api/manuals/find` | Return manual URL candidates (`find_manual_candidates`). |
| POST | `/api/manuals/extract` | Fetch or accept text, extract, validate, save rules. |
| POST | `/api/manuals/find-and-extract` | Resolve URLs (web search), fetch, extract, merge, match vehicle, save. |
| GET | `/api/rules` | Load rules for make/model/year/(trim). |
| POST | `/api/rules/save` | Validate and save rules from UI or import. |
| POST | `/api/rules/import-from-notebooklm` | Same validation path for external JSON (e.g. NotebookLM). |
| GET | `/api/vehicles` | List registered vehicles (`vehicles.json`). |
| POST | `/api/vehicles` | Create vehicle by license plate. |
| GET | `/api/vehicles/{license_plate}` | Get one vehicle. |
| PUT | `/api/vehicles/{license_plate}` | Update allowed fields. |
| GET | `/api/vehicles/{license_plate}/logs` | Recommendation log + alerts. |
| GET/PUT | `/api/vehicles/{license_plate}/maintenance-card` | Get or save current maintenance card. |
| GET | `/api/vehicles/{license_plate}/maintenance-cards` | List snapshots. |
| GET | `/api/vehicles/{license_plate}/maintenance-cards/{card_id}` | One snapshot JSON. |
| GET | `/api/vehicles/.../maintenance-cards/{card_id}/csv` and `.../{card_id}.csv` | CSV export for snapshot. |
| GET | `/api/vehicles/.../maintenance-history.csv` | All card snapshots as CSV rows. |
| POST | `/api/vehicles/.../failures` | Append failure report. |
| GET | `/api/vehicles/.../failures` | List failure reports. |
| GET | `/api/vehicles/.../failure-history.csv` | Failures as CSV. |
| GET | `/api/vehicles/.../combined-report.csv` | Maintenance events + failures aligned to nearest recommendation inputs. |
| GET | `/api/all-data.csv` | Combined report across all vehicles. |
| POST | `/api/recommend` | Run decision engine + optional ML + explanation + optional Places; log if `license_plate` set. |
| GET | `/` | Serves `frontend/index.html`. |
| — | `/static` | Static files from `frontend/`. |

---

## 6. Python modules — key functions

### 6.1 `decision_engine.py`

| Function | Description |
|----------|-------------|
| `_km_or_miles(value, unit)` | Normalizes distance to km for comparisons. |
| `_evaluate_component(component, rule_item, inputs)` | Returns status/priority/reason for one rule vs inputs (distance/time thresholds at 80% / 100%). |
| `run_decision_engine(rules, inputs, ml_output, use_severe_service)` | Builds `DecisionOutput`: evaluates chosen schedule, then fuses ML risks (manual red preserved; yellow+high ML → red; green+high ML → inspect soon; adds ML-only yellow rows for unknown components). |

### 6.2 `rule_validator.py`

| Function | Description |
|----------|-------------|
| `_validate_item(item)` | Component in allow-list; units valid; confidence 0–1; `source_quote` required if `found` and no interval. |
| `_has_actionable_interval(item)` | Requires distance or time interval when `found`. |
| `validate_extraction(rules)` | Filters invalid items; drops non-actionable; returns errors list. |
| `verifier_pass(rules)` | Soft warnings if quoted text may not contain stated numeric interval. |

### 6.3 `manual_extractor.py`

| Function | Description |
|----------|-------------|
| `extract_rules(manual_text, vehicle, ...)` | Calls OpenAI chat with `EXTRACTION_SYSTEM` / template; parses JSON into `ExtractedManualRules` (truncates long text). |

### 6.4 `explanation_generator.py`

| Function | Description |
|----------|-------------|
| `generate_explanation(vehicle, decision, user_location=...)` | Sends structured decision text to LLM with instructions to **only** explain, not change actions. |

### 6.5 `ml_predictor.py`

| Function | Description |
|----------|-------------|
| `_load_model()` | Loads `data/model.pkl`, `model_features.txt`, optional `model_components.pkl`. |
| `_get_feature_value`, `_inputs_to_feature_row` | Map `OperationalInputs` to model features (with aliases). |
| `get_ml_prediction(inputs, enable)` | Returns `MLOutput`; uses model or `_placeholder_prediction` heuristics. |
| `_placeholder_prediction(inputs)` | Rule-of-thumb risks from temperature, oil pressure, battery voltage, tire pressure. |

### 6.6 `rule_store.py`

| Function | Description |
|----------|-------------|
| `path_for_vehicle`, `save_rules`, `load_rules`, `list_stored_vehicles` | File naming under `data/rules/*.json`. |

### 6.7 `vehicle_registry.py`

| Function | Description |
|----------|-------------|
| `list_vehicles`, `get_vehicle`, `create_vehicle`, `update_vehicle`, `delete_vehicle` | Backed by `data/vehicles.json`. |

### 6.8 `vehicle_logs.py`

| Function | Description |
|----------|-------------|
| `get_log`, `append_recommendation`, `append_alert` | JSON per plate under `data/vehicle_logs/`. |

### 6.9 `maintenance_card_store.py`

| Function | Description |
|----------|-------------|
| `MAINTENANCE_CHECK_LIST` | Fixed checklist ids/labels. |
| `get_card`, `list_cards`, `get_card_snapshot`, `save_card` | Current + history snapshots; merges **last recommended check date** from recommendation log. |

### 6.10 `failure_report_store.py`

| Function | Description |
|----------|-------------|
| `list_reports`, `add_report` | Append-only failure log per plate under `data/failure_reports/`. |

### 6.11 `manual_downloader.py`

| Function | Description |
|----------|-------------|
| `fetch_manual_text`, `get_manual_text_from_url`, `fetch_manual_text_and_type`, `extract_pdf_links_from_page` | Download, cache, PDF/HTML to text, discover PDF links on hub pages. |

### 6.12 `manual_finder.py`

| Function | Description |
|----------|-------------|
| `find_manual_candidates`, `resolve_fetchable_urls` | Use `openai_web_search.find_manual_urls_with_web_search` (OpenAI Responses + web search). |

### 6.13 `google_places.py`

| Function | Description |
|----------|-------------|
| `_nearby_search`, `nearest_general_mechanic`, `nearest_dealer_service`, `nearest_brake_tire` | Nearby Search API; returns name + Google Maps query URL. |

### 6.14 `main.py` (helpers)

| Function | Description |
|----------|-------------|
| `_do_extract_and_validate` | Extract + validate without save (used in pipelines). |
| `_do_extract_and_store` | Extract + validate + save. |
| `_vehicle_matches(requested, extracted)` | Same make/model/year (case-insensitive) for rejecting wrong manuals. |
| `_merge_rules(base, new)` | Merge extractions from multiple URLs without duplicating same (component, conditions, action). |
| `_nearest_by_time` | Aligns maintenance/failure events to closest recommendation timestamp for CSV. |

### 6.15 `schemas.py` (types)

Important models: `VehicleIdentity`, `VehicleRecord`, `ExtractedRuleItem`, `ServiceSchedule`, `ExtractedManualRules`, `OperationalInputs`, `ComponentStatus`, `MLOutput`, `DecisionOutput`, `RecommendationRequest`, `RecommendationResponse`, `UserLocation`, `NearbyMechanic`, `VehicleLog`, log entry models.

### 6.16 `config.py`

Defines `DATA_DIR`, `RULES_DIR`, caches, `ALLOWED_COMPONENTS`, `VALID_DISTANCE_UNITS`, `VALID_TIME_UNITS`, env-loaded API keys and model name.

### 6.17 `manual_agent.py` (optional)

Playwright + LLM browser agent for finding manuals (used for availability reporting; the **find-and-extract** path in `main.py` uses `resolve_fetchable_urls` / web search as logged in code, not this agent’s navigation).

---

## 7. Data layout (on disk)

| Path | Content |
|------|---------|
| `data/rules/*.json` | `ExtractedManualRules` per vehicle key |
| `data/vehicles.json` | Registered vehicles by license plate |
| `data/vehicle_logs/*.json` | Recommendations and alerts |
| `data/maintenance_cards/*.json` | Current + history maintenance card snapshots |
| `data/failure_reports/*.json` | Failure reports |
| `data/model.pkl`, `model_features.txt`, `model_components.pkl` | Optional ML artifacts |
| `data/manuals_cache/` | Cached fetched manuals |
| `data/scraped_manuals/` | Optional saved full text for NotebookLM / debugging |

---

## 8. Anti-hallucination and safety-oriented measures (as implemented)

These are **design and code behaviors** intended to keep outputs grounded and avoid letting the LLM “invent” maintenance decisions.

1. **Separation of concerns** — The **decision engine** (`run_decision_engine`) computes status and priority from rules and numeric inputs (and optional ML scores). The LLM in `explanation_generator` is instructed to **only explain** that output, not suggest different actions.

2. **Closed extraction schema** — `ALLOWED_COMPONENTS` and fixed distance/time units in `config.py` and `rule_validator.py` prevent arbitrary component names or units from entering stored rules.

3. **Extraction prompt constraints** — `manual_extractor.EXTRACTION_SYSTEM` requires intervals to be **explicitly** in the text, asks the model to set `vehicle` from the document (to allow mismatch detection), and requires `source_quote` (and page when available) for found items.

4. **Post-extraction validation** — `validate_extraction` rejects invalid items and **drops** `found` items with no actionable interval. User-edited rules via UI can omit `source_quote` when a numeric interval is present (validator allows that path).

5. **Verifier warnings** — `verifier_pass` adds non-blocking warnings if the `source_quote` string might not contain the stated interval numbers (for logging / UI messaging).

6. **Manual identity check** — In `find-and-extract`, extracted vehicle identity must match the requested make/model/year (`_vehicle_matches`) before merging; otherwise that URL’s rules are skipped.

7. **Merge discipline** — `_merge_rules` avoids duplicate (component, conditions, action) rows and tracks contributing URLs.

8. **Low temperature on extraction** — `extract_rules` uses `temperature=0` for the extraction chat completion.

9. **ML fallback honesty** — If the model file is missing or features incomplete, `get_ml_prediction` uses `_placeholder_prediction` based on simple thresholds rather than silent failure.

10. **Explanation temperature** — Explanation uses `temperature=0.3` (slightly creative wording but still tied to injected structured facts).

11. **Optional geolocation** — Browser location is only requested when the user runs **Get recommendation**; it is used for nearby shop links when priority is red and `GOOGLE_PLACES_API_KEY` is set.

**Not medical/legal advice:** The app is a technical demo; always follow manufacturer guidance and professional inspection for safety-critical systems.

---

## 9. How this documentation was written (anti-hallucination for the doc itself)

- Endpoints and behaviors were taken from **`main.py`** and cross-checked against **`frontend/index.html`** for what the UI actually calls.
- Button IDs and labels were copied from the HTML.
- Python function names and roles were taken from **`grep`/`read`** of the listed modules; secondary files (e.g. `openai_web_search.py`) were not fully quoted here unless needed.
- The root **`README.md`** may mention older behavior (e.g. non–web-search fallbacks); **this doc prefers the current code** in `manual_finder.py` (OpenAI web search only for URL resolution in that path).

If you add features, update this file alongside `main.py` and `frontend/index.html` so it stays accurate.
