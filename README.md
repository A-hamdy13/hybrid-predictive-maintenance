# Hybrid Predictive Maintenance (Demo)

Combines **manual-extracted maintenance rules** with optional **telemetry-based ML** and an **LLM explanation** layer.

## Setup

1. **Python 3.10+**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   For the **AI manual-finder agent** (browser + LLM), install Chromium for Playwright once:
   ```bash
   playwright install chromium
   ```
   If you skip this, the app falls back to non-agent URL resolution (DuckDuckGo HTML + LLM-generated queries).

3. **Environment**
   - Copy `.env.example` to `.env`
   - Set `OPENAI_API_KEY=sk-...` (required for rule extraction and explanations)
   - Manual-finding uses the **OpenAI Responses API** with the **web_search** tool (same as ChatGPT “search the web”); no Bing or other search APIs are required.

## Run

```bash
uvicorn main:app --reload
```

Open **http://127.0.0.1:8000** for the UI.

## Manual layer (this implementation)

- **Vehicle list**: The UI lists all vehicles that already have extracted rules (stored in `data/rules/`). The user picks one to get recommendations.
- **Other (add new vehicle)**: For a vehicle not in the list, the user selects "Other", enters make/model/year (and optional trim). By default the app uses the **OpenAI Responses API with the web_search tool** to find official manual or maintenance-schedule URLs (PDF or page). Those URLs are then fetched, parsed, validated, and stored. The user can optionally paste a **Manual PDF or page URL** to skip the search.
- **Validation**: Extracted rules are validated (evidence, units, closed schema) and optionally run through a verifier pass before storage.
- **Decision engine**: Evaluates schedule vs operational inputs. When **Use ML** is off, the recommendation is **rules-only**.

## UI

- **Vehicle**: Dropdown of vehicles with stored rules, plus "Other (add new vehicle)". When "Other" is selected, make/model/year/trim and "Find manual & extract rules" are shown; optional field "Manual PDF or page URL" for overriding the agent.
- **Operational inputs**: Pre-filled with default test values (e.g. mileage 52 300, months since oil 7, engine temp 98, etc.); edit as needed.
- **Use ML**: Toggle on to include telemetry-based risk; off for manual rules only.
- **Get recommendation**: Requires a selected vehicle with rules; shows per-component status and LLM explanation.

## API

- `GET /api/vehicles/demo` — list demo vehicles
- `GET /api/vehicles/rules` — list vehicles with stored rules
- `GET /api/agent/available` — whether the browser agent is available (Playwright + OpenAI)
- `POST /api/manuals/find` — body: `VehicleIdentity` → manual URL candidates
- `POST /api/manuals/extract` — body: `{ "vehicle": {...}, "manual_url": "..." }` → extract and store rules
- `POST /api/manuals/find-and-extract` — body: `{ "vehicle": {...}, "manual_url": null }` → agent finds manual, then fetch → extract → validate → store; or pass `manual_url` to use that URL instead
- `GET /api/rules?make=&model=&year=` — get rules for a vehicle
- `POST /api/recommend` — body: `RecommendationRequest` (vehicle, inputs, use_ml) → decision + explanation

## Design

- **LLM** does not decide maintenance; it only **explains** the structured recommendation from the decision engine.
- **Closed schema** for extraction (allowed components, validated units).
- **Evidence required**: each rule needs `source_quote` and optional `source_page`; entries without evidence are rejected.
