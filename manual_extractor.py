"""LLM-based extraction of maintenance rules from manual text with a fixed schema."""
import json
import logging
import re
from typing import Optional

from openai import OpenAI

from config import ALLOWED_COMPONENTS, OPENAI_API_KEY, OPENAI_MODEL
from schemas import ExtractedRuleItem, ExtractedManualRules, ServiceSchedule, VehicleIdentity

log = logging.getLogger(__name__)

EXTRACTION_SYSTEM = """You extract vehicle maintenance intervals from official manual text into a strict JSON structure.

RULES:
1. Only extract intervals that are EXPLICITLY stated in the text. If not stated, use "found": false for that component.
2. In "vehicle", set make, model, and year to the vehicle this manual is actually for (as stated in the document). Do not copy the requested vehicle—identify from the text. If the manual is for a different model or year, return that so we can reject it.
3. Every extracted item MUST include: "found" (true/false), and if true: "source_quote" (exact sentence from text) and "source_page" if page number is mentioned.
4. Use ONLY these component identifiers: engine_oil, oil_filter, tire_rotation, brake_inspection, battery, coolant, transmission_fluid, air_filter, fuel_filter, belt_inspection, hose_inspection, daily_check, weekly_check.
5. For intervals use: interval_distance_value + interval_distance_unit (km or miles), and/or interval_time_value + interval_time_unit (months, weeks, hours, days, years).
6. action: use "replace", "inspect", "check", or "service" as appropriate.
7. Return valid JSON only. No markdown, no explanation."""

EXTRACTION_USER_TEMPLATE = """Vehicle: {make} {model} {year}.
Extract maintenance schedule from the following manual text. Return a single JSON object with this exact structure (use null for missing optional fields, empty arrays if none):

{{
  "vehicle": {{ "make": "...", "model": "...", "year": {year}, "trim_or_engine": null or "..." }},
  "service_schedule": {{
    "normal_service": [
      {{
        "component": "<one of allowed components>",
        "action": "replace|inspect|check|service",
        "interval_distance_value": number or null,
        "interval_distance_unit": "km" or "miles" or null,
        "interval_time_value": number or null,
        "interval_time_unit": "months|weeks|hours|days|years" or null,
        "conditions": "normal service" or null,
        "found": true,
        "source_page": number or null,
        "source_quote": "exact quote from text",
        "confidence": 0.0-1.0 or null
      }}
    ],
    "severe_service": [ same structure ]
  }}
}}

If a component is not mentioned, do NOT include it in the array, OR include it with "found": false and no interval values.
Manual text (excerpt):

{text}
"""


CHUNK_TARGET_CHARS = 12000
CHUNK_MAX_CHARS = 15000
CHUNK_OVERLAP_CHARS = 1500
MAX_CHUNKS_TO_PROCESS: int | None = None


def _split_blocks(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    blocks = [b.strip() for b in re.split(r"\n\s*\n+", raw) if b.strip()]
    return blocks or [raw]


def _build_chunks(text: str) -> list[str]:
    """Build paragraph-aware chunks with overlap."""
    blocks = _split_blocks(text)
    if not blocks:
        return []

    chunks: list[str] = []
    current: list[str] = []
    cur_len = 0

    for block in blocks:
        bl = len(block)
        sep = 2 if current else 0  # account for "\n\n"
        if current and cur_len + sep + bl > CHUNK_TARGET_CHARS:
            chunks.append("\n\n".join(current))
            # Paragraph-aware overlap from tail of just-closed chunk.
            overlap_blocks: list[str] = []
            overlap_len = 0
            for prev in reversed(current):
                extra = len(prev) + (2 if overlap_blocks else 0)
                if overlap_len + extra > CHUNK_OVERLAP_CHARS:
                    break
                overlap_blocks.insert(0, prev)
                overlap_len += extra
            current = overlap_blocks
            cur_len = overlap_len
        current.append(block)
        cur_len += bl + (2 if cur_len else 0)
        if cur_len > CHUNK_MAX_CHARS:
            chunks.append("\n\n".join(current))
            current = []
            cur_len = 0

    if current:
        chunks.append("\n\n".join(current))

    if MAX_CHUNKS_TO_PROCESS is None or len(chunks) <= MAX_CHUNKS_TO_PROCESS:
        return chunks
    return chunks[:MAX_CHUNKS_TO_PROCESS]


def _extract_single_chunk(
    client: OpenAI,
    vehicle: VehicleIdentity,
    text: str,
    *,
    model: str,
    chunk_idx: int,
    chunk_total: int,
) -> dict:
    log.info(
        "[manual_extractor] processing chunk %s/%s chars=%s",
        chunk_idx + 1,
        chunk_total,
        len(text),
    )
    user_content = EXTRACTION_USER_TEMPLATE.format(
        make=vehicle.make,
        model=vehicle.model,
        year=vehicle.year,
        text=f"[Chunk {chunk_idx + 1}/{chunk_total}]\n\n{text}",
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
    )
    raw = (resp.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    raw = raw.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(raw)
    schedule = parsed.get("service_schedule", {}) if isinstance(parsed, dict) else {}
    normal_count = len((schedule.get("normal_service") or [])) if isinstance(schedule, dict) else 0
    severe_count = len((schedule.get("severe_service") or [])) if isinstance(schedule, dict) else 0
    log.info(
        "[manual_extractor] chunk %s/%s parsed normal=%s severe=%s",
        chunk_idx + 1,
        chunk_total,
        normal_count,
        severe_count,
    )
    return parsed


def _merge_items(items: list[dict]) -> list[dict]:
    merged: dict[tuple, dict] = {}
    for d in items:
        if not isinstance(d, dict):
            continue
        found = bool(d.get("found", True))
        if not found:
            continue
        key = (
            d.get("component"),
            d.get("action"),
            d.get("conditions"),
            d.get("interval_distance_value"),
            d.get("interval_distance_unit"),
            d.get("interval_time_value"),
            d.get("interval_time_unit"),
        )
        prev = merged.get(key)
        if prev is None:
            merged[key] = d
            continue
        # Prefer item with source quote and higher confidence.
        prev_conf = prev.get("confidence") or 0
        new_conf = d.get("confidence") or 0
        prev_quote = bool(prev.get("source_quote"))
        new_quote = bool(d.get("source_quote"))
        if (new_quote and not prev_quote) or (new_conf > prev_conf):
            merged[key] = d
    return list(merged.values())


def extract_rules(
    manual_text: str,
    vehicle: VehicleIdentity,
    *,
    api_key: Optional[str] = None,
    model: str = OPENAI_MODEL,
) -> ExtractedManualRules:
    """Extract maintenance rules from manual text using LLM with schema constraint."""
    log.info(
        "[manual_extractor] start extraction for %s %s %s text_chars=%s",
        vehicle.make,
        vehicle.model,
        vehicle.year,
        len(manual_text or ""),
    )
    client = OpenAI(api_key=api_key or OPENAI_API_KEY)
    chunks = _build_chunks(manual_text)
    if not chunks:
        chunks = [manual_text.strip()[:CHUNK_TARGET_CHARS]]
    log.info(
        "[manual_extractor] built %s chunks (target=%s max=%s overlap=%s)",
        len(chunks),
        CHUNK_TARGET_CHARS,
        CHUNK_MAX_CHARS,
        CHUNK_OVERLAP_CHARS,
    )

    all_normal: list[dict] = []
    all_severe: list[dict] = []
    vehicle_data = {
        "make": vehicle.make,
        "model": vehicle.model,
        "year": vehicle.year,
        "trim_or_engine": vehicle.trim_or_engine,
    }

    for i, chunk in enumerate(chunks):
        data = _extract_single_chunk(
            client,
            vehicle,
            chunk,
            model=model,
            chunk_idx=i,
            chunk_total=len(chunks),
        )
        chunk_vehicle = data.get("vehicle", {}) or {}
        chunk_schedule = data.get("service_schedule", {}) or {}
        if i == 0:
            vehicle_data = {
                "make": chunk_vehicle.get("make", vehicle.make),
                "model": chunk_vehicle.get("model", vehicle.model),
                "year": chunk_vehicle.get("year", vehicle.year),
                "trim_or_engine": chunk_vehicle.get("trim_or_engine"),
            }
        all_normal.extend(chunk_schedule.get("normal_service", []) or [])
        all_severe.extend(chunk_schedule.get("severe_service", []) or [])

    schedule_data = {
        "normal_service": _merge_items(all_normal),
        "severe_service": _merge_items(all_severe),
    }
    log.info(
        "[manual_extractor] merged chunk outputs raw_normal=%s raw_severe=%s dedup_normal=%s dedup_severe=%s",
        len(all_normal),
        len(all_severe),
        len(schedule_data["normal_service"]),
        len(schedule_data["severe_service"]),
    )

    def _norm_item(d: dict) -> dict:
        """Ensure required fields exist so LLM omission (e.g. found:false without action) doesn't crash."""
        return {
            "component": d.get("component", "engine_oil"),
            "action": d.get("action") or "check",
            "interval_distance_value": d.get("interval_distance_value"),
            "interval_distance_unit": d.get("interval_distance_unit"),
            "interval_time_value": d.get("interval_time_value"),
            "interval_time_unit": d.get("interval_time_unit"),
            "conditions": d.get("conditions"),
            "found": bool(d.get("found", True)),
            "source_page": d.get("source_page"),
            "source_quote": d.get("source_quote"),
            "source_url": d.get("source_url"),
            "confidence": d.get("confidence"),
        }

    normal = [
        ExtractedRuleItem(**_norm_item(item))
        for item in schedule_data.get("normal_service", [])
        if isinstance(item, dict)
    ]
    severe = [
        ExtractedRuleItem(**_norm_item(item))
        for item in schedule_data.get("severe_service", [])
        if isinstance(item, dict)
    ]

    result = ExtractedManualRules(
        vehicle=VehicleIdentity(
            make=vehicle_data.get("make", vehicle.make),
            model=vehicle_data.get("model", vehicle.model),
            year=int(vehicle_data.get("year", vehicle.year)),
            trim_or_engine=vehicle_data.get("trim_or_engine"),
        ),
        service_schedule=ServiceSchedule(normal_service=normal, severe_service=severe),
    )
    log.info(
        "[manual_extractor] final extracted found_normal=%s found_severe=%s",
        sum(1 for i in result.service_schedule.normal_service if i.found),
        sum(1 for i in result.service_schedule.severe_service if i.found),
    )
    return result
