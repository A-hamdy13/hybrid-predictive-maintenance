"""
Find manual URLs using OpenAI Responses API with the web_search tool.
Uses the same capability as ChatGPT "search the web" – no Bing/DuckDuckGo/Playwright.
"""
import logging
from typing import List

from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_MODEL
from schemas import VehicleIdentity

log = logging.getLogger(__name__)


def find_manual_urls_with_web_search(vehicle: VehicleIdentity) -> List[str]:
    """
    Use OpenAI Responses API with web_search tool to find official owner manual
    or maintenance schedule URLs (PDF or web page) for the vehicle.
    Returns a deduplicated list of URLs to try (download links preferred).
    """
    if not OPENAI_API_KEY:
        log.warning("[openai_web_search] OPENAI_API_KEY not set")
        return []

    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = f"""Find the official owner's manual or maintenance schedule for this vehicle: {vehicle.make} {vehicle.model} {vehicle.year}.

Use web search to find:
1. The manufacturer's official page where you can view or download the owner manual (PDF or HTML).
2. Any direct PDF download links for the owner manual or maintenance schedule.

Return the actual URLs that a user can open to get the manual or PDF. Prefer official manufacturer domains (e.g. ford.com, toyota.com). List the best 1–5 URLs, with direct PDF links first if you find them."""

    try:
        # Responses API with web_search tool (same as ChatGPT "search the web")
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            tools=[{"type": "web_search"}],
            tool_choice={"type": "web_search"},
            include=["web_search_call.action.sources"],
        )
    except Exception as e:
        log.warning("[openai_web_search] responses.create failed: %s", e)
        raise RuntimeError(f"OpenAI web search request failed: {e}") from e

    urls: List[str] = []
    seen: set[str] = set()

    # Response has .output as list of items (SDK may return list of objects or dicts)
    output = getattr(response, "output", None)
    if not output:
        # Log raw response for debugging (e.g. model may not support web_search)
        try:
            raw = getattr(response, "model_dump", lambda: str(response))()
            log.warning("[openai_web_search] No output in response. Response keys: %s", getattr(response, "__dict__", raw) if not callable(raw) else raw)
        except Exception:
            log.warning("[openai_web_search] No output in response")
        return []

    def _get(o, key, default=None):
        if o is None:
            return default
        if isinstance(o, dict):
            return o.get(key, default)
        return getattr(o, key, default)

    for item in output:
        kind = _get(item, "type")
        # Web search call: action.sources[] has { "type": "url", "url": "..." }
        if kind == "web_search_call":
            action = _get(item, "action")
            sources = _get(action, "sources") or []
            for s in sources:
                u = (_get(s, "url") or "").strip()
                if u and u not in seen:
                    seen.add(u)
                    urls.append(u)
        # Message: content[].annotations[] with type url_citation have .url
        if kind == "message":
            content = _get(item, "content") or []
            for block in content:
                for ann in _get(block, "annotations") or []:
                    if _get(ann, "type") == "url_citation":
                        u = (_get(ann, "url") or "").strip()
                        if u and u not in seen:
                            seen.add(u)
                            urls.append(u)

    if not urls:
        # Help debug: log output item types
        try:
            types = [_get(it, "type") or type(it).__name__ for it in output]
            log.warning("[openai_web_search] Parsed 0 URLs. Output item types: %s", types)
        except Exception:
            log.warning("[openai_web_search] Parsed 0 URLs")
    else:
        log.info("[openai_web_search] Found %s URLs for %s %s %s", len(urls), vehicle.make, vehicle.model, vehicle.year)
        for i, u in enumerate(urls):
            log.info("[openai_web_search]   [%s] %s", i, u[:90])
    return urls
