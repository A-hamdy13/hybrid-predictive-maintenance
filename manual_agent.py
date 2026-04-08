"""
AI agent that searches the web and follows links until it finds a manual.
Uses Playwright (browser) + LLM to decide which link to open next.

On Windows we explicitly ensure a Proactor event loop policy before starting
Playwright so that subprocess support works correctly under uvicorn/FastAPI.
"""
import asyncio
import json
import logging
import re
import sys
from typing import Optional
from urllib.parse import quote_plus, urlparse, parse_qs, unquote

from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_MODEL
from schemas import VehicleIdentity

log = logging.getLogger(__name__)

# Playwright optional
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    sync_playwright = None
    PLAYWRIGHT_AVAILABLE = False

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
MAX_STEPS = 12
MAX_LINKS_TO_SEND = 25


def _get_search_query(vehicle: VehicleIdentity) -> str:
    """Build initial search query for the vehicle manual (natural phrasing that matches how users search)."""
    return f"{vehicle.make} {vehicle.model} {vehicle.year} owner manual"


def _resolve_ddg_redirect(href: str) -> str:
    """If href is a DuckDuckGo redirect (uddg=), return the target URL."""
    if "uddg=" in href:
        parsed = urlparse(href)
        qs = parse_qs(parsed.query)
        u = qs.get("uddg", [""])[0]
        if u:
            return unquote(u)
    return href


def _resolve_google_redirect(href: str) -> str:
    """If href is a Google redirect (google.com/url?q=...), return the target URL."""
    if "google.com/url" in href or "google.co.uk/url" in href:
        parsed = urlparse(href)
        qs = parse_qs(parsed.query)
        u = qs.get("q", [""])[0]
        if u:
            return unquote(u)
    return href


def _get_links_from_page(page) -> list[dict]:
    """Extract visible links (href + short text) from current page. Returns list of {href, text}."""
    try:
        links = page.locator("a[href^='http']").evaluate_all(
            """els => els.map(el => {
                const text = (el.innerText || '').trim().slice(0, 100);
                const href = el.href;
                return { href, text };
            }).filter(x => x.href);"""
        )
    except Exception:
        return []
    seen = set()
    out = []
    for item in links:
        if not isinstance(item, dict):
            continue
        href = (item.get("href") or "").strip()
        if not href or href in seen:
            continue
        # Resolve search-engine redirects so we get real URLs for the LLM
        if "duckduckgo.com" in href:
            href = _resolve_ddg_redirect(href)
        elif "google.com" in href or "google.co.uk" in href:
            href = _resolve_google_redirect(href)
        if "duckduckgo.com" in href or "google.com" in href or "google.co.uk" in href:
            continue
        seen.add(href)
        text = (item.get("text") or "").strip() or href[:80]
        out.append({"href": href, "text": text})
        if len(out) >= MAX_LINKS_TO_SEND:
            break
    return out


def _ask_llm(
    vehicle: VehicleIdentity,
    page_url: str,
    page_title: str,
    links: list[dict],
    step: int,
    api_key: Optional[str] = None,
) -> dict:
    """Ask LLM: is this the manual, or which link should we open next, or new search?"""
    client = OpenAI(api_key=api_key or OPENAI_API_KEY)
    links_text = "\n".join(
        f"{i}. [{links[i]['text'][:60]}] {links[i]['href'][:80]}"
        for i in range(len(links))
    )
    prompt = f"""We are looking for an OFFICIAL maintenance schedule or owner manual (PDF or web page with service intervals) for this vehicle: {vehicle.make} {vehicle.model} {vehicle.year}.

Current page:
- URL: {page_url}
- Title: {page_title}

Links on this page (index, text, url):
{links_text}

Reply with JSON only. Choose one:
1) If this page IS the actual manual content or a direct PDF, set "found_manual": true.
2) If this page only LISTS manuals (e.g. "Owner's Manual", "Open" links to PDFs), set "link_index": <0-based index> to open the best PDF or manual link—do not set found_manual for a hub page.
3) If we need a different search, set "new_search_query": "exact search string".

Prefer opening a .pdf link when present. Return only valid JSON, e.g. {{ "found_manual": true }} or {{ "link_index": 2 }} or {{ "new_search_query": "..." }}."""

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = (resp.choices[0].message.content or "").strip()
    raw = re.sub(r"^```\w*\n?", "", raw)
    raw = re.sub(r"\n?```\s*$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"found_manual": False}


def run_manual_finder_agent(
    vehicle: VehicleIdentity,
    *,
    api_key: Optional[str] = None,
    max_steps: int = MAX_STEPS,
    headless: bool = True,
) -> Optional[str]:
    """
    Use a browser + LLM to search and follow links until we find a manual URL.
    Returns the URL of the manual (PDF or page) or None if not found / Playwright unavailable.
    """
    if not PLAYWRIGHT_AVAILABLE or not (api_key or OPENAI_API_KEY):
        log.info("[manual_agent] Skipping agent: PLAYWRIGHT_AVAILABLE=%s, has_api_key=%s", PLAYWRIGHT_AVAILABLE, bool(api_key or OPENAI_API_KEY))
        return None

    log.info("[manual_agent] Starting agent for %s %s %s", vehicle.make, vehicle.model, vehicle.year)

    # Ensure a Proactor loop policy on Windows so Playwright subprocesses work.
    if sys.platform.startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())  # type: ignore[attr-defined]
        except Exception as e:
            log.warning("[manual_agent] WindowsProactorEventLoopPolicy failed: %s", e)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless, args=["--no-sandbox"])
            context = browser.new_context(user_agent=USER_AGENT)
            context.set_default_timeout(20000)
            page = context.new_page()

            try:
                query = _get_search_query(vehicle)
                search_url = "https://www.google.com/search?q=" + quote_plus(query)
                log.info("[manual_agent] Step 0: opening Google search %r", query)
                page.goto(search_url, wait_until="domcontentloaded")
                page.wait_for_timeout(2000)

                for step in range(max_steps):
                    current_url = page.url
                    title = page.title()
                    log.info("[manual_agent] Step %s: url=%s title=%s", step, current_url[:80], title[:50] if title else "")

                    # If Google is blocking automation (captcha / sorry page), bail out so we can fall back to LLM URLs.
                    if "www.google.com/sorry/" in current_url:
                        log.warning("[manual_agent] Hit Google 'sorry' / captcha page; aborting agent.")
                        return None

                    if current_url.lower().rstrip("/").endswith(".pdf"):
                        log.info("[manual_agent] Found PDF URL: %s", current_url)
                        return current_url
                    if ".pdf?" in current_url.lower():
                        log.info("[manual_agent] Found PDF (query): %s", current_url)
                        return current_url

                    links = _get_links_from_page(page)
                    if not links and step == 0:
                        try:
                            for a in page.locator("a.result__a").all()[:15]:
                                href = a.get_attribute("href")
                                if href:
                                    href = _resolve_ddg_redirect(href)
                                if href and href.startswith("http") and "duckduckgo" not in href:
                                    links.append({"href": href, "text": (a.inner_text() or "")[:80]})
                        except Exception as e:
                            log.debug("[manual_agent] result__a fallback failed: %s", e)

                    log.info("[manual_agent] Step %s: %s links on page", step, len(links))
                    for i, L in enumerate(links[:5]):
                        log.info("[manual_agent]   link[%s] %s | %s", i, (L.get("href") or "")[:70], (L.get("text") or "")[:40])

                    decision = _ask_llm(
                        vehicle, current_url, title, links,
                        step, api_key=api_key or OPENAI_API_KEY,
                    )
                    log.info("[manual_agent] LLM decision: %s", decision)

                    if decision.get("found_manual"):
                        log.info("[manual_agent] LLM said found_manual -> return %s", current_url[:80])
                        return current_url

                    if "link_index" in decision:
                        idx = int(decision["link_index"])
                        if 0 <= idx < len(links):
                            next_url = links[idx]["href"]
                            log.info("[manual_agent] Following link_index %s -> %s", idx, next_url[:80])
                            try:
                                page.goto(next_url, wait_until="domcontentloaded")
                                page.wait_for_timeout(1500)
                            except Exception as e:
                                log.warning("[manual_agent] goto failed: %s", e)
                            continue

                    if decision.get("new_search_query"):
                        q = decision["new_search_query"]
                        search_url = "https://www.google.com/search?q=" + quote_plus(q)
                        log.info("[manual_agent] New search: %s", q[:60])
                        try:
                            page.goto(search_url, wait_until="domcontentloaded")
                            page.wait_for_timeout(2000)
                        except Exception as e:
                            log.warning("[manual_agent] goto search failed: %s", e)
                        continue

                    log.info("[manual_agent] No action from LLM, stopping")
                    break

                log.info("[manual_agent] Exhausted steps or no action, returning None")
                return None

            finally:
                context.close()
                browser.close()
    except Exception as e:
        log.exception("[manual_agent] Agent failed: %s", e)
        return None


def is_agent_available() -> bool:
    """Return True if Playwright is installed and agent can run."""
    return PLAYWRIGHT_AVAILABLE and bool(OPENAI_API_KEY)
