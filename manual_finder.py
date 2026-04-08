"""Find official manual URLs for a vehicle using OpenAI Responses API + web search only."""
import logging
from typing import Optional

from schemas import ManualCandidate, VehicleIdentity

from openai_web_search import find_manual_urls_with_web_search

log = logging.getLogger(__name__)


def find_manual_candidates(
    vehicle: VehicleIdentity,
    *,
    api_key: Optional[str] = None,
    use_llm_search: bool = True,
) -> list[ManualCandidate]:
    """Return candidate manual URLs for the vehicle via OpenAI web search only."""
    _ = api_key  # OpenAI key is read in openai_web_search from config
    if not use_llm_search:
        return []
    urls = find_manual_urls_with_web_search(vehicle)
    return [
        ManualCandidate(
            url=u,
            title="Manual",
            source_type="owner_manual",
            is_official=True,
        )
        for u in urls
    ]


def resolve_fetchable_urls(
    vehicle: VehicleIdentity,
    *,
    api_key: Optional[str] = None,
) -> list[str]:
    """Return URLs to try fetching. Uses only OpenAI Responses API + web_search tool."""
    _ = api_key
    log.info("[manual_finder] resolve_fetchable_urls vehicle=%s %s %s (OpenAI web search)", vehicle.make, vehicle.model, vehicle.year)
    urls = find_manual_urls_with_web_search(vehicle)
    out = urls[:15]
    log.info("[manual_finder] resolve_fetchable_urls returning %s URLs", len(out))
    for i, u in enumerate(out):
        log.info("[manual_finder]   [%s] %s", i, u[:90])
    return out
