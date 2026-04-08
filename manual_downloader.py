"""Download manual from URL and extract text using Unstructured only."""
import hashlib
import logging
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from config import MANUALS_CACHE_DIR, UNSTRUCTURED_API_KEY, UNSTRUCTURED_API_URL

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
log = logging.getLogger(__name__)


def _cache_path(url: str) -> Path:
    key = hashlib.sha256(url.encode()).hexdigest()[:16]
    return MANUALS_CACHE_DIR / f"{key}"


def _download_bytes(url: str, *, timeout: int = 30) -> bytes:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    return r.content


def _detect_format(url: str, raw: bytes, content_type_header: str | None = None) -> tuple[str, str]:
    """Return (format_label, extension_without_dot)."""
    parsed = urlparse(url)
    path_lower = (parsed.path or "").lower()
    cth = (content_type_header or "").lower()

    if path_lower.endswith(".pdf") or raw[:4] == b"%PDF" or "application/pdf" in cth:
        return "pdf", "pdf"
    if path_lower.endswith(".html") or path_lower.endswith(".htm") or "text/html" in cth:
        return "html", "html"
    if path_lower.endswith(".txt") or "text/plain" in cth:
        return "txt", "txt"
    if path_lower.endswith(".docx") or "wordprocessingml.document" in cth:
        return "docx", "docx"
    if path_lower.endswith(".doc") or "msword" in cth:
        return "doc", "doc"
    if path_lower.endswith(".pptx") or "presentationml.presentation" in cth:
        return "pptx", "pptx"
    if path_lower.endswith(".ppt") or "ms-powerpoint" in cth:
        return "ppt", "ppt"
    if path_lower.endswith(".xlsx") or "spreadsheetml.sheet" in cth:
        return "xlsx", "xlsx"
    if path_lower.endswith(".xls") or "vnd.ms-excel" in cth:
        return "xls", "xls"
    if path_lower.endswith(".xml") or "application/xml" in cth or "text/xml" in cth:
        return "xml", "xml"
    if path_lower.endswith(".md") or "text/markdown" in cth:
        return "md", "md"
    return "unknown", "bin"


def _elements_to_text(elements: list[dict]) -> str:
    parts: list[str] = []
    for el in elements:
        text = (el or {}).get("text")
        if isinstance(text, str):
            cleaned = text.strip()
            if cleaned:
                parts.append(cleaned)
    return "\n".join(parts)


def _partition_with_unstructured(raw: bytes, *, filename: str, content_type: str | None = None) -> str:
    """
    Parse file bytes via Unstructured legacy partition endpoint.
    Returns extracted text (may be empty).
    """
    if not UNSTRUCTURED_API_KEY:
        raise RuntimeError("UNSTRUCTURED_API_KEY not set")

    files = {"files": (filename, raw, content_type or "application/octet-stream")}
    data = {
        "output_format": "application/json",
        "strategy": "auto",
    }
    headers = {
        "accept": "application/json",
        "unstructured-api-key": UNSTRUCTURED_API_KEY,
        "User-Agent": USER_AGENT,
    }

    last_error: Exception | None = None
    # Strict mode with lightweight retries for transient upstream failures.
    for _attempt in range(3):
        log.info(
            "[manual_downloader] Unstructured parse attempt %s/3 file=%s bytes=%s",
            _attempt + 1,
            filename,
            len(raw),
        )
        try:
            r = requests.post(
                UNSTRUCTURED_API_URL,
                files=files,
                data=data,
                headers=headers,
                timeout=180,
            )
            if r.status_code >= 400:
                body = (r.text or "").strip()
                log.warning(
                    "[manual_downloader] Unstructured HTTP %s on attempt %s: %s",
                    r.status_code,
                    _attempt + 1,
                    body[:220],
                )
                if r.status_code in (429, 500, 502, 503, 504):
                    last_error = RuntimeError(f"Unstructured API {r.status_code}: {body[:300]}")
                    continue
                raise RuntimeError(f"Unstructured API {r.status_code}: {body[:500]}")
            payload = r.json()
            log.info("[manual_downloader] Unstructured parse succeeded on attempt %s", _attempt + 1)
            break
        except requests.RequestException as e:
            log.warning("[manual_downloader] Unstructured request exception on attempt %s: %s", _attempt + 1, e)
            last_error = e
            continue
    else:
        raise RuntimeError(f"Unstructured request failed after retries: {last_error}")

    if not isinstance(payload, list):
        raise RuntimeError("Unstructured API returned unexpected response format")
    text = _elements_to_text(payload)
    if not text.strip():
        raise RuntimeError("Unstructured API returned empty parsed text")
    log.info(
        "[manual_downloader] Unstructured returned elements=%s text_chars=%s",
        len(payload),
        len(text),
    )
    return text


def fetch_manual_text(url: str, *, use_cache: bool = True) -> tuple[str, Optional[Path], str]:
    """Fetch URL and return (extracted_text, path_to_cached_file_if_any, content_type).
    content_type is 'pdf', 'html', or 'unknown'.
    """
    if use_cache:
        cached = _cache_path(url)
        if cached.with_suffix(".txt").exists():
            log.info("[manual_downloader] cache hit text url=%s", url[:120])
            return cached.with_suffix(".txt").read_text(encoding="utf-8", errors="replace"), cached, "cached"
        if cached.with_suffix(".pdf").exists():
            log.info("[manual_downloader] cache hit pdf url=%s", url[:120])
            raw = cached.with_suffix(".pdf").read_bytes()
            text = _partition_with_unstructured(raw, filename="manual.pdf", content_type="application/pdf")
            cached.with_suffix(".txt").write_text(text, encoding="utf-8")
            return text, cached, "pdf"

    log.info("[manual_downloader] downloading url=%s", url[:140])
    resp = requests.get(url, timeout=30, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    raw = resp.content
    format_label, ext = _detect_format(url, raw, resp.headers.get("Content-Type"))
    filename = f"manual.{ext}"
    log.info(
        "[manual_downloader] downloaded bytes=%s detected_type=%s content_type=%s",
        len(raw),
        format_label,
        (resp.headers.get("Content-Type") or "")[:80],
    )

    # Strict parser: Unstructured API only. Any error should bubble up.
    text = _partition_with_unstructured(raw, filename=filename, content_type=resp.headers.get("Content-Type"))

    if use_cache:
        p = _cache_path(url)
        p.with_suffix(f".{ext}").write_bytes(raw)
        p.with_suffix(".txt").write_text(text, encoding="utf-8")
        log.info("[manual_downloader] cached parsed text at %s.txt", p)
    log.info("[manual_downloader] parsed text chars=%s for url=%s", len(text), url[:120])
    return text, None, format_label


def get_manual_text_from_url(url: str) -> str:
    """Convenience: return only the extracted text."""
    text, _, _ = fetch_manual_text(url)
    return text


def fetch_manual_text_and_type(url: str, *, use_cache: bool = True) -> tuple[str, str]:
    """Fetch URL and return (extracted_text, content_type). content_type is 'pdf', 'html', or 'cached'."""
    text, _, content_type = fetch_manual_text(url, use_cache=use_cache)
    return text, content_type


def extract_pdf_links_from_page(url: str, *, timeout: int = 15) -> list[str]:
    """If URL is an HTML page, fetch it and return absolute URLs of links that point to PDFs.
    Prefer links whose text/href suggest 'manual', 'maintenance', 'schedule', 'owner'.
    """
    try:
        raw = _download_bytes(url, timeout=timeout)
    except Exception:
        return []
    if raw[:4] == b"%PDF":
        return []
    soup = BeautifulSoup(raw, "html.parser")
    base = url.rsplit("/", 1)[0] + "/" if "/" in urlparse(url).path else url + "/"
    base = urljoin(url, base)
    candidates: list[tuple[str, int]] = []  # (url, score)
    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href or not href.lower().endswith(".pdf"):
            continue
        full = urljoin(url, href)
        if full == url:
            continue
        text_lower = (a.get_text() or "").lower()
        score = 0
        for kw in ("maintenance", "schedule", "owner", "manual", "service"):
            if kw in text_lower or kw in href.lower():
                score += 1
        candidates.append((full, score))
    candidates.sort(key=lambda x: -x[1])
    return [u for u, _ in candidates[:10]]
