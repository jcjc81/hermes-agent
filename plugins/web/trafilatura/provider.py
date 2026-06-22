"""Trafilatura extract provider — local, static HTML → Markdown.

Fetches pages with httpx, extracts main content via trafilatura,
applies a quality gate, and returns error results for thin/blocked
content so the dispatcher falls back to Firecrawl.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

# --- constants ---

_FETCH_TIMEOUT = 20  # seconds
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

# Truly unextractable binary content types — short-circuit before download.
# PDFs are intentionally NOT blocked (Firecrawl can extract them).
_UNEXTRACTABLE_CONTENT_TYPES = {
    "image/",
    "video/",
    "audio/",
    "application/zip",
    "application/octet-stream",
}

# Block-page signatures — only matched against thin text (<500 chars).
# Longer articles mentioning these words are legitimate content.
_BLOCK_SIGNATURES = [
    "enable javascript",
    "just a moment",
    "checking your browser",
    "cf-browser-verification",
    "verify you are human",
    "captcha",
    "access denied",
]

# Thresholds
_THIN_CONTENT_CHARS = 200
_BLOCK_SIGNATURE_TEXT_CHARS = 500


def _load_safety_tools():
    """Lazy-import safety helpers to avoid circular deps at module load."""
    from tools.url_safety import is_safe_url
    from tools.website_policy import check_website_access
    return is_safe_url, check_website_access


def _check_content_type_short_circuit(headers: Any) -> Optional[str]:
    """Return an error string if the content-type is a truly unextractable binary."""
    ct = str(headers.get("content-type", "")).lower()
    for prefix in _UNEXTRACTABLE_CONTENT_TYPES:
        if ct.startswith(prefix):
            return f"non-extractable content-type: {ct}"
    return None


def _is_block_page(text: str) -> bool:
    """Check if extracted text has block-page signatures (case-insensitive)."""
    low = text.lower()
    for sig in _BLOCK_SIGNATURES:
        if sig in low:
            return True
    return False


def _has_structured_signal(result: Dict[str, Any]) -> bool:
    """Check if the extracted result carries title/metadata signal."""
    title = result.get("title", "")
    metadata = result.get("metadata")
    if title and title.strip():
        return True
    if metadata and isinstance(metadata, dict):
        for key in ("title", "author", "date", "description"):
            if metadata.get(key):
                return True
    return False


class TrafilaturaExtractProvider(WebSearchProvider):
    """Local extract provider using Trafilatura for static page extraction."""

    @property
    def name(self) -> str:
        return "trafilatura"

    @property
    def display_name(self) -> str:
        return "Trafilatura (self-hosted)"

    def is_available(self) -> bool:
        """Return True when trafilatura is importable."""
        try:
            import trafilatura  # noqa: F401
            import httpx  # noqa: F401
            return True
        except ImportError:
            return False

    def supports_search(self) -> bool:
        return False

    def supports_extract(self) -> bool:
        return True

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract content from URLs using Trafilatura.

        Returns one entry per input URL in order. Each entry has:
        - On success: {url, title, content, raw_content, metadata, ...}
        - On failure: {url, error: "..."}

        The provider always echoes the *requested* URL (not post-redirect)
        so the dispatcher can positionally align results.
        """
        is_safe_url, check_website_access = _load_safety_tools()

        results: List[Dict[str, Any]] = []

        for requested_url in urls:
            result = self._extract_single(
                requested_url, is_safe_url, check_website_access
            )
            results.append(result)

        return results

    def _extract_single(
        self,
        url: str,
        is_safe_url,
        check_website_access,
    ) -> Dict[str, Any]:
        """Extract a single URL. Returns result dict with requested URL echoed."""

        # Pre-fetch policy gate
        block_info = check_website_access(url)
        if block_info is not None:
            logger.info(
                "Blocked by website policy: %s", url
            )
            return {"url": url, "error": f"blocked by website policy: {block_info.get('reason', 'blocked')}"}

        try:
            response = httpx.get(
                url,
                headers={"User-Agent": _USER_AGENT},
                timeout=_FETCH_TIMEOUT,
                follow_redirects=True,
            )
        except httpx.TimeoutException:
            return {"url": url, "error": f"timeout after {_FETCH_TIMEOUT}s"}
        except httpx.ConnectError as exc:
            return {"url": url, "error": f"connection error: {exc}"}
        except Exception as exc:
            return {"url": url, "error": f"fetch error: {exc}"}

        # Check for error status codes
        if response.status_code in (403, 429):
            return {"url": url, "error": f"HTTP {response.status_code}"}
        if 500 <= response.status_code < 600:
            return {"url": url, "error": f"HTTP {response.status_code}"}

        # SSRF re-check: validate ALL redirect hops
        all_urls_to_check = [url]
        if response.history:
            for hop in response.history:
                all_urls_to_check.append(str(hop.url))
            all_urls_to_check.append(str(response.url))

        for checked_url in all_urls_to_check:
            if not is_safe_url(checked_url):
                return {
                    "url": url,
                    "error": "blocked: redirect to private address",
                }

        # Post-redirect policy gate
        final_url = str(response.url)
        if final_url != url:
            block_info = check_website_access(final_url)
            if block_info is not None:
                return {"url": url, "error": f"blocked by website policy (redirect): {block_info.get('reason', 'blocked')}"}

        # Content-type short-circuit for truly unextractable binaries
        ct_error = _check_content_type_short_circuit(response.headers)
        if ct_error is not None:
            return {"url": url, "error": ct_error}

        # Extract with trafilatura — pass bytes, not text
        try:
            import trafilatura
        except ImportError:
            return {"url": url, "error": "trafilatura not installed"}

        text = trafilatura.extract(
            response.content,
            output_format="markdown",
            include_links=True,
        )

        # Trafilatura returns None when no main content found
        if text is None:
            return {"url": url, "error": "no main content found (likely JS-rendered page)"}

        # Get metadata
        raw_metadata = trafilatura.extract_metadata(response.content)
        # extract_metadata can return a Document object or dict
        if hasattr(raw_metadata, 'as_dict'):
            metadata = raw_metadata.as_dict() or {}
        elif isinstance(raw_metadata, dict):
            metadata = raw_metadata
        else:
            metadata = {}

        title = metadata.get("title", "") or ""

        # Quality gate: thin content with no structured signal → fall back
        text_len = len(text.strip())
        if text_len < _THIN_CONTENT_CHARS and not _has_structured_signal({
            "title": title,
            "metadata": metadata,
        }):
            return {"url": url, "error": f"thin content ({text_len} chars, no structured signal)"}

        # Block-page signature check — only for thin-ish text
        if text_len < _BLOCK_SIGNATURE_TEXT_CHARS and _is_block_page(text):
            return {"url": url, "error": "block page detected (bot wall / captcha)"}

        return {
            "url": url,
            "title": title,
            "content": text,
            "raw_content": text,
            "metadata": metadata,
        }
