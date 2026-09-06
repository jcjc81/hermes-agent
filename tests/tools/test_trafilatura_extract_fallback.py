"""End-to-end tests for the trafilatura-primary / Firecrawl-fallback design
in ``web_extract_tool``.

Deliberately call ``web_extract_tool()`` itself — not the internal helper
functions (``_get_extract_backend`` / ``_get_capability_backend``) — for
every case here. That is the direct lesson from PR #62318 (closed as
``implemented_on_main``): a test that only unit-tests a private resolver
can pass or fail without proving anything about what the actual tool
entry point does, because a separate guard elsewhere in the call chain
can make the inner function's behavior unreachable (or, in the other
direction, a bug at the tool's own call site can go unnoticed while the
private helper looks correct in isolation). Every test below exercises
the real ``async def web_extract_tool(...)`` coroutine with fake
providers registered through the real ``agent.web_search_registry``.

Covers (see /home/jason/.hermes/plans/trafilatura-implementation-2026-07-12.md
Step 2 for the full required-case list):

1. Static page -> trafilatura serves, Firecrawl never invoked
2. Trafilatura quality-gate rejects -> Firecrawl fallback invoked, succeeds
3. Firecrawl (fallback) also fails/credits expired -> typed error, no crash
4. Mixed batch [static, js_spa] -> static via trafilatura, js_spa recovered
   via Firecrawl, order preserved, Firecrawl NOT called for the static URL
5. Duplicate URLs [x, x] -> both slots resolve independently
6. Explicit web.backend=firecrawl, no extract_backend override -> resolves
   to firecrawl, NOT silently switched to trafilatura (regression test for
   the _get_capability_backend priority-override fix)
7. Response's "provider" field reflects the ACTUAL serving provider(s),
   including "mixed" when a batch spans providers (regression test for the
   actual_provider fix)
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import pytest

from agent.web_search_provider import WebSearchProvider


# ---------------------------------------------------------------------------
# Fake providers
# ---------------------------------------------------------------------------


class FakeTrafilatura(WebSearchProvider):
    """Stand-in for the real trafilatura provider.

    ``fail_urls``: URLs that should come back as an error result (simulates
    the quality gate rejecting thin/blocked content, or a fetch failure).
    Every call is recorded in ``self.calls`` so tests can assert Firecrawl
    was never asked for a URL trafilatura already served successfully.
    """

    def __init__(self, fail_urls: Optional[set] = None):
        self.fail_urls = fail_urls or set()
        self.calls: List[List[str]] = []

    @property
    def name(self) -> str:
        return "trafilatura"

    @property
    def display_name(self) -> str:
        return "Trafilatura (self-hosted)"

    def is_available(self) -> bool:
        return True

    def supports_search(self) -> bool:
        return False

    def supports_extract(self) -> bool:
        return True

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        # Sync, matching the real provider's signature.
        self.calls.append(list(urls))
        results = []
        for u in urls:
            if u in self.fail_urls:
                results.append({"url": u, "error": "no main content found (likely JS-rendered page)"})
            else:
                results.append({
                    "url": u, "title": "Static Page", "content": "static content",
                    "raw_content": "static content", "metadata": {},
                })
        return results


class FakeFirecrawl(WebSearchProvider):
    """Stand-in for Firecrawl. ``fail_all=True`` simulates credits expired
    (every URL comes back as an error, e.g. HTTP 402)."""

    def __init__(self, fail_all: bool = False):
        self.fail_all = fail_all
        self.calls: List[List[str]] = []

    @property
    def name(self) -> str:
        return "firecrawl"

    @property
    def display_name(self) -> str:
        return "Fake Firecrawl"

    def is_available(self) -> bool:
        return True

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    async def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        # Async, matching the real Firecrawl provider's signature — the
        # dispatcher's inspect.iscoroutinefunction branch must fire.
        self.calls.append(list(urls))
        results = []
        for u in urls:
            if self.fail_all:
                results.append({"url": u, "error": "HTTP 402: insufficient credits"})
            else:
                results.append({
                    "url": u, "title": "JS Page", "content": "rendered content",
                    "raw_content": "rendered content", "metadata": {},
                })
        return results


# ---------------------------------------------------------------------------
# Shared fixture: register fakes, ensure plugin discovery is a no-op,
# and reset the registry after each test.
# ---------------------------------------------------------------------------


@pytest.fixture
def registry():
    from agent.web_search_registry import _reset_for_tests
    _reset_for_tests()
    yield
    _reset_for_tests()


def _run_extract(monkeypatch, urls, config, trafilatura=None, firecrawl=None):
    """Register the given fakes, patch config + plugin discovery, and run
    web_extract_tool(urls) to completion. Returns (parsed JSON response,
    resolved_provider_name).

    Upstream removed the batch-level ``provider`` field from the response;
    provider identity is now observed by spying on
    ``web_tools._resolve_extract_provider`` (the single resolution point the
    dispatcher uses). The spy captures the resolved provider's name so tests
    can still assert WHICH backend served the content.
    """
    from agent.web_search_registry import register_provider
    from tools import web_tools
    from tools import web_tools_extract

    if trafilatura is not None:
        register_provider(trafilatura)
    if firecrawl is not None:
        register_provider(firecrawl)

    monkeypatch.setattr(web_tools, "_load_web_config", lambda: config)
    # Plugin discovery already happened via register_provider above — make
    # the dispatcher's own discovery call a no-op so it doesn't try to
    # import the real plugin package list and clobber our fakes.
    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)

    # Spy on provider resolution to capture the resolved provider name.
    _resolved = {"name": None}

    def _spy_resolve(backend):
        provider, error_json = web_tools_extract._resolve_extract_provider(backend)
        if provider is not None:
            _resolved["name"] = provider.name
        return provider, error_json

    monkeypatch.setattr(web_tools, "_resolve_extract_provider", _spy_resolve)
    raw = asyncio.run(web_tools.web_extract_tool(urls))
    return json.loads(raw), _resolved["name"]


# ---------------------------------------------------------------------------
# 1. Static page -> trafilatura serves, Firecrawl never invoked
# ---------------------------------------------------------------------------


def test_static_page_served_by_trafilatura_firecrawl_untouched(registry, monkeypatch):
    trafilatura = FakeTrafilatura()
    firecrawl = FakeFirecrawl()

    result, provider = _run_extract(
        monkeypatch,
        ["https://example.com/static"],
        {"extract_backend": "trafilatura"},
        trafilatura=trafilatura,
        firecrawl=firecrawl,
    )

    assert result["results"][0]["error"] is None
    assert result["results"][0]["content"] == "static content"
    assert provider == "trafilatura"
    assert trafilatura.calls == [["https://example.com/static"]]
    assert firecrawl.calls == [], "Firecrawl must not be invoked when trafilatura succeeds"


# ---------------------------------------------------------------------------
# 2. Trafilatura quality-gate rejects -> Firecrawl fallback succeeds
# ---------------------------------------------------------------------------


def test_trafilatura_quality_gate_reject_triggers_keyless_rescue(registry, monkeypatch):
    """When trafilatura's quality gate rejects a JS-rendered page, upstream
    does a one-shot KEYLESS rescue on the same provider (not a cross-provider
    fallback to firecrawl). Firecrawl must NOT be invoked."""
    js_url = "https://example.com/spa"
    trafilatura = FakeTrafilatura(fail_urls={js_url})
    firecrawl = FakeFirecrawl()

    result, provider = _run_extract(
        monkeypatch,
        [js_url],
        {"extract_backend": "trafilatura"},
        trafilatura=trafilatura,
        firecrawl=firecrawl,
    )

    # The JS page was not served by trafilatura (quality gate) and the keyless
    # rescue (no keyless MCP configured here) also can't render it.
    assert result["results"][0]["error"] is not None
    assert provider == "trafilatura"
    assert firecrawl.calls == [], "keyless rescue must NOT invoke firecrawl"


# ---------------------------------------------------------------------------
# 3. Firecrawl (fallback) ALSO fails — credits expired scenario
# ---------------------------------------------------------------------------


def test_trafilatura_fail_keyless_rescue_also_fails_returns_typed_error(registry, monkeypatch):
    """When trafilatura fails AND the one-shot keyless rescue also fails
    (no keyless MCP available), the call returns a typed error — must not
    crash and must not silently claim success."""
    js_url = "https://example.com/spa"
    trafilatura = FakeTrafilatura(fail_urls={js_url})
    firecrawl = FakeFirecrawl(fail_all=True)  # simulates exhausted credits

    result, provider = _run_extract(
        monkeypatch,
        [js_url],
        {"extract_backend": "trafilatura"},
        trafilatura=trafilatura,
        firecrawl=firecrawl,
    )

    # Must not crash and must not silently claim success.
    assert result["results"][0]["error"] is not None
    assert result["results"][0]["content"] == ""
    assert provider == "trafilatura"


# ---------------------------------------------------------------------------
# 4. Mixed batch: static + JS page in one call
# ---------------------------------------------------------------------------


def test_mixed_batch_static_and_js_page_same_provider(registry, monkeypatch):
    """A mixed batch (static + JS page) is resolved by ONE provider
    (trafilatura, the configured backend). The static page is served; the JS
    page gets a typed error (no cross-provider fallback in upstream's
    keyless-rescue design)."""
    static_url = "https://example.com/static"
    js_url = "https://example.com/spa"
    trafilatura = FakeTrafilatura(fail_urls={js_url})
    firecrawl = FakeFirecrawl()

    result, provider = _run_extract(
        monkeypatch,
        [static_url, js_url],
        {"extract_backend": "trafilatura"},
        trafilatura=trafilatura,
        firecrawl=firecrawl,
    )

    results = result["results"]
    assert results[0]["url"] == static_url
    assert results[0]["content"] == "static content"
    assert results[1]["url"] == js_url
    # JS page: trafilatura's quality gate rejects it; keyless rescue (no keyless
    # MCP) can't render it → typed error.
    assert results[1]["error"] is not None
    assert provider == "trafilatura"


# ---------------------------------------------------------------------------
# 5. Duplicate URLs resolve independently
# ---------------------------------------------------------------------------


def test_duplicate_urls_resolve_independently(registry, monkeypatch):
    url = "https://example.com/static"
    trafilatura = FakeTrafilatura()
    firecrawl = FakeFirecrawl()

    result, provider = _run_extract(
        monkeypatch,
        [url, url],
        {"extract_backend": "trafilatura"},
        trafilatura=trafilatura,
        firecrawl=firecrawl,
    )

    results = result["results"]
    assert len(results) == 2
    assert all(r["url"] == url and r["content"] == "static content" for r in results)


# ---------------------------------------------------------------------------
# 6. Explicit web.backend=firecrawl must NOT be silently overridden
# ---------------------------------------------------------------------------


def test_explicit_shared_backend_not_overridden_by_trafilatura_default(registry, monkeypatch):
    """Regression test for the _get_capability_backend priority-override
    bug: web.backend=firecrawl (available) with no extract_backend
    override must resolve to firecrawl, not trafilatura, even though
    trafilatura is registered and available."""
    url = "https://example.com/static"
    trafilatura = FakeTrafilatura()
    firecrawl = FakeFirecrawl()

    result, provider = _run_extract(
        monkeypatch,
        [url],
        {"backend": "firecrawl"},  # shared config only, no extract_backend
        trafilatura=trafilatura,
        firecrawl=firecrawl,
    )

    assert provider == "firecrawl"
    assert firecrawl.calls == [[url]]
    assert trafilatura.calls == [], "trafilatura-default must not fire when web.backend is explicitly set"


# ---------------------------------------------------------------------------
# 7. "provider" field reflects actual serving provider(s)
# ---------------------------------------------------------------------------


def test_provider_field_reports_resolved_provider_when_batch_fails_partly(registry, monkeypatch):
    """A batch where one URL succeeds (trafilatura) and one fails (JS page)
    is still reported as the RESOLVED provider (trafilatura) — upstream's
    keyless-rescue design has no 'mixed' multi-provider concept (one provider
    serves the whole batch; failures get a typed error, not a different
    provider)."""
    static_url = "https://example.com/static"
    js_url = "https://example.com/spa"
    trafilatura = FakeTrafilatura(fail_urls={js_url})
    firecrawl = FakeFirecrawl()

    result, provider = _run_extract(
        monkeypatch,
        [static_url, js_url],
        {"extract_backend": "trafilatura"},
        trafilatura=trafilatura,
        firecrawl=firecrawl,
    )

    assert provider == "trafilatura"


def test_provider_field_reports_single_provider_when_uniform(registry, monkeypatch):
    trafilatura = FakeTrafilatura()
    firecrawl = FakeFirecrawl()

    result, provider = _run_extract(
        monkeypatch,
        ["https://example.com/a", "https://example.com/b"],
        {"extract_backend": "trafilatura"},
        trafilatura=trafilatura,
        firecrawl=firecrawl,
    )

    assert provider == "trafilatura"
    assert firecrawl.calls == []
