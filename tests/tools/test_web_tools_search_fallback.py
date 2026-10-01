"""Tests for the web_search runtime fallback walk.

Covers the degradation path in ``tools.web_tools.web_search_tool``:

- A primary provider that errors OR returns ``success: True`` with zero
  results must trigger the fallback walk (regression: SearXNG instances
  whose scrape engines are all suspended answer success-empty, and the
  empty list used to propagate silently — the walk only fired on errors).
- The walk order is cost-first (``_FALLBACK_PREFERENCE``): free providers
  before paid/metered ones.
- When no fallback produces results, the primary's original response is
  returned unchanged (contract preserved — the agent sees the primary's
  real outcome, not an intermediate fallback's).
- A healthy primary never triggers the walk.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

import tools.web_tools
from agent.web_search_provider import WebSearchProvider


def _make_provider(
    name: str,
    *,
    results: List[Dict[str, Any]] | None = None,
    error: str | None = None,
    available: bool = True,
):
    """Build a fake search provider with a scripted response.

    Returns ``(provider, calls)`` — ``calls`` collects every query the
    dispatcher routed to this provider.
    """
    calls: List[str] = []

    class _Fake(WebSearchProvider):
        @property
        def name(self) -> str:
            return name

        @property
        def display_name(self) -> str:
            return name.title()

        def is_available(self) -> bool:
            return available

        def supports_search(self) -> bool:
            return True

        def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
            calls.append(query)
            if error is not None:
                return {"success": False, "error": error}
            return {"success": True, "data": {"web": list(results or [])}}

    return _Fake(), calls


def _result(title: str) -> Dict[str, Any]:
    return {"title": title, "url": "https://example.com", "description": "", "position": 1}


def _run_search_tool(primary_name: str, registry: Dict[str, Any]) -> Dict[str, Any]:
    """Call web_search_tool with the registry fully mocked.

    Mocks the keyless rescue to be disabled so the one-shot rescue path is
    skipped and only the cost-first fallback walk (our local patch) is tested.
    """
    with patch("tools.web_tools._get_search_backend", return_value=primary_name), \
         patch("agent.web_search_registry.get_provider", side_effect=lambda n: registry.get(n)), \
         patch("tools.web_tools._ensure_web_plugins_loaded"), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch.object(tools.web_tools._debug, "log_call"), \
         patch.object(tools.web_tools._debug, "save"), \
         patch("tools.web_tools_rescue._keyless_rescue_enabled", return_value=False):
        return json.loads(tools.web_tools.web_search_tool("test query", 5))


class TestSearchFallbackTrigger:
    """Both hard errors and success-with-zero-results trigger the walk."""

    def test_empty_results_trigger_fallback(self):
        searxng, searxng_calls = _make_provider("searxng", results=[])
        brave, brave_calls = _make_provider("brave-free", results=[_result("brave hit")])
        registry = {"searxng": searxng, "brave-free": brave}

        data = _run_search_tool("searxng", registry)

        assert data["data"]["web"] == [_result("brave hit")]
        assert len(brave_calls) == 1

    def test_primary_error_triggers_fallback(self):
        searxng, _ = _make_provider("searxng", error="SearXNG returned HTTP 400")
        brave, brave_calls = _make_provider("brave-free", results=[_result("brave hit")])
        registry = {"searxng": searxng, "brave-free": brave}

        data = _run_search_tool("searxng", registry)

        assert data["data"]["web"] == [_result("brave hit")]
        assert len(brave_calls) == 1

    def test_healthy_primary_skips_fallback(self):
        searxng, _ = _make_provider("searxng", results=[_result("primary hit")])
        brave, brave_calls = _make_provider("brave-free", results=[_result("brave hit")])
        registry = {"searxng": searxng, "brave-free": brave}

        data = _run_search_tool("searxng", registry)

        assert data["data"]["web"] == [_result("primary hit")]
        assert brave_calls == []


class TestFallbackOrdering:
    """The walk is cost-first, with one deliberate, documented exception.

    The owned, metered firecrawl pool sits BEFORE the exa/parallel keyless
    ring: with no EXA/PARALLEL keys those slots route into a shared anonymous
    tier (throttled, not a quota we own), while keyed firecrawl is reliable
    and metered. See the ``_FALLBACK_PREFERENCE`` comment for the rationale.
    """

    def test_fallback_preference_is_free_first(self):
        from agent.web_search_registry import _FALLBACK_PREFERENCE

        # Free self-hosted/keyless tiers must come before any keyed/metered
        # provider, EXCEPT that keyed firecrawl is intentionally placed before
        # the keyless ring (see class docstring) — firecrawl is the reliable,
        # owned pool; the ring is the throttled anonymous last resort.
        free_self_hosted = ("searxng", "brave-free", "ddgs")
        # Members of the anonymous keyless ring that participate in the
        # fallback walk (keenable is ring-only and not in the walk).
        keyless_ring = ("exa", "parallel")
        metered = ("tavily", "perplexity")
        assert all(name in _FALLBACK_PREFERENCE for name in free_self_hosted + keyless_ring + metered)
        # Free self-hosted before the keyless ring and metered tiers.
        last_free = max(_FALLBACK_PREFERENCE.index(n) for n in free_self_hosted)
        first_ring_or_metered = min(
            _FALLBACK_PREFERENCE.index(n) for n in keyless_ring + metered
        )
        assert last_free < first_ring_or_metered

    def test_keyed_firecrawl_before_keyless_ring(self):
        from agent.web_search_registry import _FALLBACK_PREFERENCE

        # Keyed firecrawl is consulted before the anonymous keyless ring so a
        # healthy owned pool serves before the throttled shared tier is reached.
        assert _FALLBACK_PREFERENCE.index("firecrawl") < _FALLBACK_PREFERENCE.index("exa")

    def test_free_fallback_wins_over_paid(self):
        searxng, _ = _make_provider("searxng", results=[])
        brave, brave_calls = _make_provider("brave-free", results=[_result("free hit")])
        firecrawl, firecrawl_calls = _make_provider("firecrawl", results=[_result("paid hit")])
        registry = {"searxng": searxng, "brave-free": brave, "firecrawl": firecrawl}

        data = _run_search_tool("searxng", registry)

        assert data["data"]["web"] == [_result("free hit")]
        assert len(brave_calls) == 1
        assert firecrawl_calls == []  # paid tier never consulted

    def test_paid_fallback_used_when_free_unavailable(self):
        searxng, _ = _make_provider("searxng", results=[])
        brave, _brave_calls = _make_provider("brave-free", results=[], available=False)
        firecrawl, _fc_calls = _make_provider("firecrawl", results=[_result("paid hit")])
        registry = {"searxng": searxng, "brave-free": brave, "firecrawl": firecrawl}

        data = _run_search_tool("searxng", registry)

        assert data["data"]["web"] == [_result("paid hit")]

    def test_empty_free_fallback_is_skipped_not_accepted(self):
        searxng, _ = _make_provider("searxng", results=[])
        brave, brave_calls = _make_provider("brave-free", results=[])  # available but also empty
        firecrawl, fc_calls = _make_provider("firecrawl", results=[_result("paid hit")])
        registry = {"searxng": searxng, "brave-free": brave, "firecrawl": firecrawl}

        data = _run_search_tool("searxng", registry)

        assert data["data"]["web"] == [_result("paid hit")]
        assert len(brave_calls) == 1
        assert len(fc_calls) == 1

    def test_keyed_firecrawl_walked_before_keyless_ring(self):
        """Regression for the firecrawl-before-exa/parallel reorder: with no
        EXA/PARALLEL keys, the exa/parallel slots would route into the shared
        anonymous keyless ring. The owned, metered firecrawl pool must be
        consulted first — and the ring must never be reached when it serves."""
        searxng, _ = _make_provider("searxng", results=[])
        brave, _ = _make_provider("brave-free", error="402 Payment Required")
        ddgs, _ = _make_provider("ddgs", available=False)  # not installed
        firecrawl, fc_calls = _make_provider("firecrawl", results=[_result("keyed hit")])
        exa, exa_calls = _make_provider("exa", results=[_result("ring hit")])
        registry = {"searxng": searxng, "brave-free": brave, "ddgs": ddgs,
                    "firecrawl": firecrawl, "exa": exa}

        data = _run_search_tool("searxng", registry)

        assert data["data"]["web"] == [_result("keyed hit")]
        assert len(fc_calls) == 1
        assert exa_calls == []  # keyless ring never consulted

    def test_keyless_ring_catches_when_keyed_firecrawl_fails(self):
        """The ring remains the last resort: if the keyed firecrawl pool is
        exhausted (402), the walk must still fall through to the keyless ring."""
        searxng, _ = _make_provider("searxng", results=[])
        brave, _ = _make_provider("brave-free", error="402 Payment Required")
        firecrawl, _ = _make_provider("firecrawl", error="402 Payment Required")
        exa, exa_calls = _make_provider("exa", results=[_result("ring hit")])
        registry = {"searxng": searxng, "brave-free": brave,
                    "firecrawl": firecrawl, "exa": exa}

        data = _run_search_tool("searxng", registry)

        assert data["data"]["web"] == [_result("ring hit")]
        assert len(exa_calls) == 1


class TestAllFallbacksExhausted:
    """When nothing yields results, the primary's response is preserved."""

    def test_all_empty_returns_primary_response(self):
        searxng, _ = _make_provider("searxng", results=[])
        brave, _brave_calls = _make_provider("brave-free", results=[])
        registry = {"searxng": searxng, "brave-free": brave}

        data = _run_search_tool("searxng", registry)

        assert data == {"success": True, "data": {"web": []}}

    def test_all_failed_returns_primary_error(self):
        searxng, _ = _make_provider("searxng", error="primary boom")
        brave, _brave_calls = _make_provider("brave-free", error="fallback boom")
        registry = {"searxng": searxng, "brave-free": brave}

        data = _run_search_tool("searxng", registry)

        assert data["success"] is False
        assert data["error"] == "primary boom"
