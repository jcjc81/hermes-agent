"""Trafilatura extract-only plugin — bundled, auto-loaded."""

from __future__ import annotations

from plugins.web.trafilatura.provider import TrafilaturaExtractProvider


def register(ctx) -> None:
    """Register the Trafilatura provider with the plugin context."""
    ctx.register_web_search_provider(TrafilaturaExtractProvider())
