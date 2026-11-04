"""LOCAL PATCH (live commit 09dc1e127b): auxiliary calls on a ``model_aliases`` credential never retry
with another credential.

A ``model_aliases`` entry with its own key (``key_env`` / ``api_key``) selects a specific account, e.g.
a second Anthropic org. Auxiliary calls in that session run on the same key. When one fails, the
recovery ladder's credential refresh (401), pool rotation (429/billing) and provider fallback each
re-resolve the provider without that key, i.e. with the provider-default credential: a different
account, billed silently. ``_aux_recovery_ladder`` skips those rungs for such a key.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def is_alias_bound_credential(api_key: Any) -> bool:
    """True when ``api_key`` is the credential a ``model_aliases`` entry declares (``key_env`` /
    ``api_key``) in the active profile. Matched by value: aux calls carry the session's key, not the
    alias name. Only consulted on the failure path."""
    if not isinstance(api_key, str) or not api_key.strip():
        return False
    try:
        from hermes_cli.model_switch import DIRECT_ALIASES, _ensure_direct_aliases, direct_alias_api_key
        _ensure_direct_aliases()
        return any(
            (alias.api_key or alias.key_env) and direct_alias_api_key(alias) == api_key.strip()
            for alias in list(DIRECT_ALIASES.values())
        )
    except Exception:
        logger.debug("model_aliases credential lookup failed", exc_info=True)
        return False
