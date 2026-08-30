"""Custom / Ollama (local) provider profile.

Covers any endpoint registered as provider="custom", including local
Ollama instances and OpenAI-compatible reasoning endpoints (GLM-5.2 on
Volcengine ARK, vLLM, llama.cpp). Key quirks:
  - ollama_num_ctx → extra_body.options.num_ctx (local context window)
  - reasoning_config disabled → top-level reasoning_effort="none"
    (Ollama /v1/chat/completions ignores think=False — ollama#14820)
    + extra_body.think = False only on Ollama URLs (/api/chat and proxies)
  - reasoning_config enabled + effort → top-level reasoning_effort
    (the native OpenAI-compatible format GLM/ARK expect; unset omits it
    so the endpoint's server default applies)
"""

from typing import Any
from urllib.parse import urlparse

from providers import register_provider
from providers.base import ProviderProfile


def _looks_like_ollama_endpoint(base_url: str | None) -> bool:
    """True when ``base_url`` is an Ollama host, not a generic OpenAI-compat relay.

    ``think`` is an Ollama-native extra_body field. Strict hosts (Mistral
    ``extra=forbid``, Groq, …) reject it with HTTP 422. Match only explicit
    Ollama signatures — default port 11434, or ``ollama`` as a hostname
    label — not arbitrary localhost (llama.cpp / vLLM / LM Studio).
    """
    raw = (base_url or "").strip()
    if not raw:
        return False
    parsed = urlparse(raw if "://" in raw else f"//{raw}")
    # urlparse raises ValueError for non-integer / out-of-range ports
    # ("http://host:99999/v1" parses fine in the OpenAI client, so the URL
    # is reachable here). Treat a malformed port as "not Ollama" instead of
    # killing the whole kwargs build — same try/except shape the 11434
    # check in hermes_cli/models.py uses, not the same detection logic.
    try:
        if parsed.port == 11434:
            return True
    except ValueError:
        return False
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        return False
    if host == "ollama.com" or host.endswith(".ollama.com"):
        return True
    return "ollama" in host.split(".")


class CustomProfile(ProviderProfile):
    """Custom/Ollama local provider — think=false and num_ctx support."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        ollama_num_ctx: int | None = None,
        **ctx: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        # Ollama context window
        if ollama_num_ctx:
            options = extra_body.get("options", {})
            options["num_ctx"] = ollama_num_ctx
            extra_body["options"] = options

        # Reasoning / thinking control for custom OpenAI-compatible endpoints
        # (GLM-5.2 on Volcengine ARK, vLLM, Ollama, llama.cpp, …).
        #
#   - disabled  → top-level reasoning_effort="none"; extra_body.think
        #     = False only on Ollama URLs (Ollama's thinking-off flag).
        #     For detected vLLM / llama.cpp also emit
        #     chat_template_kwargs.enable_thinking = False — the ONLY key those
        #     backends honor to turn Qwen3-style reasoning off (they ignore
        #     ``think``; see vLLM reasoning-outputs docs).
        #   - enabled + effort set → TOP-LEVEL reasoning_effort string, the
        #     format GLM-5.2/ARK and other OpenAI-compatible reasoning APIs
        #     expect (GLM documents "high" and "max"; "max" is its default).
        #     For detected vLLM / llama.cpp also emit enable_thinking = True so
        #     the chat template actually produces reasoning content.
        #   - enabled + no effort  → omit both, so the endpoint applies its own
        #     server-side default (do NOT force a level the user didn't pick).
        #
# chat_template_kwargs is scoped to *detected* vLLM / llama.cpp only: it
        # is a chat-template concept those servers understand, and sending it to
        # GLM/ARK (also provider=custom) risks a 400. We deliberately do NOT
        # emit ``think=True`` on enable: it is an Ollama-only flag and thinking
        # is already server-default-on for these backends. The same constraint
        # applies to ``think=False`` on disable — Mistral/Groq reject unknown
        # fields (HTTP 422 extra_forbidden), so that flag stays Ollama-URL-gated.
        if reasoning_config and isinstance(reasoning_config, dict):
            _effort = (reasoning_config.get("effort") or "").strip().lower()
            _enabled = reasoning_config.get("enabled", True)

            # Detect vLLM / llama.cpp so chat-template flags only go to backends
            # that understand them. base_url is passed by the chat_completions
            # transport (see agent/transports/chat_completions.py). The probe is
            # a lightweight local /version + /models GET (~8ms warm on a LAN box)
            # and is best-effort: any failure leaves _server_type None so we fall
            # back to the safe GLM/ARK path (no chat_template_kwargs emitted).
            _base_url = ctx.get("base_url") or self.base_url or ""
            _server_type = None
            if _base_url:
                try:
                    from agent.model_metadata import detect_local_server_type
                    _server_type = detect_local_server_type(_base_url)
                except Exception:
                    _server_type = None
            _templated = _server_type in ("vllm", "llamacpp")

            if _effort == "none" or _enabled is False:
                # Ollama's /v1/chat/completions silently ignores
                # extra_body.think (only /api/chat honours it — ollama#14820)
                # but respects the top-level reasoning_effort field (#25758).
                # Always emit reasoning_effort="none"; only add think=False
                # when the URL is actually Ollama.
                top_level["reasoning_effort"] = "none"
                if _looks_like_ollama_endpoint(ctx.get("base_url")):
                    extra_body["think"] = False
                if _templated:
                    extra_body["chat_template_kwargs"] = {"enable_thinking": False}
            elif _effort:
# Templated backends (vLLM / llama.cpp) only accept the OpenAI 4-level
                # enum (none/low/medium/high) — they reject Hermes-only
                # levels (minimal/xhigh/max/ultra) with a non-retryable 400.
                # Clamp via the shared policy using a narrower supported set
                # that matches what these backends actually honor.
                # GLM/ARK and SGLang accept the wider OpenAI-compatible
                # vocabulary (including max) — use the full set for them
                # (forwarding "ultra" verbatim is a guaranteed 400, #89503).
                from agent.reasoning_effort import clamp_effort

                if _templated:
                    _supported = ("none", "low", "medium", "high")
                else:
                    from agent.reasoning_effort import (
                        OPENAI_COMPAT_WIRE_EFFORTS as _supported,
                    )

                top_level["reasoning_effort"] = clamp_effort(
                    _effort, _supported
                )
                if _templated:
                    extra_body["chat_template_kwargs"] = {"enable_thinking": True}

        return extra_body, top_level

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Custom/Ollama: base_url is user-configured; fetch if set."""
        if not (base_url or self.base_url):
            return None
        return super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)


custom = CustomProfile(
    name="custom",
    aliases=(
        "ollama",
        "local",
        "vllm",
        "llamacpp",
        "llama.cpp",
        "llama-cpp",
    ),
    env_vars=(),  # No fixed key — custom endpoint
    base_url="",  # User-configured
    # Without this, no max_tokens is sent and Ollama falls back to its internal
    # num_predict=128, truncating responses after a few tokens (#39281). This is
    # only a floor used when the user hasn't set model.max_tokens — they can
    # override per-model — so we set it generously rather than lowballing it.
    default_max_tokens=65536,
)

register_provider(custom)
