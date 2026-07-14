"""Unit tests for the custom provider profile's reasoning wiring.

``provider=custom`` covers any OpenAI-compatible endpoint the user points
Hermes at — local Ollama, vLLM, llama.cpp, and hosted reasoning APIs like
GLM-5.2 on Volcengine ARK. Before #57601's salvage, ``CustomProfile`` emitted
nothing when reasoning was *enabled*, so a configured ``reasoning_effort``
was silently dropped for every custom endpoint.

These tests pin the wire-shape contract:
  - disabled            → extra_body.think = False
  - enabled + effort    → top-level reasoning_effort (native OpenAI-compat
                          format GLM/ARK expect), passed through verbatim
                          including ``max``/``xhigh``
  - enabled + no effort → nothing emitted (endpoint's server default applies)
  - ollama_num_ctx      → extra_body.options.num_ctx, orthogonal to reasoning
"""

from __future__ import annotations

import pytest


@pytest.fixture
def custom_profile():
    """Resolve the registered custom profile via the global registry.

    Importing ``model_tools`` triggers plugin discovery, which registers the
    ``custom`` profile. Going through ``get_provider_profile`` keeps the test
    honest — if the registered class is ever downgraded to a plain
    ``ProviderProfile``, the assertions below collapse.
    """
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("custom")
    assert profile is not None, "custom provider profile must be registered"
    return profile


class TestCustomReasoningWireShape:
    """``build_api_kwargs_extras`` produces the correct wire format."""

    def test_no_reasoning_config_emits_nothing(self, custom_profile):
        """Unset reasoning → omit everything so the endpoint's default applies."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config=None, model="glm-5.2"
        )
        assert eb == {}
        assert tl == {}

    def test_disabled_sends_think_false(self, custom_profile):
        """enabled=False → extra_body.think = False (Ollama thinking-off flag)."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model="glm-5.2"
        )
        assert eb == {"think": False}
        assert tl == {}

    def test_effort_none_sends_think_false(self, custom_profile):
        """effort='none' is the disable alias → think=False, no effort."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "none"}, model="glm-5.2"
        )
        assert eb == {"think": False}
        assert tl == {}

    @pytest.mark.parametrize(
        "effort", ["minimal", "low", "medium", "high", "xhigh", "max"]
    )
    def test_enabled_effort_goes_top_level(self, custom_profile, effort):
        """enabled + effort → TOP-LEVEL reasoning_effort, passed through verbatim.

        GLM-5.2/ARK and OpenAI-compatible reasoning APIs read reasoning_effort
        as a top-level string, not nested in extra_body. ``max`` is GLM's
        native deep-reasoning level and must survive.
        """
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort}, model="glm-5.2"
        )
        assert tl == {"reasoning_effort": effort}
        assert "reasoning_effort" not in eb
        assert "think" not in eb

    def test_enabled_without_effort_emits_nothing(self, custom_profile):
        """enabled but no effort → omit; do NOT force a level the user didn't pick."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True}, model="glm-5.2"
        )
        assert eb == {}
        assert tl == {}

    def test_does_not_force_think_true_on_enable(self, custom_profile):
        """We must never send think=True on enable — it's Ollama-only and
        would 400 on GLM/vLLM endpoints that don't recognize it."""
        eb, _ = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"}, model="glm-5.2"
        )
        assert eb.get("think") is not True


class TestCustomReasoningWithNumCtx:
    """Ollama num_ctx and reasoning are independent and compose."""

    def test_num_ctx_alone(self, custom_profile):
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config=None, ollama_num_ctx=8192, model="qwen3"
        )
        assert eb == {"options": {"num_ctx": 8192}}
        assert tl == {}

    def test_num_ctx_with_effort(self, custom_profile):
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            ollama_num_ctx=8192,
            model="qwen3",
        )
        assert eb == {"options": {"num_ctx": 8192}}
        assert tl == {"reasoning_effort": "high"}


class TestCustomSlugResolution:
    """get_provider_profile(resolve_custom_slug=True) maps a `custom:<name>`
    slug to the bundled `custom` ProviderProfile.

    Rationale: `agent.provider` stores the slug verbatim (e.g. `custom:vllm-5090`)
    but the registry only knows `custom` (+ aliases). Without stripping, the
    wire path takes its legacy branch and sends NO provider default, while the
    trading profile (bare `custom`) sends 65536 -- divergent behavior for the
    same backend. This is opt-in so the six other get_provider_profile callers
    (CLI model-picker dispatch in particular) keep their current behavior.
    """

    def test_default_off_slug_returns_none(self):
        """Backward-compat: without the flag, a slug still returns None."""
        import providers
        assert providers.get_provider_profile("custom:vllm-5090") is None

    def test_opt_in_slug_resolves_to_custom(self):
        import providers
        p = providers.get_provider_profile(
            "custom:vllm-5090", resolve_custom_slug=True
        )
        assert p is not None
        assert p.name == "custom"
        assert p.get_max_tokens("Qwen3.6-27B") == 65536

    def test_opt_in_bare_custom_unchanged(self):
        """Bare `custom` resolves with or without the flag."""
        import providers
        assert providers.get_provider_profile(
            "custom", resolve_custom_slug=True
        ).name == "custom"

    def test_opt_in_bare_user_name_still_none(self):
        """A bare user-config name (no `custom:` prefix) is NOT a slug -- the
        flag must not turn it into the custom profile. (Handled by config
        normalization, not code.)"""
        import providers
        assert providers.get_provider_profile(
            "vllm-5090", resolve_custom_slug=True
        ) is None

    def test_opt_in_non_custom_provider_unchanged(self):
        """The flag only affects the `custom:` prefix; a real provider still
        resolves to its own profile, and a `custom:` slug never shadows it."""
        import providers
        assert providers.get_provider_profile(
            "nvidia", resolve_custom_slug=True
        ).name == "nvidia"

    def test_opt_in_unknown_slug_body_resolves_to_custom(self):
        """Any `custom:<anything>` maps to the custom profile -- the suffix is a
        user label, not a registry key."""
        import providers
        p = providers.get_provider_profile(
            "custom:some-random-proxy", resolve_custom_slug=True
        )
        assert p is not None and p.name == "custom"
