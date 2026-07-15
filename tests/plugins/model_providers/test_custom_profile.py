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
                          for GLM/ARK/Ollama/unknown backends, including
                          ``max``/``xhigh``
  - vLLM/llama.cpp      → clamp xhigh/max → high, minimal → low (both
    detected            reject Hermes-only levels with HTTP 400); emit
                          chat_template_kwargs.enable_thinking (True/False) —
                          the only key those backends honor to actually
                          turn Qwen3-style reasoning on/off
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


class TestVLLMLlamaCppClamp:
    """Detected vLLM/llama.cpp: clamp Hermes-only effort levels to the
    OpenAI-compatible enum and emit chat_template_kwargs.enable_thinking.

    Regression guard: without the clamp, xhigh/max/minimal → HTTP 400 on
    vLLM (confirmed against a live 192.168.12.129:8000 Qwen3.6-27B server).
    Without chat_template_kwargs, reasoning_effort alone does not make the
    chat template actually emit reasoning content.
    """

    @pytest.fixture
    def profile(self):
        import model_tools  # noqa: F401
        import providers

        return providers.get_provider_profile("custom")

    @pytest.mark.parametrize("server_type", ["vllm", "llamacpp"])
    @pytest.mark.parametrize(
        "effort,expected",
        [
            ("minimal", "low"),
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            ("xhigh", "high"),
            ("max", "high"),
        ],
    )
    def test_clamps_hermes_levels(
        self, profile, monkeypatch, server_type, effort, expected
    ):
        """Both vLLM and llama.cpp clamp xhigh/max → high, minimal → low,
        and both get enable_thinking=True — chat_template_kwargs is a
        chat-template concept both backends understand."""
        monkeypatch.setattr(
            "agent.model_metadata.detect_local_server_type",
            lambda *a, **k: server_type,
        )
        eb, tl = profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort},
            base_url="http://192.168.12.129:8000/v1",
            model="qwen3",
        )
        assert tl == {"reasoning_effort": expected}
        assert eb["chat_template_kwargs"] == {"enable_thinking": True}

    @pytest.mark.parametrize("server_type", ["vllm", "llamacpp"])
    def test_disabled_emits_enable_thinking_false(
        self, profile, monkeypatch, server_type
    ):
        """When reasoning is disabled, detected vLLM/llama.cpp get both
        think=False (harmless no-op there) AND enable_thinking=False
        (the key they actually honor)."""
        monkeypatch.setattr(
            "agent.model_metadata.detect_local_server_type",
            lambda *a, **k: server_type,
        )
        eb, tl = profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            base_url="http://192.168.12.129:8000/v1",
            model="qwen3",
        )
        assert eb == {"think": False, "chat_template_kwargs": {"enable_thinking": False}}
        assert tl == {}

    @pytest.mark.parametrize("server_type", ["vllm", "llamacpp"])
    def test_unrecognized_effort_defaults_to_high(
        self, profile, monkeypatch, server_type
    ):
        """An effort string not in the clamp table falls back to 'high'
        (the safe/most-capable default) rather than passing through an
        arbitrary unvalidated string to a backend known to 400 on it."""
        monkeypatch.setattr(
            "agent.model_metadata.detect_local_server_type",
            lambda *a, **k: server_type,
        )
        eb, tl = profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "ultra"},
            base_url="http://192.168.12.129:8000/v1",
            model="qwen3",
        )
        assert tl == {"reasoning_effort": "high"}
        assert eb["chat_template_kwargs"] == {"enable_thinking": True}


class TestNonTemplatedCustomPassthrough:
    """GLM/ARK, Ollama, and unknown/undetected backends: reasoning_effort
    passes through verbatim, no chat_template_kwargs emitted.

    Regression guard: GLM-5.2/ARK natively accept max/xhigh and would break
    if clamped; Ollama doesn't understand chat_template_kwargs.
    """

    @pytest.fixture
    def profile(self):
        import model_tools  # noqa: F401
        import providers

        return providers.get_provider_profile("custom")

    @pytest.mark.parametrize(
        "server_type,effort",
        [
            (None, "xhigh"),
            (None, "max"),
            ("ollama", "xhigh"),
            ("unknown-backend", "minimal"),
        ],
    )
    def test_passthrough_when_not_templated(
        self, profile, monkeypatch, server_type, effort
    ):
        monkeypatch.setattr(
            "agent.model_metadata.detect_local_server_type",
            lambda *a, **k: server_type,
        )
        eb, tl = profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort},
            base_url="http://example.com/v1",
            model="glm-5.2",
        )
        assert tl == {"reasoning_effort": effort}
        assert "chat_template_kwargs" not in eb

    def test_no_base_url_bypasses_detection(self, profile):
        """No base_url and no profile-level default → detection skipped
        entirely → verbatim passthrough (matches the pre-existing tests in
        TestCustomReasoningWireShape which never pass base_url)."""
        eb, tl = profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "xhigh"},
            model="glm-5.2",
        )
        assert tl == {"reasoning_effort": "xhigh"}
        assert "chat_template_kwargs" not in eb

    def test_detection_failure_falls_back_to_passthrough(self, profile, monkeypatch):
        """If detect_local_server_type raises (network blip, timeout), the
        best-effort try/except must leave _server_type None, NOT propagate
        the exception — a transient probe failure must never crash a
        chat-completion request."""
        def _raise(*a, **k):
            raise TimeoutError("probe timed out")

        monkeypatch.setattr(
            "agent.model_metadata.detect_local_server_type", _raise
        )
        eb, tl = profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "xhigh"},
            base_url="http://192.168.12.129:8000/v1",
            model="qwen3",
        )
        assert tl == {"reasoning_effort": "xhigh"}
        assert "chat_template_kwargs" not in eb
