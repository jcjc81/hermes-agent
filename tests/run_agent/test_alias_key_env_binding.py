"""A ``model_aliases`` credential bound via ``key_env`` must stay on that key.

An alias like::

    model_aliases:
      o2:
        model: claude-opus-5-5
        provider: anthropic
        key_env: ANTHROPIC_API_KEY_2

selects a *different* Anthropic account than the provider default
(``ANTHROPIC_API_KEY``). Three paths used to swap the provider-default key back
in, silently billing the wrong account while every response still looked fine:

1. the per-request credential refresh (``resolve_anthropic_token()`` →
   ``ANTHROPIC_API_KEY``), run before every Anthropic call;
2. gateway restart rehydration (only model/provider/base_url were persisted,
   so the key was re-resolved by provider);
3. credential-pool rotation on 429/billing errors (the pool is seeded from the
   provider-default env vars).

The session is bound to the env var NAME (never the value): refresh re-reads
only that var, so rotating the alias key in ``.env`` still takes effect.
"""

from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent

ALIAS_ENV = "ALIAS_TEST_ANTHROPIC_KEY"
ALIAS_KEY = "sk-ant-api03-alias-account-AAAA"
ALIAS_KEY_ROTATED = "sk-ant-api03-alias-account-BBBB"
DEFAULT_KEY = "sk-ant-api03-default-account-ZZZZ"


def _native_anthropic_agent(key_env: str = "") -> AIAgent:
    """Agent on native Anthropic holding the alias key (bare, no network)."""
    agent = AIAgent.__new__(AIAgent)
    agent.provider = "anthropic"
    agent.model = "claude-opus-5-5"
    agent.api_mode = "anthropic_messages"
    agent.base_url = "https://api.anthropic.com/v1"
    agent.api_key = ALIAS_KEY
    agent._anthropic_api_key = ALIAS_KEY
    agent._anthropic_base_url = "https://api.anthropic.com/v1"
    agent._anthropic_client = MagicMock(name="AliasClient")
    agent._is_anthropic_oauth = False
    agent._api_key_env = key_env
    return agent


@pytest.fixture
def both_keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", DEFAULT_KEY)
    monkeypatch.setenv(ALIAS_ENV, ALIAS_KEY)
    for var in ("ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)


class TestPerRequestRefresh:
    def test_refresh_keeps_alias_key_when_default_differs(self, both_keys):
        """Path 1: refresh must not swap ANTHROPIC_API_KEY into an alias-bound session."""
        agent = _native_anthropic_agent(key_env=ALIAS_ENV)
        original_client = agent._anthropic_client

        with patch("agent.anthropic_credentials.read_claude_code_credentials", return_value=None), \
                patch("agent.anthropic_adapter.build_anthropic_client") as rebuild:
            refreshed = agent._try_refresh_anthropic_client_credentials()

        assert refreshed is False
        assert agent._anthropic_api_key == ALIAS_KEY
        assert agent._anthropic_client is original_client
        rebuild.assert_not_called()

    def test_refresh_picks_up_rotated_alias_key(self, both_keys, monkeypatch):
        """Rotation still works: a new value in the BOUND var rebuilds the client with it."""
        agent = _native_anthropic_agent(key_env=ALIAS_ENV)
        monkeypatch.setenv(ALIAS_ENV, ALIAS_KEY_ROTATED)
        new_client = MagicMock(name="RotatedClient")

        with patch("agent.anthropic_adapter.build_anthropic_client", return_value=new_client) as rebuild:
            refreshed = agent._try_refresh_anthropic_client_credentials()

        assert refreshed is True
        assert agent._anthropic_api_key == ALIAS_KEY_ROTATED
        assert agent.api_key == ALIAS_KEY_ROTATED  # auxiliary calls (compression, titles) follow
        assert agent._anthropic_client is new_client
        assert rebuild.call_args.args[0] == ALIAS_KEY_ROTATED

    def test_refresh_ignores_empty_bound_var(self, both_keys, monkeypatch):
        """An emptied bound var keeps the current key — never falls back to the default."""
        agent = _native_anthropic_agent(key_env=ALIAS_ENV)
        monkeypatch.delenv(ALIAS_ENV)

        with patch("agent.anthropic_adapter.build_anthropic_client") as rebuild:
            assert agent._try_refresh_anthropic_client_credentials() is False

        assert agent._anthropic_api_key == ALIAS_KEY
        rebuild.assert_not_called()

    def test_unbound_session_keeps_default_refresh(self, both_keys):
        """No key_env (plain `o` / default model): existing refresh behaviour is unchanged."""
        agent = _native_anthropic_agent(key_env="")
        new_client = MagicMock(name="DefaultClient")

        with patch("agent.anthropic_credentials.read_claude_code_credentials", return_value=None), \
                patch("agent.anthropic_adapter.build_anthropic_client", return_value=new_client):
            assert agent._try_refresh_anthropic_client_credentials() is True

        assert agent._anthropic_api_key == DEFAULT_KEY


class TestSwitchModelBinding:
    def test_switch_model_binds_key_env_and_drops_provider_pool(self, both_keys):
        """Path 3: an alias-bound switch carries the var name and no provider credential pool."""
        agent = _native_anthropic_agent(key_env="")
        agent.client = None
        agent._client_kwargs = {}
        agent.context_compressor = None
        agent._cached_system_prompt = "cached"
        agent._primary_runtime = {}
        agent._fallback_activated = False
        agent._fallback_index = 0
        agent._fallback_chain = []
        agent._fallback_model = None
        agent._config_context_length = None
        agent.runtime_capabilities = {}
        agent._credential_pool = MagicMock(name="ProviderDefaultPool", provider="anthropic")
        agent._credential_pool_entry_id = "entry-1"

        with patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()), \
                patch("agent.credential_pool.load_pool") as load_pool:
            agent.switch_model(
                new_model="claude-opus-5-5", new_provider="anthropic", api_key=ALIAS_KEY,
                base_url="https://api.anthropic.com/v1", api_mode="anthropic_messages",
                api_key_env=ALIAS_ENV,
            )

        assert agent._api_key_env == ALIAS_ENV
        assert agent._credential_pool is None
        load_pool.assert_not_called()
        assert agent._primary_runtime.get("api_key_env") == ALIAS_ENV
