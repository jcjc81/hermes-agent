"""LOCAL PATCH tests: a ``model_aliases`` key_env override survives a gateway restart on its own key.

Ported from live commit 500f5bed363 (tests/gateway/test_session_model_override_persistence.py) onto
upstream PR #148256's implementation, plus a carry-over test for overrides already persisted by live's
code. An alias with ``key_env: X`` bills a different account than the provider default. Before the fix
only model/provider/base_url were persisted, so the restart path re-resolved credentials BY PROVIDER and
silently swapped in the provider-default key (and its credential pool).
"""

import json
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore

ALIAS_ENV = "ALIAS_TEST_ANTHROPIC_KEY"
ALIAS_KEY = "sk-ant...AAAA"
DEFAULT_KEY = "sk-ant...ZZZZ"

ALIAS_OVERRIDE = {
    "model": "claude-opus-5-5",
    "provider": "anthropic",
    "api_key": ALIAS_KEY,
    "base_url": "https://api.anthropic.com/v1",
    "api_mode": "anthropic_messages",
    "key_env": ALIAS_ENV,
}

_PROVIDER_DEFAULT_RUNTIME = {
    "api_key": DEFAULT_KEY,
    "api_mode": "anthropic_messages",
    "base_url": "https://api.anthropic.com",
    "provider": "anthropic",
    "requested_provider": "anthropic",
    "credential_pool": object(),  # provider-default pool (seeded from ANTHROPIC_API_KEY)
}


def _make_source() -> SessionSource:
    return SessionSource(platform=Platform.TELEGRAM, user_id="u1", chat_id="c1", user_name="tester", chat_type="dm")


@pytest.fixture
def stores(tmp_path, monkeypatch):
    """Build SessionStores over a shared sessions dir, without SQLite."""

    def _raise():
        raise RuntimeError("SQLite disabled in test")

    import hermes_state

    monkeypatch.setattr(hermes_state, "SessionDB", _raise)

    def _make() -> SessionStore:
        store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
        assert store._db is None
        return store

    return _make


def _make_runner(store):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {}
    runner.session_store = store
    return runner


def _restart_and_resolve(stores, session_key):
    runner = _make_runner(stores())  # simulated restart
    with patch("gateway.run._resolve_runtime_agent_kwargs_for_provider", return_value=dict(_PROVIDER_DEFAULT_RUNTIME)), \
            patch("gateway.run._credential_pool_for_provider", return_value=_PROVIDER_DEFAULT_RUNTIME["credential_pool"]):
        model, runtime = runner._resolve_session_agent_runtime(
            session_key=session_key, user_config={"model": {"default": "global-model"}},
        )
        route = runner._resolve_turn_agent_config("", model, runtime)
    return model, runtime, route


def test_alias_key_env_persists_as_name_never_value(stores, tmp_path):
    store = stores()
    entry = store.get_or_create_session(_make_source())
    store.set_model_override(entry.session_key, ALIAS_OVERRIDE)

    raw = (tmp_path / "sessions.json").read_text(encoding="utf-8")
    assert ALIAS_KEY not in raw
    assert stores().get_model_override(entry.session_key)["key_env"] == ALIAS_ENV


def test_restart_rehydrates_alias_key_not_provider_default(stores, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", DEFAULT_KEY)
    monkeypatch.setenv(ALIAS_ENV, ALIAS_KEY)
    store = stores()
    session_key = store.get_or_create_session(_make_source()).session_key
    store.set_model_override(session_key, ALIAS_OVERRIDE)

    model, runtime, route = _restart_and_resolve(stores, session_key)

    assert model == "claude-opus-5-5"
    assert runtime["api_key"] == ALIAS_KEY
    assert runtime["credential_pool"] is None
    # The binding must reach AIAgent(**route["runtime"]) or its refresh re-resolves the default.
    assert route["runtime"]["api_key"] == ALIAS_KEY
    assert route["runtime"]["api_key_env"] == ALIAS_ENV
    assert route["runtime"]["credential_pool"] is None


def test_restart_with_unset_alias_var_fails_loudly(stores, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", DEFAULT_KEY)
    monkeypatch.delenv(ALIAS_ENV, raising=False)
    store = stores()
    session_key = store.get_or_create_session(_make_source()).session_key
    store.set_model_override(session_key, ALIAS_OVERRIDE)

    runner = _make_runner(stores())
    with patch("gateway.run._resolve_runtime_agent_kwargs_for_provider", return_value=dict(_PROVIDER_DEFAULT_RUNTIME)), \
            pytest.raises(RuntimeError, match=ALIAS_ENV):
        runner._resolve_session_agent_runtime(
            session_key=session_key, user_config={"model": {"default": "global-model"}},
        )


def test_override_written_by_previous_release_keeps_its_key(stores, tmp_path, monkeypatch):
    """Carry-over: sessions.json already holds overrides written by the previous local release (live
    commit 500f5bed363) in exactly this shape: model/provider/base_url/key_env, no key value, no
    api_mode. After the upgrade such a session must still send its alias key, never the default."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", DEFAULT_KEY)
    monkeypatch.setenv(ALIAS_ENV, ALIAS_KEY)
    store = stores()
    session_key = store.get_or_create_session(_make_source()).session_key
    store.set_model_override(session_key, {"model": "placeholder", "provider": "openai", "base_url": "https://x.example/v1"})

    path = tmp_path / "sessions.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = [v for v in data.values() if isinstance(v, dict) and "model_override" in v]
    assert len(entries) == 1, "sessions.json layout changed: locate the entry's model_override"
    entries[0]["model_override"] = {
        "base_url": "https://api.anthropic.com/v1", "key_env": ALIAS_ENV,
        "model": "claude-opus-5-5", "provider": "anthropic",
    }
    path.write_text(json.dumps(data), encoding="utf-8")

    model, _runtime, route = _restart_and_resolve(stores, session_key)

    assert model == "claude-opus-5-5"
    assert (route["runtime"]["api_key"], route["runtime"]["credential_pool"], route["runtime"].get("api_key_env")) == (
        ALIAS_KEY, None, ALIAS_ENV)
