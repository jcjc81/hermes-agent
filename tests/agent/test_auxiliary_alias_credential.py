"""An auxiliary call (compression, titles, ...) that fails on a ``model_aliases`` credential must never
be retried with a different credential.

A ``model_aliases`` entry with its own key (``key_env`` / ``api_key``) selects a specific account, e.g.
a second Anthropic org. Auxiliary calls in that session run on the same key. When one fails, the
recovery ladder's credential refresh (401), pool rotation (429/billing) and provider fallback each
re-resolve the provider without that key, i.e. with the provider-default credential: a different
account, billed silently. The failure must surface instead.
"""

import pytest

ALIAS_ENV = "ALIAS_TEST_ANTHROPIC_KEY"
ALIAS_KEY = "sk-ant-api03-alias-AAAA"
DEFAULT_KEY = "sk-ant-api03-default-ZZZZ"
_ERRORS = {401: "invalid x-api-key", 429: "rate_limit_error: too many requests"}


class _StatusError(Exception):
    def __init__(self, status: int):
        super().__init__(f"Error code: {status} - {_ERRORS[status]}")
        self.status_code = status


@pytest.fixture
def anthropic_wire(monkeypatch, tmp_path):
    """Native-Anthropic aux route whose every request fails with ``wire.status``; records keys sent."""
    import agent.anthropic_adapter as anthropic_adapter
    import agent.auxiliary_client as aux
    import hermes_cli.model_switch as model_switch

    hermes_home = tmp_path / "hermes_home"
    hermes_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / "config.yaml").write_text(
        "model_aliases:\n"
        "  o2:\n"
        "    model: claude-opus-5-5\n"
        "    provider: anthropic\n"
        "    base_url: https://api.anthropic.com/v1\n"
        f"    key_env: {ALIAS_ENV}\n"
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", DEFAULT_KEY)
    monkeypatch.setenv(ALIAS_ENV, ALIAS_KEY)
    monkeypatch.setattr(model_switch, "DIRECT_ALIASES", {})  # load from this HERMES_HOME's config
    monkeypatch.setattr(aux, "_client_cache", {})
    monkeypatch.setattr(aux, "_aux_unhealthy_until", {})

    class Wire:
        status = 401
        sent: list = []

    class FakeMessages:
        def __init__(self, key):
            self.key = key

        def create(self, **_):
            Wire.sent.append(self.key)
            raise _StatusError(Wire.status)

    class FakeAnthropic:
        def __init__(self, key, base_url=None):
            self.api_key, self.base_url, self.messages = key, base_url, FakeMessages(key)

        def close(self):
            pass

    Wire.sent = []
    monkeypatch.setattr(anthropic_adapter, "build_anthropic_client",
                        lambda key, base_url=None, *a, **k: FakeAnthropic(key, base_url))
    return Wire


def _compress_on(api_key: str):
    from agent.auxiliary_client import call_llm
    # The shape ContextCompressor passes: the session's main runtime, including its key.
    main_runtime = {"model": "claude-opus-5-5", "provider": "anthropic", "base_url": "https://api.anthropic.com/v1",
                    "api_key": api_key, "api_mode": "anthropic_messages"}
    call_llm(task="compression", main_runtime=main_runtime,
             messages=[{"role": "user", "content": "summarise"}], max_tokens=16)


@pytest.mark.parametrize("status", [401, 429])
def test_alias_credential_failure_is_never_retried_with_another_key(anthropic_wire, status):
    anthropic_wire.status = status
    with pytest.raises(_StatusError) as excinfo:
        _compress_on(ALIAS_KEY)
    assert excinfo.value.status_code == status
    assert anthropic_wire.sent, "the aux call must reach the alias endpoint"
    assert set(anthropic_wire.sent) == {ALIAS_KEY}, (
        f"recovery after HTTP {status} sent a different credential: "
        f"{[k[-4:] for k in anthropic_wire.sent]} (alias key ends AAAA)")


def test_non_alias_credential_keeps_auth_refresh_recovery(anthropic_wire):
    """The guard is scoped to alias credentials: a provider-default key still gets the 401 refresh retry."""
    anthropic_wire.status = 401
    with pytest.raises(_StatusError):
        _compress_on(DEFAULT_KEY)
    assert len(anthropic_wire.sent) >= 2, "a 401 on a non-alias key must still be retried after a credential refresh"
